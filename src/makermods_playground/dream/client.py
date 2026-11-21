"""Dream1 Beta / Beta Flash through the Dream API (the ``dream`` package of ``dreamlabs-client``).

The model reads, per arm, ``state.<side>_arm`` (zeros: the zero-shot models take no joints), ``_gripper`` and
``_grip`` (the opening in [0, 1]), ``_eef_xyz`` and ``_eef_rot6`` (the measured tool pose in that arm's base frame),
plus the prompt and a window of camera frames (9 frames at 5 Hz, kept by the client): ``video.overhead`` from the
top camera, ``video.wrist_l`` / ``video.wrist_r`` from the left / right wrist cameras, each stretched to 320x176
and sent as base64 JPEG, quality 85. It answers each arm's end-effector head (``action.<side>_eef_xyz``,
``_eef_rot6``, ``_grip``) in ``per_key``. ``robot: other_embodiment`` asks for that head; ``_record: False`` keeps the
ticks out of Dream Labs' recording service. Gateway and key come from ``~/.dreamlabs/credentials`` or the
environment (``DREAM_GATEWAY_URL``, ``DREAM_API_KEY``).
"""

from __future__ import annotations

import base64
import time
import warnings
from typing import Optional
from urllib.parse import urlsplit

import numpy as np

from .scheduling import ACTION_HZ, EXECUTION_HORIZON, SYNC_ABOVE_MS

MODELS = {"beta": "dreamlabs/dream-1-beta", "flash": "dreamlabs/dream-1-beta-flash"}
VIEWS = {"overhead": "top", "wrist_l": "left", "wrist_r": "right"}   # wire view -> robot camera
SIDES = ("left", "right")
PARTS = (("arm", 6), ("gripper", 1), ("eef_xyz", 3), ("eef_rot6", 6), ("grip", 1))
IMAGE_SIZE = (320, 176)
JPEG_QUALITY = 85
NO_KEY = "no Dream API key: set DREAM_API_KEY in ~/.dreamlabs/credentials or the environment"


class PolicyError(RuntimeError):
    """A run-ending policy problem whose text never includes an observation or the key."""


def model_id(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    if model in MODELS.values():
        return model
    raise PolicyError(f"unknown model {model!r}: choose {' or '.join(MODELS)}")


def encode_frame(image: np.ndarray) -> str:
    import cv2
    small = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise PolicyError("JPEG encoding failed")
    return base64.b64encode(jpeg).decode("ascii")


def credentials():
    from dream import credentials as c
    return c.load()


def embodiment(sides) -> dict:
    return {"state": {f"{side}_{part}": {"shape": [width], "dtype": "float32"} for side in sides
                      for part, width in PARTS},
            "video": {view: {} for view in VIEWS}, "action": {}}


def retry_delay(exc) -> Optional[float]:
    """Seconds to wait before asking again after a failed request, or None when asking again cannot help."""
    import dream
    if isinstance(exc, (dream.DreamAuthError, dream.DreamBadRequest)):
        return None
    if isinstance(exc, dream.DreamSessionExpired):
        return 0.0
    if isinstance(exc, dream.DreamServerError):
        if not exc.retryable:
            return None
        return max(0.5, exc.retry_after_s) if exc.retry_after_s else 1.0
    if isinstance(exc, (dream.DreamActionError, dream.DreamTimeout, dream.DreamProtocolError)):
        return 1.0
    return None


def starts_new_session(exc) -> bool:
    """Whether the retry after ``exc`` opens a new episode."""
    import dream
    if isinstance(exc, dream.DreamSessionExpired):
        return False
    if isinstance(exc, dream.DreamServerError):
        return getattr(exc, "reason", None) not in ("rate_limited", "queue_timeout")
    return not isinstance(exc, dream.DreamActionError)


class DreamPolicy:
    """One policy session per run; each task is one episode."""

    def __init__(self, model: str, sides, timeout_ms: int = 60000, client=None, cameras=None, parked=None):
        """``sides``: the robot's arms. ``cameras``: the robot's cameras (default: every view's). The model
        always reads both arms and all three views, so an arm the robot lacks is reported at ``parked[side]``
        (a 4x4 tool pose, gripper closed) and a camera it lacks as a black image."""
        self.model = model_id(model)
        self.sides = list(sides)
        if not self.sides or set(self.sides) - set(SIDES):
            raise PolicyError("the arms must be named left and right")
        self.parked = {side: np.asarray(pose, dtype=float) for side, pose in (parked or {}).items()}
        absent = [side for side in SIDES if side not in self.sides and side not in self.parked]
        if absent:
            raise PolicyError(f"the model reads both arms: give a parked tool pose for the {', '.join(absent)} arm")
        if client is None:
            import dream
            record = credentials()
            if not record.api_key:
                raise PolicyError(NO_KEY)
            client = dream.DreamClient()
        self.client = client
        self.client.policy({
            "checkpoint": self.model, "embodiment": embodiment(SIDES), "timeout_ms": timeout_ms,
            "action_hz": ACTION_HZ, "execution_horizon": EXECUTION_HORIZON, "sync_above_ms": SYNC_ABOVE_MS,
            "robot": "other_embodiment", "_record": False,
        })
        self.cameras = [c for c in VIEWS.values() if cameras is None or c in cameras]

    @property
    def session_id(self) -> Optional[str]:
        return self.client.session_id

    def observation(self, prompt: str, state: dict) -> dict:
        """``state``: per arm ``opening`` (float), ``tool`` (4, 4)."""
        obs = {"prompt": prompt}
        for side in SIDES:
            s = state[side] if side in self.sides else {"opening": 0.0, "tool": self.parked[side]}
            opening = float(np.clip(s["opening"], 0.0, 1.0))
            tool = np.asarray(s["tool"], dtype=float)
            obs[f"state.{side}_arm"] = np.zeros(6, np.float32)
            obs[f"state.{side}_gripper"] = np.array([opening], np.float32)
            obs[f"state.{side}_grip"] = np.array([opening], np.float32)
            obs[f"state.{side}_eef_xyz"] = tool[:3, 3].astype(np.float32)
            obs[f"state.{side}_eef_rot6"] = np.concatenate([tool[:3, 0], tool[:3, 1]]).astype(np.float32)
        return obs

    def push(self, images: dict, observation: dict, timestamp: float) -> None:
        """One camera tick: ``images`` per robot camera (BGR), encoded here."""
        frames = {}
        for view, camera in VIEWS.items():
            if camera in images:
                frames[f"video.{view}"] = encode_frame(images[camera])
            elif camera in self.cameras:
                raise PolicyError(f"no {camera} camera frame for the model's {view} view")
            else:
                frames[f"video.{view}"] = encode_frame(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8))
        self.client.set_observation(frames, observation, timestamp=timestamp)

    def reset_episode(self) -> None:
        self.client.reset_episode()

    def act(self) -> tuple[dict, dict]:
        """The reply's ``per_key`` and its timing (``rtt_ms`` at least)."""
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                action = self.client.get_action()
        finally:
            timing = dict(self.client.last_timing or {"rtt_ms": (time.perf_counter() - started) * 1000})
        per_key = action.get("per_key") if isinstance(action, dict) else None
        return (per_key if isinstance(per_key, dict) else {}), timing

    def close(self) -> None:
        self.client.close()


def public_url(url) -> Optional[str]:
    """Scheme, host and port of a URL: no userinfo, path, query or fragment."""
    if not url:
        return url
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "(not a valid URL)"
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{host}{port}" if parts.scheme and host else "(not a valid URL)"


def check(models=tuple(MODELS)) -> dict:
    """Key present and the models listed for this account (one call to the platform; nothing moves)."""
    import dream
    record = credentials()
    out = {"credentials": str(record.path), "gateway": public_url(record.gateway_url), "key": bool(record.api_key),
           "models": {}}
    if not record.api_key:
        out["error"] = NO_KEY
        return out
    listed = {m.id: m for m in dream.DreamClient().get_models()}
    for name in models:
        m = listed.get(MODELS[name])
        out["models"][name] = {"id": MODELS[name], "listed": m is not None,
                               "servable": bool(m is not None and getattr(m, "servable", False))}
    return out
