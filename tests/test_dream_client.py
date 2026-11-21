import base64

import cv2
import dream
import numpy as np
import pytest

from makermods_playground.core.mask import mask_secrets, redact_secrets
from makermods_playground.dream import client as dc


class FakeDreamClient:
    """The DreamClient surface DreamPolicy uses; never touches the network."""

    def __init__(self, per_key=None):
        self.config = None
        self.ticks = []
        self.resets = 0
        self.closed = False
        self.per_key = per_key or {}
        self.last_timing = None
        self.session_id = "s-1"

    def policy(self, config):
        self.config = config

    def set_observation(self, frames, observation, timestamp=None):
        self.ticks.append((frames, observation, timestamp))

    def reset_episode(self):
        self.resets += 1

    def get_action(self):
        self.last_timing = {"rtt_ms": 321.0, "network_ms": 20.0}
        return {"per_key": self.per_key}

    def close(self):
        self.closed = True


def test_policy_config():
    fake = FakeDreamClient()
    policy = dc.DreamPolicy("flash", ["left", "right"], timeout_ms=30000, client=fake)
    c = fake.config
    assert c["checkpoint"] == "dreamlabs/dream-1-beta-flash"
    assert c["robot"] == "other_embodiment" and c["_record"] is False
    assert (c["action_hz"], c["execution_horizon"], c["sync_above_ms"], c["timeout_ms"]) == (20.0, 30, 700.0, 30000)
    assert set(c["embodiment"]["state"]) == {f"{s}_{p}" for s in ("left", "right")
                                             for p in ("arm", "gripper", "eef_xyz", "eef_rot6", "grip")}
    assert set(c["embodiment"]["video"]) == {"overhead", "wrist_l", "wrist_r"}
    assert policy.cameras == ["top", "left", "right"]
    assert dc.model_id("beta") == "dreamlabs/dream-1-beta"
    with pytest.raises(dc.PolicyError):
        dc.model_id("gamma")
    with pytest.raises(dc.PolicyError):
        dc.DreamPolicy("beta", ["middle"], client=fake)


def test_observation_and_frames():
    fake = FakeDreamClient()
    policy = dc.DreamPolicy("beta", ["left"], client=fake, parked={"right": np.eye(4)})
    tool = np.eye(4)
    tool[:3, 3] = [0.3, 0.1, 0.2]
    tool[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    obs = policy.observation("pick", {"left": {"opening": 1.3, "tool": tool}})
    assert obs["prompt"] == "pick"
    np.testing.assert_array_equal(obs["state.left_arm"], np.zeros(6, np.float32))
    assert obs["state.left_grip"][0] == 1.0 and obs["state.left_gripper"][0] == 1.0
    np.testing.assert_allclose(obs["state.left_eef_xyz"], [0.3, 0.1, 0.2])
    np.testing.assert_allclose(obs["state.left_eef_rot6"], [0, 1, 0, -1, 0, 0])
    assert all(v.dtype == np.float32 for k, v in obs.items() if k.startswith("state."))
    images = {n: np.full((480, 640, 3), 100, np.uint8) for n in ("top", "left", "right")}
    policy.push(images, obs, 12.5)
    frames, sent, stamp = fake.ticks[0]
    assert set(frames) == {"video.overhead", "video.wrist_l", "video.wrist_r"} and stamp == 12.5
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(frames["video.overhead"]), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (176, 320, 3)
    with pytest.raises(dc.PolicyError, match="right"):
        policy.push({"top": images["top"], "left": images["left"]}, obs, 13.0)


def test_act_returns_per_key_and_timing():
    fake = FakeDreamClient({"action.left_grip": [0.1]})
    policy = dc.DreamPolicy("beta", ["left"], client=fake, parked={"right": np.eye(4)})
    per_key, timing = policy.act()
    assert per_key == {"action.left_grip": [0.1]} and timing["rtt_ms"] == 321.0
    policy.reset_episode()
    policy.close()
    assert fake.resets == 1 and fake.closed and policy.session_id == "s-1"


def test_no_key_refuses_before_any_call():
    with pytest.raises(dc.PolicyError, match="no Dream API key"):
        dc.DreamPolicy("beta", ["left"], parked={"right": np.eye(4)})
    out = dc.check()
    assert out["key"] is False and out["error"] == dc.NO_KEY and out["models"] == {}


def test_retry_rules():
    warming = dream.DreamServerError("warming", "DA-2000", "pool_warming", True, 5.0)
    assert dc.retry_delay(warming) == 5.0 and dc.starts_new_session(warming)
    limited = dream.DreamServerError("slow down", "DA-2000", "rate_limited", True, None)
    assert dc.retry_delay(limited) == 1.0 and not dc.starts_new_session(limited)
    assert dc.retry_delay(dream.DreamAuthError("bad key", "DA-1000", "unknown_key")) is None
    assert dc.retry_delay(dream.DreamSessionExpired("expired", "DA-0001")) == 0.0
    assert dc.retry_delay(dream.DreamTimeout("slow")) == 1.0
    assert dc.retry_delay(ValueError("x")) is None
    assert dc.retry_delay(dream.DreamServerError("broken", "DA-2000", "internal", False, None)) is None


def test_check_shows_only_the_gateway_scheme_host_and_port(monkeypatch):
    record = type("R", (), {"path": "c", "gateway_url": "https://user:pw@gw.example:8443/v1?key=dl_live_x#f",
                            "api_key": None})()
    monkeypatch.setattr(dc, "credentials", lambda: record)
    assert dc.check()["gateway"] == "https://gw.example:8443"
    assert dc.public_url("not a url") == "(not a valid URL)"


def test_mask():
    assert mask_secrets("key dl_live_abcdef123 used") == "key dl_live_*** used"
    assert mask_secrets("Authorization: Bearer abc.def") == "Authorization: Bearer ***"
    assert mask_secrets("DREAM_API_KEY=secret123 rest") == "DREAM_API_KEY=*** rest"
    assert mask_secrets("https://user:pw@host/x") == "https://user:***@host/x"
    assert redact_secrets({"api_key": "x", "note": "token=abc", "n": 3}) == {"api_key": "***", "note": "token=***",
                                                                          "n": 3}


def test_a_one_arm_two_camera_robot_gets_stand_ins_for_the_rest():
    import numpy as np
    from makermods_playground.dream.client import DreamPolicy, PolicyError

    class Client:
        session_id = None

        def policy(self, config):
            self.config = config

        def set_observation(self, frames, observation, timestamp):
            self.frames, self.observation = frames, observation

    parked = np.eye(4)
    parked[:3, 3] = [0.2, 0.0, 0.1]
    client = Client()
    policy = DreamPolicy("beta", ["left"], client=client, cameras=["top", "left"], parked={"right": parked})
    assert set(client.config["embodiment"]["state"]) >= {"left_grip", "right_grip"}
    assert policy.cameras == ["top", "left"]
    observation = policy.observation("task", {"left": {"opening": 0.4, "tool": np.eye(4)}})
    assert observation["state.right_eef_xyz"].tolist() == [np.float32(0.2), 0.0, np.float32(0.1)]
    assert observation["state.right_grip"].tolist() == [0.0]
    image = np.full((48, 64, 3), 200, np.uint8)
    policy.push({"top": image, "left": image}, observation, 0.0)
    assert set(client.frames) == {"video.overhead", "video.wrist_l", "video.wrist_r"}
    try:
        DreamPolicy("beta", ["left"], client=Client(), cameras=["top"])
    except PolicyError as e:
        assert "parked tool pose" in str(e)
    else:
        raise AssertionError("an absent arm without a parked pose must be refused")
