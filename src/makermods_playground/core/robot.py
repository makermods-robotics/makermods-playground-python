"""Robot: one or two arms of one type (left / right) and named cameras, through the arm's SDK.

    with Robot.load() as robot:          # connect; leaving returns every arm to rest, then torque off
        robot.enable()
        obs = robot.observe()
        robot.command({"left": obs.q["left"]})

Joints are the SDK's radians, seven per arm (six joints, then the gripper). The gripper is also given as an opening
in [0, 1]: 0 closed, 1 fully open.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config
from .config import RobotDef

log = logging.getLogger(__name__)


@dataclass
class Observation:
    q: dict[str, np.ndarray]                    # side -> (7,) rad
    gripper: dict[str, float]                   # side -> opening in [0, 1] (unclipped)
    tool: dict[str, np.ndarray]                 # side -> (4, 4) tool pose in that arm's base frame
    frames: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)   # camera -> (BGR image, age s)
    feedback_age: float = 0.0                   # s, oldest joint feedback
    stamp: float = 0.0                          # time.monotonic() when read


class Robot:
    def __init__(self, definition: RobotDef, arms: Optional[dict] = None, cameras: Optional[dict] = None):
        self.definition = definition
        self.name = definition.name
        self.arm_type = definition.arm
        self.settings = definition.settings
        self.sdk = config.sdk(definition.arm)
        buses = list(arms or {})
        if arms is None:
            arms = {side: definition.arm_spec(side).build() for side in definition.sides}
            buses = [(definition.arm_spec(s).backend, definition.arm_spec(s).address) for s in arms]
        if cameras is None:
            cameras = {name: self.sdk.cameras.Camera(cid) for name, cid in definition.cameras.items()}
        self.arms = dict(arms)
        self.cameras = dict(cameras)
        self.group = self.sdk.rig.ArmGroup(self.arms, parallel=len(set(buses)) == len(buses))
        self.sides = [s for s in config.SIDES if s in self.arms]
        self.rehomed: list[str] = []              # arms enabled away from rest (enable())

    @classmethod
    def load(cls, name: Optional[str] = None) -> "Robot":
        return cls(config.load(name))

    @classmethod
    def offline(cls, arm_type: str, sides=("left", "right"), definition: Optional[RobotDef] = None) -> "Robot":
        """Arms on a mock bus, never connected: the models (limits, kinematics, IK) without hardware."""
        definition = definition or RobotDef("offline", arm_type, {s: {"backend": "gs_usb"} for s in sides})
        sdk = config.sdk(definition.arm)
        arms = {s: sdk.arm.Arm(definition.arm_spec(s).config(), sdk.mock.MockBackend()) for s in definition.sides
                if s in sides}
        return cls(definition, arms=arms, cameras={})

    # ── lifecycle ──
    def connect(self, cameras: bool = True) -> None:
        """Find every motor (torque stays off) and start the cameras."""
        self.group.connect()
        if cameras:
            try:
                for cam in self.cameras.values():
                    cam.open()
            except BaseException:
                self.close_cameras()
                self.group.disconnect()
                raise

    def enable(self, cancel: Optional[threading.Event] = None) -> None:
        """Torque on every arm, holding where it is. An arm that fails leaves the others stopped again.

        An arm comes home (stop(), disconnect()) to the SDK's rest target for its enable pose only when that pose is
        at rest: a joint further from the profile's rest pose than the SDK treats as at rest (away_from_rest())
        makes the profile's rest pose its way home instead, so torque never goes off where it was merely being held
        (``rehomed`` lists those arms). That holds too for the arms already enabled when another fails: they come home
        before the error is raised."""
        self.rehomed = []
        try:
            self.group.each(lambda arm: arm.enable(cancel=cancel))
        except BaseException:
            try:
                self._aim_home()
                self.stop()
            except Exception:
                log.exception("bringing the enabled arms home failed")
            raise
        self._aim_home()

    def _aim_home(self) -> None:
        """Make the profile's rest pose the way home of each enabled arm whose enable pose is away from it."""
        for side in self.sides:
            arm = self.arms[side]
            if arm.state.value != "enabled" or arm.rest_pose is None or not self.away_from_rest(side, arm.rest_pose):
                continue
            rest = self.rest_pose(side)
            arm.rest_pose = [r if j.return_to_rest else q for r, q, j in zip(rest, arm.rest_pose, arm.config.joints)]
            self.rehomed.append(side)

    def command(self, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Send joint targets (rad, seven per arm). Returns, per accepted arm, the target as the arm drives toward
        it: inside its joint limits."""
        out = {}
        for side, q in targets.items():
            q = np.asarray(q, dtype=float)
            if self.arms[side].set_joint_targets(q.tolist()):
                out[side] = np.clip(q, *self.limits(side))
        return out

    def hold(self) -> None:
        """Freeze every arm at its measured pose, torque on."""
        for arm in self.arms.values():
            arm.hold_current_position()

    def stop(self) -> dict[str, bool]:
        """Return every arm to its rest pose (see enable()), then torque off. False where the return could not be
        verified: that arm keeps holding."""
        return self.group.stop()

    def rest(self, cancel: Optional[threading.Event] = None) -> dict[str, bool]:
        """Each enabled arm in turn to its profile's rest pose (as the SDK's ``rest`` command), verified there, then
        torque off. Ends at the first arm whose arrival is not verified, or once ``cancel`` is set: that arm and the
        ones not yet moved keep holding (absent from the result when not tried)."""
        arrived = {}
        for side in self.sides:
            if cancel is not None and cancel.is_set():
                break
            arm = self.arms[side]
            arm.rest_pose = self.rest_pose(side)
            arrived[side] = arm.stop()
            if not arrived[side]:
                break
        return arrived

    def rest_pose(self, side: str) -> list[float]:
        """The profile's rest pose of an arm as the SDK's ``rest`` command drives to it (its rest_pose, else the zero
        pose, clipped into the joint limits), rad."""
        cfg = self.arms[side].config
        return self.sdk.move.pose_goal("rest", cfg, [0.0] * cfg.n_joints)

    def away_from_rest(self, side: str, q) -> bool:
        """Whether a joint that returns to rest is further from the profile's rest pose than the SDK treats as at
        rest: its rest window where the profile sets one (inside it the SDK's own return presses toward the
        support), else what it accepts as arrived (``rest_settle``)."""
        cfg = self.arms[side].config
        rest = self.rest_pose(side)
        near = max(cfg.rest_settle, getattr(cfg, "rest_window", None) or 0.0)
        return any(j.return_to_rest and not abs(float(v) - r) <= near for j, v, r in zip(cfg.joints, q, rest))

    def release(self) -> dict[str, dict]:
        """Torque off now, verified motor by motor. An arm away from its rest pose drops. A connected arm is
        disabled; one that is not is released on its own bus, which works with a latched fault or a motor that does
        not answer. Per arm: ``ok`` (every motor confirmed torque off), ``unconfirmed`` and ``missing`` joints."""
        out = {}
        for side in self.sides:
            arm = self.arms[side]
            if arm.state.value == "disconnected":
                result = self.sdk.bus.release(arm.backend, arm.config)
                out[side] = {k: result[k] for k in ("ok", "unconfirmed", "missing")}
                continue
            arm.disable()
            names = {m.motor_id: j.label for m, j in zip(arm.motors, arm.config.joints)}
            unconfirmed = [names.get(i, i) for i in arm.unconfirmed_disable]
            out[side] = {"ok": not unconfirmed, "unconfirmed": unconfirmed, "missing": []}
        return out

    def estop(self) -> None:
        self.group.estop()

    def detach(self) -> dict[str, dict[str, float]]:
        """Close the buses leaving torque on: each arm holds its last command (joints with a motor watchdog, the
        gripper, release once it times out; returned per arm)."""
        out = {s: self.arms[s].detach() for s in self.sides}
        self.close_cameras()
        return out

    def disconnect(self) -> dict[str, str]:
        """Close cameras and buses. An arm that is still enabled is stopped first; one whose return to rest is not
        verified is left holding. Returns the SDK's ArmGroup.close() outcome per arm: only "released" is torque off
        confirmed."""
        try:
            return self.group.close()
        finally:
            self.close_cameras()

    def close_cameras(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.close()
            except Exception:
                pass

    def __enter__(self) -> "Robot":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ── reading ──
    def joint_positions(self) -> dict[str, np.ndarray]:
        return {s: np.asarray(self.arms[s].get_joint_positions(), dtype=float) for s in self.sides}

    def commanded_positions(self) -> dict[str, np.ndarray]:
        return {s: np.asarray(self.arms[s].get_commanded_positions(), dtype=float) for s in self.sides}

    def states(self) -> dict[str, str]:
        return {s: self.arms[s].state.value for s in self.sides}

    def faults(self) -> dict[str, Optional[str]]:
        return {s: self.arms[s].fault_reason for s in self.sides}

    def torque(self) -> dict[str, str]:
        """Per arm, "on" (enabled here, or a motor reports torque on), "off" (every motor reports it off) or
        "unknown" (not connected, or a motor reports no mode or has no feedback)."""
        out = {}
        for side in self.sides:
            arm = self.arms[side]
            if arm.state.value in ("enabled", "fault"):
                out[side] = "on"
            elif arm.state.value == "disconnected":
                out[side] = "unknown"
            else:
                flags = [j["enabled"] for j in arm.status()["joints"].values()]
                out[side] = "on" if True in flags else "unknown" if None in flags else "off"
        return out

    def feedback_ages(self) -> dict[str, float]:
        """Per arm, the age (s) of its oldest joint feedback (inf where a joint has none)."""
        out = {}
        for side in self.sides:
            ages = [j["feedback_age"] for j in self.arms[side].status()["joints"].values()]
            out[side] = max((math.inf if a is None else a) for a in ages)
        return out

    def frame(self, name: str, max_age_s: Optional[float] = None) -> Optional[tuple[np.ndarray, float]]:
        """The newest frame of a camera and its age (s, monotonic clock), or None when there is none (or it is
        older than ``max_age_s``)."""
        newest = self.cameras[name].newest()
        if newest is None:
            return None
        age = newest.age_s
        return None if max_age_s is not None and age > max_age_s else (newest.image, age)

    def frames(self, max_age_s: Optional[float] = None) -> dict[str, tuple[np.ndarray, float]]:
        out = {}
        for name in self.cameras:
            got = self.frame(name, max_age_s)
            if got is not None:
                out[name] = got
        return out

    def observe(self, max_frame_age_s: Optional[float] = None) -> Observation:
        stamp = time.monotonic()
        q = self.joint_positions()
        ages = self.feedback_ages()
        return Observation(
            q=q,
            gripper={s: self.opening(s, q[s][6]) for s in self.sides},
            tool={s: self.tool_pose(s, q[s]) for s in self.sides},
            frames=self.frames(max_frame_age_s),
            feedback_age=max(ages.values(), default=math.inf),
            stamp=stamp,
        )

    def status(self) -> dict:
        """JSON-serialisable snapshot of every arm (the SDK's status plus the gripper opening) and camera."""
        arms = {}
        for side in self.sides:
            st = self.arms[side].status()
            opening = self.arms[side].gripper_opening()
            st["gripper_opening"] = round(opening, 4) if math.isfinite(opening) else None
            arms[side] = st
        cams = {}
        for name, cam in self.cameras.items():
            got = self.frame(name) if getattr(cam, "is_open", False) else None
            cams[name] = {"id": str(getattr(cam, "id", "")), "open": bool(getattr(cam, "is_open", False)),
                          "age_s": None if got is None else round(got[1], 3), "error": getattr(cam, "error", None)}
        return {"robot": self.name, "arm": self.arm_type, "arms": arms, "cameras": cams}

    # ── models ──
    def limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        joints = self.arms[side].config.joints
        return np.array([j.lo for j in joints]), np.array([j.hi for j in joints])

    def gripper_range(self, side: str) -> tuple[float, float]:
        """(closed, fully open) gripper positions, rad: the SDK's gripper mapping (fully open is the profile's
        ``open_position`` where it sets one)."""
        arm = self.arms[side]
        return arm.gripper_target(0.0), arm.gripper_target(1.0)

    def opening(self, side: str, position: float) -> float:
        """The opening (0 closed, 1 fully open, unclipped) of a gripper position (rad)."""
        closed, full = self.gripper_range(side)
        return float((position - closed) / (full - closed))

    def gripper_position(self, side: str, opening) -> np.ndarray:
        """Gripper positions (rad) for openings (clipped to [0, 1])."""
        return np.vectorize(self.arms[side].gripper_target, otypes=[float])(np.asarray(opening, dtype=float))

    def chain(self, side: str):
        return self.arms[side].chain

    def tool_pose(self, side: str, q) -> np.ndarray:
        chain = self.chain(side)
        q = np.asarray(q, dtype=float)[:chain.n]
        return chain.fk(q) if np.isfinite(q).all() else np.full((4, 4), np.nan)

    def ik(self, side: str, solver: Optional[str] = None):
        """The SDK's trajectory IK for one arm (``solver``: numpy, jax or auto = numpy; default from the settings)."""
        arm = self.arms[side]
        return self.sdk.trajectory_ik.from_config(arm.config, chain=arm.chain, solver=solver or self.settings.ik_solver)
