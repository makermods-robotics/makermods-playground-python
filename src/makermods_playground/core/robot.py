"""Robot: one or two arms of one type (left / right) and named cameras, through the arm's SDK.

    with Robot.load() as robot:          # connect; leaving returns every arm to rest, then torque off
        robot.enable()
        obs = robot.observe()
        robot.command({"left": obs.q["left"]})

Joints are the SDK's radians, seven per arm (six joints, then the gripper). The gripper is also given as an opening
in [0, 1]: 0 closed, 1 fully open.

An arm comes home in two steps (stop(), rest(), disconnect()): first its tool point moves in a straight line to where
it sits at the rest pose (the retract: low and close to the base), then the SDK's joint return to the rest pose
verifies arrival before torque goes off. The retract is skipped or cut short, never waited on, when it cannot be
solved, takes longer than RETRACT_LIMIT_S, or a joint motor reads RETRACT_MAX_TEMP_C or more; the joint return then
starts from wherever the arm is.
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

RETRACT_LIMIT_S = 15.0                    # the retract ends by then, wherever the tool is
RETRACT_SPEED = math.radians(30.0)        # rad/s of the fastest joint, as the SDK's return to rest
RETRACT_STEP_M = 0.05                     # the line is solved every 5 cm of tool travel
RETRACT_STEP_ROT = math.radians(15.0)     # and every 15 degrees of tool turn,
RETRACT_SAG_M = 0.005                     # then halved where the joint move between two poses strays this far,
RETRACT_FINE_M = 0.01                     # down to 1 cm and 5 degrees
RETRACT_FINE_ROT = math.radians(5.0)
RETRACT_TOLERANCE_M = 0.03                # a 1 cm step solved further than this from the line ends the retract,
RETRACT_JUMP = math.radians(20.0)         # and so does one that moves any joint further than this
RETRACT_MAX_POSES = 200
RETRACT_HZ = 50.0                         # joint targets per second while it moves
RETRACT_MAX_TEMP_C = 70.0                 # a joint motor at this temperature or more skips or ends the retract
RETRACT_SETTLE_S = 2.0                    # after the line: the longest wait for the arm to settle
RETRACT_STILL_S = 0.2                     # it has settled once no joint moved more than RETRACT_STILL in this long
RETRACT_STILL = math.radians(0.3)
AT_REST = "skipped: already at its rest pose"
HELD_ELSEWHERE = "already have torque on"     # the SDK's refusal to enable motors that hold for another program


class ArmHeld(Exception):
    """enable() refused an arm whose motors already hold (left holding by another program); nothing was sent to it."""


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
        self.last_retract: dict[str, str] = {}    # side -> what the last retract did
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
        held: dict[str, str] = {}
        sides = {id(arm): side for side, arm in self.arms.items()}

        def enable(arm) -> None:
            try:
                arm.enable(cancel=cancel)
            except Exception as e:
                if HELD_ELSEWHERE in str(e):
                    held[sides[id(arm)]] = str(e)
                raise

        try:
            self.group.each(enable)
        except BaseException as e:
            try:
                self._aim_home()
                self.stop()           # only arms enabled here move; an arm that was refused is left as it is
            except Exception:
                log.exception("bringing the enabled arms home failed")
            if held:
                side = next(s for s in self.sides if s in held)
                raise ArmHeld(f"the {side} arm is held by another program (its motors report torque on); nothing "
                              f"was sent to it. Bring it down with '{self.rest_command(side)}', then try again") from e
            raise
        self._aim_home()

    def rest_command(self, side: str) -> str:
        """The SDK command that brings this arm down and releases it: by its name in the SDK's rig file when the rig
        names the same bus, else by its bus."""
        spec = self.definition.arm_spec(side)
        cli = f"{self.arm_type}-arm rest"
        try:
            rig = self.sdk.rig.Rig.load()
            for name in rig.arm_names:
                theirs = rig.spec(name)
                if (theirs.backend, theirs.address) == (spec.backend, spec.address):
                    return f"{cli} --arm {name}"
        except Exception:
            pass
        if spec.backend == "socketcan":
            return f"{cli} --backend socketcan --channel {spec.channel or 'can0'}"
        if spec.port and spec.port.startswith("gs_usb"):
            return f"{cli} --port {spec.port}"
        return f"{cli} --backend {spec.backend}" + (f" --port {spec.port}" if spec.port else "")

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
        """Bring every enabled arm home (the retract, then the joint return to its rest pose, see enable()), then
        torque off. False where arrival could not be verified: that arm keeps holding."""
        self._try_retract()
        return self.group.stop()

    def rest(self, cancel: Optional[threading.Event] = None) -> dict[str, bool]:
        """Each enabled arm in turn to its profile's rest pose (the retract, then the joint return), verified there,
        then torque off. Ends at the first arm whose arrival is not verified, or once ``cancel`` is set: that arm and
        the ones not yet moved keep holding (absent from the result when not tried)."""
        arrived = {}
        for side in self.sides:
            if cancel is not None and cancel.is_set():
                break
            arm = self.arms[side]
            arm.rest_pose = self.rest_pose(side)
            self.retract([side], cancel)
            if cancel is not None and cancel.is_set():
                break
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

    def retract(self, sides=None, cancel: Optional[threading.Event] = None) -> dict[str, str]:
        """Move the tool point of each enabled arm (of ``sides``, else all, in parallel) in a straight line to where
        it sits at the arm's rest pose, turning to that pose's orientation, at RETRACT_SPEED. Never longer than
        RETRACT_LIMIT_S, and cut short once ``cancel`` is set or a joint motor reads RETRACT_MAX_TEMP_C or more: the
        arm then holds where it is. The line is solved from the commanded pose with the SDK's IK, a little ahead of
        the motion; where it has no pose near the line, or needs a joint to jump, the retract ends there. Returns
        what happened per arm."""
        sides = [s for s in (sides or self.sides) if self.arms[s].state.value == "enabled"]
        if not sides:
            return {}                 # nothing enabled: what the last retract did stays readable
        outcome: dict[str, str] = {}
        workers: dict[str, tuple[threading.Thread, threading.Event, threading.Lock]] = {}
        for side in sides:
            warm = self._warm_motor(side)
            if warm:
                outcome[side] = f"skipped: {warm}"
                continue
            ending, gate = threading.Event(), threading.Lock()
            thread = threading.Thread(target=self._retract_arm, args=(side, ending, gate, outcome), daemon=True,
                                      name=f"retract-{side}")
            thread.start()
            workers[side] = (thread, ending, gate)
        deadline = time.monotonic() + RETRACT_LIMIT_S
        ended: dict[str, str] = {}
        while any(thread.is_alive() and side not in ended for side, (thread, _, _) in workers.items()):
            for side, (thread, ending, gate) in workers.items():
                if not thread.is_alive() or side in ended:
                    continue
                why = (f"not done within {RETRACT_LIMIT_S:g} s" if time.monotonic() >= deadline else
                       "cancelled" if cancel is not None and cancel.is_set() else self._warm_motor(side))
                if why:
                    ended[side] = why
                    with gate:              # no target of this retract is sent after this
                        ending.set()
            time.sleep(0.05)
        for side, why in ended.items():
            workers[side][0].join(0.5)
            outcome.setdefault(side, f"ended early: {why}")
        for side in sides:
            log.info("%s: retract %s", side, outcome.get(side, "not run"))
        self.last_retract = outcome
        return outcome

    def retract_notes(self) -> dict[str, str]:
        """What the last retract did, per arm, where it is worth saying (it did not run to the end, and the arm was
        not simply at rest already)."""
        return {side: what for side, what in self.last_retract.items() if what not in ("done", AT_REST)}

    def _try_retract(self) -> None:
        try:
            self.retract()
        except Exception:
            log.exception("retract failed; returning in joint space")

    def _warm_motor(self, side: str) -> Optional[str]:
        """"<joint> at N °C" for the first joint motor (the gripper aside) reading RETRACT_MAX_TEMP_C or more."""
        arm = self.arms[side]
        for j, t in zip(arm.config.joints, arm.get_temperatures()):
            if j.return_to_rest and t is not None and math.isfinite(t) and t >= RETRACT_MAX_TEMP_C:
                return f"{j.label} at {t:.0f} °C"
        return None

    def _retract_arm(self, side: str, ending: threading.Event, gate: threading.Lock, outcome: dict) -> None:
        """Send the line as joint targets following the clock (a late wake-up does not slow it) while a second thread
        solves it ahead: the motion waits at the last pose solved, and a refusal partway leaves the arm there. Each
        target goes out under ``gate`` and only while ``ending`` is not set; nothing is written once it is."""
        arm = self.arms[side]
        try:
            command = np.asarray(arm.get_commanded_positions(), dtype=float)
            line = self._retract_line(side, command)
            if isinstance(line, str):
                outcome[side] = line
                return
            start, tail = line[0], command[len(line[0]):].tolist()
            path, times, solving = [start], [0.0], {"done": False, "why": None}
            lock = threading.Lock()

            def solve():
                poses = self._retract_poses(side, line, ending)
                try:
                    while True:
                        q = next(poses)
                        with lock:
                            times.append(times[-1] + float(np.max(np.abs(q - path[-1]))) / RETRACT_SPEED)
                            path.append(q)
                except StopIteration as done:
                    solving["why"] = done.value
                except Exception as e:
                    solving["why"] = f"{type(e).__name__}: {e}"
                finally:
                    solving["done"] = True

            threading.Thread(target=solve, daemon=True, name=f"retract-solve-{side}").start()
            clock, last = 0.0, time.monotonic()
            while True:
                now = time.monotonic()
                with lock:
                    done = solving["done"]
                    rows, stamps = list(path), list(times)
                clock, last = min(clock + now - last, stamps[-1]), now
                if len(rows) > 1:
                    k = min(int(np.searchsorted(stamps, clock, side="right")) - 1, len(rows) - 2)
                    span = stamps[k + 1] - stamps[k]
                    f = min(1.0, max(0.0, (clock - stamps[k]) / span)) if span > 0 else 1.0
                    q = rows[k] + (rows[k + 1] - rows[k]) * f
                    with gate:
                        if ending.is_set():
                            return
                        if not arm.move_to(q.tolist() + tail, speed=2 * RETRACT_SPEED, wait=False):
                            outcome[side] = "ended early: its targets were refused"
                            return
                elif ending.is_set():
                    return
                if done and clock >= stamps[-1]:
                    break
                time.sleep(1.0 / RETRACT_HZ)
            if solving["why"]:
                if not ending.is_set():
                    outcome[side] = (f"{solving['why']} (after {len(rows) - 1} of the poses: held there)"
                                     if len(rows) > 1 else solving["why"])
                return
            # Settled once there, or once no joint moves any more: a loaded joint keeps a standing error
            began, seen = time.monotonic(), []
            while not ending.is_set():
                now = time.monotonic()
                here = np.asarray(arm.get_joint_positions()[:len(start)], dtype=float)
                seen = [(t, q) for t, q in seen if now - t <= RETRACT_STILL_S] + [(now, here)]
                still = now - began >= RETRACT_STILL_S and seen[0][0] <= now - RETRACT_STILL_S + 0.05 and max(
                    float(np.max(np.abs(q - here))) for _, q in seen) <= RETRACT_STILL
                if still or float(np.max(np.abs(here - rows[-1]))) <= arm.config.rest_arrived:
                    outcome[side] = "done"
                    return
                if now - began >= RETRACT_SETTLE_S:
                    outcome[side] = "done, not yet settled"
                    return
                time.sleep(0.02)
        except Exception as e:
            if not ending.is_set():
                outcome[side] = f"skipped: {type(e).__name__}: {e}"

    def _retract_line(self, side: str, command: np.ndarray):
        """(start joints, start tool pose, end tool pose, turn) of the straight tool line from the commanded pose to
        the rest pose's tool pose, or why there is none."""
        arm = self.arms[side]
        if arm.rest_pose is None or not arm.config.urdf:
            return "skipped: no rest pose or no kinematics"
        chain = arm.chain
        start = command[:chain.n]
        rest = np.asarray(arm.rest_pose[:chain.n], dtype=float)
        if not np.all(np.isfinite(start)):
            return "skipped: no commanded pose"
        if float(np.max(np.abs(start - rest))) <= arm.config.rest_settle:
            return AT_REST
        ik = self.sdk.trajectory_ik.TrajectoryIK.from_config(arm.config, chain=chain)
        a, b = chain.fk(start), chain.fk(np.clip(rest, ik.lo, ik.hi))
        return start, a, b, self.sdk.kinematics.rotation_error(b[:3, :3], a[:3, :3])

    def _retract_poses(self, side: str, line, ending: threading.Event):
        """Yield joint poses along the line, consecutive ones joined by a joint move that keeps the tool near it;
        returns why it stopped short (None once the line is done or ``ending`` is set)."""
        arm = self.arms[side]
        chain = arm.chain
        start, a, b, turn = line
        ik = self.sdk.trajectory_ik.TrajectoryIK.from_config(arm.config, chain=chain)
        lo, hi = np.minimum(ik.lo, start), np.maximum(ik.hi, start)
        length, angle = float(np.linalg.norm(b[:3, 3] - a[:3, 3])), float(np.linalg.norm(turn))
        fine = max(length / RETRACT_FINE_M, angle / RETRACT_FINE_ROT)    # fine steps in the whole line

        def pose(s: float) -> np.ndarray:
            target = np.eye(4)
            target[:3, 3] = a[:3, 3] + s * (b[:3, 3] - a[:3, 3])
            target[:3, :3] = _rotation(turn * s) @ a[:3, :3]
            return target

        def strays(q0, q1) -> bool:
            """The joint move q0 -> q1 takes the tool further than RETRACT_SAG_M off the chord between its ends."""
            p0, p1 = chain.fk(q0)[:3, 3], chain.fk(q1)[:3, 3]
            return any(np.linalg.norm(chain.fk(q0 + f * (q1 - q0))[:3, 3] - (p0 + f * (p1 - p0))) > RETRACT_SAG_M
                       for f in (0.25, 0.5, 0.75))

        steps = max(2, math.ceil(length / RETRACT_STEP_M), math.ceil(angle / RETRACT_STEP_ROT))
        s0, q0, todo = 0.0, start, [k / steps for k in range(1, steps + 1)]
        for _ in range(RETRACT_MAX_POSES):
            if ending.is_set():
                return None
            if not todo:
                return None
            s1 = todo[0]
            coarse = (s1 - s0) * fine > 1.0
            solved = chain.ik(pose(s1), q0, limits=(lo, hi), max_iters=30 if coarse else 60, tol=1e-3, rot_tol=1e-2,
                              rot_weight=0.05)        # the position first: the turn only guides the wrist
            q1 = np.asarray(solved.q, dtype=float)
            off = solved.pos_err > RETRACT_TOLERANCE_M
            jump = float(np.max(np.abs(q1 - q0))) > RETRACT_JUMP
            if coarse and (off or jump or strays(q0, q1)):
                todo.insert(0, (s0 + s1) / 2)          # solve half the way first
                continue
            if off:
                return (f"refused: no pose within {RETRACT_TOLERANCE_M * 1000:g} mm of the line "
                        f"({solved.pos_err * 1000:.0f} mm)")
            if jump:
                return "refused: the line needs a joint to jump"
            yield q1
            s0, q0 = todo.pop(0), q1
        return f"refused: more than {RETRACT_MAX_POSES} poses"

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
        """Close cameras and buses. An arm that is still enabled comes home first (as stop()); one whose return to
        rest is not verified is left holding. Returns the SDK's ArmGroup.close() outcome per arm: only "released" is
        torque off confirmed."""
        try:
            self._try_retract()
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


def _rotation(vector: np.ndarray) -> np.ndarray:
    """The rotation matrix of a rotation vector (axis * angle, rad)."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
