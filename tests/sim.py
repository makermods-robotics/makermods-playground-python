"""A simulated robot: SDK arms on the SDKs' simulated motors (``<sdk>.testing``), fake cameras, a fake policy."""

from __future__ import annotations

import importlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from makermods_playground.core import config
from makermods_playground.core.robot import Robot

FIXTURES = Path(__file__).parent / "fixtures"
START = {"metal": [0.1, -0.6, 0.9, -0.2, 0.05, 0.1, 0.2], "maker": [0.1, -0.6, 0.9, 0.2, 0.05, 0.1, -0.5]}


def sim_arm(arm_type: str, start=None, cfg=None, **options):
    """(Arm, SimulatedMotors) for the shipped profile (or ``cfg``) on simulated motors that report their mode."""
    testing = importlib.import_module(f"{arm_type}_arm.testing")
    return testing.simulated_arm(cfg, START[arm_type] if start is None else start, params_supported=True, **options)


def rest_pose(arm_type: str) -> list:
    """The shipped profile's rest pose as the SDK's ``rest`` command drives to it (rad, clipped into the limits)."""
    cfg = config.sdk(arm_type).rig.ArmSpec(backend="gs_usb").config()
    return importlib.import_module(f"{arm_type}_arm.cli.move").pose_goal("rest", cfg, [0.0] * cfg.n_joints)


def frame(image, age=0.0, wall_offset=0.0):
    """An SDK camera Frame captured ``age`` s ago, its wall-clock stamp off by ``wall_offset`` s."""
    return config.sdk("metal").cameras.Frame(image, time.time() - age + wall_offset, time.monotonic() - age)


class FakeCamera:
    def __init__(self, id="fake", size=(640, 480)):
        self.id, self.size = id, size
        self.is_open = False
        self.error = None
        self.dead = False
        self.opens = 0

    def _frame(self):
        w, h = self.size
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :, 1] = int(time.time() * 50) % 255
        return img

    def open(self, timeout=5.0):
        if self.dead:
            raise OSError(f"cannot open camera {self.id}")
        self.is_open = True
        self.opens += 1

    def read(self, timeout=1.0):
        if self.dead or not self.is_open:
            raise TimeoutError(f"no frame from camera {self.id}")
        time.sleep(0.01)
        return frame(self._frame())

    def newest(self):
        if not self.is_open:
            return None
        return frame(self._frame(), age=10.0 if self.dead else 0.0)

    def close(self):
        self.is_open = False


def sim_robot(arm_type="metal", sides=("left", "right"), cameras=("top", "left", "right"), name="sim", settings=None,
              cfg=None, **options):
    """(robot, {side: SimulatedMotors}) with settings that keep a test quick and write nothing; ``cfg`` replaces the
    shipped profile, ``options`` go to every simulated motor (e.g. ``enabled=True``)."""
    definition = config.RobotDef(name, arm_type, {s: {"backend": "gs_usb", "port": f"gs_usb:{s}"} for s in sides},
                                 {c: f"usb:{c}" for c in cameras})
    definition.settings = settings or config.Settings(record=False)
    arms, motors = {}, {}
    for side in sides:
        arms[side], motors[side] = sim_arm(arm_type, cfg=cfg, **options)
    return Robot(definition, arms=arms, cameras={c: FakeCamera(c) for c in cameras}), motors


def watchdog_profile(arm_type="metal"):
    """The shipped profile with a motor watchdog on the shoulder lift, a stiff joint: it drops once commands stop."""
    from dataclasses import replace
    cfg = config.sdk(arm_type).rig.ArmSpec(backend="gs_usb").config()
    joints = list(cfg.joints)
    joints[1] = replace(joints[1], can_timeout_ms=200)
    return replace(cfg, joints=joints)


def fixture_run() -> dict:
    return json.loads((FIXTURES / "metal_run.json").read_text())


def head_near(robot, side: str, rows: int = 48, dx: float = 0.02) -> dict:
    """A head that starts at the arm's present tool pose and moves ``dx`` m along x."""
    q = robot.joint_positions()[side]
    pose = robot.tool_pose(side, q)
    s = np.linspace(0.0, 1.0, rows)
    xyz = pose[:3, 3] + np.outer(s, [dx, 0.0, 0.0])
    rot6 = np.tile(np.concatenate([pose[:3, 0], pose[:3, 1]]), (rows, 1))
    return {"eef_xyz": xyz, "eef_rot6": rot6, "grip": np.full(rows, 0.3)}


class FakePolicy:
    """A DreamRunner policy: ``heads(n)`` gives the n-th reply's head per arm; ``errors`` are raised first."""

    def __init__(self, robot, sides, latency=0.05, errors=(), cameras=("top", "left", "right"), limit=None, rows=16):
        self.robot, self.sides, self.latency = robot, list(sides), latency
        self.errors = list(errors)
        self.cameras = list(cameras)
        self.calls = 0
        self.resets = 0
        self.pushes = 0
        self.observations = []
        self.limit = limit
        self.rows = rows
        self.closed = False
        self.session_id = "sess-1"
        self.lock = threading.Lock()

    def reset_episode(self):
        self.resets += 1

    def observation(self, prompt, state):
        self.observations.append(state)
        return {"prompt": prompt}

    def push(self, images, observation, timestamp):
        assert set(images) >= set(self.cameras)
        self.pushes += 1

    def act(self):
        time.sleep(self.latency)
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.limit is not None and self.calls > self.limit:
            from makermods_playground.dream.runner import PolicyFinished
            raise PolicyFinished("no more chunks")
        per_key = {}
        for side in self.sides:
            head = head_near(self.robot, side, self.rows, dx=0.01 * (1 if self.calls % 2 else -1))
            per_key.update({f"action.{side}_{k}": v for k, v in head.items()})
        return per_key, {"rtt_ms": self.latency * 1000}

    def close(self):
        self.closed = True


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def once_enabled(robot, action, timeout=30.0):
    """Run ``action`` on a thread once every arm of ``robot`` is enabled (a fixed delay can land in a slow enable)."""
    def run():
        if wait_until(lambda: all(state == "enabled" for state in robot.states().values()), timeout):
            action()
    threading.Thread(target=run, daemon=True).start()


def sag(motors, joint, rad):
    """Make a simulated motor hold ``rad`` short of every position it is commanded to (a loaded joint's standing
    error)."""
    m = motors.motor(joint)
    handle = m.handle

    def sagging(can_id, data):
        count = len(m.log)
        replies = handle(can_id, data)
        commanded = len(m.log) > count and isinstance(m.log[-1], tuple) and m.log[-1][0] == "cmd"
        if commanded and m.enabled and m.log[-1][1] > 0:
            m.position -= rad
            return [m.feedback()]
        return replies

    m.handle = sagging


def sent(motors, since):
    """What the host sent each simulated motor after ``since`` ({joint: entries of its log}), reads aside (a RobStride
    state read is its fault-clear frame, logged "clear": refresh() sends it only to a motor reporting no fault)."""
    quiet = ("status_query", "dropped", "clear")
    return {j: [e for e in m.log[since.get(j, 0):] if e not in quiet and not (isinstance(e, tuple) and e[0] == "read")]
            for j, m in motors.motors.items()}


def log_lengths(motors):
    return {j: len(m.log) for j, m in motors.motors.items()}


def held_arm(arm_type: str):
    """(arm, SimulatedMotors) of an arm another program left holding at sim.START, which the SDK's enable() refuses
    with HeldByAnotherProgram: Metal motors report torque on; Maker motors on old firmware report no torque state
    but the shoulder is producing torque."""
    if arm_type == "metal":
        return sim_arm("metal", enabled=True)
    testing = importlib.import_module("maker_arm.testing")
    arm, motors = testing.simulated_arm(None, START["maker"], params_supported=False)
    motors.motor("shoulder_lift").torque = 2.0
    return arm, motors
