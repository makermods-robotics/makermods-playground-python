import math
import threading

import numpy as np
import pytest
import sim

from makermods_playground.core.limits import ENTRY_SPEED, GRIPPER_OPEN_MARGIN, LimitCounts, Limiter
from makermods_playground.core.servo import Servo


class StubRobot:
    """Joints that follow their commands at once, unless told otherwise."""

    def __init__(self):
        self.q = {"left": np.zeros(7)}
        self.state = "enabled"
        self.lag = 0.0
        self.age = 0.005
        self.sent = []
        self.arms = {"left": type("A", (), {"fault_reason": "boom"})()}

    def states(self):
        return {"left": self.state}

    def joint_positions(self):
        return {k: v.copy() for k, v in self.q.items()}

    def feedback_ages(self):
        return {k: self.age for k in self.q}

    def commanded_positions(self):
        return {k: v + self.lag for k, v in self.q.items()}

    def command(self, targets):
        self.sent.append(targets)
        for k, v in targets.items():
            self.q[k] = np.asarray(v, dtype=float)
        return targets


def run_for(servo, seconds):
    stop = threading.Event()
    threading.Timer(seconds, stop.set).start()
    return servo.run(stop)


def test_rate_and_samples():
    robot = StubRobot()
    seen = []
    servo = Servo(robot, 40, sample=lambda now: {"left": np.full(7, 0.1)},
                  on_tick=lambda now, phase, cmd, meas: seen.append(phase))
    assert run_for(servo, 0.5) == "stopped"
    # Never faster than 40 Hz; a late wake-up on a loaded machine drops ticks rather than bunching them.
    assert 5 <= servo.stats.ticks <= 22
    assert set(seen) == {1}
    summary = servo.stats.summary()
    assert summary["period_ms"]["p50"] >= 22
    np.testing.assert_allclose(robot.q["left"], 0.1)


def test_hold_sends_nothing():
    robot = StubRobot()
    servo = Servo(robot, 100, sample=lambda now: None)
    run_for(servo, 0.1)
    assert robot.sent == []


def test_ends_on_fault_tracking_feedback_and_check():
    robot = StubRobot()
    robot.state = "fault"
    assert Servo(robot, 100, sample=lambda now: None).run(threading.Event()) == "left arm fault: boom"
    robot = StubRobot()
    robot.lag = math.radians(20)
    assert "tracking error 20.0 deg" in Servo(robot, 100, sample=lambda now: None).run(threading.Event())
    robot = StubRobot()
    robot.q["left"][2] = math.nan
    assert Servo(robot, 100, sample=lambda now: None).run(threading.Event()) == "left arm: no joint feedback"
    robot = StubRobot()
    assert Servo(robot, 100, sample=lambda now: None, check=lambda now: "timeout").run(threading.Event()) == "timeout"


def test_stale_feedback_ends_the_loop_before_any_command():
    robot = StubRobot()
    robot.age = 0.22    # a motor silent for 220 ms: its cached position is not fresh
    servo = Servo(robot, 100, sample=lambda now: {"left": np.full(7, 0.1)}, feedback_limit=0.2)
    assert servo.run(threading.Event()) == "left arm: joint feedback 220 ms old"
    assert robot.sent == []
    robot.age = 0.01
    servo = Servo(robot, 100, sample=lambda now: {"left": np.full(7, 0.1)}, feedback_limit=0.2)
    run_for(servo, 0.05)
    assert robot.sent


def test_exception_ends_the_loop():
    def sample(now):
        raise RuntimeError("bad chunk")
    assert Servo(StubRobot(), 100, sample).run(threading.Event()) == "error: RuntimeError: bad chunk"


def test_servo_on_simulated_arms():
    robot, _ = sim.sim_robot("metal", sides=("left",), cameras=())
    robot.connect()
    robot.enable()
    start = robot.joint_positions()["left"]
    goal = start.copy()
    goal[0] += 0.05
    servo = Servo(robot, 40, sample=lambda now: {"left": goal})
    assert run_for(servo, 0.4) == "stopped"
    assert abs(robot.joint_positions()["left"][0] - goal[0]) < 1e-3
    assert robot.stop() == {"left": True}
    robot.disconnect()


def limiter():
    lo = np.r_[-np.ones(6), 0.0]
    hi = np.r_[np.ones(6), 2.0]
    return Limiter(lo, hi, closed=0.0, open_=2.0, joint_speed=np.full(6, 4.0))


def test_gripper_kept_short_of_open():
    cmd, report = limiter().limit(np.r_[np.zeros(6), 2.0], None, 0.025)
    assert cmd[6] == pytest.approx(2.0 - GRIPPER_OPEN_MARGIN) and report["position"]


def test_rate_caps():
    lim = limiter()
    last = np.zeros(7)
    cmd, report = lim.limit(np.r_[np.full(6, 0.5), 1.0], last, 0.025)
    np.testing.assert_allclose(cmd[:6], 4.0 * 0.025)
    assert cmd[6] == pytest.approx(2 * 2.0 * 0.025) and report["rate"]
    cmd, _ = lim.limit(np.r_[np.full(6, 0.5), 1.0], last, 0.025, speed_cap=0.5)
    np.testing.assert_allclose(cmd[:6], 0.5 * 0.025)
    cmd, _ = lim.limit(np.full(7, 0.5), last, 5.0)          # a stalled tick counts 0.1 s at most
    np.testing.assert_allclose(cmd[:6], 0.4)


def test_entering_from_past_a_limit():
    last = np.r_[1.2, np.zeros(6)]
    cmd, report = limiter().limit(np.zeros(7), last, 0.1)
    assert cmd[0] == pytest.approx(1.2 - ENTRY_SPEED * 0.1) and report["entering"]


def test_counts():
    counts = LimitCounts()
    counts.add([{"position": True, "rate": False, "entering": False}], 0.01)
    counts.add([{"position": False, "rate": False, "entering": False}], 0.0)
    assert counts.to_json()["limited_ticks"] == 1 and counts.to_json()["policy_ticks"] == 2


def test_rejects_bad_targets():
    with pytest.raises(ValueError):
        limiter().limit(np.full(7, np.nan), None, 0.02)
