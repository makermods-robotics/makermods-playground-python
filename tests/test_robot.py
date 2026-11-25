import http.server
import math
import threading

import numpy as np
import pytest
import sim

from makermods_playground.core import config
from makermods_playground.core.robot import Robot


@pytest.fixture(params=["metal", "maker"])
def robot(request):
    robot, motors = sim.sim_robot(request.param)
    robot.motors = motors
    yield robot
    robot.estop()
    robot.disconnect()


def test_lifecycle(robot):
    robot.connect()
    assert robot.states() == {"left": "connected", "right": "connected"}
    robot.enable()
    obs = robot.observe(max_frame_age_s=1.0)
    assert set(obs.q) == {"left", "right"} and obs.q["left"].shape == (7,)
    np.testing.assert_allclose(obs.q["left"], sim.START[robot.arm_type], atol=1e-3)
    assert obs.tool["left"].shape == (4, 4) and np.isfinite(obs.tool["left"]).all()
    assert set(obs.frames) == {"top", "left", "right"}
    assert 0 <= obs.frames["top"][1] < 1.0
    assert obs.feedback_age < 0.3
    target = obs.q["left"].copy()
    target[0] += 0.1
    target[1] = 5.0                          # past the joint limit
    sent = robot.command({"left": target})
    lo, hi = robot.limits("left")
    assert sent["left"][1] == hi[1] and sent["left"][0] == pytest.approx(target[0])
    assert sim.wait_until(lambda: abs(robot.joint_positions()["left"][0] - target[0]) < 1e-3)
    robot.hold()
    assert robot.stop() == {"left": True, "right": True}
    assert robot.states() == {"left": "connected", "right": "connected"}
    st = robot.status()
    assert st["arm"] == robot.arm_type and "gripper_opening" in st["arms"]["left"]
    assert set(st["cameras"]) == {"top", "left", "right"}


def test_gripper_opening(robot):
    closed, full = robot.gripper_range("left")
    if robot.arm_type == "metal":
        assert (closed, full) == (0.0, pytest.approx(math.radians(116.4)))
    else:
        assert (closed, full) == (pytest.approx(math.radians(-2.5)), pytest.approx(math.radians(-96.0)))
    assert robot.opening("left", closed) == 0.0
    assert robot.opening("left", full) == pytest.approx(1.0)
    assert robot.gripper_position("left", 0.5) == pytest.approx((closed + full) / 2)


def test_cancel_during_enable(robot):
    robot.connect()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Exception, match="cancel"):
        robot.enable(cancel=cancel)
    assert robot.states() == {"left": "connected", "right": "connected"}


def test_motor_fault_holds_the_arm(robot):
    robot.connect()
    robot.enable()
    robot.motors["left"].fault("elbow_flex")
    assert sim.wait_until(lambda: robot.states()["left"] == "fault", 10.0)
    assert "elbow_flex" in robot.faults()["left"]
    assert robot.states()["right"] == "enabled"


def test_unverified_return_to_rest_keeps_holding(robot):
    robot.connect()
    robot.enable()
    q = robot.joint_positions()["left"]
    q[1] -= 0.2
    robot.command({"left": q})
    assert sim.wait_until(lambda: abs(robot.joint_positions()["left"][1] - q[1]) < 1e-3)
    robot.motors["left"].offline("shoulder_lift")     # the shoulder stops answering
    rest = robot.stop()
    assert rest == {"left": False, "right": True}
    assert robot.states()["left"] in ("enabled", "fault")       # torque stays on
    assert robot.states()["right"] == "connected"


def test_camera_dropout(robot):
    robot.connect()
    assert robot.frame("left", 1.0) is not None
    robot.cameras["left"].dead = True
    assert robot.frame("left", 1.0) is None
    assert "left" not in robot.frames(1.0)


def test_camera_failure_on_connect_leaves_nothing_open(robot):
    robot.cameras["right"].dead = True
    with pytest.raises(OSError):
        robot.connect()
    assert robot.states() == {"left": "disconnected", "right": "disconnected"}
    assert not robot.cameras["top"].is_open


def test_context_manager_stops():
    robot, _ = sim.sim_robot("metal", sides=("right",), cameras=())
    with robot:
        robot.enable()
        assert robot.states() == {"right": "enabled"}
    assert robot.states() == {"right": "disconnected"}


def test_offline_models():
    robot = Robot.offline("maker", ["left"])
    assert robot.sides == ["left"]
    pose = robot.tool_pose("left", [0, -1.2, 0.81, -0.1, 0, 0, 0])
    assert np.isfinite(pose).all()
    ik = robot.ik("left", "numpy")
    assert ik.max_velocity.shape == (6,)


def test_a_wall_clock_step_does_not_hide_a_camera_dropout(robot):
    robot.connect()
    camera = robot.cameras["left"]
    camera.newest = lambda: sim.frame(np.zeros((4, 4, 3), np.uint8), age=5.0, wall_offset=3600.0)
    assert robot.frame("left", 1.0) is None and robot.frame("left")[1] >= 5.0


def test_the_gripper_follows_the_profile_calibration():
    from dataclasses import replace

    from makermods_playground.core.config import sdk
    from makermods_playground.core.limits import GRIPPER_OPEN_MARGIN, Limiter
    cfg = sdk("metal").rig.ArmSpec(backend="gs_usb").config()
    joints = list(cfg.joints)
    joints[6] = replace(joints[6], open_position=math.radians(100.0))
    robot, _ = sim.sim_robot("metal", sides=("left",), cameras=(), cfg=replace(cfg, joints=joints))
    closed, full = robot.gripper_range("left")
    assert full == pytest.approx(math.radians(100.0))
    assert robot.gripper_position("left", 1.0) == pytest.approx(full) and robot.opening("left", full) == 1.0
    limiter = Limiter.for_arm(robot, "left", np.ones(6))
    assert limiter.hi[6] == pytest.approx(math.radians(100.0) - GRIPPER_OPEN_MARGIN)


def test_disconnect_reports_an_arm_left_energized_by_an_earlier_program():
    robot, motors = sim.sim_robot("metal", sides=("left",), cameras=(), enabled=True)
    robot.connect()
    assert robot.torque() == {"left": "on"}
    assert robot.disconnect() == {"left": "energized"}
    assert all(m.enabled for m in motors["left"].motors.values())
    robot, _ = sim.sim_robot("metal", sides=("left",), cameras=())
    robot.connect()
    assert robot.torque() == {"left": "off"} and robot.disconnect() == {"left": "released"}


def test_an_arm_enabled_away_from_rest_comes_home_to_the_profile_rest_pose():
    robot, motors = sim.sim_robot("maker", cameras=())            # sim.START: away from the rest pose
    robot.connect()
    near = list(np.asarray(sim.rest_pose("maker")) + np.radians(3.0))
    robot.arms["right"], motors["right"] = sim.sim_arm("maker", start=near)    # within rest_settle of it
    robot.group.arms["right"] = robot.arms["right"]
    robot.arms["right"].connect()
    try:
        robot.enable()
        assert robot.stop() == {"left": True, "right": True}
        settle = robot.arms["left"].config.rest_settle
        np.testing.assert_allclose(motors["left"].positions[:6], sim.rest_pose("maker")[:6], atol=settle)
        np.testing.assert_allclose(motors["right"].positions[:6], near[:6], atol=settle)   # where it rested
        assert robot.rehomed == ["left"]
    finally:
        robot.estop()
        robot.disconnect()


def test_an_arm_enabled_away_from_rest_comes_home_when_the_other_fails_to_enable():
    robot, motors = sim.sim_robot("metal", cameras=())
    robot.connect()
    motors["right"].fault("wrist_flex")
    try:
        with pytest.raises(Exception, match="fault"):
            robot.enable()
        assert robot.states()["left"] == "connected"          # verified home, torque off
        np.testing.assert_allclose(motors["left"].positions[:6], sim.rest_pose("metal")[:6],
                                   atol=robot.arms["left"].config.rest_settle)
    finally:
        robot.estop()
        robot.disconnect()


def test_a_url_camera_is_read_through_the_sdk(tmp_path):
    import cv2
    jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))[1].tobytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    definition = config.RobotDef("cam", "metal", {"left": {"backend": "gs_usb", "port": "gs_usb:x"}},
                                 {"top": f"url:http://127.0.0.1:{server.server_address[1]}/frame.jpg"})
    arm, _ = sim.sim_arm("metal")
    robot = Robot(definition, arms={"left": arm})
    try:
        robot.cameras["top"].open(timeout=5.0)
        assert sim.wait_until(lambda: robot.frame("top") is not None, 5.0)
        assert robot.frame("top")[0].shape == (48, 64, 3)
    finally:
        robot.close_cameras()
        server.shutdown()
