import json
import os
import signal
import threading

import pytest
import sim
from test_config import write_rig
from test_replay import write_recording

from makermods_playground import cli
from makermods_playground.core import config
from makermods_playground.core.robot import Robot
from makermods_playground.dream import cli as dream_cli
from makermods_playground.dream import runner as runner_module


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, (json.loads(out) if "--json" in argv and out.strip() else out)


class Made(list):
    """The robots Robot.load made; ``options`` go to sim.sim_robot for the next ones, ``prepare(robot)`` runs on
    each."""
    options: dict
    prepare = None


@pytest.fixture
def sim_robots(monkeypatch):
    """Robot.load gives a simulated robot (the definition must exist, as for a real one)."""
    made = Made()
    made.options = {}

    def load(cls, name=None):
        definition = config.load(name)
        robot, motors = sim.sim_robot(definition.arm, sides=tuple(definition.sides),
                                      cameras=tuple(definition.cameras), name=definition.name, **made.options)
        robot.motors = motors
        if made.prepare:
            made.prepare(robot)
        made.append(robot)
        return robot

    monkeypatch.setattr(Robot, "load", classmethod(load))
    config.save(config.RobotDef("lab", "metal", {"left": {"backend": "gs_usb", "port": "gs_usb:A"},
                                                 "right": {"backend": "gs_usb", "port": "gs_usb:B"}},
                                {"top": "usb:t", "left": "usb:l", "right": "usb:r"}, config.Settings(record=False)))
    yield made
    for robot in made:
        robot.estop()


def test_robot_definitions(capsys, tmp_path):
    rig = write_rig(tmp_path / "rig.yaml")
    code, out = run(capsys, "robot", "init", "lab", "--arm", "metal", "--from-rig", str(rig), "--json")
    assert code == 0 and out["arms"]["left"]["port"] == "gs_usb:AAA" and out["cameras"]["left"] == "usb:lw"
    code, out = run(capsys, "robot", "init", "lab", "--arm", "metal", "--from-rig", str(rig), "--json")
    assert code == 1 and "exists" in out["error"]
    code, out = run(capsys, "robot", "init", "solo", "--arm", "maker", "--port", "right=can1",
                    "--camera", "top=usb:x", "--json")
    assert code == 0 and out["arms"] == {"right": {"backend": "socketcan", "channel": "can1"}}
    code, out = run(capsys, "robot", "list", "--json")
    assert [r["name"] for r in out["robots"]] == ["lab", "solo"]
    code, out = run(capsys, "robot", "show", "solo", "--json")
    assert out["arm"] == "maker"
    code, out = run(capsys, "robot", "remove", "solo", "--json")
    assert code == 0 and config.robot_names() == ["lab"]
    code, out = run(capsys, "robot", "show", "--json")
    assert out["name"] == "lab"
    code, out = run(capsys, "robot", "init", "x", "--arm", "metal", "--json")
    assert code == 1 and "--from-rig" in out["error"]


def test_check_and_status(capsys, sim_robots):
    code, out = run(capsys, "robot", "check", "--json")
    assert code == 0 and out["ok"] and set(out["cameras"]) == {"top", "left", "right"}
    assert out["arms"]["left"]["ok"] and "elbow_flex" in out["arms"]["left"]["temperatures"]
    assert sim_robots[-1].states() == {"left": "disconnected", "right": "disconnected"}
    code, out = run(capsys, "robot", "status", "--json")
    assert code == 0 and out["arms"]["left"]["state"] == "connected"
    assert sim_robots[-1].states() == {"left": "disconnected", "right": "disconnected"}


def test_check_reports_a_fault_and_a_dead_camera(capsys, sim_robots, monkeypatch):
    original = Robot.load.__func__

    def broken(cls, name=None):
        robot = original(cls, name)
        robot.motors["right"].fault("wrist_flex")
        robot.cameras["top"].dead = True
        return robot

    monkeypatch.setattr(Robot, "load", classmethod(broken))
    code, out = run(capsys, "robot", "check", "--json")
    assert code == 1 and not out["ok"]
    assert "fault" in out["arms"]["right"]["error"] and not out["cameras"]["top"]["ok"]


def test_motion_commands(capsys, sim_robots):
    code, out = run(capsys, "robot", "hold", "--for", "0.3", "--json")
    assert code == 1 and "--yes" in out["error"]
    code, out = run(capsys, "robot", "hold", "--for", "0.3", "--yes", "--json")
    assert code == 0 and out["then"] == "rest" and out["rest"] == {"left": True, "right": True}
    code, out = run(capsys, "robot", "hold", "--for", "0.2", "--then", "stay", "--yes", "--json")
    assert code == 0 and out["holding"]["left"] == {"gripper": 0.5}
    code, out = run(capsys, "robot", "rest", "--yes", "--json")
    assert code == 0 and out["arrived"] == {"left": True, "right": True}
    assert out["torque"] == {"left": "released", "right": "released"}
    code, out = run(capsys, "robot", "release", "--yes", "--json")
    assert code == 0 and all(r["ok"] for r in out["released"].values()) and set(out["released"]) == {"left", "right"}


def test_cameras(capsys, sim_robots, tmp_path):
    code, out = run(capsys, "cameras", "snapshot", "--out", str(tmp_path), "--json")
    assert code == 0 and set(out["saved"]) == {"top", "left", "right"}
    assert (tmp_path / "lab_top.jpg").stat().st_size > 0
    code, out = run(capsys, "cameras", "stream", "--for", "0.5", "--json")
    assert code == 0 and all(f > 10 for f in out["fps"].values())


def test_dream_check(capsys, monkeypatch):
    code, out = run(capsys, "dream", "check", "--json")
    assert code == 1 and out["key"] is False
    monkeypatch.setattr(dream_cli, "check", lambda: {
        "credentials": "c", "gateway": "g", "key": True,
        "models": {"beta": {"id": "dreamlabs/dream-1-beta", "listed": True, "servable": True}}})
    code, out = run(capsys, "dream", "check", "--json")
    assert code == 0 and out["models"]["beta"]["servable"]


def test_dream_run(capsys, sim_robots, monkeypatch):
    def factory(model, sides, timeout_ms, **stand_ins):
        return sim.FakePolicy(sim_robots[-1], sides)

    monkeypatch.setattr(runner_module, "DreamPolicy", factory)
    code, out = run(capsys, "dream", "run", "pick up the marker", "--timeout", "1.5", "--json")
    assert code == 1 and "--yes" in out["error"]
    code, out = run(capsys, "dream", "run", "pick up the marker", "--timeout", "1.5", "--model", "flash",
                    "--no-record", "--yes", "--json")
    assert code == 0 and out["stop_reason"] == "timeout" and out["chunks"] >= 1
    assert out["rest"] == {"left": True, "right": True} and out["recording"] is None
    assert sim_robots[-1].states() == {"left": "disconnected", "right": "disconnected"}


def test_dream_replay(capsys, sim_robots, tmp_path):
    folder = write_recording(tmp_path / "run")
    code, out = run(capsys, "dream", "replay", str(folder), "--dry", "--json")
    assert code == 0 and out["chunks"] == 2 and out["arms"]["left"]["ik_rad"]["p95"] < 0.025
    code, text = run(capsys, "dream", "replay", str(folder), "--dry")
    assert "IK vs recorded" in text
    code, out = run(capsys, "dream", "replay", str(folder), "--yes", "--no-record", "--json")
    assert code == 0 and out["stop_reason"] == "the recording has no more chunks" and out["chunks"] == 2


def test_rest_moves_one_arm_at_a_time_and_stops_at_the_first_unverified(capsys, sim_robots, monkeypatch):
    tried = []

    def prepare(robot):
        for side, arm in robot.arms.items():
            def stop(side=side, arm=arm, real=arm.stop):
                if arm.state.value != "enabled":
                    return real()
                tried.append(side)
                return side != "left" and real()
            arm.stop = stop

    sim_robots.prepare = prepare
    code, out = run(capsys, "robot", "rest", "--yes", "--json")
    assert code == 1 and "not verified on left, right" in out["error"]
    assert tried == ["left"]        # the right arm never started
    robot = sim_robots[-1]
    assert robot.states() == {"left": "disconnected", "right": "disconnected"}
    assert all(m.enabled for m in robot.motors["left"].motors.values() if m is not robot.motors["left"].motor(6))


def test_sigterm_ends_a_hold_through_its_return_to_rest(capsys, sim_robots):
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
    code, out = run(capsys, "robot", "hold", "--yes", "--json")
    assert code == 0 and out["then"] == "rest" and out["rest"] == {"left": True, "right": True}
    assert out["torque"] == {"left": "released", "right": "released"}


def test_a_hold_ends_on_a_fault_and_exits_with_an_error(capsys, sim_robots):
    sim_robots.prepare = lambda robot: threading.Timer(0.4, robot.motors["left"].fault, ("elbow_flex",)).start()
    code, out = run(capsys, "robot", "hold", "--then", "release", "--yes")
    assert code == 1 and "fault on left" in out


def test_stay_is_refused_before_energizing_when_a_stiff_joint_has_a_watchdog(capsys, sim_robots):
    sim_robots.options["cfg"] = sim.watchdog_profile()
    for argv in (("robot", "hold", "--for", "0.1", "--then", "stay"),
                 ("dream", "run", "x", "--then", "stay", "--no-record")):
        code, out = run(capsys, *argv, "--yes", "--json")
        assert code == 1 and "would drop shoulder_lift" in out["error"]
    assert all(r.states() == {"left": "disconnected", "right": "disconnected"} for r in sim_robots)
    assert not any(m.enabled for r in sim_robots for s in r.motors.values() for m in s.motors.values())


def test_an_unverified_arm_that_would_drop_is_kept_holding_until_release_is_typed(sim_robots, monkeypatch):
    from makermods_playground.core.cli import close_holding
    robot, motors = sim.sim_robot(sides=("left",), cameras=(), cfg=sim.watchdog_profile())
    robot.connect()
    robot.enable()
    monkeypatch.setattr("builtins.input", lambda *_: "RELEASE")
    assert close_holding(robot, lambda *_: None) == {"left": "released"}   # held until RELEASE, never detached
    safe, _ = sim.sim_robot(sides=("left",), cameras=())
    safe.connect()
    safe.enable()
    assert close_holding(safe, lambda *_: None) == {"left": "holding"}     # detached, still on
    safe.estop()


@pytest.mark.parametrize("then, words", [("release", "DROPS"), ("stay", "torque still on"),
                                         ("rest", "torque goes off")])
def test_the_hold_confirmation_names_the_ending(capsys, sim_robots, then, words):
    code, out = run(capsys, "robot", "hold", "--then", then, "--json")
    assert code == 1 and words in out["error"] and "--yes" in out["error"]


def test_release_works_on_a_faulted_arm_and_reports_a_missing_motor(capsys, sim_robots):
    sim_robots.prepare = lambda robot: robot.motors["left"].fault("wrist_flex")
    code, out = run(capsys, "robot", "release", "--yes", "--json")
    assert code == 0 and out["released"]["left"] == {"ok": True, "unconfirmed": [], "missing": []}
    sim_robots.prepare = lambda robot: robot.motors["right"].offline("wrist_roll")
    code, out = run(capsys, "robot", "release", "--yes", "--json")
    assert code == 1 and out["released"]["right"]["missing"] == ["wrist_roll"] and out["released"]["left"]["ok"]
