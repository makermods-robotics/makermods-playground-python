import json
import threading

import dream
import numpy as np
import pytest
import sim

from makermods_playground.core.config import Settings
from makermods_playground.dream import runner as runner_module
from makermods_playground.dream.runner import WARMING, DreamRunner


def make(arm_type="metal", sides=("left", "right"), record=False, tmp_path=None, **policy_kwargs):
    settings = Settings(record=record, recordings_dir=str(tmp_path or "."))
    robot, motors = sim.sim_robot(arm_type, sides=sides, settings=settings)
    robot.connect()
    policy = sim.FakePolicy(robot, sides, **policy_kwargs)
    return robot, motors, policy


@pytest.mark.parametrize("arm_type, sides", [("metal", ("left", "right")), ("maker", ("right",))])
def test_timeout_run_returns_to_rest(arm_type, sides):
    robot, _, policy = make(arm_type, sides)
    start = robot.joint_positions()
    runner = DreamRunner(robot, "pick up the marker", settings=robot.settings, policy=policy)
    result = runner.run(timeout_s=2.0)
    robot.disconnect()
    assert result.stop_reason == "timeout" and not result.error
    assert result.chunks >= 2 and result.ticks > 10      # the servo ran throughout (fewer ticks on a loaded machine)
    assert result.rest == {s: True for s in sides}
    assert result.timing["rtt_ms"]["p50"] > 0 and result.timing["ik_ms"]["p50"] > 0
    assert result.ik["pos_err_mm"]["max"] < 50      # the first chunk ramps in at 0.5 rad/s
    assert policy.resets == 1 and policy.closed and policy.pushes > 0
    assert result.pacing == "async" and result.session_ids == ["sess-1"]
    for s in sides:
        assert policy.observations[0][s]["tool"].shape == (4, 4)
        np.testing.assert_allclose(robot.joint_positions()[s][:6], start[s][:6], atol=0.05)
    state = runner.state()
    assert state["phase"] == "done" and any(e["text"].startswith("run ended") for e in state["events"])


def test_motion_follows_the_head():
    robot, _, policy = make("metal", ("left",))
    before = robot.tool_pose("left", robot.joint_positions()["left"])[:3, 3]
    moved = []
    runner = DreamRunner(robot, "move", settings=robot.settings, policy=policy)
    orig = runner._on_tick

    def spy(now, phase, commanded, measured):
        moved.append(robot.tool_pose("left", measured["left"])[:3, 3] - before)
        orig(now, phase, commanded, measured)

    runner._on_tick = spy
    runner.run(timeout_s=2.5, then="stay")
    assert max(np.linalg.norm(m) for m in moved) > 0.002
    assert robot.states() == {"left": "enabled"}
    assert robot.stop() == {"left": True}
    robot.disconnect()


def test_stop_event_and_recording(tmp_path):
    robot, _, policy = make(record=True, tmp_path=tmp_path)
    stop = threading.Event()
    threading.Timer(1.5, stop.set).start()
    result = DreamRunner(robot, "wave", settings=robot.settings, policy=policy).run(timeout_s=30, stop_event=stop)
    robot.disconnect()
    assert result.stop_reason == "stopped" and not result.error
    folder = tmp_path / result.recording.split("/")[-1]
    files = {p.name for p in folder.iterdir()}
    assert {"motion.csv", "chunks.csv", "timing.csv", "raw_chunks.npz", "ik_trace.json", "run.json",
            "top.mp4", "left.mp4", "right.mp4"} <= files
    run = json.loads((folder / "run.json").read_text())
    assert run["units"] == "rad" and run["stopped_because"] == "stopped" and run["task"] == "wave"
    trace = json.loads((folder / "ik_trace.json").read_text())
    arm = trace["chunks"][0]["arms"]["left"]
    assert {"xyz", "rot6", "grip", "q_from", "q_before", "q", "skip", "speed_cap"} <= set(arm)
    assert arm["speed_cap"] == 0.5
    raw = np.load(folder / "raw_chunks.npz")
    assert raw["left_eef_xyz"].shape[1:] == (policy.rows, 3)
    header = (folder / "motion.csv").read_text().splitlines()[0].split(",")
    assert header[:3] == ["t", "phase", "cmd_left_shoulder_pan"] and len(header) == 2 + 2 * 14


def test_policy_error_ends_the_run():
    robot, _, policy = make(errors=[dream.DreamAuthError("key refused", "DA-1000", "unknown_key")])
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=10)
    robot.disconnect()
    assert result.error and "key refused" in result.stop_reason and result.chunks == 0
    assert result.rest == {"left": True, "right": True}


def test_pool_warming_ends_at_once():
    robot, _, policy = make(errors=[dream.DreamServerError("warming", "DA-2000", "pool_warming", True, 30.0)])
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=10)
    robot.disconnect()
    assert result.stop_reason == WARMING and result.error


def test_retryable_refusal_is_asked_again():
    robot, _, policy = make(errors=[dream.DreamServerError("busy", "DA-2000", "rate_limited", True, 0.5)])
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=2.5)
    robot.disconnect()
    assert result.stop_reason == "timeout" and result.chunks >= 1 and policy.resets == 1


def test_policy_finished():
    robot, _, policy = make(limit=1)
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=20)
    robot.disconnect()
    assert result.stop_reason == "no more chunks" and not result.error and result.chunks == 1


def test_camera_dropout_ends_the_run():
    robot, _, policy = make()
    threading.Timer(1.0, lambda: setattr(robot.cameras["right"], "dead", True)).start()
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=20)
    robot.disconnect()
    assert result.stop_reason == "camera right: no fresh frame" and result.error
    assert result.rest == {"left": True, "right": True}


def test_motor_fault_ends_the_run_and_the_arm_keeps_holding():
    robot, motors, policy = make()
    threading.Timer(1.0, lambda: motors["left"].offline("shoulder_lift")).start()
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=20)
    assert result.error and result.stop_reason.startswith("left arm")      # stale feedback, or the SDK's fault
    assert result.rest == {"left": False, "right": True}
    assert sim.wait_until(lambda: robot.states()["left"] == "fault")     # the SDK's feedback watchdog, still holding
    robot.estop()
    robot.disconnect()


def test_enable_cancelled():
    robot, _, policy = make()
    stop = threading.Event()
    stop.set()
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=5, stop_event=stop)
    robot.disconnect()
    assert result.error and "cancelled" in result.stop_reason and result.ticks == 0


def test_dream_policy_is_built_from_the_model(monkeypatch):
    robot, _, fake = make()
    built = {}

    def factory(model, sides, timeout_ms, **stand_ins):
        built.update(model=model, sides=sides, timeout_ms=timeout_ms)
        return fake

    monkeypatch.setattr(runner_module, "DreamPolicy", factory)
    result = DreamRunner(robot, "x", model="flash", settings=robot.settings).run(timeout_s=1.0)
    robot.disconnect()
    assert built == {"model": "flash", "sides": ["left", "right"], "timeout_ms": 60000}
    assert result.stop_reason == "timeout"


def test_nothing_moves_without_the_cameras_the_model_needs():
    robot, _ = sim.sim_robot("metal", sides=("left",), cameras=("top", "left"))
    robot.connect()
    policy = sim.FakePolicy(robot, ["left"])
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=5)
    assert result.error and "right camera" in result.stop_reason
    assert robot.states() == {"left": "connected"} and policy.calls == 0
    robot.disconnect()


def test_the_arms_freeze_before_the_run_logs_or_waits_for_its_threads():
    robot, _, policy = make(latency=0.5)
    order = []
    hold = robot.hold
    robot.hold = lambda: (order.append("hold"), hold())[1]
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy,
                         on_event=lambda kind, text: order.append(text))
    runner.run(timeout_s=1.0)
    robot.disconnect()
    assert order.index("hold") < order.index("run ended: timeout")


def test_an_error_after_enable_still_returns_to_rest(monkeypatch):
    robot, _, policy = make()
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy)
    monkeypatch.setattr(runner, "_start_recording", lambda: (_ for _ in ()).throw(RuntimeError("disk full")))
    result = runner.run(timeout_s=5)
    assert result.error and "disk full" in result.stop_reason
    assert result.rest == {"left": True, "right": True} and robot.states() == {"left": "connected",
                                                                               "right": "connected"}
    robot.disconnect()


def test_a_reply_to_a_stale_observation_is_discarded(monkeypatch):
    monkeypatch.setattr(runner_module, "REPLY_MAX_AGE_S", 0.5)
    robot, _, policy = make(latency=0.8)     # the arms hold: nothing played caps how late a reply may land
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy)
    result = runner.run(timeout_s=3.0)
    robot.disconnect()
    assert result.chunks == 0 and policy.calls >= 2
    assert any("discarded" in e["text"] for e in runner.events)


def test_the_first_installed_chunk_is_capped_even_after_a_retry():
    robot, _, policy = make(errors=[dream.DreamServerError("busy", "DA-2000", "rate_limited", True, 0.1)])
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy)
    caps = []
    install = runner._install
    runner._install = lambda request, reply: (caps.append(reply["cap"]), install(request, reply))[1]
    runner.run(timeout_s=2.0)
    robot.disconnect()
    assert caps[0] == 0.5 and runner.chunks >= 1


def test_a_session_expiry_drops_the_old_plan_and_caps_again():
    robot, _, policy = make()
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy)
    runner.previous, runner.ramp = {"left": np.zeros((16, 6))}, False
    with runner.lock:
        runner._failed(dream.DreamSessionExpired("gone", "DA-3000", "session_expired"))
    assert runner.previous == {} and runner.ramp
    robot.disconnect()


def test_a_server_error_marked_not_retryable_ends_the_run_at_once():
    robot, _, policy = make(errors=[dream.DreamServerError("broken", "DA-2000", "internal", False, None)])
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=10)
    robot.disconnect()
    assert result.error and "broken" in result.stop_reason and policy.calls == 1


def test_a_hanging_retry_is_ended_by_the_servo(monkeypatch):
    monkeypatch.setattr(runner_module, "RETRY_FIRST_CHUNK_S", 1.0)
    robot, _, policy = make(latency=1.5, errors=[dream.DreamServerError("busy", "DA-2000", "rate_limited", True, 0.1)])
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=20)
    robot.disconnect()
    assert result.error and "retry window" in result.stop_reason
    # refused at 1.5 s, asked again at 2.0 s: the window ends the run at 2.5 s, not the reply at 3.5 s
    assert result.ticks < 3.0 * robot.settings.servo_hz


def test_the_planner_observation_is_always_installed():
    robot, _, policy = make()
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy)
    pushed = []
    policy.push = lambda images, observation, stamp: pushed.append((observation, stamp))
    runner.last_push = 1e12       # a camera tick already pushed with a later stamp
    observation = {"prompt": "x"}
    assert runner._push_frames(observation)
    assert pushed == [(observation, 1e12)] and runner.latest_observation is observation
    assert not runner._push_frames()      # the camera thread: nothing newer is due
    robot.disconnect()


def test_policy_frames_are_pushed_only_when_the_window_takes_one():
    robot, _, policy = make()
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=2.0)
    robot.disconnect()
    assert policy.pushes <= 2.0 * runner_module.POLICY_FRAME_HZ + result.chunks + 3


def test_a_finished_policy_plays_its_last_chunk_to_the_end():
    robot, _, policy = make(limit=1, rows=32)
    runner = DreamRunner(robot, "x", settings=robot.settings, policy=policy)
    left = []
    sample = runner._sample
    runner._sample = lambda now: (sample(now), left.append(runner.schedule.remaining()))[0]
    result = runner.run(timeout_s=20)
    robot.disconnect()
    assert result.stop_reason == "no more chunks" and not result.error
    assert max(left) > 10 and left[-1] == 0


def test_a_task_carrying_a_key_is_masked_in_the_live_state():
    robot, _, policy = make()
    runner = DreamRunner(robot, "use dl_live_abcdef123456 to pick", settings=robot.settings, policy=policy)
    assert "abcdef123456" not in runner.state()["task"]
    robot.disconnect()


def test_an_invalid_time_limit_or_setting_is_refused_before_enable():
    robot, _, policy = make()
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=float("nan"))
    assert result.error and "time limit" in result.stop_reason and robot.states()["left"] == "connected"
    bad = Settings(record=False, camera_max_age_s=float("nan"))
    result = DreamRunner(robot, "x", settings=bad, policy=policy).run(timeout_s=5)
    assert result.error and "camera_max_age_s" in result.stop_reason and robot.states()["left"] == "connected"
    robot.disconnect()


def test_jax_ik_compiles_before_enable_and_other_lengths_use_numpy(monkeypatch):
    robot, _, policy = make(rows=16)
    robot.settings = Settings(record=False, ik_solver="jax")
    warmed, solved = [], []
    real_ik = robot.ik

    class FakeJax(type(real_ik("left", "numpy"))):
        def warmup(self, rows):
            warmed.append((rows, robot.states()["left"]))

        def solve(self, *args, **kwargs):
            solved.append("jax")
            return super().solve(*args, **kwargs)

    def ik(side, solver=None):
        numpy_ik = real_ik(side, "numpy")
        if (solver or robot.settings.ik_solver) != "jax":
            return numpy_ik
        numpy_ik.__class__ = FakeJax
        return numpy_ik

    monkeypatch.setattr(robot, "ik", ik)
    result = DreamRunner(robot, "x", settings=robot.settings, policy=policy).run(timeout_s=1.5)
    robot.disconnect()
    assert warmed == [(48, "connected"), (48, "connected")]
    assert result.chunks >= 1 and solved == []      # 16-row replies: numpy


def test_a_video_failure_keeps_the_motion_log_and_the_run_error(tmp_path):
    from makermods_playground.core.recording import RunRecorder

    def frames():
        raise OSError("camera gone")

    recorder = RunRecorder(tmp_path, ["left"], frames, {"task": "x"}).start()
    assert sim.wait_until(lambda: recorder.error is not None)
    for i in range(5):              # the servo keeps logging after the videos stopped
        recorder.log.tick(float(i), 1, {"left": np.zeros(7)}, {"left": np.zeros(7)})
    assert recorder.finish({"error": True, "stopped_because": "fault"}) == "videos stopped: OSError: camera gone"
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["error"] is True and run["recording_error"] == "videos stopped: OSError: camera gone"
    assert len((tmp_path / "motion.csv").read_text().splitlines()) == 1 + 5
