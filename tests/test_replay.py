import json

import numpy as np
import pytest
import sim

from makermods_playground.core.robot import Robot
from makermods_playground.dream.replay import RecordedPolicy, load_recording, replay_dry
from makermods_playground.dream.runner import DreamRunner, PolicyFinished


def write_recording(folder):
    """The fixture as a recording in the degree format with a v1 trace (gripper opening in raw_chunks.npz)."""
    run = sim.fixture_run()
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({"arm_type": "metal", "prompts": [{"prompt": "pick the marker"}]}))
    chunks = [{**{k: c[k] for k in ("arrived", "observed", "played_from_row")},
               "arms": {s: {k: v for k, v in a.items() if k != "grip"} for s, a in c["arms"].items()}}
              for c in run["chunks"]]
    (folder / "ik_trace.json").write_text(json.dumps({"version": 1, "chunks": chunks}))
    raw = {s: np.array([np.column_stack([np.asarray(c["arms"][s]["q"]), c["arms"][s]["grip"]])
                        for c in run["chunks"]]) for s in ("left", "right")}
    np.savez(folder / "raw_chunks.npz", arrived=np.array([c["arrived"] for c in run["chunks"]]), **raw)
    columns = run["motion_deg"]["columns"]
    meas = [c.replace("cmd_", "meas_") for c in columns[2:]]
    lines = [",".join(columns + meas)]
    for row in run["motion_deg"]["rows"]:
        lines.append(",".join(str(v) for v in row + row[2:]))
    (folder / "motion.csv").write_text("\n".join(lines) + "\n")
    return folder


def test_load_degree_recording(tmp_path):
    rec = load_recording(write_recording(tmp_path / "run"))
    assert rec.arm_type == "metal" and rec.sides == ["left", "right"] and rec.task == "pick the marker"
    assert len(rec.chunks) == 2 and rec.chunks[0]["heads"]["left"]["eef_xyz"].shape == (48, 3)
    assert rec.motion["cmd"]["left"].shape[1] == 7
    assert np.nanmax(np.abs(rec.motion["cmd"]["left"])) < 3.2        # read as radians


def test_dry_replay_matches_the_recorded_commands(tmp_path):
    rec = load_recording(write_recording(tmp_path / "run"))
    report = replay_dry(rec, Robot.offline("metal"))
    for side in ("left", "right"):
        arm = report["arms"][side]
        assert arm["ik_rad"]["p95"] < 0.025
        assert arm["command_rad"]["n"] > 0 and arm["command_rad"]["p95"] < 0.025
        assert arm["command_rad"]["max"] < 0.1
        assert arm["gripper_rad"]["p95"] < 0.01
        assert arm["max_step_rad"]["ours"] == pytest.approx(arm["max_step_rad"]["recorded"], abs=0.01)


def test_recorded_policy_gives_the_chunks_in_order(tmp_path):
    rec = load_recording(write_recording(tmp_path / "run"))
    policy = RecordedPolicy(rec)
    per_key, timing = policy.act()
    np.testing.assert_allclose(per_key["action.left_eef_xyz"], rec.chunks[0]["heads"]["left"]["eef_xyz"])
    policy.act()
    with pytest.raises(PolicyFinished):
        policy.act()


def test_replay_drives_a_robot(tmp_path):
    rec = load_recording(write_recording(tmp_path / "run"))
    robot, _ = sim.sim_robot("metal", cameras=())
    robot.connect()
    result = DreamRunner(robot, rec.task, model="replay", settings=robot.settings,
                         policy=RecordedPolicy(rec)).run(timeout_s=30)
    robot.disconnect()
    assert result.stop_reason == "the recording has no more chunks" and result.chunks == 2
    assert result.rest == {"left": True, "right": True}


def test_a_pickled_recording_is_refused_before_anything_runs(tmp_path):
    folder = write_recording(tmp_path / "run")
    ragged = np.array([np.zeros(3), np.zeros(4)], dtype=object)
    np.savez(folder / "raw_chunks.npz", left=ragged, right=ragged)
    with pytest.raises(ValueError, match="pickle"):
        load_recording(folder)


def test_a_short_dry_replay_prints_missing_numbers(tmp_path, capsys):
    from makermods_playground import cli
    folder = write_recording(tmp_path / "run")
    lines = (folder / "motion.csv").read_text().splitlines()
    phases = [line.split(",")[1] for line in lines[1:]]
    first = phases.index("1")
    (folder / "motion.csv").write_text("\n".join([lines[0], lines[1 + first]]) + "\n")   # one policy tick
    assert cli.main(["dream", "replay", str(folder), "--dry", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["arms"]["left"]["max_step_rad"]["ours"] is None
    assert cli.main(["dream", "replay", str(folder), "--dry"]) == 0
    assert "largest step per tick - (recorded -)" in capsys.readouterr().out
