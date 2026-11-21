import math

import numpy as np
import pytest
import sim

from makermods_playground.core.robot import Robot
from makermods_playground.dream import decode
from makermods_playground.dream.scheduling import Pacer, StreamSchedule, at_cubic, at_linear


@pytest.fixture(scope="module")
def metal():
    return Robot.offline("metal")


def test_rot6_round_trip():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = q * np.sign(np.linalg.det(q))
    np.testing.assert_allclose(decode.rot6_to_matrix(decode.matrix_to_rot6(R)), R, atol=1e-12)


def test_head_from_reply_validates():
    rows = 10
    good = {"action.left_eef_xyz": np.zeros((rows, 3)), "action.left_eef_rot6": np.tile([1, 0, 0, 0, 1, 0], (rows, 1)),
            "action.left_grip": np.zeros((rows, 1))}
    heads = decode.head_from_reply(good, ["left"])
    assert heads["left"]["grip"].shape == (rows,)
    with pytest.raises(decode.DecodeError, match="left"):
        decode.head_from_reply({**good, "action.left_eef_xyz": np.full((rows, 3), np.nan)}, ["left"])
    with pytest.raises(decode.DecodeError, match="right"):
        decode.head_from_reply(good, ["left", "right"])
    other = {k.replace("left", "right"): v[:5] for k, v in good.items()}
    with pytest.raises(decode.DecodeError, match="different lengths"):
        decode.head_from_reply({**good, **other}, ["left", "right"])


def test_takeover():
    plan = np.outer(np.arange(48) * 0.01, np.ones(6))
    assert decode.takeover(None, plan[0], plan[0], 3, 48) is None
    assert decode.takeover(plan, plan[0], plan[0], 3, 40) is None                    # another length
    clock, q, b1, b2 = decode.takeover(plan, plan[10], plan[10], 3.4, 48)
    assert clock == 3 and np.allclose(q, plan[14]) and np.allclose(b1, plan[13]) and np.allclose(b2, plan[12])
    assert decode.takeover(plan, plan[10] + 1.0, plan[10], 3, 48) is None            # off the plan
    clock, q, _, _ = decode.takeover(plan, plan[46], plan[46], 10, 48)               # ran out of rows: holds
    assert clock == 0 and np.allclose(q, plan[47])
    clock, _, _, _ = decode.takeover(plan, plan[5], None, 2, 48)                      # nearest the measured pose
    assert clock == 2


def test_solve_matches_the_recorded_run(metal):
    """The recorded run's heads re-solved by the SDK's numpy IK from the recorded starts land on the recorded
    joints within the Metal URDF's JOINT5 difference to Dream's (about 0.9 deg at the tool)."""
    run = sim.fixture_run()
    for chunk in run["chunks"]:
        for side, arm in chunk["arms"].items():
            head = decode.check_head({"eef_xyz": arm["xyz"], "eef_rot6": arm["rot6"], "grip": arm["grip"]}, side)
            out = decode.solve(metal.ik(side, "numpy"), metal.sdk.trajectory_ik, head, arm["q_from"],
                               arm["q_before"], arm["skip"], arm["speed_cap"])
            diff = np.abs(out.q - np.asarray(arm["q"]))
            assert np.percentile(diff, 95) < 0.025
            assert diff.max() < 0.05
            assert out.trace["pos_err_max"] < 0.05
            if arm["speed_cap"]:
                steps = np.abs(np.diff(np.vstack([arm["q_from"], out.q]), axis=0))
                assert steps.max() <= arm["speed_cap"] / 20 + 1e-9


def test_solve_skips_rows(metal):
    arm = sim.fixture_run()["chunks"][1]["arms"]["left"]
    head = decode.check_head({"eef_xyz": arm["xyz"], "eef_rot6": arm["rot6"], "grip": arm["grip"]}, "left")
    out = decode.solve(metal.ik("left", "numpy"), metal.sdk.trajectory_ik, head, arm["q_from"], arm["q_before"], 5,
                       None)
    assert out.q.shape == (48, 6)
    np.testing.assert_allclose(out.q[:5], np.tile(arm["q_from"], (5, 1)))
    assert out.trace["skip"] == 5


def test_cubic_and_linear():
    rows = np.array([[0.0], [1.0], [4.0], [9.0]])
    assert at_cubic(rows, 2)[0] == 4.0
    assert at_cubic(rows, -1)[0] == 0.0 and at_cubic(rows, 10)[0] == 9.0
    assert at_linear(rows, 1.5)[0] == 2.5


def chunk(value, rows=20):
    return {"left": np.tile(np.r_[np.full(6, value), 0.5], (rows, 1)) + np.c_[np.linspace(0, 0.1, rows)[:, None]
                                                                                * np.ones((1, 6)), np.zeros(rows)]}


def test_schedule_plays_to_the_last_row_and_holds():
    s = StreamSchedule(ease_in_s=0.0, ease_out_s=0.2)
    s.install(chunk(0.0), 0.0, 0)
    t, last = 0.0, None
    while True:
        t += 0.025
        target = s.sample(t)
        if target is None:
            break
        last = target["left"]
    np.testing.assert_allclose(last[:6], 0.1)
    assert s.remaining() == 0 and t == pytest.approx(19 / 20 + 0.2, abs=0.3)


def test_schedule_ease_in_starts_at_the_held_command():
    s = StreamSchedule(ease_in_s=0.6, ease_out_s=0.2)
    s.install(chunk(0.5), 0.0, 0, origin={"left": np.zeros(7)})
    first = s.sample(0.025)["left"]
    assert np.all(np.abs(first[:6]) < 0.01)
    late = s.sample(0.7)["left"]
    assert late[0] > 0.5


def test_a_chunk_installed_mid_motion_joins_the_outgoing_command():
    s = StreamSchedule(ease_in_s=0.6, ease_out_s=0.2)
    s.install(chunk(0.0), 0.0, 0)
    t, out = 0.0, None
    for _ in range(10):
        t += 0.025
        out = s.sample(t)["left"]
    new = chunk(0.08)
    new["left"][:, 6] = 0.9                     # the gripper too
    s.install(new, t, 0, origin={"left": out}, ramp=False)
    first = s.sample(t + 0.025)["left"]
    assert np.max(np.abs(first - out)) < 0.01    # no jump at the handoff, joints or gripper
    assert first[0] > out[0]                     # and still moving: no ramp from standstill
    late = s.sample(t + 0.7)["left"]
    assert late[6] == pytest.approx(0.9) and late[0] > 0.08


def test_schedule_installs_past_skipped_rows():
    s = StreamSchedule()
    s.install(chunk(0.0), 0.0, 7)
    assert s.position == 7 and s.executed() == 0 and s.remaining() == 12


def test_pacer():
    p = Pacer()
    assert p.mode == "async" and p.refill_due(0, 0)
    p.record(0.3)
    assert not p.refill_due(10, 30) and p.refill_due(24, 20)
    for _ in range(3):
        p.record(1.0)
    assert p.mode == "sync" and not p.refill_due(40, 2) and p.refill_due(40, 0)
    assert math.isclose(min(p.rtts), 0.3)
