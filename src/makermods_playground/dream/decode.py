"""End-effector chunks to joint rows.

A reply carries, per arm, a head of tool poses (``eef_xyz`` m, ``eef_rot6`` the rotation's first two columns) and
``grip`` openings, at one row per 1/ACTION_HZ s of playback, in that arm's base frame: the frame of the SDK chain's
world, with the tool point the profile's ``tcp``. Per arm the head is smoothed (Savitzky-Golay), cut at the row
where it takes over from the plan the arm is on and continued past its end, then solved by the SDK's trajectory IK
from the commands there, so the new rows continue the old motion. The takeover follows Dream Labs'
dreamlabs-client seam smoothing (Apache-2.0, see NOTICE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .scheduling import ACTION_HZ

SMOOTH_WINDOW = 15
SMOOTH_POLYORDER = 3
MAX_OFFSET_RAD = 0.5      # joint-space distance (L2) beyond which an arm is off the plan it was given
FIRST_CHUNK_SPEED = 0.5   # rad/s cap on an episode's first chunk, so the arms ramp in
MAX_ROWS = 256


class DecodeError(ValueError):
    pass


@dataclass
class Solved:
    q: np.ndarray            # (T, 6) joint rows, rad
    grip: np.ndarray         # (T,) opening in [0, 1]
    trace: dict              # what was solved from what, for ik_trace.json


def head_from_reply(per_key: dict, sides) -> dict[str, dict[str, np.ndarray]]:
    """Per arm: ``eef_xyz`` (T, 3), ``eef_rot6`` (T, 6), ``grip`` (T,) from a reply's ``action.<side>_*`` keys."""
    heads, horizon = {}, None
    for side in sides:
        try:
            xyz = np.asarray(per_key.get(f"action.{side}_eef_xyz"), dtype=float)
            rot6 = np.asarray(per_key.get(f"action.{side}_eef_rot6"), dtype=float)
            grip = np.asarray(per_key.get(f"action.{side}_grip"), dtype=float)
        except (TypeError, ValueError):
            xyz = rot6 = grip = np.zeros(0)
        heads[side] = check_head({"eef_xyz": xyz, "eef_rot6": rot6, "grip": grip}, side)
        rows = len(heads[side]["eef_xyz"])
        if horizon is not None and rows != horizon:
            raise DecodeError("the arms' chunks have different lengths")
        horizon = rows
    return heads


def check_head(head: dict, side: str) -> dict:
    xyz, rot6 = np.asarray(head["eef_xyz"], dtype=float), np.asarray(head["eef_rot6"], dtype=float)
    grip = np.asarray(head["grip"], dtype=float)
    grip = grip.reshape(-1) if grip.ndim == 2 and grip.shape[-1] == 1 else grip
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or not 2 <= len(xyz) <= MAX_ROWS or rot6.shape != (len(xyz), 6)
            or grip.shape != (len(xyz),) or not all(np.isfinite(v).all() for v in (xyz, rot6, grip))):
        raise DecodeError(f"{side}: the reply has no finite end-effector head (action.{side}_eef_xyz, _eef_rot6, "
                          "_grip)")
    return {"eef_xyz": xyz, "eef_rot6": rot6, "grip": grip}


def rot6_to_matrix(rot6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3): Gram-Schmidt on the two columns."""
    a1, a2 = rot6[..., :3], rot6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def matrix_to_rot6(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def head_poses(xyz: np.ndarray, rot6: np.ndarray) -> np.ndarray:
    T = np.broadcast_to(np.eye(4), (*xyz.shape[:-1], 4, 4)).copy()
    T[..., :3, :3] = rot6_to_matrix(rot6)
    T[..., :3, 3] = xyz
    return T


# ── takeover: where a new chunk continues the plan the arm is on ──
def nearest_row(rows: np.ndarray, q: np.ndarray) -> Optional[int]:
    """Row nearest ``q`` (of equal rows, the first, unless they run to the end), or None beyond MAX_OFFSET_RAD."""
    distances = np.linalg.norm(rows - q, axis=1)
    index = int(np.argmin(distances))
    if (distances[index:] == distances[index]).all():
        index = len(distances) - 1
    return index if distances[index] <= MAX_OFFSET_RAD else None


def takeover(previous: Optional[np.ndarray], measured: np.ndarray, commanded: Optional[np.ndarray],
             rows_in_flight: float, length: int):
    """(clock, q, before, before2) for a new chunk of ``length`` rows over the ``previous`` plan (T, 6): the rows the
    arm runs while the chunk is in flight (``clock``), the plan's command there and the two before it. None when
    there is no plan of that length or the arm is off it."""
    if previous is None or len(previous) != length:
        return None
    if commanded is None:
        anchor = nearest_row(previous, measured)
    elif np.linalg.norm(measured - commanded) > MAX_OFFSET_RAD:
        anchor = None
    else:
        row = nearest_row(previous, commanded)
        anchor = None if row is None else min(row + 1, length - 1)
    if anchor is None:
        return None
    clock = min(int(round(rows_in_flight)), length - 1 - anchor)
    at = anchor + clock
    return clock, previous[at], previous[max(at - 1, 0)], previous[max(at - 2, 0)]


def continued(poses: np.ndarray, skip: int, extend_poses) -> np.ndarray:
    """The poses from ``skip`` on, padded back to the same length by continuing the last rows' motion."""
    kept = poses[skip:]
    if len(kept) < 2:
        return np.concatenate([kept, np.repeat(poses[-1:], skip, axis=0)])
    return np.asarray(extend_poses(kept, skip))


def solve(ik, trajectory_ik, head: dict, q_from, q_before, skip: int, speed_cap: Optional[float],
          rate_hz: float = ACTION_HZ) -> Solved:
    """One arm's head solved to joint rows: ``skip`` copies of ``q_from``, then the solve of the rows from ``skip``
    on, continuing from ``q_from`` with ``q_before`` (newest first) as the motion into it."""
    xyz, rot6 = head["eef_xyz"], head["eef_rot6"]
    rows = len(xyz)
    poses = trajectory_ik.smooth_poses(head_poses(xyz, rot6), SMOOTH_WINDOW, SMOOTH_POLYORDER)
    padded = continued(poses, skip, trajectory_ik.extend_poses)
    q_from = np.asarray(q_from, dtype=float)
    result = ik.solve(padded, q_from, np.asarray(q_before, dtype=float), rate_hz=rate_hz, speed_cap=speed_cap)
    q = np.concatenate([np.repeat(q_from[None], skip, axis=0), result.q[:rows - skip]])
    trace = {"xyz": xyz.tolist(), "rot6": rot6.tolist(), "grip": head["grip"].tolist(), "q_from": q_from.tolist(),
             "q_before": np.asarray(q_before).tolist(), "skip": int(skip),
             "speed_cap": None if speed_cap is None or not math.isfinite(speed_cap) else speed_cap,
             "hz": rate_hz, "solve": "traj", "q": q.tolist(),
             "pos_err_max": float(np.max(result.pos_err)), "rot_err_max": float(np.max(result.rot_err)),
             "capped_ticks": int(np.count_nonzero(result.capped))}
    return Solved(q=q, grip=np.clip(head["grip"], 0.0, 1.0), trace=trace)
