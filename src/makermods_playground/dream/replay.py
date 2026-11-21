"""Recorded runs played again without the Dream API.

``RecordedPolicy`` hands a DreamRunner a recording's replies in order, so a run can be repeated on a robot.
``replay_dry`` re-solves each recorded reply from the joint state it was solved from, plays the result through the
same scheduling and limits at the recorded servo ticks, and compares it with what the recorded run commanded.

Recordings in degrees (``motion.csv`` without ``units: rad`` in run.json) and their ``ik_trace.json`` v1 (head in
the trace, gripper opening in ``raw_chunks.npz``) are read as well as this package's own.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.limits import Limiter
from . import decode
from .runner import PolicyFinished
from .scheduling import ACTION_HZ, StreamSchedule


@dataclass
class Recording:
    path: Path
    arm_type: str
    sides: list
    task: str
    chunks: list                       # {arrived, observed, skipped, heads: {side: head}, trace: {side: {...}}}
    motion: dict = field(default_factory=dict)   # t (N,), phase (N,), cmd/meas {side: (N, 7) rad}


def load_recording(path) -> Recording:
    path = Path(path).expanduser()
    if not (path / "ik_trace.json").exists():
        raise FileNotFoundError(f"{path}: no ik_trace.json (a run with no policy chunk)")
    info = json.loads((path / "run.json").read_text()) if (path / "run.json").exists() else {}
    trace = json.loads((path / "ik_trace.json").read_text())
    raw = np.load(path / "raw_chunks.npz", allow_pickle=False) if (path / "raw_chunks.npz").exists() else None
    chunks = []
    for i, entry in enumerate(trace["chunks"]):
        heads, traces = {}, {}
        for side, arm in entry["arms"].items():
            if "grip" in arm:
                grip = np.asarray(arm["grip"], dtype=float)
            elif raw is not None and f"{side}_grip" in raw.files:
                grip = np.asarray(raw[f"{side}_grip"][i], dtype=float)
            else:
                grip = np.asarray(raw[side][i], dtype=float)[:, 6]
            heads[side] = decode.check_head({"eef_xyz": arm["xyz"], "eef_rot6": arm["rot6"], "grip": grip}, side)
            traces[side] = arm
        chunks.append({"arrived": entry["arrived"], "observed": entry["observed"],
                       "skipped": int(entry.get("played_from_row") or 0), "heads": heads, "trace": traces})
    sides = [s for s in ("left", "right") if s in chunks[0]["heads"]] if chunks else []
    prompts = info.get("prompts") or []
    task = info.get("task") or (prompts[0]["prompt"] if prompts else "")
    rec = Recording(path, info.get("arm_type") or "metal", sides, task, chunks)
    if (path / "motion.csv").exists():
        rec.motion = _motion(path / "motion.csv", sides, degrees=info.get("units") != "rad")
    return rec


def _motion(path: Path, sides, degrees: bool) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
    scale = math.pi / 180 if degrees else 1.0

    def column(kind, side):
        return np.array([[float(r[f"{kind}_{side}_{j}"] or "nan") for j in joints] for r in rows]) * scale

    return {"t": np.array([float(r["t"]) for r in rows]), "phase": np.array([int(float(r["phase"])) for r in rows]),
            "cmd": {s: column("cmd", s) for s in sides}, "meas": {s: column("meas", s) for s in sides}}


class RecordedPolicy:
    """A recording's replies, in order, as a DreamRunner's policy. Needs no camera."""

    cameras: list = []

    def __init__(self, recording: Recording):
        self.recording = recording
        self.sides = recording.sides
        self.index = 0
        self.session_id = None

    def reset_episode(self) -> None:
        pass

    def observation(self, prompt: str, state: dict) -> dict:
        return {"prompt": prompt}

    def push(self, images, observation, timestamp) -> None:
        pass

    def act(self):
        if self.index >= len(self.recording.chunks):
            raise PolicyFinished("the recording has no more chunks")
        chunk = self.recording.chunks[self.index]
        self.index += 1
        per_key = {f"action.{s}_{k}": h[k] for s, h in chunk["heads"].items()
                   for k in ("eef_xyz", "eef_rot6", "grip")}
        return per_key, {"rtt_ms": 0.0}

    def close(self) -> None:
        pass


def _stats(values) -> dict:
    a = np.abs(np.concatenate([np.ravel(v) for v in values])) if values else np.zeros(0)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"median": None, "p95": None, "max": None, "n": 0}
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "n": int(a.size)}


def replay_dry(recording: Recording, robot, ease_in_s: float = 0.6, ease_out_s: float = 0.2) -> dict:
    """Numbers only: per arm, the IK difference to the recorded solve, the command difference to the recorded
    commands (policy ticks), and the largest joint step per tick of each. ``robot`` needs arms (connected or not)."""
    sides = recording.sides
    iks = {s: robot.ik(s, "numpy") for s in sides}
    limiters = {s: Limiter.for_arm(robot, s, iks[s].max_velocity) for s in sides}
    ik_diff = {s: [] for s in sides}
    solved = []
    for chunk in recording.chunks:
        rows = {}
        for side in sides:
            tr = chunk["trace"][side]
            out = decode.solve(iks[side], robot.sdk.trajectory_ik, chunk["heads"][side], tr["q_from"],
                               tr["q_before"], int(tr.get("skip") or 0), tr.get("speed_cap"))
            if "q" in tr:
                ik_diff[side].append(out.q - np.asarray(tr["q"]))
            rows[side] = np.column_stack([out.q, robot.gripper_position(side, out.grip)])
        solved.append((chunk, rows))
    report = {"recording": str(recording.path), "chunks": len(recording.chunks), "arms": {}}
    motion = recording.motion
    if not motion or not len(motion["t"]):
        for side in sides:
            report["arms"][side] = {"ik_rad": _stats(ik_diff[side])}
        return report
    schedule = StreamSchedule(ease_in_s, ease_out_s, ACTION_HZ)
    t, phase = motion["t"], motion["phase"]
    last = {s: motion["cmd"][s][0].copy() for s in sides}
    caps, k = {}, 0
    ours = {s: [] for s in sides}
    theirs = {s: [] for s in sides}
    previous_t = t[0]
    for i, now in enumerate(t):
        if phase[i] == 2:
            break
        while k < len(solved) and solved[k][0]["arrived"] <= now:
            chunk, rows = solved[k]
            holding = schedule.remaining() == 0
            schedule.install(rows, chunk["arrived"], chunk["skipped"], dict(last), ramp=holding)
            caps = {s: chunk["trace"][s].get("speed_cap") for s in sides}
            k += 1
        target = schedule.sample(now)
        if target:
            for s in sides:
                last[s], _ = limiters[s].limit(target[s], last[s], now - previous_t, caps.get(s))
        previous_t = now
        if phase[i] == 1:
            for s in sides:
                ours[s].append(last[s].copy())
                theirs[s].append(motion["cmd"][s][i])
    for side in sides:
        a, b = np.array(ours[side]), np.array(theirs[side])
        diff = a - b if len(a) else np.zeros((0, 7))
        report["arms"][side] = {
            "ik_rad": _stats(ik_diff[side]),
            "command_rad": _stats([diff[:, :6]]),
            "gripper_rad": _stats([diff[:, 6]]),
            "max_step_rad": {"ours": float(np.max(np.abs(np.diff(a[:, :6], axis=0)))) if len(a) > 1 else None,
                             "recorded": float(np.max(np.abs(np.diff(b[:, :6], axis=0)))) if len(b) > 1 else None},
            "per_joint_ik_p95_rad": (np.percentile(np.abs(np.concatenate(ik_diff[side])), 95, axis=0).round(4).tolist()
                                     if ik_diff[side] else None),
        }
    return report
