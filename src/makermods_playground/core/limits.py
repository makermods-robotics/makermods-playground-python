"""Policy command limits applied before a target goes to an arm, on top of what the SDK's Arm does itself
(joint limits, its max_velocity, startup sync).

Per arm and tick: the gripper is kept GRIPPER_OPEN_MARGIN short of fully open; each joint moves at most its speed
cap (the chunk's own cap, e.g. the first chunk's ramp-in, under the IK's per-joint speeds) and the gripper at most
GRIPPER_STROKES_PER_S; a value whose last command is past its limit comes in at ENTRY_SPEED at most.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GRIPPER_OPEN_MARGIN = math.radians(2.0)
GRIPPER_STROKES_PER_S = 2.0
ENTRY_SPEED = math.radians(5.0)   # rad/s
MAX_DT = 0.1                      # s, the longest tick a rate cap counts
EPS = 1e-9


@dataclass
class LimitCounts:
    ticks: int = 0
    limited: int = 0
    position: int = 0
    rate: int = 0
    entering: int = 0
    max_joint_change: float = 0.0     # rad

    def add(self, reports: list[dict], change: float) -> None:
        position, rate = any(r["position"] for r in reports), any(r["rate"] for r in reports)
        self.ticks += 1
        self.position += position
        self.rate += rate
        self.entering += any(r["entering"] for r in reports)
        self.limited += position or rate
        self.max_joint_change = max(self.max_joint_change, change)

    def to_json(self) -> dict:
        return {"policy_ticks": self.ticks, "limited_ticks": self.limited, "position_clamped_ticks": self.position,
                "rate_clamped_ticks": self.rate, "entering_ticks": self.entering,
                "max_joint_change_rad": round(self.max_joint_change, 5)}


class Limiter:
    """``lo``/``hi`` (7,) joint limits, ``closed``/``open`` the gripper's ends, ``joint_speed`` (6,) rad/s."""

    def __init__(self, lo, hi, closed: float, open_: float, joint_speed):
        self.lo, self.hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        toward = math.copysign(1.0, open_ - closed)
        margin = open_ - toward * GRIPPER_OPEN_MARGIN
        if toward > 0:
            self.hi[6] = min(self.hi[6], margin)
        else:
            self.lo[6] = max(self.lo[6], margin)
        self.joint_speed = np.asarray(joint_speed, dtype=float)
        self.gripper_speed = GRIPPER_STROKES_PER_S * abs(open_ - closed)

    @classmethod
    def for_arm(cls, robot, side: str, joint_speed) -> "Limiter":
        lo, hi = robot.limits(side)
        closed, full = robot.gripper_range(side)
        return cls(lo, hi, closed, full, joint_speed)

    def limit(self, target, last, dt: float, speed_cap=None) -> tuple[np.ndarray, dict]:
        """(command, report) for one arm's target from ``last`` (its previous command, or None)."""
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("policy target must be seven finite values")
        clipped = np.clip(target, self.lo, self.hi)
        report = {"position": bool(np.any(np.abs(clipped - target) > EPS)), "rate": False, "entering": False}
        if last is None:
            return clipped, report
        last = np.asarray(last, dtype=float)
        span = min(max(dt, 0.0), MAX_DT)
        caps = np.r_[np.minimum(self.joint_speed, math.inf if speed_cap is None else speed_cap), self.gripper_speed]
        outside = (last < self.lo - EPS) | (last > self.hi + EPS)
        caps = np.where(outside, np.minimum(caps, ENTRY_SPEED), caps) * span
        command = np.where(np.isfinite(last), last + np.clip(clipped - last, -caps, caps), clipped)
        report["rate"] = bool(np.any(np.abs(command - clipped) > EPS))
        report["entering"] = bool(outside.any())
        return command, report
