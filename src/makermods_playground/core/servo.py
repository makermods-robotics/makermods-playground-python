"""Fixed-rate loop sending joint targets: read the joints, check them, sample the target, limit it, command it.

The SDK's Arm runs its own faster loop that keeps targets inside the joint limits, caps speed, adds gravity
feed-forward, regulates the gripper's grip and holds the pose on a fault; this loop only decides what the target is
and when a run has to end.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

Targets = dict[str, np.ndarray]


@dataclass
class ServoStats:
    ticks: int = 0
    misses: int = 0
    max_late_ms: float = 0.0
    periods: list = field(default_factory=list)   # s between ticks

    def summary(self) -> dict:
        p = np.asarray(self.periods[1:] or [math.nan]) * 1000
        return {"ticks": self.ticks, "misses": self.misses, "max_late_ms": round(self.max_late_ms, 1),
                "period_ms": {"p50": _r(np.percentile(p, 50)), "p95": _r(np.percentile(p, 95)),
                              "max": _r(np.max(p))}}


def _r(x: float) -> Optional[float]:
    return round(float(x), 2) if math.isfinite(x) else None


class Servo:
    """``sample(now)`` gives the target per arm (None: hold the last command); ``limit(targets, last, dt)`` the
    commands for them; ``check(now)`` a reason to end, or None; ``on_tick(now, phase, commanded, measured)`` sees
    every tick (phase 1 = policy target, 0 = hold). Joint feedback older than ``feedback_limit`` ends the run before
    anything is commanded; ``sampled_at`` is when the tick's oldest joint feedback was measured."""

    def __init__(self, robot, rate_hz: float, sample: Callable[[float], Optional[Targets]],
                 limit: Optional[Callable] = None, check: Optional[Callable[[float], Optional[str]]] = None,
                 on_tick: Optional[Callable] = None, tracking_limit: float = math.radians(12.0),
                 feedback_limit: float = 0.2):
        self.robot, self.rate_hz = robot, rate_hz
        self.sample, self.limit, self.check, self.on_tick = sample, limit, check, on_tick
        self.tracking_limit, self.feedback_limit = tracking_limit, feedback_limit
        self.stats = ServoStats()
        self.last: Targets = {}
        self.measured: Targets = {}
        self.sampled_at = 0.0
        self.lock = threading.Lock()

    def run(self, stop: threading.Event) -> str:
        """Run until ``stop`` is set or a check ends it; returns why it ended."""
        period = 1.0 / self.rate_hz
        with self.lock:
            self.last = self.robot.commanded_positions()
        previous = deadline = time.monotonic()
        while not stop.is_set():
            tick = time.monotonic()
            try:
                reason = self._tick(tick, tick - previous)
            except Exception as e:
                reason = f"error: {type(e).__name__}: {e}"
            if reason:
                return reason
            if self.stats.ticks > 1:
                self.stats.periods.append(tick - previous)
            previous = tick
            now = time.monotonic()
            due = deadline + period
            if now > due:
                self.stats.misses += 1
                self.stats.max_late_ms = max(self.stats.max_late_ms, (now - due) * 1000)
            deadline = max(due, now)
            stop.wait(max(0.0, deadline - time.monotonic()))
        return "stopped"

    def _tick(self, now: float, dt: float) -> Optional[str]:
        self.stats.ticks += 1
        robot = self.robot
        for side, state in robot.states().items():
            if state != "enabled":
                return f"{side} arm {state}: {robot.arms[side].fault_reason or 'not enabled'}"
        ages = robot.feedback_ages()
        measured = robot.joint_positions()
        for side, q in measured.items():
            if not np.isfinite(q).all():
                return f"{side} arm: no joint feedback"
            if ages[side] > self.feedback_limit:
                return f"{side} arm: joint feedback {ages[side] * 1000:.0f} ms old"
        commanded = robot.commanded_positions()
        for side, q in commanded.items():
            gap = float(np.max(np.abs(q[:6] - measured[side][:6])))
            if gap > self.tracking_limit:
                return f"{side} arm: tracking error {math.degrees(gap):.1f} deg; motion stopped"
        with self.lock:
            self.measured, self.sampled_at = measured, now - max(ages.values(), default=0.0)
        reason = self.check(now) if self.check else None
        if reason:
            return reason
        targets = self.sample(now)
        if targets:
            commands = self.limit(targets, dict(self.last), dt) if self.limit else targets
            applied = robot.command(commands)
            with self.lock:
                self.last = {**self.last, **applied}
        if self.on_tick:
            self.on_tick(now, 1 if targets else 0, dict(self.last), measured)
        return None

    def snapshot(self) -> tuple[Targets, Targets, float]:
        """(measured, last commands, when the oldest feedback was measured) as of the latest tick."""
        with self.lock:
            return ({k: v.copy() for k, v in self.measured.items()}, {k: v.copy() for k, v in self.last.items()},
                    self.sampled_at)
