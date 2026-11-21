"""Chunk playback and request pacing.

Chunk rows play one after another at ACTION_HZ; a new chunk is installed over the one playing, past the rows the
arms ran since its observation, and sampled at the servo rate (Catmull-Rom for the joints, linear for the gripper).
The pacer decides when the next chunk is asked for: in async mode while a chunk still plays, so the reply lands in
time; in sync mode once it has played. Pacing follows Dream Labs' dreamlabs-client (Apache-2.0, see NOTICE).
"""

from __future__ import annotations

import collections
import math
import statistics

import numpy as np

ACTION_HZ = 20.0          # rows per second a chunk plays at
EXECUTION_HORIZON = 30    # async: rows of a chunk run before the next takes over
SYNC_ABOVE_MS = 700.0     # median round trip past which pacing is sync
RTT_WINDOW = 5            # round trips the mode (median) and the async lead (shortest) are taken from
RUNWAY_MARGIN_S = 0.25    # async: never let the queued rows drop below the round trip plus this


def _exact(position: float) -> float:
    return float(round(position)) if abs(position - round(position)) < 1e-9 else position


def at_cubic(rows: np.ndarray, position: float) -> np.ndarray:
    """Catmull-Rom between rows at a fractional row ``position``, held at the ends."""
    position = _exact(min(max(position, 0.0), len(rows) - 1))
    if len(rows) < 2:
        return np.array(rows[0], dtype=float)
    i = min(int(position), len(rows) - 2)
    x = position - i
    p0, p1, p2, p3 = rows[max(i - 1, 0)], rows[i], rows[i + 1], rows[min(i + 2, len(rows) - 1)]
    return 0.5 * (2 * p1 + (p2 - p0) * x + (2 * p0 - 5 * p1 + 4 * p2 - p3) * x * x
                  + (3 * p1 - p0 - 3 * p2 + p3) * x ** 3)


def at_linear(rows: np.ndarray, position: float) -> np.ndarray:
    position = _exact(min(max(position, 0.0), len(rows) - 1))
    low = int(position)
    high = min(low + 1, len(rows) - 1)
    return rows[low] + (position - low) * (rows[high] - rows[low])


class StreamSchedule:
    """Chunks played as a stream. A chunk installed while the arms hold eases in: playback speed rises from 0 to 1x
    over ``ease_in_s`` while the joints' offset from the held command fades out. A chunk installed over a playing
    one joins the outgoing command the same way at full speed: the offset (joints and gripper) fades out over
    ``ease_in_s``, starting with zero slope so the command keeps moving with the new plan. The last stretch of a
    chunk slows to a stop on its last row over ``ease_out_s``; a chunk that lands in time takes over before then."""

    def __init__(self, ease_in_s: float = 0.6, ease_out_s: float = 0.2, hz: float = ACTION_HZ):
        self.ease_in_s, self.ease_out_s, self.hz = ease_in_s, ease_out_s, hz
        self.rows = None
        self.position = 0.0   # playback position in rows of the current chunk
        self.played = 0.0     # rows played since the run started, across chunks
        self.start = 0.0
        self.last = 0.0
        self.ease = None

    @property
    def length(self) -> int:
        return min(len(r) for r in self.rows.values()) if self.rows else 0

    def offset(self, chunks: dict, skipped: int, origin: dict) -> dict:
        """Per arm, ``origin`` (the outgoing command) minus where ``chunks`` start playing from row ``skipped``."""
        out = {}
        for name, rows in chunks.items():
            rows = np.asarray(rows, dtype=float)
            if name in origin:
                position = float(min(max(0, skipped), len(rows) - 1))
                start = at_cubic(rows, position)
                start[6] = at_linear(rows, position)[6]
                out[name] = np.asarray(origin[name], dtype=float) - start
        return out

    def install(self, chunks: dict, now: float, skipped: int, origin: dict | None = None, ramp: bool = True) -> None:
        """Play ``chunks`` (rows per arm) from row ``skipped`` on, from ``now``, joining ``origin`` (the outgoing
        command per arm); ``ramp``: the arms hold, so playback speeds up from 0."""
        self.rows = {k: np.array(v, dtype=float) for k, v in chunks.items()}
        self.position = self.start = float(min(max(0, skipped), self.length - 1))
        self.last = now
        self.ease = None
        if origin and self.ease_in_s > 0:
            offset = self.offset(self.rows, skipped, origin)
            for value in offset.values() if ramp else ():
                value[6] = 0.0   # from a hold the gripper goes straight for the chunk, at the limiter's speed
            self.ease = {"at": now, "offset": offset, "ramp": ramp}

    def clear(self) -> None:
        self.rows = None
        self.ease = None

    def remaining(self) -> int:
        return max(0, math.ceil(self.length - 1 - self.position - 1e-9)) if self.rows else 0

    def executed(self) -> int:
        return math.floor(self.position - self.start + 1e-9)

    def sample(self, now: float) -> dict | None:
        """Target per arm at ``now``, or None (hold) once the chunk has played."""
        dt = max(0.0, now - self.last)
        self.last = now
        if not self.rows or not self.remaining():
            return None
        end = self.length - 1
        since = now - self.ease["at"] if self.ease else math.inf
        middle = since - dt / 2
        ramping = self.ease is not None and self.ease["ramp"]
        speed = max(0.0, middle) / self.ease_in_s if ramping and middle < self.ease_in_s else 1.0
        if self.ease_out_s > 0:   # v = sqrt(2 a d) for the deceleration that stops in ease_out_s from full speed
            speed = min(speed, math.sqrt(2 * (end - self.position) / (self.hz * self.ease_out_s)))
        step = dt * self.hz * speed
        if end - (self.position + step) < 0.02:
            step = end - self.position
        self.position += step
        self.played += step
        fade = 1 - (since / self.ease_in_s) ** 2 * (3 - 2 * since / self.ease_in_s) if since < self.ease_in_s else 0.0
        if fade == 0:
            self.ease = None
        target = {}
        for name, rows in self.rows.items():
            value = at_cubic(rows, self.position)
            value[6] = at_linear(rows, self.position)[6]
            offset = self.ease["offset"].get(name) if self.ease else None
            if offset is not None:
                value += fade * offset
            target[name] = value
        return target


class Pacer:
    """Sync or async from the median of the recent round trips; ``refill_due`` says when to ask again."""

    def __init__(self, execution_horizon: int = EXECUTION_HORIZON, hz: float = ACTION_HZ,
                 sync_above_ms: float = SYNC_ABOVE_MS):
        self.execution_horizon, self.hz, self.sync_above_ms = execution_horizon, hz, sync_above_ms
        self.mode = "async"
        self.rtts: collections.deque = collections.deque(maxlen=RTT_WINDOW)

    def record(self, rtt_s: float) -> None:
        self.rtts.append(rtt_s)
        self.mode = "sync" if statistics.median(self.rtts) * 1000 > self.sync_above_ms else "async"

    def refill_due(self, executed: int, remaining: int) -> bool:
        if remaining <= 0:
            return True
        if self.mode == "sync":
            return False
        rtt = min(self.rtts, default=0.0)
        return (executed + math.ceil(rtt * self.hz) >= self.execution_horizon
                or remaining / self.hz <= rtt + RUNWAY_MARGIN_S)
