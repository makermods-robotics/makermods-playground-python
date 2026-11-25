"""DreamRunner: one task on a robot, from enable to the end of the run.

Three threads: the servo loop (the caller's thread) samples the playing chunk at the servo rate and sends the
limited targets; the planner thread builds each observation, asks the policy, solves the reply to joint rows and
installs them; the camera thread hands the policy's frame window a camera tick each time one is due. Whatever ends
the run, the arms freeze at their measured pose first, then return to rest.

    runner = DreamRunner(robot, "pick up the marker", model="beta")
    result = runner.run(timeout_s=60)          # torque on, run, back to rest, torque off
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..core.config import Settings
from ..core.limits import LimitCounts, Limiter
from ..core.mask import mask_secrets
from ..core.recording import RunRecorder, new_folder
from ..core.robot import ArmHeld
from ..core.servo import Servo
from . import decode
from .client import DreamPolicy, retry_delay, starts_new_session
from .scheduling import ACTION_HZ, Pacer, StreamSchedule

RETRY_FIRST_CHUNK_S = 120.0   # how long the arms hold while a retryable refusal is asked again, before the first chunk
RETRY_MID_RUN_S = 30.0        # and after it
OFF_PLAN = math.radians(0.5)  # a joint limited this far from its target sends the next chunk from the command
JOIN_LIMIT = math.radians(15.0)   # a chunk starting further than this from the outgoing command is replanned
REPLY_MAX_AGE_S = 5.0         # a reply to an older observation is dropped and a fresh one sent
CHUNK_ROWS = (48,)            # Dream1's reply length: the JAX IK compiles it before the arms are enabled
CAMERA_HZ = 30.0
POLICY_FRAME_HZ = 5.0         # the policy's frame window keeps frames at this rate
PUSH_PERIOD_S = 1.0 / POLICY_FRAME_HZ - 0.5 / CAMERA_HZ   # a camera tick is pushed once this much newer
EVENTS_KEPT = 100
log = logging.getLogger(__name__)
WARMING = "Dream Labs has no live serve for this model yet; try again later"


class PolicyFinished(Exception):
    """The policy has nothing more to give (a replayed recording ran out)."""


def _pct(values, scale=1.0) -> dict:
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float) * scale
    if not len(a):
        return {"p50": None, "p95": None, "max": None}
    return {k: round(float(v), 2) for k, v in (("p50", np.percentile(a, 50)), ("p95", np.percentile(a, 95)),
                                                ("max", a.max()))}


@dataclass
class RunResult:
    stop_reason: str
    error: bool = False
    chunks: int = 0
    ticks: int = 0
    elapsed_s: float = 0.0
    timing: dict = field(default_factory=dict)
    ik: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    pacing: Optional[str] = None
    rest: Optional[dict] = None
    recording: Optional[str] = None
    session_ids: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


class DreamRunner:
    """``policy`` replaces the Dream API (a replayed recording, a test double); ``on_event(kind, text)`` hears every
    event the run logs; ``state()`` is a live snapshot for a UI."""

    def __init__(self, robot, task: str, model: str = "beta", settings: Optional[Settings] = None, policy=None,
                 on_event: Optional[Callable[[str, str], None]] = None):
        self.robot, self.task, self.model = robot, task, model
        self.settings = settings or robot.settings
        self.policy = policy
        self.on_event = on_event
        self.sides = list(robot.sides)
        self.lock = threading.RLock()
        self.events: list[dict] = []
        self.phase = "idle"
        self.started = 0.0
        self.timeout_s = 0.0
        self.schedule = StreamSchedule(self.settings.ease_in_s, self.settings.ease_out_s)
        self.pacer = Pacer()
        self.counts = LimitCounts()
        self.requests: queue.Queue = queue.Queue()
        self.in_flight: Optional[dict] = None
        self.next_request = 0.0
        self.new_session = True      # the next request starts a new episode on the policy
        self.ramp = True             # the next chunk installed is capped at FIRST_CHUNK_SPEED
        self.finished: Optional[str] = None   # the policy has no more chunks: end once the installed ones played
        self.limited = False
        self.retrying_since: Optional[float] = None
        self.end_reason: Optional[str] = None
        self.end_error = False
        self.chunks = 0
        self.speed_caps: dict = {}
        self.previous: dict = {}
        self.ik_estimate = 0.0
        self.latest_observation = None
        self.last_push = -math.inf
        self.push_lock = threading.Lock()   # one publisher of observations at a time
        self.warmed: Optional[set] = None   # chunk lengths the JAX IK compiled; None: numpy
        self.rtts: list = []
        self.ik_ms: list = []
        self.ik_errors: list = []
        self.session_ids: list = []
        self.recorder: Optional[RunRecorder] = None
        self.servo: Optional[Servo] = None
        self.done = threading.Event()

    # ── public ──
    def run(self, timeout_s: float, stop_event: Optional[threading.Event] = None, then: str = "rest") -> RunResult:
        """Enable (if needed), run the task until ``timeout_s``, ``stop_event``, a fault or the policy ends it, then
        ``then``: "rest" (home to the rest pose, see Robot.enable(), torque off) or "stay" (hold, torque on). Every
        check runs before the arms are enabled; once they are, any error still ends the same way."""
        if then not in ("rest", "stay"):
            raise ValueError("then must be rest or stay")
        stop_event = stop_event or threading.Event()
        self.timeout_s = timeout_s
        self._set_phase("enabling")
        try:
            if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) \
                    or timeout_s <= 0:
                raise ValueError(f"the time limit must be a positive number of seconds, got {timeout_s!r}")
            self.settings.validate()
            self.iks = {s: self.robot.ik(s) for s in self.sides}
            self.numpy_iks = {s: self.robot.ik(s, "numpy") for s in self.sides}
            self.limiters = {s: Limiter.for_arm(self.robot, s, self.iks[s].max_velocity) for s in self.sides}
            if self.policy is None:
                self.policy = self._dream_policy()
            missing = [c for c in self.policy.cameras if c not in self.robot.cameras]
            if missing:
                raise ValueError(f"the model needs the {', '.join(missing)} camera(s); {self.robot.name} has "
                                 f"{', '.join(self.robot.cameras) or 'none'}")
            self._warm(stop_event)
            if any(s != "enabled" for s in self.robot.states().values()):
                self.robot.enable(cancel=stop_event)
        except Exception as e:
            self._event("system", f"start failed: {e}" if isinstance(e, ArmHeld) else
                        f"start failed: {type(e).__name__}: {e}")
            self._set_phase("done")
            if self.policy is not None:
                self.policy.close()
            enabled = [s for s, state in self.robot.states().items() if state in ("enabled", "fault")]
            rest = None
            if then == "rest" and enabled:     # an arm never enabled here (refused, not reached) is left as it is
                rest = {s: ok for s, ok in self.robot.stop().items() if s in enabled}
            return RunResult(stop_reason=mask_secrets(f"start failed: {e}"), error=True, rest=rest)
        self.started = time.monotonic()
        reason, error, threads, rest = "error", True, [], None
        try:
            self._start_recording()
            self.servo = Servo(self.robot, self.settings.servo_hz, self._sample, self._limit,
                               lambda now: self._check(now, stop_event), self._on_tick,
                               math.radians(self.settings.tracking_limit_deg), self.settings.feedback_max_age_s)
            threads = [threading.Thread(target=self._planner, daemon=True, name="planner"),
                       threading.Thread(target=self._cameras, daemon=True, name="policy-cameras")]
            for t in threads:
                t.start()
            self._set_phase("running")
            self._event("user", self.task)
            try:
                reason = self.servo.run(threading.Event())
            except KeyboardInterrupt:
                reason = "stopped"
            finally:
                self._freeze()
            error = self.end_error if reason == self.end_reason else reason not in ("timeout", "stopped")
        except Exception as e:
            reason, error = f"error: {type(e).__name__}: {e}", True
        finally:
            self._freeze()
            self.done.set()
            with self.lock:
                self.schedule.clear()
            self._event("system", f"run ended: {reason}")
            for t in threads:
                t.join(timeout=2.0)
            rest = self._wind_down(then)
        elapsed = time.monotonic() - self.started
        ticks = self.servo.stats.ticks if self.servo else 0
        result = RunResult(
            stop_reason=mask_secrets(reason), error=error, chunks=self.chunks, ticks=ticks,
            elapsed_s=round(elapsed, 2),
            timing={"servo": self.servo.stats.summary() if self.servo else {}, "rtt_ms": _pct(self.rtts, 1000),
                    "ik_ms": _pct(self.ik_ms)},
            ik={"pos_err_mm": _pct([e[0] for e in self.ik_errors], 1000),
                "rot_err_deg": _pct([math.degrees(e[1]) for e in self.ik_errors]),
                "capped_ticks": int(sum(e[2] for e in self.ik_errors))},
            limits=self.counts.to_json(), pacing=self.pacer.mode, rest=rest,
            recording=str(self.recorder.folder) if self.recorder else None, session_ids=list(self.session_ids))
        if self.recorder:
            problem = self.recorder.finish({
                "stopped_because": result.stop_reason, "error": error, "rest": rest,
                "stopped_at": datetime.now().isoformat(timespec="seconds"), "command_limits": result.limits,
                "session_ids": result.session_ids, "timing": result.timing})
            if problem:
                self._event("system", f"recording incomplete: {problem}")
        self._set_phase("done")
        return result

    def _dream_policy(self) -> DreamPolicy:
        """The model reads two arms and three views: an arm this robot lacks is reported parked at the pose its
        own arm starts from, a camera it lacks as a black image."""
        if not self.robot.cameras:
            raise ValueError(f"{self.robot.name} has no camera; the model needs at least one")
        start = self.robot.joint_positions()
        parked = {side: self.robot.tool_pose(self.sides[0], start[self.sides[0]])
                  for side in ("left", "right") if side not in self.sides}
        policy = DreamPolicy(self.model, self.sides, self.settings.request_timeout_ms,
                             cameras=self.robot.cameras, parked=parked)
        stand_ins = [f"{side} arm" for side in parked] + [f"{c} camera" for c in ("top", "left", "right")
                                                         if c not in self.robot.cameras]
        if stand_ins:
            self._event("warning", f"the model is given stand-ins for the {', '.join(stand_ins)}")
        return policy

    def state(self) -> dict:
        """A JSON-serialisable snapshot for live status."""
        with self.lock:
            measured, commanded, _ = self.servo.snapshot() if self.servo else ({}, {}, 0.0)
            elapsed = time.monotonic() - self.started if self.started else 0.0
            return {
                "phase": self.phase, "task": mask_secrets(self.task), "model": self.model, "chunks": self.chunks,
                "elapsed_s": round(elapsed, 1), "remaining_s": max(0.0, round(self.timeout_s - elapsed, 1)),
                "pacing": self.pacer.mode, "retrying": self.retrying_since is not None,
                "stop_reason": self.end_reason, "recording": str(self.recorder.folder) if self.recorder else None,
                "measured": {k: np.round(v, 4).tolist() for k, v in measured.items()},
                "commanded": {k: np.round(v, 4).tolist() for k, v in commanded.items()},
                "events": list(self.events),
            }

    # ── servo thread ──
    def _check(self, now: float, stop_event: threading.Event) -> Optional[str]:
        if stop_event.is_set():
            return "stopped"
        with self.lock:
            window = RETRY_FIRST_CHUNK_S if self.chunks == 0 else RETRY_MID_RUN_S
            if self.retrying_since is not None and now - self.retrying_since > window:
                self._end(f"the policy did not answer within the {window:g} s retry window", error=True)
            if self.finished and self.schedule.remaining() == 0:
                self._end(self.finished)
        if self.end_reason:
            return self.end_reason
        if now - self.started >= self.timeout_s:
            return "timeout"
        for name in self.policy.cameras:
            if self.robot.frame(name, self.settings.camera_max_age_s) is None:
                return f"camera {name}: no fresh frame"
        return None

    def _sample(self, now: float):
        with self.lock:
            return self.schedule.sample(now)

    def _limit(self, targets: dict, last: dict, dt: float) -> dict:
        commands, reports, change = {}, [], 0.0
        for side, target in targets.items():
            commands[side], report = self.limiters[side].limit(target, last.get(side), dt, self.speed_caps.get(side))
            reports.append(report)
            change = max(change, float(np.max(np.abs(commands[side][:6] - target[:6]))))
        self.counts.add(reports, change)
        if change >= OFF_PLAN:
            self.limited = True
        return commands

    def _on_tick(self, now: float, phase: int, commanded: dict, measured: dict) -> None:
        if self.recorder:
            self.recorder.log.tick(now, phase, commanded, measured)
        with self.lock:
            if self.in_flight is not None or now < self.next_request or self.end_reason or self.finished:
                return
            if not self.pacer.refill_due(self.schedule.executed(), self.schedule.remaining()):
                return
            self.in_flight = {"measured": measured, "commanded": commanded, "sampled_at": self.servo.sampled_at,
                              "sent_at": now, "played": self.schedule.played, "limited": self.limited}
            self.limited = False
            self.requests.put(self.in_flight)

    # ── planner thread ──
    def _planner(self) -> None:
        try:
            while not self.done.is_set():
                try:
                    request = self.requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    reply = self._answer(request)
                except Exception as e:
                    with self.lock:
                        self.in_flight = None
                        self._failed(e)
                    continue
                with self.lock:
                    self.in_flight = None
                    if reply is not None:
                        self._install(request, reply)
        finally:
            try:
                self.policy.close()
            except Exception:
                pass

    def _answer(self, request: dict) -> Optional[dict]:
        """The request's observation sent and the reply solved, or None when it is not fit to send (its joint
        feedback too old, no fresh camera tick): the next tick asks again."""
        robot, policy = self.robot, self.policy
        with self.lock:
            reset, self.new_session = self.new_session, False
        if reset:
            with self.push_lock:
                self.latest_observation = None
                self.last_push = -math.inf
                policy.reset_episode()
            with self.lock:
                self.previous = {}
                self.ramp = True
        with self.lock:
            cap = decode.FIRST_CHUNK_SPEED if self.ramp else None
        if time.monotonic() - request["sampled_at"] > self.settings.feedback_max_age_s:
            return None
        measured, commanded = request["measured"], request["commanded"]
        state = {s: {"opening": robot.opening(s, measured[s][6]), "tool": robot.tool_pose(s, measured[s])}
                 for s in self.sides}
        if not self._push_frames(policy.observation(self.task, state)):
            return None
        started = time.monotonic()
        per_key, timing = policy.act()
        heads = decode.head_from_reply(per_key, self.sides)
        observed_ago = started - request["sampled_at"]
        rtt_s = max((timing.get("rtt_ms") or 0) / 1000 + observed_ago, time.monotonic() - request["sampled_at"])
        rtt_s += self.ik_estimate
        ik_started = time.perf_counter()
        with self.lock:
            previous = {} if request["limited"] else dict(self.previous)
        chunks, traces = {}, {}
        for side in self.sides:
            q_meas, q_cmd = measured[side][:6], commanded.get(side, measured[side])[:6]
            length = len(heads[side]["eef_xyz"])
            taken = decode.takeover(previous.get(side), q_meas, q_cmd, rtt_s * ACTION_HZ, length)
            if taken is not None:
                skip, q_from, before = taken[0], taken[1], [taken[2], taken[3]]
            else:
                skip, q_from = 0, (q_cmd if request["limited"] else q_meas)
                before = [q_from, q_from]
            ik = self.iks[side] if self.warmed is None or length in self.warmed else self.numpy_iks[side]
            solved = decode.solve(ik, robot.sdk.trajectory_ik, heads[side], q_from, before, skip, cap)
            chunks[side] = np.column_stack([solved.q, robot.gripper_position(side, solved.grip)])
            traces[side] = solved.trace
        self.ik_estimate = time.perf_counter() - ik_started
        timing = {**timing, "ik_ms": self.ik_estimate * 1000}
        return {"chunks": chunks, "traces": traces, "heads": heads, "timing": timing, "cap": cap,
                "session_id": getattr(policy, "session_id", None)}

    def _install(self, request: dict, reply: dict) -> None:
        """Caller holds the lock."""
        if self.done.is_set() or self.end_reason:
            return
        now = time.monotonic()
        window = RETRY_FIRST_CHUNK_S if self.chunks == 0 else RETRY_MID_RUN_S
        if self.retrying_since is not None and now - self.retrying_since > window:
            self._end(f"the policy answered after the {window:g} s retry window", error=True)
            return
        self.retrying_since = None
        rtt = now - request["sent_at"]
        self.pacer.record(rtt)
        self.rtts.append(rtt)
        self.ik_ms.append(reply["timing"]["ik_ms"])
        for trace in reply["traces"].values():
            self.ik_errors.append((trace["pos_err_max"], trace["rot_err_max"], trace["capped_ticks"]))
        session = reply.get("session_id")
        if session and session not in self.session_ids:
            self.session_ids.append(session)
        length = min(len(rows) for rows in reply["chunks"].values())
        age = now - request["sampled_at"]
        skipped = min(int(self.schedule.played - request["played"]), math.floor(age * ACTION_HZ + 0.5))
        if age > REPLY_MAX_AGE_S or skipped >= length - 1:
            self._event("system", f"discarded a chunk {age:.1f} s after its observation: too late for its "
                                  f"{length}-row plan; asking again")
            return
        holding = self.schedule.remaining() == 0
        origin = self.servo.snapshot()[1]
        if not holding:
            offset = self.schedule.offset(reply["chunks"], skipped, origin)
            worst = max(float(np.max(np.abs(o[:6]))) for o in offset.values())
            if worst > JOIN_LIMIT:
                self.previous = {}
                self._event("system", f"discarded a chunk starting {math.degrees(worst):.0f} deg from the outgoing "
                                      "command; asking again from the command")
                return
        self.schedule.install(reply["chunks"], now, skipped, origin, ramp=holding)
        self.speed_caps = {s: reply["cap"] for s in self.sides}
        self.previous = {s: reply["chunks"][s][:, :6] for s in self.sides}
        self.ramp = False
        self.chunks += 1
        if self.recorder:
            timing = reply["timing"]
            self.recorder.log.chunk(now, request["sampled_at"], skipped, reply["heads"], reply["traces"], {
                "total": rtt * 1000, "request": timing.get("rtt_ms"), "network": timing.get("network_ms"),
                "serve_infer": timing.get("serve_infer_ms"), "ik": timing.get("ik_ms"),
                "other": None if timing.get("rtt_ms") is None else rtt * 1000 - timing["rtt_ms"] - timing["ik_ms"]})
        if self.chunks == 1:
            self._event("system", "executing motion")

    def _failed(self, exc: Exception) -> None:
        """A request that failed. Caller holds the lock."""
        if isinstance(exc, PolicyFinished):
            self.finished = mask_secrets(str(exc) or "the policy finished")
            return
        reason = getattr(exc, "reason", None)
        name = type(exc).__name__
        trace_id = getattr(exc, "trace_id", None)
        if trace_id:
            self._event("policy", f"{name} {getattr(exc, 'code', '') or ''} {reason or ''} trace_id {trace_id}")
        if reason == "pool_warming":
            self._end(WARMING, error=True)
            return
        try:
            delay = retry_delay(exc)
        except ImportError:
            delay = None
        self.new_session = self.new_session or starts_new_session(exc)
        if name == "DreamSessionExpired":
            self.schedule.clear()
            self.previous = {}
            self.ramp = True
        now = time.monotonic()
        if delay is not None:
            first = self.retrying_since is None
            self.retrying_since = now if first else self.retrying_since
            window = RETRY_FIRST_CHUNK_S if self.chunks == 0 else RETRY_MID_RUN_S
            if now + delay - self.retrying_since <= window:
                self.next_request = now + delay
                if first:
                    self._event("system", f"policy not ready ({reason or name}); holding and asking again for up "
                                          f"to {window:g} s")
                return
        text = mask_secrets(f"{name}: {exc}")
        self._end(text[:400], error=True)

    def _end(self, reason: str, error: bool = False) -> None:
        if self.end_reason is None:
            self.end_reason, self.end_error = mask_secrets(reason), error

    # ── camera thread ──
    def _cameras(self) -> None:
        period = 1.0 / CAMERA_HZ
        while not self.done.wait(period):
            try:
                self._push_frames()
            except Exception as e:
                self._event("system", f"camera frames not passed to the policy: {e}")
                self.done.wait(1.0)

    def _push_frames(self, observation=None) -> bool:
        """Hand the policy a camera tick: with ``observation`` (the planner's new one) always, its timestamp never
        before the last; without, the latest observation with the newest frames once a tick is due. One publisher at
        a time. True when the tick was handed over (or the policy takes no frames)."""
        with self.push_lock:
            if observation is None:
                observation = self.latest_observation
                if observation is None:
                    return False
            if not self.policy.cameras:
                self.latest_observation = observation
                return True
            frames = self.robot.frames(self.settings.camera_max_age_s)
            if any(name not in frames for name in self.policy.cameras):
                return False
            stamp = time.monotonic() - max(frames[name][1] for name in self.policy.cameras)
            if observation is not self.latest_observation:
                stamp = max(stamp, self.last_push)
            elif stamp < self.last_push + PUSH_PERIOD_S:
                return False
            self.policy.push({n: frames[n][0] for n in self.policy.cameras}, observation, stamp)
            self.last_push, self.latest_observation = stamp, observation
            return True

    # ── bookkeeping ──
    def _start_recording(self) -> None:
        if not self.settings.record:
            return
        try:
            folder = new_folder(Path(self.settings.recordings_dir), self.robot.name)
            info = {"robot": self.robot.name, "arm_type": self.robot.arm_type, "model": self.model,
                    "sides": self.sides, "task": self.task, "started_at": datetime.now().isoformat(timespec="seconds"),
                    "settings": asdict(self.settings)}
            self.recorder = RunRecorder(folder, self.sides, lambda: {n: f[0] for n, f in self.robot.frames().items()},
                                        info).start()
            self._event("system", f"recording to {folder}")
        except Exception as e:
            self.recorder = None
            self._event("system", f"recording could not start ({e}); the run continues unrecorded")

    def _warm(self, stop_event: threading.Event) -> None:
        """With the JAX IK (asked for by name), compile Dream1's chunk length before the arms are enabled; a reply of
        another length is solved with numpy."""
        if self.settings.ik_solver != "jax":
            return
        self._event("system", "compiling the JAX IK (the arms stay off)")
        for rows in CHUNK_ROWS:
            for side in self.sides:
                if stop_event.is_set():
                    raise RuntimeError("cancelled while compiling the IK")
                self.iks[side].warmup(rows)
        self.warmed = set(CHUNK_ROWS)

    def _freeze(self) -> None:
        """Every enabled arm holds its measured pose now (the SDK drops any target it was pursuing)."""
        try:
            self.robot.hold()
        except Exception as e:
            log.error("hold failed: %s", e)

    def _wind_down(self, then: str) -> Optional[dict]:
        """``then`` once the run is over: "rest" brings every arm home (Robot.stop(): the tool low and close to the base
        first, then the joint return; an arm whose return is not verified keeps holding), "stay" keeps them holding.
        Returns, for "rest", whether each arm's return was verified."""
        if then != "rest":
            return None
        self._set_phase("returning")
        try:
            rest = self.robot.stop()
        except Exception as e:
            self._event("system", f"return to rest failed: {type(e).__name__}: {e}")
            rest = {s: state not in ("enabled", "fault") for s, state in self.robot.states().items()}
        for side, what in self.robot.retract_notes().items():
            self._event("system", f"{side} arm retract {what}; joint return from where it was")
        if not all(rest.values()):
            self._event("system", "return to rest not verified: "
                        + ", ".join(s for s, ok in rest.items() if not ok) + " still holding")
        return rest

    def _set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def _event(self, kind: str, text: str) -> None:
        entry = {"t": round(time.time(), 3), "kind": kind, "text": mask_secrets(text)}
        with self.lock:
            self.events = (self.events + [entry])[-EVENTS_KEPT:]
        if self.on_event:
            try:
                self.on_event(kind, entry["text"])
            except Exception:
                pass
