"""The page's robot: the selected definition, its connection, one run at a time and what the last run left.

A run is a ``DreamRunner`` on its own thread, ending with the return to rest. It stops when Stop is pressed, when no
page has sent a heartbeat for ``heartbeat_s`` while it runs (any open page's heartbeat keeps it going), and when the
server shuts down; an identify wiggle stops the same way. An arm whose torque off is not confirmed (its return to rest
not verified, or found energized on connecting) is listed in ``holding`` until a release confirms it. Requests that
start motion or release name the robot the page shows and the selection ``revision`` it saw.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..core import cli as core_cli
from ..core import config
from ..core.mask import mask_secrets
from ..core.robot import ArmHeld, Robot
from ..dream.runner import DreamRunner
from . import devices

log = logging.getLogger(__name__)

HEARTBEAT_S = 2.0
EVENTS_KEPT = 50


class Busy(Exception):
    """Refused in the present state (HTTP 409)."""


def _only_robot() -> Optional[str]:
    names = config.robot_names()
    return names[0] if len(names) == 1 else None


class Session:
    def __init__(self, robot: Optional[str] = None, heartbeat_s: float = HEARTBEAT_S):
        if robot is not None:
            config.load(robot)
        self.selected = robot or _only_robot()
        self.revision = 0    # changes with every change of the selection
        self.heartbeat_s = heartbeat_s
        self.lock = threading.RLock()
        self.robot: Optional[Robot] = None
        self.runner: Optional[DreamRunner] = None
        self.pending: Optional[dict] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.result: Optional[dict] = None
        self.holding: dict[str, list[str]] = {}
        self.identifying: Optional[str] = None
        self.identify_stop = threading.Event()
        self.events: deque = deque(maxlen=EVENTS_KEPT)
        self.beat = 0.0
        self.closing = threading.Event()
        self.previews = devices.Previews()
        threading.Thread(target=self._watch_heartbeat, daemon=True, name="heartbeat").start()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # ── robot definitions ──
    def select(self, name: str) -> None:
        with self.lock:
            self._idle()
            config.load(name)
            if self.robot is not None and self.robot.name != name:
                if self.holding.get(self.robot.name):
                    raise Busy(self._holding_text(self.robot.name))
                self._disconnect()
            if name != self.selected:
                self._select(name)

    def save_robot(self, name: str, arm: Optional[str], arms: Optional[dict], cameras: Optional[dict],
                   record: Optional[bool]) -> config.RobotDef:
        """Create or change a definition. Arms are adapter addresses (``gs_usb:<serial>``); an arm whose address is
        unchanged keeps its entry. Only ``record`` may change while that robot is connected."""
        with self.lock:
            existing = config.load(name) if name in config.robot_names() else None
            if existing is None and (arm is None or not arms):
                raise ValueError("a new robot needs its arm type and at least one arm")
            connected = self.robot is not None and self.robot.name == name
            if connected and (arm is not None or arms is not None or cameras is not None):
                raise Busy(f"{name} is connected: disconnect it first")
            if self.holding.get(name) and (arm is not None or arms is not None):
                raise Busy(self._holding_text(name))
            arm = arm or existing.arm
            entries = dict(existing.arms) if existing and arms is None and arm == existing.arm else {}
            for side, address in (arms or {}).items():
                if not address:
                    continue
                if existing and existing.arm == arm and side in existing.arms \
                        and existing.arm_spec(side).address == address:
                    entries[side] = existing.arms[side]
                else:
                    entries[side] = {"backend": "gs_usb", "port": address}
            if cameras is None:
                cameras = existing.cameras if existing else {}
            settings = existing.settings if existing else config.Settings()
            if record is not None:
                settings = replace(settings, record=record)
            robot = config.RobotDef(name, arm, entries, {k: v for k, v in cameras.items() if v}, settings)
            config.save(robot)
            if connected:
                self.robot.settings = replace(self.robot.settings, record=settings.record)
            if self.selected is None:
                self._select(name)
            return robot

    def remove_robot(self, name: str) -> None:
        with self.lock:
            if self.robot is not None and self.robot.name == name:
                raise Busy(f"{name} is connected: disconnect it first")
            if self.holding.get(name):
                raise Busy(self._holding_text(name))
            config.remove(name)
            if self.selected == name:
                self._select(_only_robot())

    def check(self) -> dict:
        """Read-only check of the selected robot: motors answer, faults, temperatures, cameras."""
        with self.lock:
            self._idle()
            if self.robot is not None:
                raise Busy("disconnect first: the check opens each arm and camera itself")
            if self.identifying:
                raise Busy("an arm is being identified")
            robot = Robot.load(self._require_selected())
            self.previews.close(robot.definition.cameras.values())
            return core_cli.check(robot)

    # ── connection ──
    def connect(self) -> None:
        """Find every motor (torque stays off) and start the cameras. An arm whose motors report torque on (left
        holding by an earlier program) is listed as holding."""
        with self.lock:
            if self.robot is not None:
                return
            if self.closing.is_set():
                raise Busy("the server is shutting down")
            if self.identifying:
                raise Busy("an arm is being identified")
            robot = Robot.load(self._require_selected())
            ids = set(robot.definition.cameras.values())
            self.previews.blocked = ids
            self.previews.close(ids)
            try:
                robot.connect()
            except BaseException:
                self.previews.blocked = set()
                raise
            self.robot = robot
            torque = robot.torque()
            on = [side for side, state in torque.items() if state == "on"]
            if on:
                self.holding[robot.name] = on
                self._event(f"connected {robot.name}: {self._holding_text(robot.name)}")
            else:
                self._event(f"connected {robot.name}: torque " + ("off" if set(torque.values()) == {"off"} else
                                                                   "off not confirmed (a motor reports no mode)"))

    def disconnect(self) -> None:
        with self.lock:
            self._idle()
            if self.robot is not None and self.holding.get(self.robot.name):
                raise Busy(self._holding_text(self.robot.name))
            self._disconnect()

    def _disconnect(self) -> None:
        robot, self.robot = self.robot, None
        self.previews.blocked = set()
        if robot is not None:
            outcomes = robot.disconnect()
            energized = [side for side, outcome in outcomes.items() if outcome != "released"]
            if energized:
                self.holding[robot.name] = energized
                self._event(f"disconnected {robot.name}: torque off not confirmed on the {' and '.join(energized)} "
                            f"arm ({', '.join(outcomes[s] for s in energized)})")
            else:
                self._event(f"disconnected {robot.name}")

    def release(self, robot: str, revision: int) -> dict:
        """Torque off now on every arm of the selected robot, verified motor by motor; an arm away from rest drops.
        Works on an arm with a latched fault or a motor that does not answer. An arm whose torque off is not
        confirmed stays listed as holding."""
        with self.lock:
            self._idle()
            self._check_selection(robot, revision)
            if self.identifying:
                raise Busy("an arm is being identified")
            name = self._require_selected()
            connected = self.robot is not None and self.robot.name == name
            target = self.robot if connected else Robot.load(name)
            released = target.release()
            failed = [side for side, r in released.items() if not r["ok"]]
            if failed:
                self.holding[name] = failed
                self._event(f"{name}: torque off NOT confirmed on the {' and '.join(failed)} arm: "
                            + "; ".join(f"{s}: unconfirmed {r['unconfirmed']}, no reply from {r['missing']}"
                                        for s, r in released.items() if not r["ok"])
                            + ". Cut motor power if it still holds")
            else:
                self.holding.pop(name, None)
                self._event(f"{name}: torque off (released)")
            return released

    # ── runs ──
    def run(self, task: str, model: str, timeout_s: float, robot: str, revision: int) -> None:
        with self.lock:
            self._idle()
            if self.closing.is_set():
                raise Busy("the server is shutting down")
            if self.identifying:
                raise Busy("an arm is being identified")
            self._check_selection(robot, revision)
            name = self._require_selected()
            if self.holding.get(name):
                raise Busy(self._holding_text(name))
            self.stop_event = threading.Event()
            self.beat = time.monotonic()
            self.runner, self.result = None, None
            self.pending = {"phase": "connecting", "task": mask_secrets(task), "model": model, "events": []}
            self.thread = threading.Thread(target=self._run, args=(name, task, model, timeout_s, self.stop_event),
                                           name="run")
            self.thread.start()

    def stop(self, reason: str = "Stop pressed") -> None:
        """End the run (the arms return to rest) and any identify wiggle."""
        if self.identifying and not self.identify_stop.is_set():
            self.identify_stop.set()
            self._event(f"stopping the wiggle: {reason}")
        if self.running and not self.stop_event.is_set():
            self.stop_event.set()
            self._event(f"stopping: {reason}")

    def heartbeat(self) -> None:
        self.beat = time.monotonic()

    def _run(self, name: str, task: str, model: str, timeout_s: float, stop: threading.Event) -> None:
        try:
            self.connect()
            if stop.is_set():
                self.result = {"stop_reason": "stopped", "error": False}
                return
            self.runner = DreamRunner(self.robot, task, model=model)
            result = self.runner.run(timeout_s, stop, then="rest")
            self.result = result.to_json()
        except Exception as e:
            text = mask_secrets(str(e) if isinstance(e, ArmHeld) else f"{type(e).__name__}: {e}")
            self.result = {"stop_reason": text, "error": True}
            self._event(f"run failed: {text}")
        finally:
            self.pending = None
            self._note_holding(name)

    def _note_holding(self, name: str) -> None:
        """After a run: every arm of the connected robot whose torque is on (or not known to be off) is holding."""
        robot = self.robot
        if robot is None or robot.name != name:
            return
        try:
            torque = robot.torque()
        except Exception as e:
            log.error("torque state not read: %s", e)
            torque = {side: "unknown" for side in robot.sides}
        on = [side for side, state in torque.items() if state != "off"]
        if on:
            self.holding[name] = on
            self._event(self._holding_text(name))

    def _watch_heartbeat(self) -> None:
        while not self.closing.wait(0.2):
            if (self.running or self.identifying) and time.monotonic() - self.beat > self.heartbeat_s:
                self.stop("no page has answered for "
                          f"{self.heartbeat_s:g} s")

    # ── reading ──
    def status(self) -> dict:
        robot, runner, pending = self.robot, self.runner, self.pending
        arms = torque = None
        if robot is not None:
            try:
                arms, torque = robot.status(), robot.torque()
            except Exception as e:
                arms = {"error": str(e)}
        name = self.selected
        return {
            "selected": name,
            "revision": self.revision,
            "connected": robot is not None,
            "torque": torque,
            "running": self.running,
            "stopping": self.running and self.stop_event.is_set(),
            "run": runner.state() if runner is not None else (dict(pending) if pending else None),
            "result": self.result,
            "holding": list(self.holding.get(name, [])) if name else [],
            "robot": arms,
            "identifying": self.identifying,
            "events": list(self.events),
            "heartbeat_s": self.heartbeat_s,
        }

    def recordings_dir(self) -> Path:
        settings = config.load(self.selected).settings if self.selected else config.Settings()
        return Path(settings.recordings_dir).expanduser()

    def recordings(self) -> list[dict]:
        """Run folders, newest first: what ran, how it ended, the files written."""
        root = self.recordings_dir()
        out = []
        if not root.is_dir():
            return out
        for folder in sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: p.name,
                             reverse=True):
            info = {}
            try:
                info = json.loads((folder / "run.json").read_text())
            except (OSError, ValueError):
                pass
            out.append({"name": folder.name, "robot": info.get("robot"), "task": info.get("task"),
                        "model": info.get("model"), "started_at": info.get("started_at"),
                        "stopped_because": info.get("stopped_because"), "error": info.get("error"),
                        "files": sorted(p.name for p in folder.iterdir() if p.is_file() and not p.is_symlink())})
        return out

    # ── identify ──
    def identify(self, arm_type: str, port: str) -> dict:
        """Wiggle the gripper of the arm on ``port``. Stop, a lost heartbeat and shutdown cut it short; the result's
        ``released`` says whether the gripper's torque off was confirmed."""
        with self.lock:
            self._idle()
            if self.robot is not None:
                raise Busy("disconnect the robot first: its adapters are in use")
            if self.identifying:
                raise Busy("another arm is being identified")
            if self.closing.is_set():
                raise Busy("the server is shutting down")
            self.identifying = port
            self.identify_stop = threading.Event()
            self.beat = time.monotonic()
        try:
            result = devices.wiggle(arm_type, port, self.identify_stop)
        finally:
            self.identifying = None
        if not result.get("released"):
            self._event(f"{port}: the gripper's torque off was NOT confirmed; cut motor power if it still holds")
        return result

    # ── shutdown ──
    def close(self) -> None:
        """Stop a run (it returns the arms to rest) and any wiggle, then disconnect: an arm whose return is not
        verified is left holding, never released here."""
        self.closing.set()
        self.stop("the server is shutting down")
        thread = self.thread
        if thread is not None:
            thread.join()
        with self.lock:
            self._disconnect()
            for name, sides in self.holding.items():
                log.warning("%s", self._holding_text(name))
        self.previews.close_all()

    # ── helpers ──
    def _idle(self) -> None:
        if self.running:
            raise Busy("a run is in progress: stop it first")

    def _select(self, name: Optional[str]) -> None:
        self.selected, self.runner, self.result = name, None, None
        self.revision += 1

    def _check_selection(self, robot: str, revision: int) -> None:
        if robot != self.selected or revision != self.revision:
            raise Busy(f"the selected robot changed (now {self.selected or 'none'}): reload the page and try again")

    def _require_selected(self) -> str:
        if not self.selected:
            raise Busy("select a robot first")
        return self.selected

    def _holding_text(self, name: str) -> str:
        sides = self.holding.get(name, [])
        return (f"{name}: the {' and '.join(sides)} arm{'s are' if len(sides) > 1 else ' is'} still holding (torque "
                "off not confirmed). Support it, then release")

    def _event(self, text: str) -> None:
        self.events.append({"t": round(time.time(), 3), "kind": "system", "text": mask_secrets(text)})
