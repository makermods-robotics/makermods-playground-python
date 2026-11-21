"""Run recordings: one folder per run with an MP4 per camera, the motion log and the policy trace.

Files (written by the recorder's own thread, so the servo loop never waits on encoding or disk):
``<camera>.mp4``; ``motion.csv`` (every servo tick: commanded and measured joints, rad); ``chunks.csv``;
``timing.csv``; ``raw_chunks.npz`` (each reply's end-effector head per arm: ``<side>_eef_xyz``, ``<side>_eef_rot6``,
``<side>_grip``); ``ik_trace.json`` (per chunk and arm: head, IK start, speed cap, solved joints); ``run.json``.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .mask import redact_secrets

RECORD_FPS = 15
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
TIMING_KEYS = ("total", "request", "network", "serve_infer", "ik", "other")


def _num(x, digits=5) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def new_folder(root, robot_name: str) -> Path:
    folder = Path(root).expanduser() / f"{datetime.now():%Y%m%d-%H%M%S}_{robot_name}"
    n = 1
    while folder.exists():
        n += 1
        folder = folder.with_name(f"{folder.name.split('~')[0]}~{n}")
    folder.mkdir(parents=True)
    return folder


class RunLog:
    def __init__(self, sides):
        self.sides = list(sides)
        self.lock = threading.Lock()
        self.motion, self.chunks, self.timing, self.heads, self.ik = [], [], [], [], []

    def tick(self, t: float, phase: int, commanded: dict, measured: dict) -> None:
        row = [t, phase]
        for side in self.sides:
            for values in (commanded.get(side), measured.get(side)):
                row.extend([np.nan] * 7 if values is None else [float(v) for v in values])
        with self.lock:
            self.motion.append(row)

    def chunk(self, arrived: float, observed: float, skipped: int, heads: dict, trace: dict,
              timing: Optional[dict] = None) -> None:
        with self.lock:
            self.chunks.append((arrived, observed, arrived - observed, skipped))
            self.timing.append((arrived, timing or {}))
            self.heads.append((arrived, observed, heads))
            self.ik.append({"arrived": arrived, "observed": observed, "played_from_row": skipped, "arms": trace})

    def write(self, folder: Path) -> list[str]:
        with self.lock:
            motion, chunks, timing, heads, ik = (list(x) for x in (self.motion, self.chunks, self.timing,
                                                                   self.heads, self.ik))
        written = []

        def csv(name, header, rows):
            (folder / name).write_text("\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n")
            written.append(name)

        if motion:
            names = [f"{kind}_{side}_{j}" for side in self.sides for kind in ("cmd", "meas") for j in JOINTS]
            csv("motion.csv", ["t", "phase", *names],
                [[f"{r[0]:.4f}", str(int(r[1]))] + [_num(v) for v in r[2:]] for r in motion])
        if chunks:
            csv("chunks.csv", ["arrived", "observed", "latency_s", "skipped"],
                [[f"{a:.4f}", f"{o:.4f}", f"{lat:.4f}", str(s)] for a, o, lat, s in chunks])
            csv("timing.csv", ["arrived"] + [f"{k}_ms" for k in TIMING_KEYS],
                [[f"{a:.4f}"] + [_num(t.get(k), 1) for k in TIMING_KEYS] for a, t in timing])
        if heads:
            arrays = {"arrived": np.array([a for a, _, _ in heads]), "observed": np.array([o for _, o, _ in heads])}
            for side in heads[0][2]:
                for part in ("eef_xyz", "eef_rot6", "grip"):
                    rows = [np.asarray(h[side][part], dtype=float) for _, _, h in heads]
                    longest = max(len(r) for r in rows)   # shorter replies padded with NaN rows
                    arrays[f"{side}_{part}"] = np.stack([np.concatenate([r, np.full((longest - len(r),) + r.shape[1:],
                                                                                    np.nan)]) for r in rows])
            np.savez(folder / "raw_chunks.npz", **arrays)
            written.append("raw_chunks.npz")
        if ik:
            (folder / "ik_trace.json").write_text(json.dumps({"version": 2, "units": "rad", "chunks": ik}))
            written.append("ik_trace.json")
        return written


class RunRecorder:
    """``frames()`` returns {camera: newest image}; each is written at RECORD_FPS, so a camera with no newer image
    repeats its last one and the video keeps time. A video failure stops the videos only: the motion log and
    run.json are written once the run finishes. ``error`` is the recording's own problem, kept apart from the run's."""

    def __init__(self, folder: Path, sides, frames: Callable[[], dict], info: dict, fps: int = RECORD_FPS):
        self.folder, self.frames, self.fps = Path(folder), frames, fps
        self.log = RunLog(sides)
        self.info = dict(info)
        self.ending: dict = {}
        self.error: Optional[str] = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="recorder")

    def start(self) -> "RunRecorder":
        self.thread.start()
        return self

    def finish(self, ending: dict) -> Optional[str]:
        """Write everything and wait until it is written; returns the recording's error, if any."""
        self.ending = dict(ending)
        self.stop_event.set()
        self.thread.join()
        return self.error

    def _run(self) -> None:
        import cv2
        writers = {}
        period = 1.0 / self.fps
        deadline = time.monotonic()
        try:
            while not self.stop_event.is_set():
                for name, image in (self.frames() or {}).items():
                    if name not in writers:
                        h, w = image.shape[:2]
                        writers[name] = cv2.VideoWriter(str(self.folder / f"{name}.mp4"),
                                                        cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                    writers[name].write(image)
                deadline = max(deadline + period, time.monotonic())
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
        except Exception as e:
            self.error = f"videos stopped: {type(e).__name__}: {e}"
        finally:
            for w in writers.values():
                w.release()
        self.stop_event.wait()
        try:
            written = self.log.write(self.folder)
        except Exception as e:
            written = []
            self.error = self.error or f"motion log not saved: {type(e).__name__}: {e}"
        info = {**self.info, **self.ending, "units": "rad", "fps": self.fps,
                "videos": {n: f"{n}.mp4" for n in writers}, "files": written, "recording_error": self.error}
        (self.folder / "run.json").write_text(json.dumps(redact_secrets(info), indent=2, default=str))
