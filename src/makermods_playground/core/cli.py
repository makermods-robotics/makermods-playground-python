"""``playground robot ...`` and ``playground cameras ...``."""

from __future__ import annotations

import json
import math
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import config
from .robot import Robot


class CommandError(Exception):
    """Ends a command with exit status 1 and this message."""


def emit(args, data: dict, text: str = "") -> None:
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, default=_jsonable))
    elif text:
        print(text)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def confirm(args, question: str) -> None:
    """Ask before anything moves; ``--yes`` answers for scripts."""
    if getattr(args, "yes", False):
        return
    if getattr(args, "json", False) or not sys.stdin.isatty():
        raise CommandError(f"{question} Pass --yes to confirm.")
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        raise CommandError("cancelled")


@contextmanager
def signals():
    """An event that SIGINT and SIGTERM set, instead of raising, until the block ends: a command that moves installs
    it before connecting and keeps it until its cleanup is done, so a signal ends the motion but never interrupts the
    return to rest or the release."""
    stop = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield stop
        return
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield stop
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def wait(seconds, stop: threading.Event, robot=None) -> None:
    """Until ``seconds`` pass (forever when None), ``stop`` is set, or an arm of ``robot`` leaves "enabled" (a
    fault)."""
    deadline = math.inf if seconds is None else time.monotonic() + seconds
    while not stop.is_set() and time.monotonic() < deadline:
        if robot is not None and any(s != "enabled" for s in robot.states().values()):
            return
        stop.wait(min(0.1, max(0.0, deadline - time.monotonic())))


def progress(args):
    """Progress text: stdout, or stderr when stdout carries JSON."""
    return (lambda *parts: print(*parts, file=sys.stderr, flush=True)) if getattr(args, "json", False) else print


def check_stay(robot: Robot, then: str) -> None:
    """Refuse "stay" before anything is energized when a stiff joint has a motor watchdog (the SDK's check): it
    would drop once commands stop."""
    robot.sdk.common.check_stay(robot.arms.values(), then)


def close_holding(robot: Robot, say=print) -> dict[str, str]:
    """Close a robot whose arms may still be energized, never releasing one: an enabled arm is left holding (the
    SDK's leave_holding: detached, or, when a stiff joint would drop on its watchdog, this process keeps holding it
    until RELEASE is typed); then every bus closes. Returns the outcome per arm (only "released" and "rest" are
    torque off confirmed)."""
    held = {}
    for side, arm in robot.arms.items():
        if arm.state.value in ("enabled", "fault"):
            held[side] = robot.sdk.safety.leave_holding(arm, say)
    return {**robot.disconnect(), **held}


def finish(robot: Robot, then: str, say=print) -> dict:
    """End a command that enabled the arms: rest (home to the rest pose, then torque off; see Robot.enable()), stay
    (torque on) or release (off now)."""
    if then == "stay":
        return {"then": "stay", "holding": robot.detach()}
    if then == "release":
        released = robot.release()
        robot.disconnect()
        bad = [s for s, r in released.items() if not r["ok"]]
        if bad:
            raise CommandError(f"torque off not confirmed on {', '.join(bad)}: {released}. Cut motor power if they "
                               "still hold")
        return {"then": "release", "released": released}
    rest = robot.stop()
    for side, what in robot.retract_notes().items():
        say(f"{side}: retract {what}; joint return from where it was")
    torque = close_holding(robot, say)
    if not all(rest.values()):
        raise CommandError("return to rest not verified on " + ", ".join(s for s, ok in rest.items() if not ok)
                           + ": still holding. Support the arm, then run 'playground robot release --yes'")
    return {"then": "rest", "rest": rest, "torque": torque}


ENDINGS = {"rest": "then they return to rest (an arm enabled away from its rest pose to the profile's rest pose) and "
                    "torque goes off",
           "stay": "then the program exits with torque still on: release them with 'playground robot release'",
           "release": "then torque goes off where they are: an arm that is not supported DROPS"}


# ── robot ──
def cmd_init(args) -> None:
    if args.name in config.robot_names() and not args.force:
        raise CommandError(f"robot {args.name!r} exists; pass --force to replace it")
    if args.from_rig is not None:
        robot = config.from_rig(args.name, args.arm, args.from_rig or None, args.arms, args.cameras)
    else:
        if not args.port:
            raise CommandError("give --from-rig, or the arms with --port left=gs_usb:<serial> [--port right=...]")
        arms = {}
        for item in args.port:
            side, _, address = item.partition("=")
            arms[side] = _arm_entry(address)
        cameras = dict(item.partition("=")[::2] for item in args.camera or [])
        robot = config.RobotDef(args.name, args.arm, arms, cameras)
    path = config.save(robot)
    emit(args, {"name": robot.name, "path": path, **robot.to_dict()},
         f"wrote {path}\n{_yaml(robot)}")


def _arm_entry(address: str) -> dict:
    if address.startswith("gs_usb"):
        return {"backend": "gs_usb", **({"port": address} if ":" in address else {})}
    if address.startswith("/dev/") or address.upper().startswith("COM"):
        return {"backend": "slcan", "port": address}
    return {"backend": "socketcan", "channel": address}


def _yaml(robot) -> str:
    import yaml
    return yaml.safe_dump(robot.to_dict(), sort_keys=False).rstrip()


def cmd_list(args) -> None:
    names = config.robot_names()
    rows = []
    for name in names:
        try:
            r = config.load(name)
            rows.append({"name": name, "arm": r.arm, "arms": r.sides, "cameras": list(r.cameras)})
        except config.ConfigError as e:
            rows.append({"name": name, "error": str(e)})
    emit(args, {"robots": rows, "dir": config.robots_dir()},
         "\n".join(f"{r['name']}: " + (r.get("error") or f"{r['arm']}, arms {' '.join(r['arms'])}, cameras "
                                        f"{' '.join(r['cameras']) or 'none'}") for r in rows)
         or f"no robots in {config.robots_dir()}")


def cmd_show(args) -> None:
    robot = config.load(args.name or args.robot)
    emit(args, {"name": robot.name, "path": robot.path, **robot.to_dict()}, f"# {robot.path}\n{_yaml(robot)}")


def cmd_remove(args) -> None:
    path = config.remove(args.name)
    emit(args, {"removed": args.name, "path": path}, f"removed {path}")


def check(robot: Robot, camera_timeout: float = 5.0) -> dict:
    """Read-only preflight of a robot that is not connected: motors answer without faults, temperatures, cameras
    deliver frames. Each arm and camera is opened and closed again; torque is never switched on."""
    report = {"robot": robot.name, "arm": robot.arm_type, "arms": {}, "cameras": {}, "ok": True}
    for side, arm in robot.arms.items():
        entry = {"ok": False}
        try:
            arm.connect()
            st = arm.status()
            temps = {n: j["temperature"] for n, j in st["joints"].items()}
            faults = {n: j["fault"] for n, j in st["joints"].items() if j["fault"]}
            hot = {}
            for j in arm.config.joints:
                t = temps.get(j.label)
                limit = j.enable_temperature_c or j.max_temperature_c
                if t is not None and limit is not None and t >= limit:
                    hot[j.label] = t
            entry.update(ok=not faults and not hot, temperatures=temps, faults=faults, too_hot=hot,
                         positions={n: j["position"] for n, j in st["joints"].items()})
        except Exception as e:
            entry["error"] = str(e)
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass
        report["arms"][side] = entry
    for name, cam in robot.cameras.items():
        entry = {"id": str(cam.id), "ok": False}
        try:
            cam.open(timeout=camera_timeout)
            frame, _ = cam.read(timeout=camera_timeout)
            entry.update(ok=True, size=[int(frame.shape[1]), int(frame.shape[0])])
        except Exception as e:
            entry["error"] = str(e)
        finally:
            cam.close()
        report["cameras"][name] = entry
    report["ok"] = all(e["ok"] for e in [*report["arms"].values(), *report["cameras"].values()])
    return report


def cmd_check(args) -> None:
    """Read-only preflight: motors answer without faults, temperatures, cameras deliver frames."""
    robot = Robot.load(args.robot)
    report = check(robot, args.camera_timeout)
    lines = [f"{robot.name} ({robot.arm_type})"]
    for side, e in report["arms"].items():
        lines.append(f"  arm {side}: " + ("OK" if e["ok"] else "FAIL") + (f" -- {e['error']}" if "error" in e else "")
                     + (f" faults {e['faults']}" if e.get("faults") else "")
                     + (f" too hot {e['too_hot']}" if e.get("too_hot") else "")
                     + (f" max {max(v for v in e['temperatures'].values() if v is not None):.0f} °C"
                        if e.get("temperatures") else ""))
    for name, e in report["cameras"].items():
        state = "OK " + "x".join(map(str, e["size"])) if e["ok"] else f"FAIL -- {e['error']}"
        lines.append(f"  camera {name}: {state}")
    emit(args, report, "\n".join(lines))
    if not report["ok"]:
        raise SystemExit(1)


def cmd_status(args) -> None:
    robot = Robot.load(args.robot)
    robot.connect(cameras=False)
    try:
        st = robot.status()
    finally:
        robot.group.disconnect()
    lines = []
    for side, arm in st["arms"].items():
        lines.append(f"{side}: {arm['state']}" + (f" ({arm['fault_reason']})" if arm["fault_reason"] else "")
                     + (f", gripper {arm['gripper_opening']:.2f} open" if arm["gripper_opening"] is not None else ""))
        for name, j in arm["joints"].items():
            pos = "-" if j["position"] is None else f"{math.degrees(j['position']):+7.1f}°"
            temp = "-" if j["temperature"] is None else f"{j['temperature']:.0f} °C"
            lines.append(f"  {name:14s} {pos}  {temp}" + (f"  FAULT {j['fault']}" if j["fault"] else ""))
    emit(args, st, "\n".join(lines))


def cmd_hold(args) -> None:
    robot = Robot.load(args.robot)
    check_stay(robot, args.then)
    confirm(args, "Torque on: the arms hold where they are" + (f" for {args.for_s:g} s" if args.for_s else "")
            + f"; {ENDINGS[args.then]}.")
    say = progress(args)
    with signals() as stop:
        robot.connect(cameras=False)
        try:
            robot.enable(cancel=stop)
            for side in robot.rehomed:
                say(f"{side}: not at its rest pose; it ends at the profile's rest pose, not here")
            say("holding" + (f" for {args.for_s:g} s" if args.for_s else "; Ctrl-C ends"))
            wait(args.for_s, stop, robot)
            faults = {s: f for s, f in robot.faults().items() if f}
        except BaseException:
            close_holding(robot, say)
            raise
        out = finish(robot, args.then, say)
    emit(args, {"held_s": args.for_s, "faults": faults, **out},
         f"done ({out['then']})" + "".join(f"\nfault on {s}: {f}" for s, f in faults.items()))
    if faults:
        raise SystemExit(1)


def cmd_rest(args) -> None:
    """Move each arm in turn to its profile's rest pose (zeros where the profile names none), verified, then torque
    off. The first arm not verified there keeps holding, and so do the ones not yet moved."""
    robot = Robot.load(args.robot)
    confirm(args, "The arms move one at a time to their rest pose; each goes torque off once it is verified there.")
    say = progress(args)
    with signals() as stop:
        robot.connect(cameras=False)
        arrived = {}
        try:
            robot.enable(cancel=stop)
            arrived = robot.rest(stop)
        finally:
            torque = close_holding(robot, say)
    if set(arrived) != set(robot.sides) or not all(arrived.values()):
        raise CommandError("rest pose not verified on " + ", ".join(s for s in robot.sides if not arrived.get(s))
                           + ": left holding. Support the arm, then run 'playground robot release --yes'")
    emit(args, {"arrived": arrived, "torque": torque}, "at rest, torque off")


def cmd_release(args) -> None:
    confirm(args, "Torque off now: an arm that is not at rest drops. Support it first.")
    robot = Robot.load(args.robot)
    with signals():
        released = robot.release()
    lines = [f"{s}: torque off confirmed" if r["ok"] else
             f"{s}: torque off NOT confirmed" + (f" on {', '.join(map(str, r['unconfirmed']))}" if r["unconfirmed"]
                                                 else "") + (f"; no reply from {', '.join(r['missing'])}"
                                                             if r["missing"] else "") for s, r in released.items()]
    emit(args, {"released": released}, "\n".join(lines))
    if not all(r["ok"] for r in released.values()):
        raise SystemExit(1)


# ── cameras ──
def cmd_snapshot(args) -> None:
    import cv2
    robot = Robot.load(args.robot)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    saved, errors = {}, {}
    for name, cam in robot.cameras.items():
        try:
            cam.open()
            frame, _ = cam.read()
            path = out / f"{robot.name}_{name}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"could not write {path}")
            saved[name] = str(path)
        except Exception as e:
            errors[name] = str(e)
        finally:
            cam.close()
    emit(args, {"saved": saved, "errors": errors},
         "\n".join([f"{n}: {p}" for n, p in saved.items()] + [f"{n}: FAIL -- {e}" for n, e in errors.items()]))
    if errors:
        raise SystemExit(1)


def cmd_stream(args) -> None:
    """Measure each camera's frame rate over ``--for`` seconds."""
    robot = Robot.load(args.robot)
    counts, errors = {}, {}
    stop = threading.Event()

    def count(name, cam):
        try:
            cam.open()
            counts[name] = 0
            while not stop.is_set():
                cam.read(timeout=1.0)
                counts[name] += 1
        except Exception as e:
            errors[name] = str(e)

    threads = [threading.Thread(target=count, args=(n, c), daemon=True) for n, c in robot.cameras.items()]
    started = time.monotonic()
    for t in threads:
        t.start()
    wait(args.for_s, stop)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    robot.close_cameras()
    elapsed = time.monotonic() - started
    fps = {n: round(c / elapsed, 1) for n, c in counts.items()}
    emit(args, {"fps": fps, "seconds": round(elapsed, 2), "errors": errors},
         "\n".join([f"{n}: {f} fps" for n, f in fps.items()] + [f"{n}: FAIL -- {e}" for n, e in errors.items()]))


def add_parsers(sub, common) -> None:
    robot = sub.add_parser("robot", help="robot definitions, status and motion").add_subparsers(dest="action",
                                                                                              required=True)
    p = robot.add_parser("init", parents=[common], help="write a robot definition")
    p.add_argument("name")
    p.add_argument("--arm", choices=config.ARM_TYPES, required=True)
    p.add_argument("--from-rig", nargs="?", const="", default=None, metavar="RIG",
                   help="the SDK's rig file (default: its usual place)")
    p.add_argument("--arms", nargs="+", choices=config.SIDES)
    p.add_argument("--cameras", nargs="*", choices=config.CAMERAS)
    p.add_argument("--port", action="append", metavar="SIDE=ADDRESS", help="gs_usb:<serial>, /dev/ttyACM0 or can0")
    p.add_argument("--camera", action="append", metavar="NAME=ID", help="top=usb:<path> or top=url:<http URL>")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)
    robot.add_parser("list", parents=[common], help="the robots defined").set_defaults(func=cmd_list)
    p = robot.add_parser("show", parents=[common], help="a robot's definition")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_show)
    p = robot.add_parser("remove", parents=[common], help="delete a robot definition")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove)
    p = robot.add_parser("check", parents=[common], help="read-only preflight of motors and cameras")
    p.add_argument("--camera-timeout", type=float, default=5.0)
    p.set_defaults(func=cmd_check)
    robot.add_parser("status", parents=[common], help="joints, temperatures, faults (torque unchanged)") \
        .set_defaults(func=cmd_status)
    p = robot.add_parser("hold", parents=[common], help="torque on, holding the pose")
    p.add_argument("--for", dest="for_s", type=float, metavar="S")
    p.add_argument("--then", choices=("rest", "stay", "release"), default="rest")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_hold)
    p = robot.add_parser("rest", parents=[common], help="move to the rest pose, then torque off")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_rest)
    p = robot.add_parser("release", parents=[common], help="torque off now (the arm drops if not at rest)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_release)

    cams = sub.add_parser("cameras", help="the robot's cameras").add_subparsers(dest="action", required=True)
    p = cams.add_parser("snapshot", parents=[common], help="one JPEG per camera")
    p.add_argument("--out", default=".")
    p.set_defaults(func=cmd_snapshot)
    p = cams.add_parser("stream", parents=[common], help="measure each camera's frame rate")
    p.add_argument("--for", dest="for_s", type=float, default=5.0, metavar="S")
    p.set_defaults(func=cmd_stream)

