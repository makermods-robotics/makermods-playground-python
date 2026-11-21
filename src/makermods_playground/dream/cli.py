"""``playground dream ...``."""

from __future__ import annotations

from dataclasses import replace

from ..core import config
from ..core.cli import CommandError, check_stay, close_holding, confirm, emit, progress, signals
from ..core.robot import Robot
from .client import MODELS, check
from .replay import RecordedPolicy, load_recording, replay_dry
from .runner import DreamRunner


def cmd_check(args) -> None:
    """Key present and models listed; the only command that calls the Dream API, and it moves nothing."""
    out = check()
    lines = [f"credentials: {out['credentials']}", f"gateway: {out['gateway']}",
             f"key: {'set' if out['key'] else 'missing'}"]
    for name, m in out["models"].items():
        lines.append(f"{name}: {m['id']} " + ("servable" if m["servable"] else "listed, not servable" if m["listed"]
                                               else "not available to this key"))
    if out.get("error"):
        lines.append(out["error"])
    emit(args, out, "\n".join(lines))
    if not out["key"] or not all(m["listed"] for m in out["models"].values()):
        raise SystemExit(1)


THEN = {"rest": "return to the pose they started at and go torque off",
        "stay": "stay holding with torque on after the program exits"}


def _robot(args) -> Robot:
    robot = Robot.load(args.robot)
    if getattr(args, "no_record", False):
        robot.settings = replace(robot.settings, record=False)
    return robot


def _drive(args, robot: Robot, runner: DreamRunner) -> None:
    """Run with SIGINT and SIGTERM ending the run (never its return to rest), then close the robot: "stay" leaves
    the arms holding, otherwise an arm whose return to rest was not verified is left holding without a release."""
    with signals() as stop:
        robot.connect()
        try:
            result = runner.run(args.timeout, stop, then=args.then)
        finally:
            if args.then == "stay":
                robot.detach()
                closed = {}
            else:
                closed = close_holding(robot, progress(args))
    data = {**result.to_json(), "torque": closed}
    if not args.json:
        lines = [f"ended: {result.stop_reason}", f"chunks {result.chunks}, servo ticks {result.ticks}, "
                 f"{result.elapsed_s:g} s, pacing {result.pacing}"]
        rtt = result.timing.get("rtt_ms", {})
        if rtt.get("p50") is not None:
            lines.append(f"round trip p50 {rtt['p50']:.0f} ms, p95 {rtt['p95']:.0f} ms; IK p50 "
                         f"{result.timing['ik_ms']['p50']:.0f} ms")
        if result.rest is not None:
            lines.append("rest: " + ", ".join(f"{s} {'verified' if ok else 'NOT verified, still holding'}"
                                              for s, ok in result.rest.items()))
        lines += [f"{s}: torque off not confirmed ({o})" for s, o in closed.items() if o not in ("released", "rest")]
        if result.recording:
            lines.append(f"recording: {result.recording}")
        emit(args, data, "\n".join(lines))
    else:
        emit(args, data)
    if result.error or (result.rest and not all(result.rest.values())) \
            or any(o not in ("released", "rest") for o in closed.values()):
        raise SystemExit(1)


def cmd_run(args) -> None:
    robot = _robot(args)
    check_stay(robot, args.then)
    confirm(args, f"The {robot.arm_type} arms of {robot.name} will move under the policy for up to "
                  f"{args.timeout:g} s, then {THEN[args.then]}. Workspace clear, hand on the e-stop?")
    runner = DreamRunner(robot, args.task, model=args.model,
                         on_event=None if args.json else lambda kind, text: print(f"[{kind}] {text}", flush=True))
    _drive(args, robot, runner)


def cmd_replay(args) -> None:
    recording = load_recording(args.recording)
    if args.dry:
        definition = config.load(args.robot) if args.robot else None
        if definition is not None and (definition.arm != recording.arm_type
                                       or set(recording.sides) - set(definition.sides)):
            raise CommandError(f"the recording drives {recording.arm_type} arms {', '.join(recording.sides)}; "
                               f"{definition.name} has {definition.arm} arms {', '.join(definition.sides)}")
        robot = Robot.offline(recording.arm_type, recording.sides, definition)
        report = replay_dry(recording, robot)
        if args.json:
            emit(args, report)
            return

        def fmt(stats):
            return "-" if stats.get("median") is None else (f"{stats['median']:.4f} / {stats['p95']:.4f} / "
                                                            f"{stats['max']:.4f}")

        def num(x):
            return "-" if x is None else f"{x:.4f}"

        lines = [f"{report['recording']}: {report['chunks']} chunks (rad: median / p95 / max)"]
        for side, a in report["arms"].items():
            lines.append(f"  {side}: IK vs recorded {fmt(a['ik_rad'])}")
            if "command_rad" in a:
                steps = a["max_step_rad"]
                lines.append(f"  {side}: commands vs recorded {fmt(a['command_rad'])}, gripper "
                             f"{fmt(a['gripper_rad'])}; largest step per tick {num(steps['ours'])} "
                             f"(recorded {num(steps['recorded'])})")
        emit(args, report, "\n".join(lines))
        return
    robot = _robot(args)
    if robot.arm_type != recording.arm_type:
        raise CommandError(f"the recording is of {recording.arm_type} arms, {robot.name} has {robot.arm_type} arms")
    if set(recording.sides) != set(robot.sides):
        raise CommandError(f"the recording drives {', '.join(recording.sides)}; {robot.name} has "
                           f"{', '.join(robot.sides)}")
    check_stay(robot, args.then)
    confirm(args, f"The arms of {robot.name} will replay {len(recording.chunks)} recorded chunks, then "
                  f"{THEN[args.then]}.")
    runner = DreamRunner(robot, recording.task or "replay", model="replay", policy=RecordedPolicy(recording),
                         on_event=None if args.json else lambda kind, text: print(f"[{kind}] {text}", flush=True))
    _drive(args, robot, runner)


def add_parsers(sub, common) -> None:
    dream = sub.add_parser("dream", help="Dream1 policies").add_subparsers(dest="action", required=True)
    dream.add_parser("check", parents=[common], help="API key and models (moves nothing)").set_defaults(
        func=cmd_check)
    p = dream.add_parser("run", parents=[common], help="run a task on the robot")
    p.add_argument("task")
    p.add_argument("--timeout", type=float, default=60.0, metavar="S")
    p.add_argument("--model", choices=tuple(MODELS), default="beta")
    p.add_argument("--then", choices=("rest", "stay"), default="rest")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_run)
    p = dream.add_parser("replay", parents=[common], help="a recorded run's chunks without the API")
    p.add_argument("recording")
    p.add_argument("--dry", action="store_true", help="numbers only; nothing connects")
    p.add_argument("--timeout", type=float, default=120.0, metavar="S")
    p.add_argument("--then", choices=("rest", "stay"), default="rest")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_replay)
