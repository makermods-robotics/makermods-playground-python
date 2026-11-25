"""``playground``: robot and camera commands, Dream policies."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .core import cli as core_cli
from .core.mask import mask_secrets
from .core.robot import ArmHeld
from .dream import cli as dream_cli


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print JSON")
    common.add_argument("--robot", metavar="NAME", help="robot definition (default: the only one)")
    p = argparse.ArgumentParser(prog="playground", description="Run Dream Labs policies on MakerMods arms.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    core_cli.add_parsers(sub, common)
    dream_cli.add_parsers(sub, common)
    ui = sub.add_parser("ui", help="the web page: run tasks, set up robots")
    ui.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    ui.add_argument("--port", type=int, default=8080)
    ui.add_argument("--robot", metavar="NAME", help="robot selected at start (default: the only one)")
    ui.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="a Host name the page is reached at (repeatable); needed with a network --host")
    ui.add_argument("--token", help="the token that drives the arms (default $PLAYGROUND_TOKEN); needed with a "
                                    "network --host")
    ui.set_defaults(func=cmd_ui)
    return p


def cmd_ui(args) -> None:
    try:
        from .ui.app import serve
    except ImportError as e:
        raise core_cli.CommandError(f"the UI needs its extra: pip install 'makermods-playground[ui]' ({e})") from e
    try:
        serve(args.host, args.port, args.robot, tuple(args.allow_host), args.token)
    except ValueError as e:
        raise core_cli.CommandError(str(e)) from e


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(args)
    except SystemExit as e:
        return int(e.code or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        plain = isinstance(e, (core_cli.CommandError, ArmHeld))
        text = mask_secrets(str(e) if plain else f"{type(e).__name__}: {e}")
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": text}))
        else:
            print(f"error: {text}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
