"""Robot definition files and run settings.

One YAML file per robot in ``$MAKERMODS_PLAYGROUND_HOME/robots`` (default ``~/.config/makermods-playground``)::

    arm: metal                                   # metal | maker: which SDK drives the arms
    arms:
      left:  {backend: gs_usb, port: "gs_usb:<serial>"}    # fields of the SDK rig's arm entries
      right: {backend: gs_usb, port: "gs_usb:<serial>"}
    cameras: {top: "usb:<path>", left: "usb:<path>", right: "url:<http URL>"}   # the SDK's camera ids
    settings: {servo_hz: 40}                     # optional, see Settings
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import yaml

ARM_TYPES = ("metal", "maker")
SIDES = ("left", "right")
CAMERAS = ("top", "left", "right")
# SDK rig camera name -> robot camera name.
RIG_CAMERAS = {"top": "top", "left_wrist": "left", "right_wrist": "right"}


class ConfigError(ValueError):
    pass


@dataclass
class Settings:
    servo_hz: float = 40.0            # joint targets per second while a policy runs
    ease_in_s: float = 0.6            # playback speeds up from a hold over this
    ease_out_s: float = 0.2           # and slows to a stop on a chunk's last row over this
    camera_max_age_s: float = 1.0     # an older frame counts as a camera dropout
    feedback_max_age_s: float = 0.2   # joint feedback older than this is not an observation
    tracking_limit_deg: float = 12.0  # command vs measured gap (joints 1-6) that ends a run
    ik_solver: str = "auto"           # numpy | jax | auto (numpy): the SDK's trajectory IK; JAX only when named
    request_timeout_ms: int = 60000   # one Dream API request
    record: bool = True               # write a recording per run
    recordings_dir: str = "recordings"

    # (low, high) for each number; a value outside, or not finite, is refused.
    RANGES = {"servo_hz": (10, 200), "ease_in_s": (0, 5), "ease_out_s": (0, 2), "camera_max_age_s": (0.05, 10),
              "feedback_max_age_s": (0.01, 1), "tracking_limit_deg": (1, 45), "request_timeout_ms": (1000, 600000)}

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "Settings":
        raw = dict(raw or {})
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ConfigError(f"settings: unknown keys {unknown}")
        s = cls(**raw)
        s.validate()
        return s

    def validate(self) -> None:
        for name, (low, high) in self.RANGES.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not low <= value <= high:
                raise ConfigError(f"settings.{name} must be a number in [{low:g}, {high:g}], got {value!r}")
        if self.ik_solver not in ("numpy", "jax", "auto"):
            raise ConfigError("settings.ik_solver must be numpy, jax or auto")


@dataclass
class RobotDef:
    name: str
    arm: str
    arms: dict[str, dict]
    cameras: dict[str, str] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)
    path: Optional[Path] = None

    def validate(self) -> None:
        if self.arm not in ARM_TYPES:
            raise ConfigError(f"{self.name}: arm must be one of {', '.join(ARM_TYPES)}, got {self.arm!r}")
        if not self.arms or set(self.arms) - set(SIDES):
            raise ConfigError(f"{self.name}: arms must be left and/or right, got {sorted(self.arms)}")
        if set(self.cameras) - set(CAMERAS):
            raise ConfigError(f"{self.name}: cameras must be among {', '.join(CAMERAS)}, got {sorted(self.cameras)}")
        rig = sdk(self.arm).rig
        for side, entry in self.arms.items():
            try:
                rig.ArmSpec(**entry)
            except (TypeError, rig.RigError) as e:
                raise ConfigError(f"{self.name}: arms.{side}: {e}") from e

    def arm_spec(self, side: str):
        return sdk(self.arm).rig.ArmSpec(**self.arms[side])

    @property
    def sides(self) -> list[str]:
        return [s for s in SIDES if s in self.arms]

    def to_dict(self) -> dict:
        out = {"arm": self.arm, "arms": {s: dict(self.arms[s]) for s in self.sides}, "cameras": dict(self.cameras)}
        changed = {k: v for k, v in asdict(self.settings).items() if v != getattr(Settings(), k)}
        if changed:
            out["settings"] = changed
        return out

    @classmethod
    def from_dict(cls, name: str, raw: dict, path: Optional[Path] = None) -> "RobotDef":
        if not isinstance(raw, dict) or set(raw) - {"arm", "arms", "cameras", "settings"}:
            raise ConfigError(f"{path or name}: expected a mapping with arm, arms, cameras and settings")
        robot = cls(name, raw.get("arm"), dict(raw.get("arms") or {}),
                    {str(k): str(v) for k, v in (raw.get("cameras") or {}).items()},
                    Settings.from_dict(raw.get("settings")), path)
        robot.validate()
        return robot


def sdk(arm_type: str) -> SimpleNamespace:
    """The SDK modules for an arm type: rig, arm, bus, cameras, trajectory_ik, kinematics, errors, the mock CAN backend
    and the CLI's helpers (``safety``, ``common``, ``move``)."""
    if arm_type not in ARM_TYPES:
        raise ConfigError(f"unknown arm type {arm_type!r}")
    package = f"{arm_type}_arm"
    modules = {m: importlib.import_module(f"{package}.{m}")
               for m in ("rig", "arm", "bus", "cameras", "trajectory_ik", "kinematics")}
    modules["errors"] = importlib.import_module(f"{package}.errors")
    modules["mock"] = importlib.import_module(f"{package}.transport.mock")
    modules["safety"] = importlib.import_module(f"{package}.cli.safety")
    modules["common"] = importlib.import_module(f"{package}.cli.common")
    modules["move"] = importlib.import_module(f"{package}.cli.move")
    return SimpleNamespace(**modules)


def home() -> Path:
    env = os.environ.get("MAKERMODS_PLAYGROUND_HOME")
    return Path(env).expanduser() if env else Path.home() / ".config" / "makermods-playground"


def robots_dir() -> Path:
    return home() / "robots"


def robot_names() -> list[str]:
    return sorted(p.stem for p in robots_dir().glob("*.yaml"))


def load(name: Optional[str] = None) -> RobotDef:
    """The named robot, or the only one there is."""
    if name is None:
        names = robot_names()
        if len(names) != 1:
            raise ConfigError("no robot defined: run 'playground robot init NAME --arm metal --from-rig'" if not names
                              else f"several robots ({', '.join(names)}): pass --robot NAME")
        name = names[0]
    path = robots_dir() / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"no robot named {name!r} ({path})")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML ({e})") from e
    return RobotDef.from_dict(name, raw, path)


def save(robot: RobotDef) -> Path:
    robot.validate()
    if not robot.name or "/" in robot.name or robot.name.startswith("."):
        raise ConfigError(f"invalid robot name {robot.name!r}")
    path = robots_dir() / f"{robot.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(robot.to_dict(), sort_keys=False))
    tmp.replace(path)
    robot.path = path
    return path


def remove(name: str) -> Path:
    path = robots_dir() / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"no robot named {name!r}")
    path.unlink()
    return path


def from_rig(name: str, arm_type: str, rig_path=None, arms=None, cameras=None) -> RobotDef:
    """A robot from the SDK's rig file: its arms named left/right and cameras top / left_wrist / right_wrist."""
    rig = sdk(arm_type).rig.Rig.load(rig_path, missing_ok=False)
    wanted_arms = list(arms or [s for s in SIDES if s in rig.arms])
    missing = [s for s in wanted_arms if s not in rig.arms]
    if missing or not wanted_arms:
        raise ConfigError(f"rig {rig.path} has no arm named {', '.join(missing) or 'left or right'}")
    by_role = {role: rig.cameras[n] for n, role in RIG_CAMERAS.items() if n in rig.cameras}
    wanted_cameras = list(cameras if cameras is not None else by_role)
    missing = [c for c in wanted_cameras if c not in by_role]
    if missing:
        raise ConfigError(f"rig {rig.path} has no camera for {', '.join(missing)}")
    entries = {}
    for side in wanted_arms:
        entry = rig.arms[side].to_yaml()
        if entry.get("profile") and rig.base and not Path(entry["profile"]).expanduser().is_absolute():
            entry["profile"] = str(rig.base / entry["profile"])
        entries[side] = entry
    robot = RobotDef(name, arm_type, entries,
                     {c: by_role[c] for c in wanted_cameras})
    robot.validate()
    return robot
