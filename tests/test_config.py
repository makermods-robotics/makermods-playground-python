import pytest
import yaml

from makermods_playground.core import config
from makermods_playground.core.config import ConfigError, RobotDef, Settings


def write_rig(path, arm_type="metal", **extra):
    rig = config.sdk(arm_type).rig.Rig.from_dict({
        "arms": {"left": {"backend": "gs_usb", "port": "gs_usb:AAA"},
                 "right": {"backend": "gs_usb", "port": "gs_usb:BBB"}},
        "cameras": {"top": "usb:top", "left_wrist": "usb:lw", "right_wrist": "usb:rw"}, **extra})
    rig.save(path)
    return path


def test_round_trip(playground_home):
    robot = RobotDef("lab", "metal", {"left": {"backend": "gs_usb", "port": "gs_usb:AAA"}}, {"top": "usb:top"},
                     Settings(servo_hz=50, record=False))
    path = config.save(robot)
    assert path == playground_home / "robots" / "lab.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw == {"arm": "metal", "arms": {"left": {"backend": "gs_usb", "port": "gs_usb:AAA"}},
                   "cameras": {"top": "usb:top"}, "settings": {"servo_hz": 50, "record": False}}
    loaded = config.load()
    assert loaded.to_dict() == robot.to_dict()
    assert loaded.sides == ["left"] and loaded.settings.servo_hz == 50
    assert config.robot_names() == ["lab"]
    config.remove("lab")
    assert config.robot_names() == []


def test_one_robot_is_the_default_several_need_a_name():
    with pytest.raises(ConfigError, match="no robot defined"):
        config.load()
    for name in ("a", "b"):
        config.save(RobotDef(name, "maker", {"right": {"backend": "socketcan", "channel": "can1"}}))
    with pytest.raises(ConfigError, match="several robots"):
        config.load()
    assert config.load("b").arm == "maker"
    with pytest.raises(ConfigError, match="no robot named"):
        config.load("c")


@pytest.mark.parametrize("raw, message", [
    ({"arm": "so101", "arms": {"left": {"backend": "gs_usb"}}}, "arm must be"),
    ({"arm": "metal", "arms": {"middle": {"backend": "gs_usb"}}}, "left and/or right"),
    ({"arm": "metal", "arms": {}}, "left and/or right"),
    ({"arm": "metal", "arms": {"left": {"backend": "pigeon"}}}, "backend must be"),
    ({"arm": "metal", "arms": {"left": {"backend": "gs_usb", "speed": 3}}}, "arms.left"),
    ({"arm": "metal", "arms": {"left": {"backend": "gs_usb"}}, "cameras": {"front": "usb:x"}}, "cameras must be"),
    ({"arm": "metal", "arms": {"left": {"backend": "gs_usb"}}, "settings": {"warp": 9}}, "unknown keys"),
    ({"arm": "metal", "arms": {"left": {"backend": "gs_usb"}}, "settings": {"ik_solver": "magic"}}, "ik_solver"),
    ({"arm": "metal", "arms": {"left": {"backend": "gs_usb"}}, "extra": 1}, "expected a mapping"),
])
def test_invalid_files(raw, message):
    with pytest.raises(ConfigError, match=message):
        RobotDef.from_dict("x", raw)


def test_from_rig(tmp_path):
    rig = write_rig(tmp_path / "rig.yaml")
    robot = config.from_rig("lab", "metal", rig)
    assert robot.arms == {"left": {"backend": "gs_usb", "port": "gs_usb:AAA"},
                          "right": {"backend": "gs_usb", "port": "gs_usb:BBB"}}
    assert robot.cameras == {"top": "usb:top", "left": "usb:lw", "right": "usb:rw"}
    reduced = config.from_rig("one", "metal", rig, arms=["left"], cameras=["top", "left"])
    assert reduced.sides == ["left"] and reduced.cameras == {"top": "usb:top", "left": "usb:lw"}
    with pytest.raises(ConfigError, match="no camera"):
        config.from_rig("x", "metal", write_rig(tmp_path / "r2.yaml", leaders={}), cameras=["top", "back"])


def test_from_rig_resolves_profile_paths(tmp_path):
    rig = config.sdk("maker").rig.Rig.from_dict({"arms": {"left": {"backend": "gs_usb", "profile": "mine.yaml"}}})
    rig.save(tmp_path / "rig.yaml")
    robot = config.from_rig("m", "maker", tmp_path / "rig.yaml")
    assert robot.arms["left"]["profile"] == str(tmp_path / "mine.yaml")


def test_missing_rig(tmp_path):
    with pytest.raises(Exception, match="no rig file"):
        config.from_rig("x", "metal", tmp_path / "none.yaml")


@pytest.mark.parametrize("key, value", [("camera_max_age_s", float("nan")), ("tracking_limit_deg", float("inf")),
                                        ("feedback_max_age_s", 0.0), ("ease_in_s", -1.0),
                                        ("request_timeout_ms", "60000"), ("servo_hz", True)])
def test_settings_refuse_values_that_would_disable_a_stop(key, value):
    with pytest.raises(ConfigError, match=key):
        Settings.from_dict({key: value})
