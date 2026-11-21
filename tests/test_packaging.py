import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_arm_sdks_are_pinned_to_a_revision():
    project = tomllib.loads(PYPROJECT.read_text())
    sdks = [d for d in project["project"]["dependencies"] if d.split()[0] in ("metal-arm", "maker-arm")]
    assert len(sdks) == 2
    assert all(re.fullmatch(r"\S+ @ git\+https://\S+?@\S+", d) for d in sdks), sdks
    assert set(project["tool"]["uv"]["sources"]) == {"metal-arm", "maker-arm"}
