import os

os.environ["DREAM_LOG"] = "off"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def slow_replies(monkeypatch):
    """A starved test machine (a hosted Windows runner, a small ARM board) can take longer than the runner's
    REPLY_MAX_AGE_S to solve a reply while the simulated arms' control loops share its interpreter, and every reply
    would be dropped as too late. The tests check what a run does with its replies, not how fast this computer solves
    them; test_a_reply_to_a_stale_observation_is_discarded sets its own limit."""
    from makermods_playground.dream import runner
    monkeypatch.setattr(runner, "REPLY_MAX_AGE_S", 60.0)


@pytest.fixture(autouse=True)
def playground_home(tmp_path, monkeypatch):
    """Robot files and recordings go to a temporary directory; the user's are never read or written."""
    home = tmp_path / "home"
    monkeypatch.setenv("MAKERMODS_PLAYGROUND_HOME", str(home))
    monkeypatch.setenv("METAL_ARM_RIG", str(tmp_path / "no-metal-rig.yaml"))
    monkeypatch.setenv("MAKER_ARM_RIG", str(tmp_path / "no-maker-rig.yaml"))
    monkeypatch.setenv("DREAM_API_KEY", "")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "no-credentials"))
    return home
