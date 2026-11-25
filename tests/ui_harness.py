"""The web app on simulated robots: Robot.load gives sim arms and fake cameras for the saved definition, the Dream
API is the fake policy, and requests carry the page token. ``harness`` is a fixture body for
``pytest.fixture(ui_harness.harness)``."""

from __future__ import annotations

import sim
from fastapi.testclient import TestClient

from makermods_playground.core import config
from makermods_playground.core.robot import Robot
from makermods_playground.dream import runner as runner_module
from makermods_playground.ui import app as app_module

ARMS = {"left": {"backend": "gs_usb", "port": "gs_usb:A"}, "right": {"backend": "gs_usb", "port": "gs_usb:B"}}
CAMERAS = {"top": "usb:t", "left": "usb:l", "right": "usb:r"}


def save_robot(name="lab", arm="metal", arms=None, cameras=None, **settings) -> config.RobotDef:
    settings.setdefault("record", False)
    definition = config.RobotDef(name, arm, dict(arms or ARMS), dict(CAMERAS if cameras is None else cameras),
                                 config.Settings(**settings))
    config.save(definition)
    return definition


class Harness:
    def __init__(self, monkeypatch):
        self.robots: list[Robot] = []
        self.policies: list[sim.FakePolicy] = []
        self.policy_kwargs: dict = {}
        self.sim_options: dict = {}   # for every simulated motor of the robots loaded next
        self.prepare = None           # prepare(robot) on each robot loaded next, before it connects
        harness = self

        def load(cls, name=None):
            definition = config.load(name)
            robot, motors = sim.sim_robot(definition.arm, sides=tuple(definition.sides),
                                          cameras=tuple(definition.cameras), name=definition.name,
                                          settings=definition.settings, **harness.sim_options)
            robot.definition = definition
            robot.motors = motors
            if harness.prepare:
                harness.prepare(robot)
            harness.robots.append(robot)
            return robot

        def policy(model, sides, timeout_ms, **stand_ins):
            made = sim.FakePolicy(harness.robots[-1], sides, **harness.policy_kwargs)
            made.model = model
            harness.policies.append(made)
            return made

        monkeypatch.setattr(Robot, "load", classmethod(load))
        monkeypatch.setattr(runner_module, "DreamPolicy", policy)

    def client(self, **kwargs) -> TestClient:
        client = TestClient(app_module.create_app(**kwargs))
        client.__enter__()
        client.headers[app_module.TOKEN_HEADER] = client.get("/api/session").json()["token"]
        return client

    @property
    def robot(self) -> Robot:
        return self.robots[-1]


def harness(monkeypatch):
    made = Harness(monkeypatch)
    clients = []
    original = made.client

    def client(**kwargs):
        c = original(**kwargs)
        clients.append(c)
        return c

    made.client = client
    yield made
    for c in clients:
        c.__exit__(None, None, None)
    for robot in made.robots:
        robot.estop()


def wait_status(client, predicate, timeout=45.0) -> dict:      # a run's ending can take ~20 s on a slow computer
    status = {}

    def check():
        nonlocal status
        client.post("/api/heartbeat")
        status = client.get("/api/status").json()
        return predicate(status)

    assert sim.wait_until(check, timeout), status
    return status
