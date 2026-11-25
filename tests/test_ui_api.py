import json
import threading

import pytest
import sim
import ui_harness
from fastapi.testclient import TestClient
from ui_harness import save_robot, wait_status

from makermods_playground.core import config
from makermods_playground.ui import app as app_module
from makermods_playground.ui import devices

harness = pytest.fixture(ui_harness.harness)


def done(status) -> bool:
    return not status["running"] and status["result"] is not None


def shown(client) -> dict:
    """The robot and selection revision a page shows."""
    listed = client.get("/api/robots").json()
    return {"robot": listed["selected"], "revision": listed["revision"]}


def run(client, **body):
    return client.post("/api/run", json={**shown(client), **body})


def release(client, **body):
    return client.post("/api/release", json={**shown(client), **body})


def test_token_and_host(harness):
    save_robot()
    client = harness.client()
    assert client.post("/api/connect", headers={"x-playground-token": "wrong"}).status_code == 403
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 400
    page = client.get("/")
    assert page.status_code == 200 and "script-src 'self'" in page.headers["content-security-policy"]


def test_robot_definitions(harness):
    client = harness.client()
    assert client.get("/api/robots").json()["robots"] == []
    assert client.put("/api/robots/lab", json={"arm": "metal"}).status_code == 400
    r = client.put("/api/robots/lab", json={"arm": "metal", "arms": {"left": "gs_usb:A", "right": "gs_usb:B"},
                                            "cameras": {"top": "usb:t", "left": None}})
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "lab", "arm": "metal", "arms": {"left": "gs_usb:A", "right": "gs_usb:B"},
                        "cameras": {"top": "usb:t"}, "record": True, "recordings_dir": "recordings"}
    listed = client.get("/api/robots").json()
    assert listed["selected"] == "lab" and [x["name"] for x in listed["robots"]] == ["lab"]

    # an unchanged arm keeps its entry; a new address replaces one; omitted fields are kept
    definition = config.load("lab")
    definition.arms["left"]["profile"] = "custom.yaml"
    config.save(definition)
    r = client.put("/api/robots/lab", json={"arms": {"left": "gs_usb:A", "right": "gs_usb:C"}})
    assert r.json()["arms"] == {"left": "gs_usb:A", "right": "gs_usb:C"}
    saved = config.load("lab")
    assert saved.arms["left"]["profile"] == "custom.yaml" and saved.arms["right"] == {"backend": "gs_usb",
                                                                                      "port": "gs_usb:C"}
    assert saved.cameras == {"top": "usb:t"}
    r = client.put("/api/robots/lab", json={"cameras": {"top": "usb:x", "left": "usb:y"}, "record": False})
    assert r.json()["cameras"] == {"top": "usb:x", "left": "usb:y"} and r.json()["record"] is False
    assert client.put("/api/robots/lab", json={"arms": {"left": None}}).status_code == 400   # no arm left

    r = client.put("/api/robots/solo", json={"arm": "maker", "arms": {"right": "gs_usb:D"}})
    assert r.status_code == 200 and r.json()["arms"] == {"right": "gs_usb:D"}
    assert client.post("/api/robots/nobody/select").status_code == 400
    assert client.post("/api/robots/solo/select").json()["selected"] == "solo"
    assert client.delete("/api/robots/solo").json() == {"removed": "solo", "selected": "lab"}
    assert client.delete("/api/robots/solo").status_code == 400
    assert config.robot_names() == ["lab"]


def test_connect_status_and_disconnect(harness):
    save_robot()
    client = harness.client()
    status = client.get("/api/status").json()
    assert status["selected"] == "lab" and not status["connected"] and status["robot"] is None
    status = client.post("/api/connect").json()
    assert status["connected"] and harness.robot.states() == {"left": "connected", "right": "connected"}
    arm = status["robot"]["arms"]["left"]
    assert arm["state"] == "connected" and "elbow_flex" in arm["joints"] and arm["gripper_opening"] is not None
    assert status["robot"]["cameras"]["top"]["open"]
    assert client.put("/api/robots/lab", json={"cameras": {"top": "usb:z"}}).status_code == 409
    assert client.put("/api/robots/lab", json={"record": True}).status_code == 200
    assert harness.robot.settings.record
    assert client.delete("/api/robots/lab").status_code == 409
    assert client.post("/api/check").status_code == 409
    status = client.post("/api/disconnect").json()
    assert not status["connected"] and harness.robot.states() == {"left": "disconnected", "right": "disconnected"}


def test_run_lifecycle(harness):
    save_robot()
    client = harness.client()
    assert run(client, task="  ").status_code == 422
    assert run(client, task="x", model="gamma").status_code == 422
    status = run(client, task="pick up the marker", model="flash", timeout_s=2).json()
    assert status["running"] and status["run"]["task"] == "pick up the marker"
    running = wait_status(client, lambda s: (s["run"] or {}).get("phase") == "running")
    assert running["connected"] and running["run"]["model"] == "flash"
    assert run(client, task="again").status_code == 409
    assert client.post("/api/disconnect").status_code == 409
    assert client.post("/api/robots/lab/select").status_code == 409
    status = wait_status(client, done)
    result = status["result"]
    assert result["stop_reason"] == "timeout" and not result["error"] and result["chunks"] >= 1
    assert result["rest"] == {"left": True, "right": True} and status["holding"] == []
    assert status["run"]["phase"] == "done" and any(e["text"] == "pick up the marker" for e in status["run"]["events"])
    assert harness.policies[-1].model == "flash"
    assert status["connected"] and harness.robot.states() == {"left": "connected", "right": "connected"}
    json.dumps(status)


def test_stop(harness):
    save_robot()
    client = harness.client()
    run(client, task="wave", timeout_s=60)
    wait_status(client, lambda s: (s["run"] or {}).get("phase") == "running")
    assert client.post("/api/stop").json()["stopping"]
    status = wait_status(client, done)
    assert status["result"]["stop_reason"] == "stopped" and status["result"]["rest"] == {"left": True, "right": True}
    assert any("Stop pressed" in e["text"] for e in status["events"])


def test_heartbeat_lost_stops_the_run(harness):
    save_robot()
    client = harness.client(heartbeat_s=0.5)
    run(client, task="wave", timeout_s=60)
    wait_status(client, lambda s: (s["run"] or {}).get("phase") == "running")    # heartbeats until then
    assert sim.wait_until(lambda: not client.get("/api/status").json()["running"], 15)
    status = client.get("/api/status").json()
    assert status["result"]["stop_reason"] == "stopped" and status["result"]["rest"] == {"left": True, "right": True}
    assert any("no page has answered" in e["text"] for e in status["events"])


def test_fault_leaves_the_arm_holding_until_a_release_is_confirmed(harness):
    save_robot()
    client = harness.client()
    run(client, task="wave", timeout_s=60)
    wait_status(client, lambda s: (s["run"] or {}).get("phase") == "running")
    threading.Timer(0.5, lambda: harness.robot.motors["left"].offline("shoulder_lift")).start()
    status = wait_status(client, done)
    assert status["result"]["error"] and status["result"]["stop_reason"].startswith("left arm")
    assert status["holding"] == ["left"] and status["torque"]["left"] == "on"
    assert run(client, task="again").status_code == 409
    assert "still holding" in client.post("/api/disconnect").json()["detail"]
    assert release(client).status_code == 422
    assert release(client, confirm=False).status_code == 400
    status = release(client, confirm=True).json()      # the shoulder still does not answer: not confirmed
    assert status["holding"] == ["left"] and "NOT confirmed" in status["events"][-1]["text"]
    harness.robot.motors["left"].offline("shoulder_lift", False)
    status = release(client, confirm=True).json()
    assert status["holding"] == [] and harness.robot.states() == {"left": "connected", "right": "connected"}
    assert status["torque"] == {"left": "off", "right": "off"}
    assert not client.post("/api/disconnect").json()["connected"]


def test_release_when_disconnected_connects_for_it(harness):
    save_robot()
    client = harness.client()
    status = release(client, confirm=True).json()
    assert not status["connected"] and harness.robot.states() == {"left": "disconnected", "right": "disconnected"}


def test_start_failure_is_reported(harness):
    save_robot(cameras={"top": "usb:t"})
    client = harness.client()
    run(client, task="wave", timeout_s=5)
    status = wait_status(client, done)
    assert status["result"]["error"] and "camera" in status["result"]["stop_reason"]
    assert harness.robot.states() == {"left": "connected", "right": "connected"}


def test_shutdown_stops_the_run_and_rests(harness):
    save_robot()
    client = harness.client()
    run(client, task="wave", timeout_s=60)
    wait_status(client, lambda s: (s["run"] or {}).get("phase") == "running")
    session = client.app.state.session
    client.__exit__(None, None, None)
    assert not session.running and session.result["stop_reason"] == "stopped"
    assert session.result["rest"] == {"left": True, "right": True}
    assert harness.robot.states() == {"left": "disconnected", "right": "disconnected"}


def test_check(harness):
    save_robot()
    client = harness.client()
    report = client.post("/api/check").json()
    assert report["ok"] and set(report["cameras"]) == {"top", "left", "right"} and report["arms"]["left"]["ok"]
    assert harness.robot.states() == {"left": "disconnected", "right": "disconnected"}


def test_identify_wiggles_only_the_gripper(harness, monkeypatch):
    save_robot()
    arms = []

    def build(arm_type, port):
        arm, motors = sim.sim_arm(arm_type)
        arms.append((arm_type, port, arm, motors))
        return arm

    monkeypatch.setattr(devices, "build_arm", build)
    monkeypatch.setattr("metal_arm.cli.identify.STEP_S", 0.001)
    client = harness.client(heartbeat_s=30)
    r = client.post("/api/identify", json={"arm": "metal", "port": "gs_usb:A"})
    assert r.status_code == 200, r.text
    assert r.json()["joint"] == "gripper" and r.json()["released"] and r.json()["moved"] > 0.1
    arm_type, port, arm, motors = arms[-1]
    assert (arm_type, port) == ("metal", "gs_usb:A") and arm.state.value == "disconnected"
    assert motors.positions[:6] == pytest.approx(sim.START["metal"][:6], abs=1e-6)
    assert not any(m.enabled for m in motors.motors.values())
    client.post("/api/connect")
    assert client.post("/api/identify", json={"arm": "metal", "port": "gs_usb:A"}).status_code == 409
    assert client.post("/api/identify", json={"arm": "so101", "port": "gs_usb:A"}).status_code == 422


def test_devices(harness, monkeypatch):
    monkeypatch.setattr("metal_arm.cli.ports.gs_usb_adapters", lambda: (
        [{"serial": "A", "port": "gs_usb:A", "usb_path": "1-2", "product": "candleLight"}], None))
    monkeypatch.setattr("metal_arm.cli.ports.camera_list", lambda: (
        [{"id": "usb:t", "index": 0, "name": "Cam", "path": "/dev/video0"}], None))
    client = harness.client()
    assert client.get("/api/devices").json() == {
        "adapters": [{"port": "gs_usb:A", "product": "candleLight", "usb_path": "1-2"}], "adapters_error": None,
        "cameras": [{"id": "usb:t", "name": "Cam", "path": "/dev/video0"}], "cameras_error": None}
    assert client.get("/api/devices", params={"arm": "so101"}).status_code == 422


def _first_frame(client, url):
    session = client.app.state.session
    timer = threading.Timer(0.6, session.closing.set)
    timer.start()
    r = client.get(url)
    timer.join()
    session.closing.clear()
    return r


def test_camera_streams(harness, monkeypatch):
    save_robot()
    opened = []

    def camera(camera_id):
        opened.append(camera_id)
        return sim.FakeCamera(camera_id)

    monkeypatch.setattr(devices, "open_camera", camera)
    client = harness.client()
    assert client.get("/api/cameras/top/stream").status_code == 404
    r = _first_frame(client, "/api/devices/cameras/usb:spare/stream")
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace") and b"image/jpeg" in r.content
    assert opened == ["usb:spare"]
    client.post("/api/connect")
    assert b"image/jpeg" in _first_frame(client, "/api/cameras/top/stream").content
    assert b"image/jpeg" in _first_frame(client, "/api/devices/cameras/usb:l/stream").content
    assert opened == ["usb:spare"]   # the robot's camera is served from the connection, not opened again
    assert client.get("/api/cameras/nope/stream").status_code == 404


def test_recordings(harness, tmp_path):
    save_robot(record=True, recordings_dir=str(tmp_path / "rec"))
    client = harness.client()
    assert client.get("/api/recordings").json()["recordings"] == []
    run(client, task="wave", timeout_s=1.5)
    status = wait_status(client, done)
    assert status["result"]["recording"]
    listed = client.get("/api/recordings").json()
    assert listed["dir"] == str(tmp_path / "rec")
    entry = listed["recordings"][0]
    assert entry["task"] == "wave" and entry["model"] == "beta" and entry["stopped_because"] == "timeout"
    assert {"run.json", "motion.csv", "top.mp4"} <= set(entry["files"])
    r = client.get(f"/api/recordings/{entry['name']}/run.json")
    assert r.status_code == 200 and r.json()["task"] == "wave"
    assert client.get(f"/api/recordings/{entry['name']}/nope.csv").status_code == 404
    assert client.get(f"/api/recordings/{entry['name']}/..%2Frun.json").status_code == 404
    assert client.get("/api/recordings/../x").status_code == 404


def test_an_arm_found_energized_on_connect_is_holding(harness):
    save_robot()
    harness.sim_options = {"enabled": True}
    client = harness.client()
    status = client.post("/api/connect").json()
    assert status["holding"] == ["left", "right"] and status["torque"] == {"left": "on", "right": "on"}
    assert "still holding" in status["events"][-1]["text"]
    assert run(client, task="wave").status_code == 409
    assert client.post("/api/disconnect").status_code == 409
    session = client.app.state.session
    session.robot.disconnect()                          # as if the connection had gone: the arms stay listed
    session.robot = None
    assert client.delete("/api/robots/lab").status_code == 409
    assert client.put("/api/robots/lab", json={"arms": {"left": "gs_usb:Z"}}).status_code == 409
    status = release(client, confirm=True).json()       # released on its own bus
    assert status["holding"] == [] and not status["connected"]


def test_a_network_address_needs_named_hosts_and_an_operator_token(harness, monkeypatch):
    monkeypatch.delenv(app_module.TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match="allow-host"):
        app_module.serve("0.0.0.0", robot=None, token="t" * 20)
    with pytest.raises(ValueError, match="token"):
        app_module.serve("0.0.0.0", allow_hosts=("lab.local",), token="short")
    with pytest.raises(ValueError, match="'\\*'"):
        app_module.create_app(hosts=("*",))
    save_robot()
    secret = "s" * 24
    client = TestClient(app_module.create_app(hosts=("lab.local",), token=secret))
    with client:
        assert client.get("/api/session").json() == {"token": None}
        assert client.post("/api/connect").status_code == 403
        assert client.post("/api/connect", headers={app_module.TOKEN_HEADER: secret}).status_code == 200
        assert client.get("/api/status", headers={"host": "lab.local:8080"}).status_code == 200
        assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 400
        client.post("/api/disconnect", headers={app_module.TOKEN_HEADER: secret})


def test_run_and_release_are_bound_to_the_robot_the_page_shows(harness):
    save_robot()
    save_robot("solo", arms={"right": {"backend": "gs_usb", "port": "gs_usb:C"}})
    client = harness.client(robot="lab")
    seen = shown(client)
    assert client.post("/api/robots/solo/select").status_code == 200    # another tab
    r = client.post("/api/run", json={**seen, "task": "wave"})
    assert r.status_code == 409 and "changed" in r.json()["detail"]
    r = client.post("/api/release", json={**seen, "confirm": True})
    assert r.status_code == 409 and not client.get("/api/status").json()["running"]
    assert client.post("/api/run", json={**shown(client), "task": "x", "robot": "lab"}).status_code == 409


def test_identify_is_cut_short_by_stop_and_blocks_conflicting_requests(harness, monkeypatch):
    save_robot()
    monkeypatch.setattr(devices, "build_arm", lambda arm_type, port: sim.sim_arm(arm_type)[0])
    monkeypatch.setattr("metal_arm.cli.identify.STEP_S", 0.05)     # a wiggle of several seconds
    client = harness.client()
    answer = {}
    thread = threading.Thread(target=lambda: answer.update(
        client.post("/api/identify", json={"arm": "metal", "port": "gs_usb:A"}).json()))
    thread.start()
    assert sim.wait_until(lambda: client.get("/api/status").json()["identifying"] == "gs_usb:A")
    assert release(client, confirm=True).status_code == 409
    assert client.post("/api/check").status_code == 409
    assert run(client, task="wave").status_code == 409
    client.post("/api/stop")
    thread.join(5)
    assert answer["stopped"] and answer["released"]


def test_a_lost_heartbeat_cuts_the_identify_wiggle_short(harness, monkeypatch):
    save_robot()
    arms = []
    monkeypatch.setattr(devices, "build_arm", lambda arm_type, port: arms.append(sim.sim_arm(arm_type)) or arms[-1][0])
    monkeypatch.setattr("metal_arm.cli.identify.STEP_S", 0.05)
    client = harness.client(heartbeat_s=0.3)
    answer = client.post("/api/identify", json={"arm": "metal", "port": "gs_usb:A"}).json()
    assert answer["stopped"] and answer["released"]
    assert not any(m.enabled for m in arms[-1][1].motors.values())


def test_a_pending_run_shows_the_task_masked(harness):
    save_robot()
    client = harness.client()
    status = run(client, task="use dl_live_abcdef123456 here", timeout_s=1).json()
    assert "abcdef123456" not in json.dumps(status)
    wait_status(client, done)


def test_recordings_do_not_follow_links_out_of_the_folder(harness, tmp_path):
    root, outside = tmp_path / "rec", tmp_path / "outside"
    save_robot(recordings_dir=str(root))
    (root / "run1").mkdir(parents=True)
    (root / "run1" / "run.json").write_text("{}")
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (root / "linked").symlink_to(outside)
    (root / "run1" / "secret.txt").symlink_to(outside / "secret.txt")
    client = harness.client()
    listed = client.get("/api/recordings").json()["recordings"]
    assert [r["name"] for r in listed] == ["run1"] and listed[0]["files"] == ["run.json"]
    assert client.get("/api/recordings/linked/secret.txt").status_code == 404
    assert client.get("/api/recordings/run1/secret.txt").status_code == 404
    assert client.get("/api/recordings/run1/run.json").status_code == 200


def test_a_camera_frame_is_encoded_once_for_every_viewer(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "jpeg", lambda image: calls.append(image) or b"jpg")
    cache, image = app_module.JpegCache(), object()
    assert cache.get("top", image) == cache.get("top", image) == b"jpg"
    assert len(calls) == 1
    cache.get("top", object())
    assert len(calls) == 2


def test_closing_a_preview_that_is_still_opening_waits_until_its_device_is_free(monkeypatch):
    import threading
    import time

    from makermods_playground.ui import devices

    events, release = [], threading.Event()

    class Slow:
        def open(self):
            events.append("opening")
            release.wait(10)
            events.append("opened")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(devices, "open_camera", lambda camera_id: Slow())
    previews = devices.Previews()
    previews.watch("usb:top")
    while "opening" not in events:
        time.sleep(0.005)
    threading.Timer(0.3, release.set).start()
    previews.blocked = {"usb:top"}
    previews.close(["usb:top"])                       # as connect() does before the robot opens the camera
    assert events == ["opening", "opened", "closed"]


def test_a_preview_camera_that_cannot_be_made_reports_its_error(monkeypatch):
    import time

    from makermods_playground.ui import devices

    def refuse(camera_id):
        raise ValueError(f"not a camera id: {camera_id}")

    monkeypatch.setattr(devices, "open_camera", refuse)
    previews = devices.Previews()
    previews.watch("bad")
    deadline = time.monotonic() + 5.0
    while previews.error("bad") is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert previews.error("bad") == "not a camera id: bad"
    started = time.monotonic()
    previews.close(["bad"])
    assert time.monotonic() - started < 1.0
