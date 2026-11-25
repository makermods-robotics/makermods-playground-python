"""What is plugged in (CAN adapters, cameras), camera previews, and the gripper wiggle that shows which arm an
adapter drives. Listing never opens a device."""

from __future__ import annotations

import importlib
import threading
from typing import Optional

from ..core import config

WIGGLE_DEGREES = 15.0
PREVIEW_IDLE_S = 5.0    # a preview camera closes this long after its last viewer left
PREVIEW_OPEN_WAIT_S = 15.0   # the longest close() waits for a preview camera that is still being opened


def _sdk_module(arm_type: str, name: str):
    config.sdk(arm_type)    # validates the arm type
    return importlib.import_module(f"{arm_type}_arm.{name}")


def discover(arm_type: str = "metal") -> dict:
    """gs_usb adapters and cameras present now, from USB descriptors and sysfs."""
    ports = _sdk_module(arm_type, "cli.ports")
    adapters, adapters_error = ports.gs_usb_adapters()
    cameras, cameras_error = ports.camera_list()
    return {
        "adapters": [{"port": a["port"], "product": a.get("product") or "", "usb_path": a.get("usb_path") or ""}
                     for a in adapters if a.get("port")],
        "adapters_error": adapters_error,
        "cameras": [{"id": c["id"], "name": c.get("name") or c["id"], "path": c.get("path")} for c in cameras],
        "cameras_error": cameras_error,
    }


def build_arm(arm_type: str, port: str):
    """The arm on a gs_usb adapter with its shipped profile (not connected)."""
    return config.sdk(arm_type).rig.ArmSpec(backend="gs_usb", port=port).build()


def wiggle(arm_type: str, port: str, stop: threading.Event) -> dict:
    """Open and close the gripper of the arm on ``port`` a few degrees, three times, at low gain (the SDK's identify
    wiggle). Only the gripper is energized and it is always released; ``stop`` cuts it short (``stopped`` in the
    result). ``released``: the gripper's torque off was confirmed."""
    sdk_wiggle = _sdk_module(arm_type, "cli.identify").wiggle
    arm = build_arm(arm_type, port)
    arm.connect()
    try:
        index = arm.joint_names.index("gripper") if "gripper" in arm.joint_names else len(arm.joint_names) - 1
        try:
            return sdk_wiggle(arm, index, WIGGLE_DEGREES, lambda pose: None, stop)
        except Exception:
            if not stop.is_set():
                raise
            # cut short: the wiggle released the gripper; its last reply says whether that was confirmed
            return {"joint": arm.joint_names[index], "moved": 0.0, "stopped": True,
                    "released": arm.motors[index].confirms("disabled")}
    finally:
        arm.disconnect()


def open_camera(camera_id: str):
    return config.sdk("metal").cameras.Camera(camera_id)


class Previews:
    """Cameras opened only while a page shows them, in the background; each closes PREVIEW_IDLE_S after its last
    viewer leaves. Ids in ``blocked`` (the connected robot's) are never opened here."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cameras: dict[str, dict] = {}
        self.blocked: set[str] = set()

    def watch(self, camera_id: str) -> None:
        with self.lock:
            entry = self.cameras.get(camera_id)
            if entry is None:
                entry = self.cameras[camera_id] = {"camera": None, "viewers": 0, "error": None, "opening": False,
                                                   "opened": threading.Event()}
            entry["viewers"] += 1
            if entry["camera"] is None and not entry["opening"] and camera_id not in self.blocked:
                entry["opening"] = True
                entry["opened"].clear()
                threading.Thread(target=self._open, args=(camera_id, entry), daemon=True,
                                 name="camera-preview").start()

    def unwatch(self, camera_id: str) -> None:
        with self.lock:
            entry = self.cameras.get(camera_id)
            if entry is None:
                return
            entry["viewers"] -= 1
            if entry["viewers"] <= 0:
                timer = threading.Timer(PREVIEW_IDLE_S, self._close_idle, args=(camera_id, entry))
                timer.daemon = True
                timer.start()

    def frame(self, camera_id: str):
        entry = self.cameras.get(camera_id)
        camera = entry and entry["camera"]
        if camera is None:
            return None
        newest = camera.newest()
        return None if newest is None else newest[0]

    def error(self, camera_id: str) -> Optional[str]:
        entry = self.cameras.get(camera_id)
        return entry and entry["error"]

    def close(self, camera_ids) -> None:
        """Close these cameras now (the robot is about to open them). A camera still being opened holds its device
        until the open ends, so this waits for that (at most PREVIEW_OPEN_WAIT_S); the opener then closes it."""
        for camera_id in camera_ids:
            with self.lock:
                entry = self.cameras.pop(camera_id, None)
                opening = bool(entry and entry["opening"])
            if opening:
                entry["opened"].wait(PREVIEW_OPEN_WAIT_S)
            elif entry and entry["camera"] is not None:
                entry["camera"].close()

    def close_all(self) -> None:
        self.close(list(self.cameras))

    def _open(self, camera_id: str, entry: dict) -> None:
        try:
            self._open_camera(camera_id, entry)
        finally:
            entry["opened"].set()     # the device is free again, or this preview holds it

    def _open_camera(self, camera_id: str, entry: dict) -> None:
        try:
            camera = open_camera(camera_id)
            camera.open()
        except Exception as e:
            with self.lock:
                entry.update(opening=False, error=str(e))
            return
        with self.lock:
            entry["opening"] = False
            if self.cameras.get(camera_id) is entry and camera_id not in self.blocked:
                entry.update(camera=camera, error=None)
                return
        camera.close()

    def _close_idle(self, camera_id: str, entry: dict) -> None:
        with self.lock:
            if entry["viewers"] > 0 or self.cameras.get(camera_id) is not entry:
                return
            del self.cameras[camera_id]
        if entry["camera"] is not None:
            entry["camera"].close()
