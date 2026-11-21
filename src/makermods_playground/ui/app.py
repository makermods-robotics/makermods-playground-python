"""The web page: pick a robot, watch its cameras and joints, run a task, set up robots. ``playground ui``.

Requests that change anything carry a token in ``x-playground-token``, so another site open in the same browser
cannot drive the arms, and the Host header must be one this server answers to. Served on this computer only, the
token is the page token from ``GET /api/session``; served on a network address, it is one the operator supplies
(``--token``) and types into the page, and the Host names are an explicit list (``--allow-host``).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..core import config
from ..dream.client import MODELS
from . import devices
from .session import HEARTBEAT_S, Busy, Session

STATIC = Path(__file__).parent / "static"
STREAM_PERIOD_S = 0.1
JPEG_QUALITY = 80
TOKEN_HEADER = "x-playground-token"
TOKEN_ENV = "PLAYGROUND_TOKEN"
MIN_TOKEN_LENGTH = 16
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")
log = logging.getLogger(__name__)


class Selection(BaseModel):
    """The robot the page shows and the selection revision it saw (``GET /api/robots`` or a select)."""
    robot: str = Field(min_length=1, max_length=200)
    revision: int


class RunRequest(Selection):
    task: str = Field(min_length=1, max_length=1000)
    model: Literal[tuple(MODELS)] = "beta"
    timeout_s: float = Field(default=60.0, ge=1, le=3600)


class RobotRequest(BaseModel):
    arm: Optional[Literal[config.ARM_TYPES]] = None
    arms: Optional[dict[Literal[config.SIDES], Optional[str]]] = None
    cameras: Optional[dict[Literal[config.CAMERAS], Optional[str]]] = None
    record: Optional[bool] = None


class ConfirmRequest(Selection):
    confirm: bool


class IdentifyRequest(BaseModel):
    arm: Literal[config.ARM_TYPES]
    port: str = Field(min_length=1, max_length=200)


def robot_view(definition: config.RobotDef) -> dict:
    return {"name": definition.name, "arm": definition.arm,
            "arms": {s: definition.arm_spec(s).address for s in definition.sides},
            "cameras": dict(definition.cameras), "record": definition.settings.record,
            "recordings_dir": definition.settings.recordings_dir}


def jpeg(image) -> Optional[bytes]:
    import cv2
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return data.tobytes() if ok else None


class JpegCache:
    """The JPEG of each stream's newest image, encoded once however many pages watch it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest: dict = {}   # key -> (image, JPEG)

    def get(self, key, image) -> Optional[bytes]:
        with self.lock:
            cached = self.latest.get(key)
            if cached is not None and cached[0] is image:
                return cached[1]
            data = jpeg(image)
            self.latest[key] = (image, data)
            return data


def mjpeg(request: Request, frame, closing: threading.Event, cache: JpegCache, key) -> StreamingResponse:
    """A multipart JPEG stream for an <img>: each new frame ``frame()`` gives, until the page or server goes."""
    async def parts():
        last = None
        yield b"--frame\r\n"
        while not closing.is_set() and not await request.is_disconnected():
            image = frame()
            if image is not None and image is not last:
                last = image
                data = await asyncio.to_thread(cache.get, key, image)
                if data:
                    yield (b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                           + data + b"\r\n--frame\r\n")
            await asyncio.sleep(STREAM_PERIOD_S)

    return StreamingResponse(parts(), media_type="multipart/x-mixed-replace; boundary=frame")


def _plain(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(404, "not found")
    return name


def create_app(robot: Optional[str] = None, hosts=(), heartbeat_s: float = HEARTBEAT_S,
               token: Optional[str] = None) -> FastAPI:
    """``hosts``: Host names answered besides this computer's own; ``token``: one the operator supplies (never
    handed out by ``/api/session``), else a page token is made and handed to any page this server answers."""
    if "*" in hosts:
        raise ValueError("name each host the page is reached at; '*' is not accepted")
    session = Session(robot, heartbeat_s)
    page_token = token is None
    token = token or secrets.token_urlsafe(32)
    jpegs = JpegCache()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(session.close)

    app = FastAPI(title="MakerMods playground", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.session = session
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted({*LOOPBACK_HOSTS, "testserver", *hosts}))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method not in ("GET", "HEAD") and not secrets.compare_digest(
                request.headers.get(TOKEN_HEADER, ""), token):
            return JSONResponse({"detail": "Reload the page to reconnect."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; script-src 'self'; "
            "frame-ancestors 'none'")
        return response

    @app.exception_handler(Busy)
    async def busy(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(config.ConfigError)
    async def config_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def failure(request, exc):
        log.exception("%s %s failed", request.method, request.url.path)
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/session")
    def session_token():
        return {"token": token if page_token else None}

    # ── robots ──
    @app.get("/api/robots")
    def robots():
        rows = []
        for name in config.robot_names():
            try:
                rows.append(robot_view(config.load(name)))
            except config.ConfigError as e:
                rows.append({"name": name, "error": str(e)})
        return {"robots": rows, "selected": session.selected, "revision": session.revision,
                "dir": str(config.robots_dir())}

    @app.put("/api/robots/{name}")
    def save_robot(name: str, body: RobotRequest):
        definition = session.save_robot(name, body.arm, body.arms, body.cameras, body.record)
        return robot_view(definition)

    @app.delete("/api/robots/{name}")
    def remove_robot(name: str):
        session.remove_robot(name)
        return {"removed": name, "selected": session.selected}

    @app.post("/api/robots/{name}/select")
    def select(name: str):
        session.select(name)
        return session.status()

    @app.post("/api/check")
    def check():
        return session.check()

    # ── connection and runs ──
    @app.get("/api/status")
    def status():
        return session.status()

    @app.post("/api/connect")
    def connect():
        session.connect()
        return session.status()

    @app.post("/api/disconnect")
    def disconnect():
        session.disconnect()
        return session.status()

    @app.post("/api/run")
    def run(body: RunRequest):
        task = body.task.strip()
        if not task:
            raise HTTPException(422, "Enter a task")
        session.run(task, body.model, body.timeout_s, body.robot, body.revision)
        return session.status()

    @app.post("/api/stop")
    def stop():
        session.stop()
        return session.status()

    @app.post("/api/heartbeat")
    def heartbeat():
        session.heartbeat()
        return {"ok": True}

    @app.post("/api/release")
    def release(body: ConfirmRequest):
        if not body.confirm:
            raise HTTPException(400, "Releasing drops an arm that is not at rest: confirm it")
        session.release(body.robot, body.revision)
        return session.status()

    @app.get("/api/cameras/{name}/stream")
    def camera_stream(name: str, request: Request):
        robot = session.robot
        if robot is None or name not in robot.cameras:
            raise HTTPException(404, f"no live camera {name}")

        def frame():
            got = robot.frame(name) if session.robot is robot else None
            return None if got is None else got[0]

        return mjpeg(request, frame, session.closing, jpegs, ("robot", name))

    # ── recordings ──
    @app.get("/api/recordings")
    def recordings():
        return {"dir": str(session.recordings_dir()), "recordings": session.recordings()}

    @app.get("/api/recordings/{run}/{file}")
    def recording_file(run: str, file: str):
        root = session.recordings_dir().resolve()
        path = session.recordings_dir() / _plain(run) / _plain(file)
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file() \
                or not path.resolve().is_relative_to(root):
            raise HTTPException(404, "not found")
        return FileResponse(path)

    # ── setup ──
    @app.get("/api/devices")
    def list_devices(arm: Literal[config.ARM_TYPES] = "metal"):
        return devices.discover(arm)

    @app.get("/api/devices/cameras/{camera_id:path}/stream")
    def device_stream(camera_id: str, request: Request):
        robot = session.robot
        role = next((r for r, cid in robot.definition.cameras.items() if cid == camera_id), None) if robot else None
        if role is not None:
            def frame():
                got = robot.frame(role) if session.robot is robot else None
                return None if got is None else got[0]
            return mjpeg(request, frame, session.closing, jpegs, ("robot", role))

        previews = session.previews

        async def watched():
            previews.watch(camera_id)
            try:
                async for part in mjpeg(request, lambda: previews.frame(camera_id), session.closing, jpegs,
                                        ("device", camera_id)).body_iterator:
                    yield part
            finally:
                previews.unwatch(camera_id)

        return StreamingResponse(watched(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/identify")
    def identify(body: IdentifyRequest):
        return session.identify(body.arm, body.port)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def serve(host: str = "127.0.0.1", port: int = 8080, robot: Optional[str] = None, allow_hosts=(),
          token: Optional[str] = None) -> None:
    """Serve until SIGINT/SIGTERM. The signal stops a run at once; the arms return to rest before the process ends.
    On a network address the page is reached only by the ``allow_hosts`` names and driven only with ``token`` (or
    $PLAYGROUND_TOKEN), which the operator types into the page."""
    import uvicorn

    token = token or os.environ.get(TOKEN_ENV) or None
    if _loopback(host):
        app = create_app(robot, hosts=allow_hosts, token=token)
    else:
        if not allow_hosts:
            raise ValueError(f"serving on {host} needs --allow-host NAME for each name the page is reached at")
        if not token or len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"serving on {host} needs --token (or ${TOKEN_ENV}) of at least {MIN_TOKEN_LENGTH} "
                             "characters: whoever has it can drive the arms")
        app = create_app(robot, hosts=allow_hosts, token=token)
        print(f"warning: serving on {host}: anyone who reaches {', '.join(allow_hosts)} with the token can drive the "
              "arms", flush=True)
    session: Session = app.state.session

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            session.closing.set()
            session.stop("the server is shutting down")
            super().handle_exit(sig, frame)

    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"playground on http://{shown}:{port}/", flush=True)
    Server(uvicorn.Config(app, host=host, port=port, access_log=False, timeout_graceful_shutdown=2)).run()
