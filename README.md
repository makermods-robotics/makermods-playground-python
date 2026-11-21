# makermods-playground

Run Dream Labs' Dream1 policies (Beta, Beta Flash) on MakerMods arms: one or two Metal arms or Maker arms, with a
top camera and a wrist camera per arm. Built on the public arm SDKs,
[`metal-arm`](https://github.com/makermods-robotics/metal-arm-sdk) and
[`maker-arm`](https://github.com/makermods-robotics/maker-arm-sdk), and on `dreamlabs-client` for the Dream API.

Three layers:

- **core**: a `Robot` (arms named `left` / `right`, cameras `top` / `left` / `right`), robot definition files, a
  fixed-rate servo loop, command limits, run recordings.
- **dream**: the Dream API client, end-effector chunks to joint commands through the SDK's trajectory IK, chunk
  scheduling and pacing, and `DreamRunner`.
- **ui**: a local web page (`playground ui`) to set up robots, watch them and run tasks.

## Install

Python 3.10 or newer.

```bash
pip install "makermods-playground @ git+https://github.com/makermods-robotics/makermods-playground-python"
```

This installs the arm SDK revisions pinned in `pyproject.toml`. The arm SDKs need libusb and, on Linux, their udev rule for gs_usb adapters: see their READMEs.

## First run

Set up the arms with their SDK first (`metal-arm rig init --detect`, `metal-arm identify --assign`), so its rig file
names the arms `left` / `right` and the cameras `top` / `left_wrist` / `right_wrist`. Then:

```bash
playground robot init lab --arm metal --from-rig      # ~/.config/makermods-playground/robots/lab.yaml
playground robot check                                # read-only: motors, faults, temperatures, cameras
playground robot status                               # joints, temperatures (torque unchanged)
playground cameras snapshot --out /tmp                # one JPEG per camera
playground robot hold --for 5 --yes                   # torque on 5 s, back to rest, torque off
playground dream check                                # API key and models (moves nothing)
playground dream run "pick up the black marker" --timeout 60
```

A robot file:

```yaml
arm: metal                                     # metal | maker
arms:
  left:  {backend: gs_usb, port: "gs_usb:<serial>"}   # the SDK rig's arm fields
  right: {backend: gs_usb, port: "gs_usb:<serial>"}   # optional: a one-arm robot has one side
cameras: {top: "usb:<path>", left: "usb:<path>", right: "usb:<path>"}
settings: {servo_hz: 40}                       # optional
```

Settings are checked when the file loads (a value that is not a finite number in range is refused). `ik_solver: jax`
uses the SDK's JAX trajectory IK (its `jax` extra), compiled for Dream1's chunk length before the arms are enabled;
the default solves with numpy.

`--arms left` and `--cameras top left` on `robot init` make reduced robots. With one robot defined, `--robot` can be
left out.

## Commands

Every command takes `--json` and `--robot NAME`.

| Command | What it does |
|---|---|
| `robot init NAME --arm metal\|maker [--from-rig [RIG]]` | Write a robot file (or `--port left=gs_usb:<serial> --camera top=usb:<path>`). |
| `robot list`, `robot show`, `robot remove NAME` | Robot files. |
| `robot check` | Read-only preflight: motors answer, no faults, temperatures, cameras deliver frames. |
| `robot status` | Joints, gripper opening, temperatures, faults. Torque unchanged. |
| `robot hold [--for S] [--then rest\|stay\|release]` | Torque on, holding the pose, until the time, Ctrl-C, SIGTERM or a fault (exit status 1). `stay` is refused before anything is energized when a stiff joint has a motor watchdog. |
| `robot rest` | Each arm in turn to the profile's rest pose, verified there, then torque off. The first arm not verified keeps holding, and so do the ones not yet moved. |
| `robot release` | Torque off now, confirmed motor by motor; works with a motor fault or a motor that does not answer, and names every motor not confirmed (exit status 1). |
| `cameras snapshot [--out DIR]`, `cameras stream [--for S]` | One JPEG per camera; frame rates. |
| `dream check` | API key present, models listed. The only command that calls the Dream API without moving. |
| `dream run "TASK" [--timeout S] [--model beta\|flash] [--then rest\|stay] [--no-record]` | Run a task. |
| `dream replay RECORDING [--dry]` | A recorded run's chunks through decode, IK and scheduling, without the API. `--dry` prints how close the result is to what the run commanded; without it the robot moves. |
| `ui [--host H] [--port P]` | The web page, below (needs the `ui` extra). |

Commands that move ask first, naming how they end; `--yes` answers for scripts. SIGINT and SIGTERM end the motion,
never the return to rest or the release that follows.

## Python

```python
from makermods_playground import Robot

with Robot.load() as robot:            # connects; leaving returns every arm to rest, then torque off
    robot.enable()
    obs = robot.observe()              # obs.q[side]: 7 joints (rad), obs.gripper[side]: opening 0..1,
                                       # obs.tool[side]: 4x4 tool pose, obs.frames[name]: (image, age s)
    q = obs.q["left"].copy()
    q[5] += 0.1
    robot.command({"left": q})
```

```python
from makermods_playground import Robot
from makermods_playground.dream.runner import DreamRunner

robot = Robot.load()
robot.connect()
try:
    result = DreamRunner(robot, "pick up the black marker", model="beta").run(timeout_s=60)
    print(result.stop_reason, result.chunks, result.recording)
finally:
    robot.disconnect()
```

`DreamRunner.state()` is a live snapshot (phase, chunks, pacing, joints, events); `run(stop_event=...)` ends a run
from another thread.

## Safety

- **The arms have no brakes.** They drop when torque goes off away from rest. Keep a hardware emergency stop on
  motor power and the workspace clear.
- **Every run ends the same way:** the arms freeze where they are, then go back to the pose they were enabled at,
  then torque off. If that return cannot be verified, the arm keeps holding and the command exits with status 1;
  support it, then `playground robot release --yes`. Where a stiff joint would drop on its motor watchdog, the
  command keeps holding the arm until you type RELEASE. "Torque off" is said only when every motor confirms it.
- **A fault holds the arm.** The SDK holds the pose on stale feedback, a motor fault or overtemperature (the gripper
  lets go); the run ends. Joint feedback older than 0.2 s, a camera without a fresh frame (judged on the monotonic
  clock), a joint lagging its command by more than 12°, a policy that has not answered within its retry window or
  the time limit end the run too. A reply that lands after its plan has passed is dropped and asked again.
- **The first chunk ramps in** at 0.5 rad/s per joint; later chunks continue the motion the arms are on.
- The tool points come from the Dream IK spec; check them on your arm.

## Web page

```bash
pip install "makermods-playground[ui] @ git+https://github.com/makermods-robotics/makermods-playground-python"
playground ui                       # http://127.0.0.1:8080/ ; --port, --robot NAME, --host
```

It listens on this computer only. To serve it on the network, name each host the page is reached at and give a
token of at least 16 characters, which the page asks for (anyone with it can drive the arms):

```bash
PLAYGROUND_TOKEN=... playground ui --host 0.0.0.0 --allow-host lab.local --allow-host 192.168.1.20
```

- **Run**: pick a robot (top right), type a task, choose Dream1 Beta or Beta Flash and a time limit, Run. Run
  connects the robot (torque off) and starts the run; the task bar shows the phase, time left and chunks; Stop (or
  Esc) ends it and the arms return to rest. After a run the robot stays connected, torque off: Disconnect frees it.
- **Watch**: the cameras live (the camera picture from each device while disconnected), joints, gripper opening,
  temperatures, faults and the run's events under "Arms and events".
- **Set up**: New robot (name, Metal or Maker) opens its setup: pick the left and right arm's CAN adapter from the
  adapters plugged in; Wiggle opens and closes that arm's gripper a few degrees, three times, at low gain (only the
  gripper motor is energized, and it is always released), so you can see which arm an adapter drives. Assign the
  cameras in the camera panel (camera button): drag a live preview onto Top, Left or Right. Check is the read-only
  `robot check`. Delete removes the robot file.
- **Recordings**: the recordings button lists the runs (task, model, how it ended) with links to their files and
  videos, and turns recording on or off for the selected robot (`settings.record`).

Safety: one run at a time. A run stops when Stop is pressed, when no open page has sent its heartbeat for 2 s (the
tab closed, the computer slept), at its time limit, on a fault, and when the server gets SIGINT or SIGTERM; it then
returns the arms to rest before the server exits. The heartbeat is any open page's: closing the tab that started a
run while another page of this server stays open keeps the run going (Stop it from that page). A Wiggle stops the
same way. An arm whose torque off is not confirmed (its return to rest not verified, or found with torque on when
connecting) keeps holding and the page says so: Release turns its torque off after you confirm (the arm drops if it
is not supported), and the arm stays listed until every motor confirms it. Runs, Disconnect and robot changes are
refused while an arm is still holding or a run is under way. Run and Release name the robot the page shows: if
another page changed the selection, they are refused.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/session` | The page token (none when served on the network: the operator's); requests other than GET carry it in `x-playground-token`. |
| GET | `/api/robots` | Robot definitions (`arm`, arm adapters, cameras, `record`), the selected one and the selection `revision`. |
| PUT | `/api/robots/{name}` | Create or change a definition: `{arm, arms: {left, right}, cameras: {top, left, right}, record}`; fields left out stay. |
| DELETE | `/api/robots/{name}` | Remove a definition. |
| POST | `/api/robots/{name}/select` | Select a robot (disconnects the previous one). |
| POST | `/api/check` | Read-only check of the selected robot (not connected). |
| GET | `/api/status` | Connection, run state (`DreamRunner.state()`), last result, arms holding, torque per arm (`on`, `off`, `unknown`), `Robot.status()`, events, `revision`. |
| POST | `/api/connect`, `/api/disconnect` | Connect (torque off) or disconnect the selected robot. |
| POST | `/api/run` | Start a run: `{task, model: beta\|flash, timeout_s, robot, revision}`. |
| POST | `/api/stop` | End the run (the arms return to rest) or a Wiggle. |
| POST | `/api/heartbeat` | The page is still open (sent every 0.5 s during a run). |
| POST | `/api/release` | `{confirm: true, robot, revision}`: torque off now on the selected robot, confirmed motor by motor. |
| GET | `/api/cameras/{name}/stream` | MJPEG of a connected robot's camera. |
| GET | `/api/devices?arm=metal\|maker` | gs_usb adapters and cameras plugged in (nothing is opened). |
| GET | `/api/devices/cameras/{id}/stream` | MJPEG preview of one camera, opened while watched. |
| POST | `/api/identify` | `{arm, port}`: wiggle the gripper of the arm on that adapter; `released` says whether its torque off was confirmed. |
| GET | `/api/recordings` | Run recordings of the selected robot, newest first. |
| GET | `/api/recordings/{run}/{file}` | One file of a recording. |

## Recordings

Each run writes `recordings/<time>_<robot>/`: `top.mp4`, `left.mp4`, `right.mp4`, `motion.csv` (every servo tick,
commanded and measured joints, rad), `chunks.csv`, `timing.csv`, `raw_chunks.npz` (each reply's end-effector head),
`ik_trace.json` (what each chunk was solved from) and `run.json` (`error` is the run's, `recording_error` the
recording's own). `--no-record` turns it off. `dream replay` reads recordings without executing anything stored in
them (no pickled data).

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ../metal-arm-sdk -e ../maker-arm-sdk -e '.[ui,dev]'
.venv/bin/python -m pytest -q
```

The tests drive simulated motors (the SDKs' `testing` module), fake cameras and a fake policy, the web page's API
through FastAPI's test client; they never open a device or call the Dream API. `pyproject.toml` points uv at sibling
SDK checkouts (`[tool.uv.sources]`) instead of the pinned revisions.
