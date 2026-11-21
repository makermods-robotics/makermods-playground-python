// Arm types, camera roles and the provider: every call to the server (/api/*) goes through it.

export const ARMS = {
  metal: { id: "metal", label: "Metal Arm", blurb: "7-DOF Damiao arm on CAN", image: "/static/assets/arms/metal.jpg" },
  maker: { id: "maker", label: "Maker Arm", blurb: "7-DOF RobStride arm on CAN", image: "/static/assets/arms/maker.jpg" },
};
export const ARM_TYPES = ["metal", "maker"];
export const SIDES = ["left", "right"];
export const CAMERA_ROLES = ["top", "left", "right"];
export const MODELS = { beta: "Dream1 Beta", flash: "Dream1 Beta Flash" };

export const noCameras = () => ({ top: null, left: null, right: null });
export function cleanCameras(value) {
  const out = noCameras();
  for (const role of CAMERA_ROLES) if (typeof value?.[role] === "string" && value[role]) out[role] = value[role];
  return out;
}

const UNREACHABLE = "Can’t reach the playground server.";

const TOKEN_KEY = "playground-token";

// The token the operator gave a server on a network address (it hands out none): asked once per tab.
function operatorToken(retry) {
  let saved = null;
  try {
    saved = retry ? null : sessionStorage.getItem(TOKEN_KEY);
  } catch {
    /* storage unavailable */
  }
  const value = saved || window.prompt("This playground is served on the network. Enter its token to drive the arms:") || "";
  try {
    sessionStorage.setItem(TOKEN_KEY, value);
  } catch {
    /* storage unavailable */
  }
  return value;
}

export function createProvider() {
  let token = null; // requests that change anything carry it: the page token, or the operator's
  let rejected = false;

  const session = () =>
    (token ??= fetch("/api/session")
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error())))
      .then((body) => body.token ?? operatorToken(rejected))
      .catch(() => {
        token = null;
        throw new Error(UNREACHABLE);
      }));

  // Same-origin JSON. Errors carry the server's detail text, ready for a toast.
  async function request(path, { method = "GET", body } = {}, retried = false) {
    const init = { method };
    if (method !== "GET") {
      init.headers = { "content-type": "application/json", "x-playground-token": await session() };
      init.body = JSON.stringify(body ?? {});
    }
    let res;
    try {
      res = await fetch(path, init);
    } catch {
      throw new Error(UNREACHABLE);
    }
    if (res.ok) return res.json();
    // A restarted server issues a new token: fetch it and send once more.
    if (res.status === 403 && method !== "GET" && !retried) {
      token = null;
      rejected = true;
      return request(path, { method, body }, true);
    }
    let detail = null;
    try {
      detail = (await res.json()).detail;
    } catch {
      /* not JSON */
    }
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    const message = typeof detail === "string" && detail ? detail : `The server answered ${res.status}.`;
    throw Object.assign(new Error(message), { status: res.status });
  }

  const post = (path, body) => request(path, { method: "POST", body });

  // Run and Release name the robot this page shows and the selection revision it saw, so a change made in another
  // tab is refused rather than acted on.
  const shown = () => ({ robot: api.selected, revision: api.revision });

  const api = {
    selected: null, // the server's selected robot
    revision: null, // the server's selection revision when it was read
    async listRobots() {
      const { robots, selected, revision } = await request("/api/robots");
      api.selected = selected;
      api.revision = revision;
      return (robots || []).filter((r) => !r.error && ARMS[r.arm]).map((r) => ({ ...r, cameras: cleanCameras(r.cameras) }));
    },
    // Creates or changes a definition; fields left out stay as they are.
    saveRobot: (name, changes) => request(`/api/robots/${encodeURIComponent(name)}`, { method: "PUT", body: changes }),
    deleteRobot: (name) => request(`/api/robots/${encodeURIComponent(name)}`, { method: "DELETE" }),
    async selectRobot(name) {
      const status = await post(`/api/robots/${encodeURIComponent(name)}/select`);
      api.selected = name;
      api.revision = status.revision;
      return status;
    },
    async getCameras(name) {
      const robot = (await api.listRobots()).find((r) => r.name === name);
      if (!robot) throw new Error(`No robot named “${name}”.`);
      return robot.cameras;
    },
    async saveCameras(name, cameras) {
      return cleanCameras((await api.saveRobot(name, { cameras: cleanCameras(cameras) })).cameras);
    },
    setRecord: (name, record) => api.saveRobot(name, { record: Boolean(record) }),
    // Adapters and cameras plugged in now; nothing is opened.
    devices: (arm = "metal") => request(`/api/devices?${new URLSearchParams({ arm })}`),
    async listCameras() {
      const { cameras } = await api.devices();
      return (cameras || []).map((c) => ({ id: c.id, name: c.name || c.id, preview: { image: previewUrl(c.id) } }));
    },
    // Opens and closes the gripper on that adapter, then releases it.
    identify: (arm, port) => post("/api/identify", { arm, port }),
    check: () => post("/api/check"),
    status: () => request("/api/status"),
    connect: () => post("/api/connect"),
    disconnect: () => post("/api/disconnect"),
    run: (task, model, timeoutS) => post("/api/run", { task, model, timeout_s: timeoutS, ...shown() }),
    stop: () => post("/api/stop"),
    heartbeat: () => post("/api/heartbeat"),
    release: () => post("/api/release", { confirm: true, ...shown() }),
    recordings: () => request("/api/recordings"),
    recordingFile: (run, file) => `/api/recordings/${encodeURIComponent(run)}/${encodeURIComponent(file)}`,
  };
  return api;
}

// A camera on this computer, streamed while shown (the robot's own stream while it is connected).
export const previewUrl = (id) => `/api/devices/cameras/${encodeURIComponent(id)}/stream`;
// A connected robot's camera by role.
export const streamUrl = (role) => `/api/cameras/${encodeURIComponent(role)}/stream`;
