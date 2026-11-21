// The page: prompt, model and time limit, Run / Stop, the connection line, arms and events, recordings, settings.
// It polls /api/status, shares it as the document event "pg:status" ({ detail: { status } }), and sends a heartbeat
// while a run is under way: the server stops the run when no page has sent one for a moment.

import { initRobots, setWorking as setRobotWorking } from "./robots.js";
import { errorToast, toast, glyph, h, confirmDialog, messageOf } from "./ui.js";
import { initCameras } from "./cameras.js";
import { ARMS, MODELS } from "./data.js";
import { initTutorial } from "./tutorial.js";

const $ = (id) => document.getElementById(id);
const store = {
  get(key, fallback) {
    try {
      return localStorage.getItem(`playground-${key}`) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`playground-${key}`, value);
    } catch {
      /* not remembered */
    }
  },
};

// ---------- Prompt ----------
const composer = $("composer");
const prompt = $("prompt");
const run = $("run");
const stopBtn = $("stop");
const hint = $("prompt-hint");
const model = $("model");
const limit = $("time-limit");
const ask = document.querySelector(".pg-ask");

model.value = MODELS[store.get("model", "beta")] ? store.get("model", "beta") : "beta";
limit.value = String(Number(store.get("time-limit", "60")) || 60);
model.addEventListener("change", () => store.set("model", model.value));
limit.addEventListener("change", () => {
  const seconds = Math.min(3600, Math.max(1, Math.round(Number(limit.value) || 60)));
  limit.value = String(seconds);
  store.set("time-limit", limit.value);
});

const holding = () => (status?.holding || []).length > 0;
const busy = () => Boolean(starting || status?.running);

function fit() {
  const working = ask.dataset.state === "working";
  prompt.style.height = "auto";
  prompt.style.height = `${prompt.scrollHeight}px`;
  run.hidden = working;
  run.disabled = !prompt.value.trim() || working || !api?.selected || holding() || !reachable;
  run.title = holding() ? "An arm is still holding: release it first." : !api?.selected ? "Select a robot first." : "";
  model.disabled = limit.disabled = working;
  hint.hidden = Boolean(prompt.value);
}

function setWorking(text, label) {
  const working = Boolean(text);
  ask.dataset.state = working ? "working" : "idle";
  setRobotWorking(working);
  $("task").hidden = !working;
  $("task-label").textContent = label;
  $("task-text").textContent = text || "";
  fit();
}

prompt.addEventListener("input", fit);
prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!api || !text || run.disabled || busy()) return;
  startRun(text); // the prompt stays in the box so it can be sent again
});

stopBtn.addEventListener("click", () => stopRun());
document.addEventListener("keydown", (event) => {
  // Esc stops a run even with a dialog open: stopping must not depend on focus.
  if (event.key === "Escape" && busy()) stopRun();
});

// ---------- Status, heartbeat, run ----------

let api = null; // the provider robots.js announces with "pg:robot"
let robotArm = null;
let status = null;
let reachable = true;
let starting = false; // a Run request on its way
let pollTimer;
let failures = 0;
let shownTask = null;
let shownResult = null;

document.addEventListener("pg:robot", (event) => {
  const next = event.detail?.api || null;
  if (!next) return;
  const changed = next !== api;
  api = next;
  robotArm = event.detail.robot?.arm ?? null;
  renderLive();
  if (changed) poll();
  if (recordings.open) loadRecordings();
});

function applyStatus(next) {
  // How a run ended, said once as it happens.
  const result = next.result;
  const key = result ? JSON.stringify([result.stop_reason, next.run?.task, result.elapsed_s]) : null;
  if (status && result && key !== shownResult && !next.running) {
    const text = `Run ended: ${result.stop_reason}`;
    if (result.error) errorToast(text);
    else toast(text);
  }
  shownResult = key;
  status = next;
  reachable = true;
  document.dispatchEvent(new CustomEvent("pg:status", { detail: { status } }));
  renderLive();
}

async function poll() {
  clearTimeout(pollTimer);
  const provider = api;
  if (!provider) return;
  try {
    const next = await provider.status();
    if (provider !== api) return;
    applyStatus(next);
    failures = 0;
  } catch {
    if (provider !== api) return;
    failures += 1;
    if (reachable) {
      reachable = false;
      renderLive();
    }
  }
  if (provider !== api) return;
  const delay = reachable ? (busy() ? 250 : 500) : Math.min(5000, 500 * 2 ** Math.min(failures, 4));
  pollTimer = setTimeout(poll, delay);
}

// The server stops a run or a wiggle when it hasn't heard from any open page for heartbeat_s.
let beating = false;
setInterval(() => {
  if (!api || !(busy() || status?.identifying) || beating) return;
  beating = true;
  api.heartbeat().catch(() => { }).finally(() => (beating = false));
}, 500);

async function startRun(text) {
  starting = true;
  renderLive();
  try {
    applyStatus(await api.run(text, model.value, Number(limit.value) || 60));
  } catch (error) {
    errorToast(error);
  } finally {
    starting = false;
    renderLive();
    poll();
  }
}

async function stopRun() {
  try {
    applyStatus(await api.stop());
  } catch (error) {
    errorToast(error);
  }
}

const PHASES = {
  connecting: ["Connecting", null],
  enabling: ["Starting the arms", null],
  running: ["Working on", null],
  returning: ["Finishing", "Returning to rest, then torque off"],
};

function renderLive() {
  renderLink();
  renderArms();
  renderEvents();
  const r = status?.run;
  let label = "Working on";
  let text = null;
  if (starting) [label, text] = ["Starting", prompt.value.trim() || "Your task"];
  else if (status?.running && r) {
    const [phaseLabel, phaseText] = PHASES[r.phase] || ["Working on", null];
    label = phaseLabel;
    text = phaseText || r.task || "Your task";
    if (r.phase === "running" && r.remaining_s != null) label = `${phaseLabel} · ${Math.ceil(r.remaining_s)} s left · ${r.chunks} chunk${r.chunks === 1 ? "" : "s"}`;
    if (status.stopping && r.phase !== "returning") label = "Stopping";
  }
  stopBtn.disabled = Boolean(status?.stopping);
  const key = text ? `${label}\n${text}` : "";
  if (key !== shownTask) {
    shownTask = key;
    setWorking(text, label);
  } else fit();
}

// ---------- Connection line ----------
const link = $("link");
const linkText = $("link-text");
const linkActions = $("link-actions");
const linkBtn = (label, onClick) => h("button", { class: "pg-link-btn", type: "button", onClick }, label);
// "torque off" only when every motor reports it.
function torqueText(torque) {
  const sides = Object.entries(torque || {});
  if (sides.length && sides.every(([, state]) => state === "off")) return "torque off";
  const on = sides.filter(([, state]) => state === "on").map(([side]) => side);
  return on.length ? `torque on: ${on.join(" and ")}` : "torque state unknown";
}
const armsWord = () => `${(ARMS[robotArm]?.label || "Robot").split(" ")[0]} arms`;

function renderLink() {
  link.hidden = !api;
  if (!api) return;
  const s = status;
  let text = "Not connected · Run connects the arms";
  const actions = [];
  link.dataset.holding = "false";
  if (!reachable) text = "Can’t reach the playground server";
  else if (!api.selected) text = "Select or add a robot at the top right";
  else if (!s) text = "Checking the arms…";
  else if (holding()) {
    link.dataset.holding = "true";
    text = `The ${s.holding.join(" and ")} arm${s.holding.length > 1 ? "s are" : " is"} still holding: torque off isn’t confirmed. Support it, then`;
    actions.push(linkBtn("Release", releaseArms));
  } else if (s.running) text = `${armsWord()} · running`;
  else if (s.connected) {
    text = `${armsWord()} · connected · ${torqueText(s.torque)}`;
    actions.push(linkBtn("Disconnect", () => act(api.disconnect)));
  } else actions.push(linkBtn("Connect", () => act(api.connect)));
  if (s && !s.running && s.result?.error && !holding()) text += ` · last run: ${s.result.stop_reason}`;
  if (s?.identifying) text += ` · wiggling ${s.identifying}`;
  linkText.textContent = text;
  link.title = text;
  linkActions.replaceChildren(...actions);
}

let acting = false;
async function act(call) {
  if (acting) return;
  acting = true;
  linkActions.querySelectorAll("button").forEach((b) => (b.disabled = true));
  try {
    applyStatus(await call());
  } catch (error) {
    errorToast(error);
  } finally {
    acting = false;
    renderLink();
  }
}

async function releaseArms() {
  const sides = (status?.holding || []).join(" and ");
  const ok = await confirmDialog({
    title: `Release the ${sides} arm?`,
    body: "Torque goes off at once: an arm that isn’t at rest drops. Hold it, or make sure it is supported, before you release it.",
    confirm: "Torque off now",
    cancel: "Keep holding",
  });
  if (ok) act(api.release);
}

// ---------- Arms and events ----------
const live = $("live");
const armsEl = $("arms");
const eventsEl = $("events");
const deg = (rad) => (rad == null ? "–" : `${((rad * 180) / Math.PI).toFixed(1)}°`);
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });

function renderArms() {
  const arms = status?.robot?.arms;
  live.hidden = !arms && !(status?.events?.length || status?.run?.events?.length);
  if (!live.open) return;
  if (!arms) return armsEl.replaceChildren();
  armsEl.replaceChildren(...Object.entries(arms).map(([side, arm]) => {
    const joints = Object.entries(arm.joints || {});
    const temps = joints.map(([, j]) => j.temperature).filter((t) => t != null);
    const grip = arm.gripper_opening == null ? "" : ` · gripper ${Math.round(arm.gripper_opening * 100)}% open`;
    const temp = temps.length ? ` · ${Math.round(Math.max(...temps))} °C` : "";
    return h("div", { class: "pg-arm" },
      h("p", { class: "pg-arm-head" }, h("span", {}, `${side.charAt(0).toUpperCase()}${side.slice(1)} arm`), h("span", {}, `${arm.state}${grip}${temp}`)),
      arm.fault_reason && h("p", { class: "pg-arm-fault" }, arm.fault_reason),
      joints.map(([name, j]) => h("p", { class: "pg-joint-row" }, h("span", {}, name.replace(/_/g, " ")), h("span", {}, deg(j.position)))));
  }));
}

function renderEvents() {
  if (!live.open) return;
  const all = [...(status?.events || []), ...(status?.run?.events || [])].sort((a, b) => a.t - b.t).slice(-8);
  eventsEl.replaceChildren(...all.map((e) => h("li", {}, `${clock(e.t)} ${e.kind === "user" ? "task" : e.kind}: ${e.text}`)));
}
live.addEventListener("toggle", renderLive);

// Example tasks take turns inside the empty box.
const HINTS = [
  "Pick up the black marker",
  "Put the orange cube in the tray",
  "Hand the cube to the other arm",
  "Take the cube out of the tray",
  "Push the tray to the left",
];
let hintIndex = 0;
hint.textContent = HINTS[0];
setInterval(() => {
  if (hint.hidden || document.hidden) return;
  hint.classList.remove("is-entering");
  hint.classList.add("is-leaving");
  setTimeout(() => {
    hintIndex = (hintIndex + 1) % HINTS.length;
    hint.textContent = HINTS[hintIndex];
    hint.classList.remove("is-leaving");
    hint.classList.add("is-entering");
  }, 160);
}, 3200);

// ---------- Recordings ----------
const recordings = $("recordings");
const recordSwitch = $("record");
const runsEl = $("runs");
$("open-recordings").append(glyph("film"));
$("open-recordings").addEventListener("click", () => {
  recordings.showModal();
  loadRecordings();
});
$("recordings-done").addEventListener("click", () => recordings.close());
recordings.addEventListener("click", (event) => {
  if (event.target === recordings) recordings.close();
});

let recordOn = null;
async function loadRecordings() {
  const provider = api;
  if (!provider) return;
  try {
    const [robots, listed] = await Promise.all([provider.listRobots(), provider.recordings()]);
    if (provider !== api) return;
    const robot = robots.find((r) => r.name === provider.selected);
    recordOn = robot ? robot.record : null;
    renderRecordSwitch(listed.dir);
    runsEl.replaceChildren(...(listed.recordings.length ? listed.recordings.map(runRow) : [h("li", { class: "pg-runs-empty" }, `No recordings yet in ${listed.dir}.`)]));
  } catch (error) {
    errorToast(error);
  }
}

function renderRecordSwitch(dir) {
  recordSwitch.setAttribute("aria-checked", String(Boolean(recordOn)));
  recordSwitch.disabled = recordOn === null;
  $("record-note").textContent = recordOn === null
    ? "Select a robot to choose whether its runs are recorded."
    : `Cameras, joints and the policy’s replies, one folder per run in ${dir}.`;
}

recordSwitch.addEventListener("click", async () => {
  if (recordSwitch.disabled || !api?.selected) return;
  recordSwitch.disabled = true;
  try {
    recordOn = (await api.setRecord(api.selected, !recordOn)).record;
  } catch (error) {
    errorToast(error);
  }
  recordSwitch.disabled = false;
  recordSwitch.setAttribute("aria-checked", String(Boolean(recordOn)));
});

function runRow(r) {
  const when = r.started_at ? r.started_at.replace("T", " ") : r.name;
  const meta = [when, r.robot, MODELS[r.model] || r.model, r.stopped_because ? `ended: ${r.stopped_because}` : "no run.json (still recording, or cut short)"].filter(Boolean).join(" · ");
  return h("li", { class: "pg-run" },
    h("p", { class: "pg-run-task" }, r.task || r.name),
    h("p", { class: "pg-run-meta" }, meta),
    h("p", { class: "pg-run-files" }, r.files.map((f) => h("a", { href: api.recordingFile(r.name, f), target: "_blank", rel: "noopener" }, f))));
}

// ---------- Settings ----------
$("open-settings").append(glyph("gear"));
$("open-cameras").append(glyph("camera"));
const settings = $("settings");
$("open-settings").addEventListener("click", () => settings.showModal());
settings.addEventListener("click", (event) => {
  if (event.target === settings) settings.close();
});

// The saved appearance ("system", "light" or "dark"); theme.js applies the same rule before first paint.
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const savedTheme = () => store.get("theme", "system");
const themeButtons = document.querySelectorAll("[data-theme-choice]");
function showTheme(choice) {
  document.documentElement.dataset.theme = choice === "system" ? (darkQuery.matches ? "dark" : "light") : choice;
  themeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeChoice === choice)));
}
themeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    store.set("theme", button.dataset.themeChoice);
    showTheme(button.dataset.themeChoice);
  });
});
darkQuery.addEventListener("change", () => showTheme(savedTheme()));
showTheme(savedTheme());

initCameras(); // listens for "pg:robot", so it starts before the robots load
initRobots().catch((error) => toast(messageOf(error)));
initTutorial();
setWorking(null, "");
