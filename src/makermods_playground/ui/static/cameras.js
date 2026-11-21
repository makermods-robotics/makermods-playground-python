// Camera panel: assign the cameras on this computer to Top / Left / Right.
//
// The panel floats over the page from the right and lists the cameras that aren't assigned. While it is open the
// page's own top / left / right frames are the places to assign: drag a camera from the list onto a frame, between
// frames (swap) or from a frame back to the list (unassign); each filled frame also gets an unassign button. Without
// drag, click a camera to select it, then click a frame. Assignments belong to the selected robot and save straight
// away. The provider and robot arrive with robots.js's "pg:robot" event; main.js's "pg:status" event says when the
// robot is connected, and each frame then shows that camera's live stream from the connection.

import { CAMERA_ROLES, noCameras, cleanCameras, streamUrl } from "./data.js";
import { h, glyph, errorToast } from "./ui.js";
import { CameraAssignmentGuide } from "./camera-guide.js";

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const state = {
  api: null,
  robot: null, // name of the selected robot, or null
  cameras: [], // [{ id, name, preview }]
  scanned: false, // false until the first listCameras() for this provider answers
  scanning: false,
  assigned: noCameras(),
  open: false,
  selected: null, // click-to-place
  drag: null, // while a drag is in progress
  status: null, // the server's latest status, or null
};

const els = {};
const panelMedia = new Map(); // camera id → media element in the panel (reused across renders)
const frameMedia = new Map(); // camera id (id|role for a second frame) → media element on the main page
let eventId = 0;
let hideTimer;
const ledgers = new WeakMap();
function ledger(api, robot) {
  let byRobot = ledgers.get(api);
  if (!byRobot) ledgers.set(api, (byRobot = new Map()));
  let entry = byRobot.get(robot);
  if (!entry) byRobot.set(robot, (entry = { confirmed: null, local: null, edits: 0, pending: 0, round: 0, queue: Promise.resolve() }));
  return entry;
}
const showing = (api, robot) => api === state.api && robot === state.robot;

// ---------- Public ----------

export function initCameras() {
  const button = document.getElementById("open-cameras");
  const drawer = document.getElementById("camera-drawer");
  els.frames = [...document.querySelectorAll(".pg-cam-frame[data-camera]")];
  els.stage = document.querySelector(".pg-stage");
  if (!button || !drawer) return;
  els.button = button;
  els.drawer = drawer;

  buildDrawer();
  els.button.addEventListener("click", () => (state.open ? close() : open()));
  document.addEventListener("keydown", onEscape, true);
  document.addEventListener("pg:robot", onRobot);
  document.addEventListener("pg:status", onStatus);
  // Fallbacks for drags whose source was replaced mid-drag (sources also clean up themselves).
  document.addEventListener("dragend", endDrag);
  document.addEventListener("drop", endDrag);
  document.addEventListener("pointerdown", () => state.drag && endDrag(), true);
  initStage();
  renderFrames();
}

// ---------- Provider and robot ----------

async function onRobot(event) {
  const { api, robot } = event.detail || {};
  const id = ++eventId;
  const providerChanged = api !== state.api;
  endDrag(); // a drag from the previous robot or provider can't land on this one
  state.api = api || null;
  state.robot = robot?.name ?? null;
  state.selected = null;
  if (providerChanged) {
    state.cameras = [];
    state.scanned = false;
    scan(); // renders when the list arrives
  }
  let assigned = noCameras();
  if (robot && api) {
    // robots.js keeps its own copy of each robot, so read the saved assignment from the server.
    // A read that overlaps a local edit or a pending save is older than what the user chose: the
    // ledger's local value wins, and the save queue reconciles the view when it settles.
    const entry = ledger(api, robot.name);
    const edits = entry.edits;
    const racing = entry.pending > 0;
    let read = null;
    if (api.getCameras) {
      try {
        read = cleanCameras(await api.getCameras(robot.name));
      } catch {
        /* robot just removed: show what we know */
      }
    }
    if (!racing && !entry.pending && entry.edits === edits && read) {
      entry.confirmed = read;
      entry.local = read;
    }
    assigned = entry.local ?? read ?? cleanCameras(robot.cameras);
  }
  if (id !== eventId) return;
  state.assigned = assigned;
  render();
}

// Only what the frames show matters here, so re-render only when that changes.
let statusKey = "";
function onStatus(event) {
  state.status = event.detail?.status ?? null;
  const key = CAMERA_ROLES.map((role) => `${liveStream(role)}|${liveDetail(role)}`).join(",");
  if (key === statusKey) return;
  statusKey = key;
  renderFrames();
}

const liveConnected = () => Boolean(state.api) && Boolean(state.status?.connected);
const captureOf = (role) => state.status?.robot?.cameras?.[role] || null;

// Which stream a frame shows: connected, and that camera is open.
function liveStream(role) {
  if (!liveConnected()) return null;
  return captureOf(role)?.open ? streamUrl(role) : null;
}

// Connected but no stream for a role: a short reason.
function liveNote(role) {
  if (!liveConnected()) return "";
  const capture = captureOf(role);
  if (!capture) return "No camera assigned";
  if (!capture.error) return "Waiting for the camera";
  return String(capture.error).split(/[.:;]/)[0].trim() || "Camera unavailable";
}

// The whole error behind liveNote, as the note's tooltip.
function liveDetail(role) {
  return liveConnected() ? String(captureOf(role)?.error ?? "") : "";
}

async function scan() {
  const api = state.api;
  if (!api?.listCameras) return;
  state.scanning = true;
  render();
  let list = [];
  try {
    list = await api.listCameras();
  } catch {
    list = [];
  }
  if (api !== state.api) return; // the provider changed while scanning
  endDrag(); // the render below replaces the cards, including one being dragged
  state.cameras = list;
  state.scanned = true;
  state.scanning = false;
  if (state.selected && !cameraById(state.selected.id)) state.selected = null;
  render();
}

const cameraById = (id) => state.cameras.find((c) => c.id === id) || null;
// Only cameras that are actually connected count as assigned.
const assignedCamera = (role) => cameraById(state.assigned[role]);
const unassignedCameras = () => state.cameras.filter((c) => !CAMERA_ROLES.some((role) => state.assigned[role] === c.id));
const anyAssigned = () => CAMERA_ROLES.some((role) => assignedCamera(role));

// ---------- Changes ----------

// Move camera `id` from "list" or a role to a role or "list". A camera dropped on a filled spot
// swaps: the one it displaces goes where the dragged camera came from (back to the list from the list).
function move(id, from, to) {
  const camera = cameraById(id);
  if (!state.robot || !camera || from === to) return;
  // Ignore a stale source (the assignment changed since the drag or pick began).
  const assignedTo = CAMERA_ROLES.find((role) => state.assigned[role] === id) ?? "list";
  if (assignedTo !== from) return;
  const next = { ...state.assigned };
  if (to === "list") {
    if (from === "list") return;
    next[from] = null;
    commit(next, `${camera.name} moved back to the list.`);
    return;
  }
  const displaced = next[to];
  next[to] = id;
  if (from !== "list") next[from] = displaced ?? null;
  commit(next, `${camera.name} is now the ${to} camera.`);
}

function commit(next, message) {
  const { api, robot } = state;
  if (!api || !robot) return;
  els.guide.hide();
  const entry = ledger(api, robot);
  if (!entry.confirmed && !entry.pending) entry.confirmed = state.assigned;
  entry.local = next;
  entry.edits += 1;
  entry.pending += 1;
  const round = entry.round;
  state.assigned = next;
  state.selected = null;
  render();
  announce(message);

  entry.queue = entry.queue.then(async () => {
    try {
      if (round !== entry.round) return; // an earlier save failed and this edit was built on it
      const saved = await api.saveCameras(robot, next);
      entry.confirmed = cleanCameras(saved ?? next);
    } catch (error) {
      // Go back to what the provider last confirmed; drop the saves queued behind this one.
      entry.round += 1;
      entry.local = entry.confirmed ?? noCameras();
      if (showing(api, robot)) {
        state.assigned = entry.local;
        render();
      }
      errorToast(`Couldn’t save cameras. ${error instanceof Error ? error.message : ""}`.trim());
    } finally {
      entry.pending -= 1;
      // Once every save has settled, the view shows exactly what the ledger holds.
      if (!entry.pending && showing(api, robot) && !sameCameras(state.assigned, entry.local)) {
        state.assigned = entry.local;
        render();
      }
    }
  });
}

const sameCameras = (a, b) => CAMERA_ROLES.every((role) => (a?.[role] ?? null) === (b?.[role] ?? null));

function unassignAll() {
  if (!anyAssigned()) return;
  commit(noCameras(), "All cameras moved back to the list.");
  els.list.focus?.();
}

// ---------- Open / close ----------

function open() {
  if (state.open) return;
  state.open = true;
  els.guide.open();
  clearTimeout(hideTimer);
  render();
  els.drawer.hidden = false;
  els.drawer.getBoundingClientRect(); // start the slide from the closed position
  els.drawer.classList.add("is-open");
  els.button.setAttribute("aria-expanded", "true");
  els.close.focus({ preventScroll: true });
}

function close({ returnFocus = true } = {}) {
  if (!state.open) return;
  const hadFocus = els.drawer.contains(document.activeElement);
  state.open = false;
  els.guide.hide();
  state.selected = null;
  endDrag();
  els.drawer.classList.remove("is-open");
  els.button.setAttribute("aria-expanded", "false");
  renderFrames(); // the page's frames leave assign mode now, not when the slide ends
  if (returnFocus || hadFocus) els.button.focus({ preventScroll: true });
  const hide = () => {
    if (state.open) return;
    els.drawer.hidden = true;
    render(); // drops the panel's previews now that nobody can see them
  };
  if (reducedMotion.matches) hide();
  else hideTimer = setTimeout(hide, 340); // a hair past --duration-base
}

// Esc clears a click-to-place selection first, then closes the panel. It runs in the capture phase
// so closing the panel doesn't also stop a running task; menus and dialogs keep their own Esc.
function onEscape(event) {
  if (event.key !== "Escape" || !state.open || event.defaultPrevented) return;
  if (document.querySelector("dialog[open]")) return;
  const menu = document.getElementById("robot-menu");
  if (menu && !menu.hidden) return;
  event.stopPropagation();
  if (state.selected) {
    const back = state.selected;
    state.selected = null;
    render();
    (back.from === "list" ? els.drawer.querySelector(`[data-select="${back.id}"]`) : frameOf(back.from))?.focus();
    return;
  }
  close();
}

// ---------- Drawer ----------

function buildDrawer() {
  const drawer = els.drawer;
  drawer.classList.add("pg-cams-drawer");
  drawer.setAttribute("aria-labelledby", "camera-drawer-title");
  drawer.removeAttribute("aria-label");

  els.note = h("p", { class: "pg-cams-note", id: "camera-drawer-note" });
  els.close = h("button", { class: "pg-cams-close", type: "button", "aria-label": "Close cameras", title: "Close", onClick: () => close() }, glyph("close"));
  els.status = h("p", { class: "pg-visually-hidden", role: "status", "aria-live": "polite" });

  els.unassignAll = h("button", { class: "hr-btn hr-btn-secondary hr-btn-sm", type: "button", onClick: unassignAll }, "Unassign all");
  els.rescan = h("button", { class: "hr-btn hr-btn-secondary hr-btn-sm", type: "button", onClick: scan }, "Rescan");
  els.grid = h("div", { class: "pg-cams-grid" });
  els.guide = new CameraAssignmentGuide();
  els.listNote = h("p", { class: "pg-cams-empty-note" });
  els.list = h("div", { class: "pg-cams-list", tabindex: "-1" }, els.grid, els.listNote, h("p", { class: "pg-cams-list-hint", "aria-hidden": "true" }, "Drop here to unassign"));
  dropTarget(els.list, "list");

  els.head = h("div", { class: "pg-cams-head" }, h("div", { class: "pg-cams-head-copy" }, h("h2", { class: "pg-cams-title", id: "camera-drawer-title" }, "Cameras"), els.note), els.close);
  els.body = h("div", {
    class: "pg-cams-body",
    onScroll: () => els.head.classList.toggle("is-scrolled", els.body.scrollTop > 0),
  }, h("section", { class: "pg-cams-section", "aria-labelledby": "camera-list-title" }, h("div", { class: "pg-cams-section-head" }, h("h3", { class: "eyebrow pg-cams-section-title", id: "camera-list-title" }, "On this computer"), els.rescan), els.list));
  els.body.prepend(els.guide.element);
  els.foot = h("div", { class: "pg-cams-foot" }, els.unassignAll);
  drawer.append(els.head, els.body, els.foot, els.status);
}

function render() {
  renderFrames();
  renderPanel();
}

function renderPanel() {
  if (!els.drawer) return;
  const used = new Set();
  const robot = Boolean(state.robot);
  const picking = state.selected ? cameraById(state.selected.id) : null;

  els.note.textContent = !robot
    ? "Select a robot first to assign cameras."
    : picking
      ? `Now click top, left or right on the page for ${picking.name}.`
      : "Drag a camera onto top, left or right on the page.";
  els.drawer.dataset.picking = String(Boolean(picking));
  els.foot.hidden = !robot;
  els.unassignAll.disabled = !anyAssigned();

  // Cameras on this computer that aren't assigned (the assigned ones show on the page).
  const list = robot ? unassignedCameras() : state.cameras;
  els.rescan.disabled = state.scanning;
  els.list.setAttribute("aria-busy", String(state.scanning));
  els.grid.replaceChildren(...(state.open ? list.map((camera) => card(camera, used, robot)) : []));
  els.grid.hidden = list.length === 0;
  els.listNote.hidden = list.length > 0;
  els.listNote.textContent = !state.scanned
    ? "Looking for cameras…"
    : state.cameras.length === 0
      ? "No cameras found. Plug one in, then rescan."
      : "All cameras are assigned.";

  disposeUnused(panelMedia, used);
}

function card(camera, used, interactive) {
  const selected = state.selected?.from === "list" && state.selected.id === camera.id;
  // Each card streams its own device, so you can see which camera is which before assigning.
  const frame = h("span", { class: "hr-media hr-media-sm pg-cams-card-frame" }, media(camera, panelMedia, used));
  const caption = h("span", { class: "pg-cams-caption" }, camera.name);
  if (!interactive) return h("div", { class: "pg-cams-card is-static" }, frame, caption);
  const el = h("button", {
    class: "pg-cams-card",
    type: "button",
    draggable: "true",
    "aria-pressed": String(selected),
    "aria-label": `${camera.name}. Select, then choose top, left or right.`,
    dataset: { select: camera.id },
    onClick: () => pickCard(camera.id),
  }, frame, caption);
  dragSource(el, () => ({ id: camera.id, from: "list" }));
  return el;
}

// ---------- Click to place ----------

function pickCard(id) {
  els.guide.hide();
  state.selected = state.selected?.from === "list" && state.selected.id === id ? null : { id, from: "list" };
  render();
  els.drawer.querySelector(`[data-select="${id}"]`)?.focus({ preventScroll: true });
}

// A frame on the page, clicked (or Enter / Space) while a camera is picked, takes that camera.
function placeOnFrame(role) {
  const sel = state.selected;
  if (!state.open || !sel) return;
  if (sel.from === role) {
    state.selected = null;
    render();
  } else {
    move(sel.id, sel.from, role);
  }
  frameOf(role)?.focus({ preventScroll: true });
}

// ---------- Drag and drop ----------

const DRAG_TYPE = "application/x-makermods-camera";

// `source()` answers { id, from } for what a drag started here would carry, or null (no drag).
function dragSource(el, source) {
  el.addEventListener("dragend", endDrag); // fires on the source even if it left the document
  el.addEventListener("dragstart", (event) => {
    const picked = source();
    if (!picked || !event.dataTransfer) {
      event.preventDefault();
      return;
    }
    const { id, from } = picked;
    els.guide.hide();
    state.drag = { id, from, api: state.api, robot: state.robot };
    state.selected = null;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData(DRAG_TYPE, id);
    event.dataTransfer.setData("text/plain", cameraById(id)?.name || id);
    // Mark targets after the browser has taken its drag image.
    requestAnimationFrame(() => {
      if (!state.drag) return;
      el.classList.add("is-dragging");
      els.drawer.dataset.drag = from === "list" ? "list" : "slot";
      els.frames.forEach((frame) => frame.classList.toggle("is-drop-target", accepts(frame.dataset.camera)));
    });
  });
}

// A target accepts our own drags only: list → frame, frame → frame, frame → list, started for the
// robot and provider still shown, carrying our payload, of a camera that is still connected.
function accepts(target, event) {
  const drag = state.drag;
  if (!drag || !state.open || !state.robot) return false;
  if (drag.api !== state.api || drag.robot !== state.robot || !cameraById(drag.id)) return false;
  if (event && !event.dataTransfer?.types.includes(DRAG_TYPE)) return false;
  if (target === "list") return drag.from !== "list";
  return drag.from !== target;
}

function dropTarget(el, target) {
  el.addEventListener("dragover", (event) => {
    if (!accepts(target, event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    el.classList.add("is-over");
  });
  el.addEventListener("dragleave", (event) => {
    if (!el.contains(event.relatedTarget)) el.classList.remove("is-over");
  });
  el.addEventListener("drop", (event) => {
    if (!accepts(target, event) || event.dataTransfer?.getData(DRAG_TYPE) !== state.drag?.id) {
      endDrag();
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const { id, from } = state.drag;
    endDrag();
    move(id, from, target);
  });
}

function endDrag() {
  if (!state.drag && !els.drawer?.dataset.drag) return;
  state.drag = null;
  delete els.drawer.dataset.drag;
  document.querySelectorAll(".is-over, .is-drop-target, .pg-cams-drawer .is-dragging, .pg-cam-frame.is-dragging").forEach((el) => el.classList.remove("is-over", "is-drop-target", "is-dragging"));
}

// ---------- Main page frames ----------

// Which camera is large in the middle and which two sit either side (top = left, bottom = right): each figure's data-slot.
// A side camera's swap button trades places with the middle one. Remembered in this browser.
const SLOTS = ["main", "top", "bottom"];
const ARRANGEMENT_KEY = "playground-camera-slots";
const figureOf = (role) => els.frames.find((frame) => frame.dataset.camera === role)?.closest(".pg-cam");
const roleIn = (slot) => els.frames.find((frame) => frame.closest(".pg-cam")?.dataset.slot === slot)?.dataset.camera;
const frameOf = (role) => els.frames.find((frame) => frame.dataset.camera === role);

function initStage() {
  for (const frame of els.frames) {
    const role = frame.dataset.camera;
    // While the panel is open the frames are where cameras get assigned.
    dropTarget(frame, role);
    dragSource(frame, () => {
      const camera = state.open && state.robot ? assignedCamera(role) : null;
      return camera ? { id: camera.id, from: role } : null;
    });
    frame.addEventListener("click", (event) => {
      if (!event.target.closest("button")) placeOnFrame(role);
    });
    frame.addEventListener("keydown", (event) => {
      if (event.target !== frame || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      placeOnFrame(role);
    });
    frame.append(h("button", {
      class: "pg-cam-remove",
      type: "button",
      hidden: true,
      onClick: () => {
        const camera = assignedCamera(role);
        if (camera) move(camera.id, role, "list");
        els.list.focus({ preventScroll: true }); // the button hides with the camera gone
      },
    }, glyph("close"), "Unassign"), h("button", {
      class: "pg-cam-swap",
      type: "button",
      "aria-label": `Show the ${role} camera in the middle`,
      title: "Show in the middle",
      onClick: () => toMain(role),
    }, glyph("swap")));
  }
  const saved = (localStorage.getItem(ARRANGEMENT_KEY) || "").split(",");
  const valid = saved.length === SLOTS.length && CAMERA_ROLES.every((role) => saved.includes(role));
  if (valid) saved.forEach((role, i) => (figureOf(role).dataset.slot = SLOTS[i]));
}

function toMain(role) {
  const figure = figureOf(role);
  const main = figureOf(roleIn("main"));
  if (!figure || !main || figure === main) return;
  const swap = () => {
    main.dataset.slot = figure.dataset.slot;
    figure.dataset.slot = "main";
    localStorage.setItem(ARRANGEMENT_KEY, SLOTS.map(roleIn).join(","));
    renderFrames();
    // The pressed button left with its camera; focus the one that took its place.
    if (refocus) main.querySelector(".pg-cam-swap")?.focus({ preventScroll: true });
  };
  const refocus = figure.contains(document.activeElement);
  if (document.startViewTransition && !reducedMotion.matches) document.startViewTransition(swap);
  else swap();
}

function renderFrames() {
  const used = new Set();
  const assigning = state.open && Boolean(state.robot);
  const picking = assigning && state.selected ? cameraById(state.selected.id) : null;
  els.stage?.classList.toggle("is-assigning", assigning);
  if (els.stage) els.stage.dataset.picking = String(Boolean(picking));
  // With no camera anywhere, one line in the middle frame says where to assign them; the side frames stay quiet.
  const none = !state.robot || !anyAssigned();
  for (const frame of els.frames || []) {
    const role = frame.dataset.camera;
    const empty = frame.querySelector(".pg-cam-empty");
    const camera = state.robot ? assignedCamera(role) : null;
    // Connected, a frame shows its role's stream (or why there is none), never a preview.
    const stream = liveStream(role);
    const shown = liveConnected() ? null : camera;
    frame.querySelectorAll(":scope > img").forEach((el) => {
      if (el.dataset.stream !== undefined) {
        if (el.dataset.stream !== stream) closeStream(el);
      } else if (!shown || el.dataset.cameraId !== shown.id)
        el.remove();
    });
    if (stream && !frame.querySelector(":scope > img[data-stream]")) {
      frame.prepend(h("img", { class: "pg-cam-stream", src: stream, alt: "", draggable: "false", dataset: { stream } }));
    } else if (shown) {
      // One element per frame: a camera assigned to two roles is shown in both (an element can't be in two places).
      const el = media(shown, frameMedia, used, used.has(shown.id) ? `${shown.id}|${role}` : shown.id);
      if (el.parentElement !== frame) frame.prepend(el);
    }
    if (empty) {
      const main = frame.closest(".pg-cam")?.dataset.slot === "main";
      const note = liveNote(role);
      empty.hidden = Boolean(stream || shown) || (none && !main && !assigning && !note);
      if (assigning) empty.textContent = "Drop a camera here";
      else if (note && !(none && main && !liveConnected())) empty.textContent = note;
      else if (none && main) empty.replaceChildren("Assign cameras with ", glyph("camera"), " at the top right.");
      else empty.textContent = "No camera";
      const detail = !assigning && note && liveDetail(role) !== note ? liveDetail(role) : "";
      if (detail) empty.title = detail;
      else empty.removeAttribute("title");
    }

    // Assign mode: filled frames drag and can be unassigned; with a camera picked, frames are click targets.
    frame.draggable = assigning && Boolean(camera);
    const remove = frame.querySelector(".pg-cam-remove");
    remove.hidden = !(assigning && camera);
    if (camera) remove.setAttribute("aria-label", `Unassign ${camera.name}`);
    if (picking) {
      frame.tabIndex = 0;
      frame.setAttribute("role", "button");
      frame.setAttribute("aria-label", `Place ${picking.name} as ${role}`);
    } else {
      frame.removeAttribute("tabindex");
      frame.removeAttribute("role");
      frame.removeAttribute("aria-label");
    }
  }
  disposeUnused(frameMedia, used);
}

// ---------- Media ----------

// One preview element (the camera's live MJPEG stream) per camera per place, reused across renders so a moved camera
// keeps streaming. `key` names the element in its pool.
function media(camera, pool, used, key = camera.id) {
  used.add(key);
  let el = pool.get(key);
  if (!el) {
    el = h("img", { src: camera.preview?.image || "", alt: "", draggable: "false", dataset: { cameraId: camera.id } });
    pool.set(key, el);
  }
  return el;
}

// Previews that aren't shown are closed and released, not left streaming off-screen.
function disposeUnused(pool, used) {
  for (const [id, el] of pool) {
    if (used.has(id)) continue;
    pool.delete(id);
    if (el.getAttribute("src")?.startsWith("/api/")) el.src = ""; // closes a live stream
    el.remove();
  }
}

// A live MJPEG <img> keeps its connection open until its source is cleared.
function closeStream(el) {
  el.src = "";
  el.remove();
}

function announce(message) {
  if (!els.status) return;
  els.status.textContent = "";
  requestAnimationFrame(() => (els.status.textContent = message));
}
