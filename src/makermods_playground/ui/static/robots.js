// Robot picker (top right) and robot setup: create a robot (name, arm type), pick each arm's CAN adapter (Wiggle
// shows which arm an adapter drives), check it read-only, delete it. Cameras are assigned in the camera panel.

import { ARMS, ARM_TYPES, SIDES, CAMERA_ROLES, createProvider } from "./data.js";
import { h, glyph, errorToast, toast, confirmDialog, smallDialog, messageOf } from "./ui.js";

let api;
let robots = [];
let selected = null;
let working = false;
let loading = true;
let loadError = "";
let connected = false; // the server has the selected robot connected
let choosing = false; // a robot switch is on its way to the server

const els = {};
const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);

// ---------- Public ----------

export function initRobots() {
  const corner = document.getElementById("robot-corner");
  els.setup = h("button", { class: "pg-robot-icon", type: "button", onClick: () => selected && openSetup(selected, false) }, glyph("sliders"));
  els.dot = h("i", { class: "pg-dot", "aria-hidden": "true" });
  els.name = h("span", { class: "pg-robot-name" });
  els.trigger = h("button", { class: "pg-robot-trigger", type: "button", "data-guide": "robot-picker", "aria-haspopup": "menu", "aria-expanded": "false", "aria-controls": "robot-menu", onClick: toggleMenu }, els.dot, els.name, glyph("chevron"));
  els.menu = h("div", { class: "pg-menu", id: "robot-menu", role: "menu", "aria-label": "Robots", hidden: true, onKeydown: menuKeys });
  corner.append(els.setup, h("span", { class: "pg-robot-rule", "aria-hidden": "true" }), els.trigger, els.menu);

  document.addEventListener("click", (event) => {
    if (!els.menu.hidden && !corner.contains(event.target) && !event.target?.closest?.("[data-tutorial-control]")) closeMenu();
  });
  document.addEventListener("pg:status", (event) => {
    const next = Boolean(event.detail?.status?.connected);
    if (next === connected) return;
    connected = next;
    renderCorner();
  });
  api = createProvider();
  return refresh();
}

// Other modules follow the provider and the selected robot through the document event "pg:robot", detail
// { api, robot }: robot is a copy, or null when none is selected.
function announce() {
  const robot = selected ? JSON.parse(JSON.stringify(selected)) : null;
  document.dispatchEvent(new CustomEvent("pg:robot", { detail: { api, robot } }));
}

export function setWorking(on) {
  working = on;
  renderCorner();
}

export const refreshRobots = () => refresh();

// ---------- Corner ----------

let refreshId = 0;
async function refresh() {
  const id = ++refreshId; // a newer refresh wins over a slower older one
  loading = true;
  renderCorner();
  let list = [];
  loadError = "";
  try {
    list = await api.listRobots();
  } catch (error) {
    loadError = messageOf(error);
  }
  if (id !== refreshId) return;
  robots = list;
  selected = robots.find((r) => r.name === api.selected) ?? null;
  loading = false;
  renderCorner();
  if (!els.menu.hidden) renderMenu();
  announce();
}

const sidesText = (robot) => SIDES.filter((s) => robot.arms[s]).join(" and ") || "no arms";
const camerasText = (robot) => CAMERA_ROLES.filter((r) => robot.cameras[r]).join(", ") || "no cameras";
const armLine = (robot) => `${ARMS[robot.arm].label} · ${sidesText(robot)} · ${camerasText(robot)}`;

function renderCorner() {
  if (!els.trigger) return;
  const none = !loading && robots.length === 0;
  els.setup.disabled = !selected;
  const setupLabel = selected ? `Set up ${selected.name}` : "Select a robot first";
  els.setup.setAttribute("aria-label", setupLabel);
  els.setup.title = setupLabel;
  els.name.textContent = loading ? "Loading…" : selected ? selected.name : none ? "Add a new robot" : "Select a robot";
  els.dot.dataset.state = working ? "working" : !selected ? "none" : connected ? "live" : "ready";
  els.trigger.title = selected ? armLine(selected) : "";
}

function toggleMenu() {
  if (els.menu.hidden) openMenu();
  else closeMenu();
}

function openMenu() {
  renderMenu();
  els.menu.hidden = false;
  els.trigger.setAttribute("aria-expanded", "true");
  els.menu.querySelector('[aria-checked="true"], [role^="menuitem"]')?.focus();
}

function closeMenu(returnFocus = false) {
  if (!els.menu || els.menu.hidden) return;
  els.menu.hidden = true;
  els.trigger.setAttribute("aria-expanded", "false");
  if (returnFocus) els.trigger.focus();
}

function menuKeys(event) {
  const items = [...els.menu.querySelectorAll('[role^="menuitem"]:not([disabled])')];
  const i = items.indexOf(document.activeElement);
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const step = event.key === "ArrowDown" ? 1 : -1;
    items[(i + step + items.length) % items.length]?.focus();
  } else if (event.key === "Escape") {
    event.stopPropagation();
    closeMenu(true);
  } else if (event.key === "Tab") {
    closeMenu();
  }
}

function renderMenu() {
  const rows = robots.map((robot) => h("button", {
    class: "pg-menu-robot",
    type: "button",
    role: "menuitemradio",
    "aria-checked": String(robot.name === selected?.name),
    tabindex: "-1",
    disabled: choosing || working,
    onClick: () => choose(robot),
  }, h("i", { class: "pg-dot", "aria-hidden": "true", dataset: { state: "ready" } }), h("span", { class: "pg-menu-robot-copy" }, h("span", { class: "pg-menu-robot-name" }, robot.name), h("span", { class: "pg-menu-robot-meta" }, armLine(robot))), robot.name === selected?.name && glyph("check")));
  const empty = loadError || "No robots yet. Add your Metal or Maker arms to get started.";
  els.menu.replaceChildren(h("p", { class: "eyebrow pg-menu-head" }, "Robots"), rows.length ? h("div", { class: "pg-menu-list" }, rows) : h("p", { class: "pg-menu-empty" }, empty), h("hr", { class: "pg-menu-rule" }),
    h("button", { class: "pg-menu-robot pg-menu-new", "data-guide": "new-robot", type: "button", role: "menuitem", tabindex: "-1", disabled: Boolean(loadError), onClick: newRobot }, glyph("plus"), "New robot"));
}

// Switching robots disconnects the previous one (refused during a run). On failure the selection stays.
async function choose(robot) {
  closeMenu(true);
  if (robot.name === selected?.name || choosing) return;
  const provider = api;
  choosing = true;
  try {
    await provider.selectRobot(robot.name);
    if (provider !== api) return;
    selected = robot;
    renderCorner();
    announce();
  } catch (error) {
    if (provider === api) errorToast(error);
  } finally {
    choosing = false;
  }
}

// ---------- Create ----------

// New robot: name and arm type, then its setup opens; Save there writes it (a robot needs at least one arm).
async function newRobot() {
  closeMenu(true);
  const draft = await createDialog();
  if (draft) openSetup(draft, true);
}

function createDialog() {
  return smallDialog((done, titleId) => {
    const d = { name: "", arm: ARM_TYPES[0] };
    const errorId = `${titleId}-error`;
    const nameId = `${titleId}-name`;
    const error = h("p", { class: "pg-field-error", id: errorId, role: "alert" });
    const nameInput = h("input", {
      id: nameId, class: "pg-input", type: "text", autocomplete: "off", spellcheck: "false", required: true,
      placeholder: "e.g. lab", "aria-describedby": errorId,
      onInput: () => {
        d.name = nameInput.value.trim();
        error.textContent = "";
        paint();
      },
    });
    const armsEl = h("div", { class: "pg-arm-choices", role: "group", "aria-label": "Arm type" });
    const nextBtn = h("button", { class: "hr-btn hr-btn-primary", type: "submit" }, "Continue");

    function paint() {
      const focused = armsEl.contains(document.activeElement) ? document.activeElement.dataset.focus : null;
      armsEl.replaceChildren(...ARM_TYPES.map((id) => h("button", {
        class: "pg-arm-choice",
        type: "button",
        "aria-pressed": String(d.arm === id),
        dataset: { focus: `arm-${id}` },
        onClick: () => {
          d.arm = id;
          paint();
        },
      }, h("span", { class: "hr-media hr-media-sm pg-arm-choice-media" }, h("img", { src: ARMS[id].image, alt: "" })), h("span", { class: "pg-arm-choice-name" }, ARMS[id].label), h("span", { class: "pg-arm-choice-blurb" }, ARMS[id].blurb))));
      if (focused) armsEl.querySelector(`[data-focus="${focused}"]`)?.focus();
      nextBtn.disabled = !d.name;
    }

    function onSubmit(event) {
      event.preventDefault();
      if (!d.name) return;
      if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(d.name)) {
        error.textContent = "Use letters, digits, dot, dash or underscore.";
        return nameInput.focus();
      }
      if (robots.some((r) => r.name === d.name)) {
        error.textContent = `A robot named “${d.name}” already exists.`;
        return nameInput.focus();
      }
      done({ name: d.name, arm: d.arm, arms: {}, cameras: {}, record: true });
    }

    paint();
    return h("form", { class: "pg-dialog-body pg-create", "data-guide": "robot-create", onSubmit }, h("h2", { class: "title", id: titleId }, "New robot"), h("div", { class: "pg-field" }, h("label", { class: "pg-field-label", for: nameId }, "Name"), nameInput, error), h("div", { class: "pg-field" }, h("span", { class: "pg-field-label", "aria-hidden": "true" }, "Arm type"), armsEl), h("div", { class: "pg-dialog-actions" }, h("button", { class: "hr-btn hr-btn-secondary", type: "button", onClick: () => done(null) }, "Cancel"), nextBtn));
  }, "pg-dialog-md");
}

// ---------- Setup (adapters, check) ----------

function openSetup(source, isNew) {
  const robot = JSON.parse(JSON.stringify(source));
  const arm = ARMS[robot.arm];
  const rescanBtn = h("button", { class: "hr-btn hr-btn-secondary hr-btn-sm", type: "button", onClick: () => rescan() }, "Rescan");
  const checkBtn = h("button", { class: "hr-btn hr-btn-secondary hr-btn-sm", type: "button", onClick: () => runCheck() }, "Check");
  const s = {
    isNew,
    saved: { ...robot.arms },
    draft: { ...robot.arms },
    adapters: [],
    adapterError: null,
    scanning: true,
    wiggling: null, // the side whose gripper is moving
    checking: false,
    report: null,
    saving: false,
    closed: false,
  };
  const dirty = () => s.isNew || SIDES.some((side) => (s.draft[side] || null) !== (s.saved[side] || null));
  const listed = (port) => s.adapters.some((a) => a.port === port);

  const dialog = h("dialog", { class: "pg-dialog pg-dialog-lg", "aria-labelledby": "setup-title", dataset: { guide: "robot-setup" } });
  const armsEl = h("section", { class: "pg-section", "aria-labelledby": "arms-title" });
  const checkEl = h("section", { class: "pg-section", "aria-labelledby": "check-title" });
  const footStatus = h("p", { class: "pg-setup-status", role: "status" });
  const saveBtn = h("button", { class: "hr-btn hr-btn-primary", type: "button", onClick: save }, "Save");
  const deleteBtn = h("button", { class: "hr-btn hr-btn-ghost hr-btn-sm pg-setup-delete", type: "button", onClick: remove }, "Delete robot");

  dialog.append(
    h("header", { class: "pg-setup-head" }, h("div", { class: "pg-setup-title" }, h("h2", { class: "title", id: "setup-title" }, `Set up ${robot.name}`)), h("p", { class: "pg-setup-meta" }, arm.label)),
    h("div", { class: "pg-setup-body" }, armsEl, checkEl),
    h("footer", { class: "pg-setup-foot" }, deleteBtn, footStatus, h("div", { class: "pg-setup-foot-actions" }, h("button", { class: "hr-btn hr-btn-secondary", type: "button", onClick: requestClose }, "Close"), saveBtn)),
  );
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    requestClose();
  });
  dialog.addEventListener("close", () => {
    s.closed = true;
    dialog.remove();
    refresh();
  });
  document.body.append(dialog);
  dialog.showModal();

  function renderAll() {
    renderArms();
    renderCheck();
    renderFoot();
  }

  function renderFoot() {
    const hasArm = SIDES.some((side) => s.draft[side]);
    deleteBtn.hidden = s.isNew;
    footStatus.textContent = !hasArm ? "Pick at least one arm’s adapter" : dirty() ? "Unsaved changes" : "Saved";
    saveBtn.disabled = s.saving || !dirty() || !hasArm || Boolean(s.wiggling);
    saveBtn.textContent = s.saving ? "Saving…" : "Save";
  }

  function sectionHead(num, id, title, ...actions) {
    return h("div", { class: "pg-section-head" }, h("span", { class: "eyebrow pg-section-num" }, num), h("h3", { class: "pg-section-title", id }, title), h("div", { class: "pg-section-actions" }, actions));
  }

  // ----- 01 Arms -----

  async function rescan() {
    s.scanning = true;
    renderArms();
    try {
      const found = await api.devices(robot.arm);
      s.adapters = found.adapters || [];
      s.adapterError = found.adapters_error;
    } catch (error) {
      s.adapters = [];
      errorToast(error);
    }
    s.scanning = false;
    if (!s.closed) renderArms();
  }

  function renderArms() {
    rescanBtn.disabled = s.scanning || Boolean(s.wiggling);
    rescanBtn.textContent = s.scanning ? "Scanning…" : "Rescan";
    const none = !s.scanning && s.adapters.length === 0;
    const note = "Pick the CAN adapter of each arm. Wiggle opens and closes that arm’s gripper a little (only the gripper gets torque, then it is released), so you can see which arm it is.";
    armsEl.replaceChildren(
      sectionHead("01", "arms-title", "Arms", h("div", { class: "pg-usb-actions", "data-guide": "usb-device" }, rescanBtn)),
      h("p", { class: "pg-section-note" }, note),
      none ? h("p", { class: "pg-empty-line" }, s.adapterError ? `No adapters found: ${s.adapterError}` : "No CAN adapters found. Plug them in and rescan.") : "",
      h("div", { class: "pg-slots" }, SIDES.map(slotCell)),
    );
  }

  function slotCell(side) {
    const port = s.draft[side];
    const detected = Boolean(port) && listed(port);
    const state = !port ? "empty" : s.scanning ? "checking" : detected ? "ready" : "missing";
    const statusText = { empty: "No adapter", checking: "Checking…", ready: "Detected", missing: "Not detected" }[state];
    const other = SIDES.find((x) => x !== side);
    const options = s.adapters.map((a) => a.port);
    if (port && !options.includes(port)) options.unshift(port);
    const select = h("select", {
      class: "pg-select pg-select-mono",
      "aria-label": `${cap(side)} arm adapter`,
      disabled: Boolean(s.wiggling),
      onChange: (event) => assign(side, event.target.value || null),
    }, h("option", { value: "" }, "No arm on this side"), options.map((p) => {
      const product = s.adapters.find((a) => a.port === p)?.product;
      const suffix = s.draft[other] === p ? ` · ${other} arm` : p === port && !detected && !s.scanning ? " · not detected" : product ? ` · ${product}` : "";
      return h("option", { value: p }, p + suffix);
    }));
    select.value = port || "";
    const busy = s.wiggling === side;
    return h("div", { class: "pg-slot", dataset: { state, busy: String(busy) } },
      h("div", { class: "pg-slot-head" }, h("p", { class: "pg-slot-name" }, `${cap(side)} arm`), h("p", { class: "pg-slot-state" }, h("i", { class: "pg-dot", "aria-hidden": "true", dataset: { state: state === "ready" ? "ready" : "setup" } }), statusText)),
      h("div", { class: "pg-select-wrap" }, select),
      busy
        ? h("div", { class: "pg-slot-live", role: "status" }, h("p", {}, "Wiggling the gripper. Watch which arm moves."))
        : h("div", { class: "pg-slot-actions" }, h("button", {
          class: "hr-btn hr-btn-secondary hr-btn-sm",
          type: "button",
          disabled: Boolean(s.wiggling) || !detected || connected,
          title: connected ? "Disconnect the robot first." : detected ? "Opens and closes this arm’s gripper a little, then releases it." : "Pick a detected adapter first.",
          onClick: () => wiggle(side, port),
        }, "Wiggle")),
    );
  }

  // An adapter the other side holds swaps sides, after asking.
  async function assign(side, port) {
    const other = SIDES.find((x) => x !== side);
    if (port && s.draft[other] === port) {
      const ok = await confirmDialog({ title: "Swap arms?", body: `${port} is the ${other} arm’s adapter. Swap the two sides?`, confirm: "Swap" });
      if (!ok || s.closed) return renderArms();
      s.draft[other] = s.draft[side] || null;
    }
    s.draft[side] = port;
    renderArms();
    renderFoot();
  }

  async function wiggle(side, port) {
    s.wiggling = side;
    renderArms();
    renderFoot();
    try {
      const result = await api.identify(robot.arm, port);
      const moved = `The gripper on ${port} moved ${Math.round((result.moved * 180) / Math.PI)}°`;
      if (s.closed) return;
      if (result.released) toast(`${moved} and was released.`);
      else errorToast(`${moved}; its torque off was NOT confirmed. Cut motor power if it still holds.`);
    } catch (error) {
      errorToast(error);
    } finally {
      s.wiggling = null;
      if (!s.closed) {
        renderArms();
        renderFoot();
      }
    }
  }

  // ----- 02 Check -----

  function renderCheck() {
    checkBtn.disabled = s.checking || s.isNew || dirty() || robot.name !== api.selected;
    checkBtn.textContent = s.checking ? "Checking…" : "Check";
    const note = s.isNew || dirty()
      ? "Save first, then check the robot."
      : robot.name !== api.selected
        ? "Select this robot to check it."
        : "Read-only: each motor answers, no faults, temperatures, each camera sends a picture. Torque stays off.";
    checkEl.replaceChildren(sectionHead("02", "check-title", "Check", checkBtn), h("p", { class: "pg-section-note" }, note), s.report ? reportList(s.report) : "");
  }

  function reportList(report) {
    const rows = [];
    for (const [side, entry] of Object.entries(report.arms || {})) {
      const temps = Object.values(entry.temperatures || {}).filter((t) => t != null);
      const detail = entry.error || [Object.keys(entry.faults || {}).length ? `faults: ${Object.entries(entry.faults).map(([j, f]) => `${j} ${f}`).join(", ")}` : "", Object.keys(entry.too_hot || {}).length ? `too hot: ${Object.keys(entry.too_hot).join(", ")}` : "", temps.length ? `max ${Math.round(Math.max(...temps))} °C` : ""].filter(Boolean).join(" · ");
      rows.push(h("li", {}, h("span", { class: "pg-check-name" }, `${cap(side)} arm`), h("span", { class: "pg-check-state" }, entry.ok ? "OK" : "FAIL"), h("span", { class: "pg-check-detail" }, detail)));
    }
    for (const [name, entry] of Object.entries(report.cameras || {})) {
      rows.push(h("li", {}, h("span", { class: "pg-check-name" }, `${cap(name)} camera`), h("span", { class: "pg-check-state" }, entry.ok ? "OK" : "FAIL"), h("span", { class: "pg-check-detail" }, entry.ok ? entry.size.join("×") : entry.error || "")));
    }
    return h("ul", { class: "pg-check" }, rows);
  }

  async function runCheck() {
    s.checking = true;
    s.report = null;
    renderCheck();
    try {
      s.report = await api.check();
      if (!s.closed) (s.report.ok ? toast : errorToast)(s.report.ok ? `${robot.name}: everything answers` : `${robot.name}: the check found a problem`);
    } catch (error) {
      errorToast(error);
    } finally {
      s.checking = false;
      if (!s.closed) renderCheck();
    }
  }

  // ----- Save / delete / close -----

  async function save() {
    s.saving = true;
    renderFoot();
    try {
      const body = { arms: Object.fromEntries(SIDES.map((side) => [side, s.draft[side] || null])) };
      if (s.isNew) Object.assign(body, { arm: robot.arm, record: true });
      const saved = await api.saveRobot(robot.name, body);
      s.saved = { ...saved.arms };
      s.draft = { ...saved.arms };
      toast(s.isNew ? `Created ${robot.name}` : "Saved");
      if (s.isNew && !api.selected) api.selected = robot.name;
      s.isNew = false;
      await refresh();
    } catch (error) {
      errorToast(error);
    }
    s.saving = false;
    if (!s.closed) renderAll();
  }

  async function remove() {
    const ok = await confirmDialog({
      title: `Delete ${robot.name}?`,
      body: "Its definition file is removed. Recordings and the arms’ own settings are kept.",
      confirm: "Delete robot",
    });
    if (!ok || s.closed) return;
    try {
      await api.deleteRobot(robot.name);
    } catch (error) {
      return errorToast(error);
    }
    toast(`Deleted ${robot.name}`);
    dialog.close();
  }

  async function requestClose() {
    if (s.wiggling) return toast("Wait for the gripper to finish.");
    if (dirty() && SIDES.some((side) => s.draft[side])) {
      const ok = await confirmDialog({
        title: "Discard unsaved changes?",
        body: s.isNew ? `${robot.name} hasn’t been created yet.` : "Your adapter changes haven’t been saved.",
        confirm: "Discard",
        cancel: "Keep editing",
      });
      if (!ok) return;
    }
    dialog.close();
  }

  renderAll();
  rescan();
}
