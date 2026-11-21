// Small shared UI pieces for dashboard pg: a DOM builder, toasts, and confirm / prompt dialogs.

// h("button", { class: "hr-btn", onClick }, "Label") — props starting with "on" become listeners,
// `value` / `checked` are set as properties, false/null props and children are skipped.
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "value" || key === "checked") continue;
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  if (props && "value" in props) el.value = props.value ?? "";
  if (props && "checked" in props) el.checked = Boolean(props.checked);
  return el;
}

// Inline glyphs, single ink (currentColor), drawn at a 1.5px stroke at 16px.
// Glyphs on a 24-unit grid (gear, camera: Lucide, ISC licence) use a 2.25-unit stroke to match.
const GLYPHS = {
  chevron: '<path d="M4 6l4 4 4-4"/>',
  check: '<path d="M3.5 8.5l3 3 6-7"/>',
  plus: '<path d="M8 3v10M3 8h10"/>',
  close: '<path d="M4 4l8 8M12 4l-8 8"/>',
  copy: '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5V3.5a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2"/>',
  sliders: '<path d="M2.5 5h5.5M12 5h1.5M2.5 11h1.5M8 11h5.5"/><circle cx="10" cy="5" r="1.75"/><circle cx="6" cy="11" r="1.75"/>',
  swap: '<path d="M11 2.5L13.5 5 11 7.5M13.5 5h-11M5 8.5L2.5 11 5 13.5M2.5 11h11"/>',
  film: '<rect x="2" y="3.5" width="12" height="9" rx="1.5"/><path d="M6.75 6.25v3.5L9.75 8z"/>',
  gear: {
    grid: 24,
    svg: '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
  },
  camera: {
    grid: 24,
    svg: '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
  },
};
export function glyph(name) {
  const def = typeof GLYPHS[name] === "string" ? { grid: 16, svg: GLYPHS[name] } : GLYPHS[name];
  const span = document.createElement("span");
  span.className = "pg-glyph";
  span.setAttribute("aria-hidden", "true");
  span.innerHTML = `<svg viewBox="0 0 ${def.grid} ${def.grid}" fill="none" stroke="currentColor" stroke-width="${1.5 * (def.grid / 16)}" stroke-linecap="round" stroke-linejoin="round">${def.svg}</svg>`;
  return span;
}

// An error's message, whatever was thrown.
export const messageOf = (error) => (error instanceof Error ? error.message : String(error ?? ""));

// ---------- Toasts ----------
let toastRegion;
export function toast(message) {
  if (!toastRegion) {
    toastRegion = h("div", { class: "pg-toasts", role: "status", "aria-live": "polite" });
    document.body.append(toastRegion);
  }
  // Toasts must sit above an open modal dialog, which lives in the top layer.
  const host = [...document.querySelectorAll("dialog[open]")].pop() || document.body;
  if (toastRegion.parentElement !== host) host.append(toastRegion);
  const item = h("p", { class: "pg-toast" }, message);
  toastRegion.append(item);
  setTimeout(() => {
    item.classList.add("is-leaving");
    setTimeout(() => item.remove(), 320);
  }, 3200);
}

// Errors stack on the right, under the header and clear of the task bar and Run, and stay until dismissed, their
// text selectable, with Copy and Dismiss buttons: the operator often needs the exact message (to send it, or to read
// it after looking at the arms). The same message again bumps a ×N count on the one already shown instead of
// stacking a copy, and at most ERRORS_SHOWN stay (the oldest goes), so a fault that repeats every tick can never
// flood the page.
const ERRORS_SHOWN = 6;
let errorRegion;
export function errorToast(message) {
  persistentToast(message instanceof Error ? message.message : String(message ?? ""));
}

function persistentToast(shown) {
  const text = shown.trim() || "Something went wrong.";
  if (!errorRegion) {
    errorRegion = h("div", { class: "pg-error-toasts", role: "alert", "aria-live": "assertive" });
  }
  const host = [...document.querySelectorAll("dialog[open]")].pop() || document.body;
  if (errorRegion.parentElement !== host) host.append(errorRegion);
  const region = errorRegion;
  const same = [...region.querySelectorAll(".pg-error-toast")].find((e) => e.dataset.message === text);
  if (same) {
    const count = Number(same.dataset.count ?? "1") + 1;
    same.dataset.count = String(count);
    same.querySelector(".pg-error-count").textContent = `×${count}`;
    region.append(same); // newest last
    return;
  }
  const copyLabel = h("span", {}, "Copy");
  const copy = h("button", { class: "pg-error-btn", type: "button", "aria-label": "Copy the error" }, glyph("copy"), copyLabel);
  const dismiss = h(
    "button",
    { class: "pg-error-btn", type: "button", "aria-label": "Dismiss the error notification" },
    "Dismiss",
  );
  const item = h(
    "div",
    { class: "pg-error-toast", dataset: { message: text, count: "1" } },
    h("p", { class: "pg-error-text" }, text),
    h("div", { class: "pg-error-actions" }, h("span", { class: "pg-error-count" }), copy, dismiss),
  );
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      copyLabel.textContent = "Copied";
    } catch {
      // No clipboard (plain http off this computer, or no permission): select the text so Cmd+C copies it.
      const range = document.createRange();
      range.selectNodeContents(item.querySelector(".pg-error-text"));
      getSelection()?.removeAllRanges();
      getSelection()?.addRange(range);
      copyLabel.textContent = "Selected";
    }
    setTimeout(() => (copyLabel.textContent = "Copy"), 1500);
  });
  dismiss.addEventListener("click", () => {
    item.remove();
    renderDismissAll(region);
  });
  region.append(item);
  while (region.querySelectorAll(".pg-error-toast").length > ERRORS_SHOWN) region.querySelector(".pg-error-toast")?.remove();
  renderDismissAll(region);
}

function renderDismissAll(region) {
  const items = region.querySelectorAll(".pg-error-toast");
  let all = region.querySelector(".pg-error-dismiss-all");
  if (items.length < 2) return all?.remove();
  if (!all) {
    all = h("button", { class: "pg-error-dismiss-all", type: "button" }, "Dismiss all");
    all.addEventListener("click", () => {
      region.querySelectorAll(".pg-error-toast").forEach((e) => e.remove());
      renderDismissAll(region);
    });
    region.prepend(all);
  }
}

// ---------- Dialogs ----------
// A modal built on demand; resolves with the chosen value and removes itself. build(done, titleId)
// returns the contents; done(value) closes it. Escape or a click outside resolves null.
let dialogCount = 0;
export function smallDialog(build, size = "pg-dialog-sm") {
  return new Promise((resolve) => {
    const titleId = `pg-dialog-title-${++dialogCount}`;
    const dialog = h("dialog", { class: `pg-dialog ${size}`, "aria-labelledby": titleId });
    let result = null;
    const done = (value) => {
      result = value;
      dialog.close();
    };
    dialog.append(build(done, titleId));
    dialog.addEventListener("close", () => {
      dialog.remove();
      resolve(result);
    });
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
    document.body.append(dialog);
    dialog.showModal();
  });
}

// confirmDialog({ title, body, confirm, cancel }) → true when confirmed.
export function confirmDialog({ title, body, confirm = "Confirm", cancel = "Cancel" }) {
  return smallDialog((done, titleId) =>
    h(
      "div",
      { class: "pg-dialog-body" },
      h("h2", { class: "title", id: titleId }, title),
      body && h("p", { class: "pg-dialog-text" }, body),
      h(
        "div",
        { class: "pg-dialog-actions" },
        h("button", { class: "hr-btn hr-btn-secondary", type: "button", onClick: () => done(false) }, cancel),
        h("button", { class: "hr-btn hr-btn-primary", type: "button", autofocus: true, onClick: () => done(true) }, confirm),
      ),
    ),
  ).then(Boolean);
}
