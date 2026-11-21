// The first-run guide: a non-modal coach card that follows the real controls (the robot menu, New robot, the setup
// dialog's Rescan), with a "?" launcher to replay it. It starts on its own until finished or skipped once.

import { h } from "./ui.js";

const COMPLETED = "playground-tutorial-complete";

const TITLES = ["Add your first robot", "Create a robot", "Make it yours", "Pick each arm’s adapter"];
const COPY = [
  "Open the robot menu. We’ll show you how to connect your arms.",
  "Choose New robot to add your Metal Arms or Maker Arms.",
  "Give your robot a name, choose its arm type, then continue.",
  "Plug in the CAN adapters and press Rescan. Pick the left and right arm’s adapter; Wiggle opens that arm’s gripper so you can see which one it is. Then save, and assign the cameras with the camera button.",
];

export function initTutorial() {
  const replay = h("button", { type: "button", class: "pg-icon-btn pg-guide-launcher", "aria-label": "Setup guide", title: "Setup guide", onClick: start }, "?");
  document.body.append(replay);
  const ring = h("div", { class: "pg-guide-ring", "aria-hidden": "true" });
  const count = h("p", { class: "pg-guide-count" });
  const title = h("h2", { id: "setup-guide-title" });
  const text = h("p", { class: "pg-guide-copy" });
  const next = h("button", { type: "button", class: "hr-btn hr-btn-primary hr-btn-sm", "data-tutorial-control": "", onClick: advance }, "Next");
  const card = h("section", { class: "pg-guide-card", role: "region", "aria-labelledby": title.id, "aria-live": "polite" },
    count, title, text, h("div", { class: "pg-guide-actions" },
      h("button", { type: "button", class: "hr-btn hr-btn-ghost hr-btn-sm", "data-tutorial-control": "", onClick: finish }, "Skip tutorial"), next));
  const overlay = h("div", { class: "pg-guide", popover: "manual", "data-tutorial-control": "" }, ring, card);
  let active = false;
  let frame = 0;
  let step = -1;
  let target = null;
  let highlighted = null;

  function finish() {
    active = false; cancelAnimationFrame(frame);
    highlighted?.classList.remove("pg-guide-glow"); highlighted = null;
    overlay.hidePopover?.(); overlay.remove(); card.remove();
    try { localStorage.setItem(COMPLETED, "true"); } catch { /* not remembered: it starts again next time */ }
    replay.focus();
  }
  function start() {
    if (active) return;
    active = true; step = -1;
    overlay.append(card);
    document.body.append(overlay);
    overlay.showPopover?.();
    draw();
  }
  const find = (selector) => document.querySelector(selector);
  function advance() {
    if (step <= 1) { target?.click(); return; }
    if (step === 2) {
      const form = find('[data-guide="robot-create"]');
      if (form?.reportValidity()) form.requestSubmit();
      return;
    }
    finish();
  }
  function draw() {
    if (!active) return;
    frame = requestAnimationFrame(draw);
    const dialog = [...document.querySelectorAll("dialog[open]")].at(-1);
    const host = dialog ?? document.body;
    if (overlay.parentElement !== host) {
      overlay.hidePopover?.(); host.append(overlay); overlay.showPopover?.();
    }
    let current = 0;
    if (find('[data-guide="robot-setup"][open]')) { current = 3; target = find('[data-guide="usb-device"]'); }
    else if (find('[data-guide="robot-create"]')) { current = 2; target = find('[data-guide="robot-create"] input'); }
    else if (find("#robot-menu:not([hidden])")) { current = 1; target = find('[data-guide="new-robot"]'); }
    else target = find('[data-guide="robot-picker"]');
    const modalRect = dialog?.getBoundingClientRect();
    const besideModal = modalRect && modalRect.left >= 352;
    const inline = current === 2 && dialog && !besideModal;
    card.classList.toggle("is-inline", Boolean(inline));
    if (inline) {
      const form = find('[data-guide="robot-create"]');
      if (card.parentElement !== form) form.insertBefore(card, form.firstElementChild?.nextSibling ?? null);
    } else if (card.parentElement !== overlay) overlay.append(card);
    if (current !== step) {
      step = current;
      count.textContent = `GET CONNECTED · ${step + 1} OF 4`;
      title.textContent = TITLES[step];
      text.textContent = COPY[step];
      next.textContent = step === 2 ? "Continue" : step === 3 ? "Got it" : "Next";
    }
    if (highlighted !== target) {
      highlighted?.classList.remove("pg-guide-glow"); highlighted = target;
      if (step === 3) highlighted?.classList.add("pg-guide-glow");
    }
    const rect = target?.getBoundingClientRect();
    ring.hidden = !rect?.width || !rect.height;
    if (!ring.hidden && rect) {
      Object.assign(ring.style, { left: `${rect.left - 7}px`, top: `${rect.top - 7}px`, width: `${rect.width + 14}px`, height: `${rect.height + 14}px` });
    }
    const width = Math.min(320, innerWidth - 32);
    const height = card.offsetHeight;
    let left = rect?.width ? rect.right - width : innerWidth - width - 16;
    if (besideModal && current >= 2) left = modalRect.left - width - 16;
    left = Math.max(16, Math.min(left, innerWidth - width - 16));
    let top = rect?.height ? rect.bottom + 22 : 90;
    if (top + height > innerHeight - 16) top = (rect?.top ?? innerHeight) - height - 22;
    Object.assign(card.style, { width: `${width}px`, left: `${left}px`, top: `${Math.max(16, Math.min(top, innerHeight - height - 16))}px` });
  }
  document.addEventListener("keydown", (event) => {
    if (active && event.key === "Escape" && !document.querySelector("dialog[open]")) finish();
  });
  let completed = false;
  try { completed = localStorage.getItem(COMPLETED) === "true"; } catch { /* storage blocked: show it */ }
  if (!completed) start();
}
