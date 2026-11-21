// The camera panel's first-open guide: how to drag a camera into place. Shown once per browser.

import { glyph, h } from "./ui.js";

const SEEN = "playground-camera-guide-seen";

// An illustrative drag only: never uses a real device or changes an assignment.
export class CameraAssignmentGuide {
  element;
  shown = false;

  constructor() {
    const diagram = h("div", { class: "pg-cams-guide-demo", "aria-hidden": "true" },
      h("span", { class: "pg-cams-guide-source" }, glyph("camera"), "Camera"),
      h("span", { class: "pg-cams-guide-arrow" }, "↓"),
      h("div", { class: "pg-cams-guide-targets" }, ...["Left", "Top", "Right"].map((label) =>
        h("span", { class: "pg-cams-guide-target" }, label))),
      h("span", { class: "pg-cams-guide-ghost" }, glyph("camera")));
    this.element = h("section", { class: "pg-cams-guide", hidden: true, "aria-labelledby": "camera-guide-title" },
      h("h3", { id: "camera-guide-title" }, "Drag a camera into place"),
      h("p", null, "Drag a camera from this panel onto the Left, Top or Right view on the page."),
      diagram,
      h("p", { class: "pg-cams-guide-alternative" }, "You can also select a camera, then select a view."),
      h("button", { type: "button", class: "hr-btn hr-btn-secondary hr-btn-sm", onClick: () => this.hide() }, "Got it"));
  }

  open() {
    let seen = this.shown;
    try { seen ||= localStorage.getItem(SEEN) === "true"; } catch { /* Private storage can be unavailable. */ }
    this.element.hidden = seen;
    if (seen) return;
    this.shown = true;
    try { localStorage.setItem(SEEN, "true"); } catch { /* Remember for this page if persistence is blocked. */ }
  }

  hide() { this.element.hidden = true; }
}
