// Loaded blocking in <head>: apply the saved theme before first paint so there is no flash.
(function () {
  var t = "system";
  try {
    t = localStorage.getItem("playground-theme") || "system";
  } catch (e) {
    /* storage blocked: follow the system */
  }
  if (t === "system") t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = t;
})();
