/* Poll, diff, patch.
 *
 * The rule this file exists to obey: never touch a panel whose content did not
 * change. An earlier board re-wrote the whole document every two seconds, which
 * looked fine and quietly destroyed keyboard focus, text selection and scroll
 * position every time it fired. Here the server hands back a content hash per
 * panel; a panel whose hash matches what is already on screen is skipped
 * entirely, so an idle board performs zero DOM writes forever.
 */
(function () {
  "use strict";

  var PANELS = ["metric", "call", "guard", "wall"];
  var INTERVAL = 2000;
  var stateKey = document.body.dataset.stateKey || "";
  var dot = document.getElementById("poll-dot");
  var label = document.getElementById("poll-state");
  var keyOut = document.getElementById("poll-key");
  var misses = 0;

  function flash() {
    if (!dot) return;
    dot.classList.remove("is-stale");
    dot.classList.add("is-hit");
    setTimeout(function () { dot.classList.remove("is-hit"); }, 320);
  }

  function patch(payload) {
    if (payload.state_key === stateKey) return false;
    stateKey = payload.state_key;
    document.body.dataset.stateKey = stateKey;
    if (keyOut) keyOut.textContent = stateKey;

    var touched = 0;
    PANELS.forEach(function (name) {
      var incoming = payload.panels[name];
      var node = document.getElementById("panel-" + name);
      if (!incoming || !node) return;
      // Scroll position is ours to keep; the server does not know about it.
      var scroller = node.querySelector(".wall-grid, .rfs");
      var top = scroller ? scroller.scrollTop : 0;
      if (node.dataset.key === incoming.key) return;
      node.dataset.key = incoming.key;
      node.innerHTML = incoming.html;
      var next = node.querySelector(".wall-grid, .rfs");
      if (next && top) next.scrollTop = top;
      touched++;
    });
    return touched > 0;
  }

  function poll() {
    fetch("/api/state", { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then(function (payload) {
        misses = 0;
        if (label) label.textContent = "live";
        if (dot) dot.classList.remove("is-stale");
        if (patch(payload)) flash();
      })
      .catch(function () {
        misses++;
        // One dropped poll on conference wifi is noise. Three is a fact.
        if (misses >= 3) {
          if (dot) dot.classList.add("is-stale");
          if (label) label.textContent = "stale — last good render held";
        }
      });
  }

  /* The clock ticks locally so the server payload stays free of wall-clock and
     the state key stays stable when nothing has actually happened. */
  function tick() {
    var el = document.querySelector(".clock[data-since]");
    if (!el) return;
    var since = parseFloat(el.dataset.since);
    if (!since) return;
    var s = Math.max(0, Math.floor(Date.now() / 1000 - since));
    el.textContent =
      String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }

  setInterval(poll, INTERVAL);
  setInterval(tick, 1000);
  tick();
})();
