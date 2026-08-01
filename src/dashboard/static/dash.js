/* EXPEDITOR — polling and clocks.
 *
 * Three jobs and no framework:
 *
 *  1. Re-fetch /board every 2s and swap it in *only if the bytes changed*.
 *     Nothing time-varying is rendered server-side, so an idle board produces
 *     an identical fragment and the DOM is left alone — which is what keeps a
 *     tabbed-to Approve button focused instead of being destroyed twice a
 *     second. When a write is unavoidable, focus is restored by data-fk.
 *
 *  2. Tick the elapsed clocks locally from an epoch, so the numbers move at
 *     1Hz without a request and keep moving if the network dies.
 *
 *  3. Intercept the Approve/Deny form so a press is instant. If anything here
 *     throws or JS never loads, the form is a real POST to a real route and
 *     the button still works. That path is the fallback for the voice
 *     approval channel; it does not get to depend on this file.
 */
(function () {
  "use strict";

  var POLL_MS = 2000;
  var board = document.getElementById("board");
  var sync = document.getElementById("sync");
  var lastKey = null;
  var misses = 0;

  /* The server still renders a first value into every clock, so the board is
     readable before this file runs and with JavaScript off entirely. Those
     digits change every second, though, and comparing them would make every
     poll look like a change — and every poll would then blow away whatever the
     operator had tabbed to. So the comparison ignores the inside of anything
     carrying data-elapsed; JS owns those from the epoch. */
  var CLOCK_TEXT = /(data-elapsed="[^"]*"[^>]*>)[^<]*/g;
  function stateKey(html) {
    return html.replace(CLOCK_TEXT, "$1");
  }

  /* -- clocks ---------------------------------------------------------- */

  function mmss(sec) {
    sec = Math.max(0, Math.floor(sec));
    var m = Math.floor(sec / 60), s = sec % 60;
    return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }

  function tickClocks() {
    var now = Date.now() / 1000;
    var nodes = board.querySelectorAll("[data-elapsed]");
    for (var i = 0; i < nodes.length; i++) {
      var started = parseFloat(nodes[i].getAttribute("data-elapsed"));
      if (!isNaN(started)) nodes[i].textContent = mmss(now - started);
    }
    var wall = document.getElementById("wallclock");
    if (wall) wall.textContent = new Date().toLocaleTimeString([], { hour12: false });
  }

  /* -- polling --------------------------------------------------------- */

  function applyFragment(html) {
    var key = stateKey(html);
    if (key === lastKey) return;
    lastKey = key;

    var active = document.activeElement;
    var fk = active && active.getAttribute ? active.getAttribute("data-fk") : null;
    var scrollLeft = {};
    var rails = board.querySelectorAll(".rec-rail");
    for (var i = 0; i < rails.length; i++) scrollLeft[i] = rails[i].scrollLeft;

    board.innerHTML = html;

    if (fk) {
      var again = board.querySelector('[data-fk="' + CSS.escape(fk) + '"]');
      if (again) again.focus();
    }
    var newRails = board.querySelectorAll(".rec-rail");
    for (var j = 0; j < newRails.length; j++) {
      if (scrollLeft[j]) newRails[j].scrollLeft = scrollLeft[j];
    }
    tickClocks();
  }

  function poll() {
    return fetch("/board", { headers: { "X-Requested-With": "poll" }, cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (html) {
        misses = 0;
        if (sync) sync.classList.remove("stale");
        applyFragment(html);
      })
      .catch(function () {
        /* A dropped poll costs two seconds and nothing else — the last render
           stays on the projector. Only say so after a few in a row. */
        misses += 1;
        if (misses >= 2 && sync) sync.classList.add("stale");
      });
  }

  /* -- approve / deny -------------------------------------------------- */

  document.addEventListener("submit", function (ev) {
    var form = ev.target;
    if (!form.matches || !form.matches(".ticket-act")) return;
    var pressed = document.activeElement;
    var decision =
      pressed && pressed.name === "decision" ? pressed.value : "deny";

    ev.preventDefault();
    var body = new URLSearchParams();
    body.set("decision", decision);

    /* Lock both buttons so a double press cannot be read as a second answer. */
    var buttons = form.querySelectorAll("button");
    for (var i = 0; i < buttons.length; i++) buttons[i].disabled = true;

    fetch(form.action, { method: "POST", body: body, redirect: "follow" })
      .then(poll)
      .catch(function () {
        /* Network refused the decision — un-press so the operator can retry
           rather than believing an unsent answer was recorded. */
        for (var k = 0; k < buttons.length; k++) buttons[k].disabled = false;
      });
  });

  /* -- go -------------------------------------------------------------- */

  tickClocks();
  setInterval(tickClocks, 1000);
  /* One immediate poll so lastKey is set from a real fragment rather than
     from the inline render, which the browser will have re-serialised. */
  poll();
  setInterval(poll, POLL_MS);
  /* Catch up immediately when the laptop wakes or the tab comes forward. */
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) poll();
  });
})();
