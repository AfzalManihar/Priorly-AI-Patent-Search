/* History & memory: the signed-in user's own searches and notes. All user text is written with textContent. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var PAGE = 15, offset = 0, total = 0, memoryOn = false;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function btn(label, onClick, cls) {
    var b = el("button", "btn " + (cls || "ghost"), label);
    b.type = "button"; b.addEventListener("click", onClick);
    return b;
  }
  function say(id, text, bad) {
    var n = $(id); if (!n) return;
    n.textContent = text || ""; n.style.color = bad ? "var(--danger, #c44)" : "";
  }
  function when(iso) { var d = new Date(iso); return isNaN(d) ? "" : d.toLocaleString(); }
  function fail(id) { return function (e) { say(id, e && e.message ? e.message : "Something went wrong.", true); }; }

  function openSearch(searchId, noteId) {
    say(noteId, "Opening saved search…");
    PriorlyAPI.get("/api/search/" + encodeURIComponent(searchId)).then(function (saved) {
      say(noteId, "");
      if (!window.PriorlyOpenSaved || !window.PriorlyOpenSaved(saved)) say(noteId, "That search has no stored results.", true);
    }).catch(fail(noteId));
  }

  /* ---------- history ---------- */
  function histRow(item) {
    var row = el("div", "mem-item");
    var text = el("div", "mem-text");
    text.appendChild(el("div", "mem-claim", item.claim));
    text.appendChild(el("div", "note", when(item.created) + " · " + item.results + " results"));
    var actions = el("div", "mem-actions");
    actions.appendChild(btn("Open", function () { openSearch(item.search_id, "histNote"); }));
    actions.appendChild(btn("Re-run", function () {
      if (window.PriorlyOpenSaved) {
        PriorlyAPI.get("/api/search/" + encodeURIComponent(item.search_id)).then(function (saved) {
          document.querySelector('.menu [data-view="search"]').click();
          $("claim").value = (saved.request || {}).claim || "";
          $("claim").dispatchEvent(new Event("input"));
          $("runBtn").focus();
          say("histNote", "Claim loaded. Press Run to search again; this uses new tokens.");
        }).catch(fail("histNote"));
      }
    }));
    actions.appendChild(btn("Delete", function () {
      if (!window.confirm("Delete this search and any report built from it?")) return;
      PriorlyAPI.del("/api/search/" + encodeURIComponent(item.search_id)).then(function () { loadHistory(true); }).catch(fail("histNote"));
    }, "ghost danger"));
    row.appendChild(text); row.appendChild(actions);
    return row;
  }
  function loadHistory(reset) {
    if (reset) { offset = 0; $("histList").textContent = ""; }
    say("histNote", "Loading…");
    var q = $("histQuery").value.trim();
    PriorlyAPI.get("/api/history?limit=" + PAGE + "&offset=" + offset + (q ? "&q=" + encodeURIComponent(q) : "")).then(function (page) {
      total = page.total;
      page.items.forEach(function (i) { $("histList").appendChild(histRow(i)); });
      offset += page.items.length;
      $("histMore").hidden = offset >= total;
      say("histNote", total ? "Showing " + offset + " of " + total : "No saved searches yet.");
    }).catch(fail("histNote"));
  }

  /* ---------- notes ---------- */
  function noteRow(n) {
    var row = el("div", "mem-item");
    var text = el("div", "mem-text");
    text.appendChild(el("div", "mem-claim", n.text));
    text.appendChild(el("div", "note", "Updated " + when(n.updated)));
    var actions = el("div", "mem-actions");
    actions.appendChild(btn("Edit", function () {
      var next = window.prompt("Edit note", n.text);
      if (next == null || !next.trim()) return;
      PriorlyAPI.put("/api/memory/notes/" + encodeURIComponent(n.id), { text: next.trim() }).then(loadNotes).catch(fail("noteNote"));
    }));
    actions.appendChild(btn("Delete", function () {
      PriorlyAPI.del("/api/memory/notes/" + encodeURIComponent(n.id)).then(loadNotes).catch(fail("noteNote"));
    }, "ghost danger"));
    row.appendChild(text); row.appendChild(actions);
    return row;
  }
  function paintMemoryState() {
    $("memStatus").textContent = memoryOn
      ? "Memory is ON. Your notes are matched against your claims on this server and are never sent to the AI model."
      : "Memory is OFF. Turn it on to save notes. Notes you saved earlier stay hidden until you do.";
    $("memToggle").textContent = memoryOn ? "Turn memory off" : "Turn memory on";
    $("noteAdd").disabled = !memoryOn; $("noteText").disabled = !memoryOn;
    $("noteText").placeholder = memoryOn
      ? "Write a note, for example a technology area you work in or how you prefer results"
      : "Turn memory on to write notes";
  }
  function loadNotes() {
    PriorlyAPI.get("/api/memory/notes").then(function (r) {
      memoryOn = !!r.memory_enabled; paintMemoryState();
      var list = $("noteList"); list.textContent = "";
      r.notes.forEach(function (n) { list.appendChild(noteRow(n)); });
      if (!r.notes.length) list.appendChild(el("p", "note", "No notes yet."));
      say("noteNote", "");
    }).catch(fail("noteNote"));
  }
  $("noteForm").addEventListener("submit", function (e) {
    e.preventDefault();
    var t = $("noteText").value.trim(); if (!t) return;
    PriorlyAPI.post("/api/memory/notes", { text: t }).then(function () { $("noteText").value = ""; loadNotes(); }).catch(fail("noteNote"));
  });
  $("noteClear").addEventListener("click", function () {
    if (!window.confirm("Delete every saved note? This cannot be undone.")) return;
    PriorlyAPI.del("/api/memory/notes?confirm=yes").then(loadNotes).catch(fail("noteNote"));
  });
  $("memToggle").addEventListener("click", function () {
    var turningOn = !memoryOn;
    if (turningOn && !window.confirm("Allow Priorly to save your personal notes and show them while you search? They stay in your account and are not sent to the AI model.")) return;
    PriorlyAPI.getProfile().then(function (r) {  // send the whole profile so no other field is blanked
      var p = r.profile;
      return PriorlyAPI.saveProfile({ email: p.email || "", display_name: p.display_name || "", learning_topic: p.learning_topic || "", memory_enabled: turningOn });
    }).then(loadNotes).catch(fail("noteNote"));
  });
  $("histMore").addEventListener("click", function () { loadHistory(false); });
  var timer = null;
  $("histQuery").addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(function () { loadHistory(true); }, 350); });

  /* refresh whenever the view becomes visible (navigation, hash restore or Re-run) */
  var view = $("v-memory");
  new MutationObserver(function () { if (view.classList.contains("on")) { loadHistory(true); loadNotes(); } })
    .observe(view, { attributes: true, attributeFilter: ["class"] });
  if (view.classList.contains("on")) { loadHistory(true); loadNotes(); }
})();
