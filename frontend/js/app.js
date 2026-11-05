(function () {
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var RRF_K = 60;
  var running = false;
  var live = false;            /* true when the API answered /api/health */
  var lastRun = null;
  var shortlist = {};
  var curEval = null;
  var savedRemote = [];

  /* Demo only (no API): these toggles hide results the real stage would have found. */
  var NEEDS_DECOMP = { "US 8,904,567 B1": 1, "US 9,230,771 B1": 1 };
  var NEEDS_HYDE = { "US 8,332,104 B2": 1, "US 9,876,540 B2": 1 };

  var STAGES = [
    { key: "decompose", n: "Decompose", on: function (o) { return o.decompose; } },
    { key: "hyde", n: "Expand", on: function (o) { return o.hyde; } },
    { key: "retrieve", n: "Retrieve", on: function () { return true; } },
    { key: "fuse", n: "Fuse", on: function (o) { return o.hybrid; } },
    { key: "rerank", n: "Rerank", on: function (o) { return o.rerank; } },
    { key: "explain", n: "Explain", on: function () { return true; } }
  ];

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function toast(msg) {
    var t = $("#toast");
    t.textContent = msg;
    t.classList.add("on");
    clearTimeout(toast.h);
    toast.h = setTimeout(function () { t.classList.remove("on"); }, 2600);
  }

  /* ---------- Theme ---------- */
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("priorly-theme", t); } catch (e) {}
    $("#themeIcon").innerHTML = icon(t === "dark" ? "sun" : "moon");
    $("#themeLabel").textContent = t === "dark" ? "Light mode" : "Dark mode";
  }
  setTheme(document.documentElement.getAttribute("data-theme") || "light");
  $("#themeBtn").addEventListener("click", function () {
    setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
    if ($("#v-eval").classList.contains("on") && curEval) renderChart();
  });

  /* ---------- Mode bar: live API or demo data ---------- */
  function setMode(health) {
    live = !!health;
    var bar = $("#modeBar");
    if (live) {
      bar.className = "modebar live";
      bar.innerHTML = '<span class="dot"></span><span><b>Live API.</b> ' + health.corpus.patents.toLocaleString() +
        " patents, " + health.corpus.chunks.toLocaleString() + " chunks. Vector: " + esc(health.dense_backend) +
        ". Rerank: " + esc(health.rerank_backend) + ". LLM: " + esc(health.llm_backend) + ".</span>";
      $("#wsInfo").textContent = health.corpus.patents.toLocaleString() + " patents indexed";
    } else {
      bar.className = "modebar demo";
      bar.innerHTML = '<span class="dot"></span><span><b>Demo data.</b> The API is not reachable, so results come from a built-in ' +
        "sample. Start the backend (<code>uvicorn app.main:app</code>) and reload to run real searches.</span>";
    }
  }

  /* ---------- Navigation ---------- */
  function show(view) {
    $$(".view").forEach(function (v) { v.classList.toggle("on", v.id === "v-" + view); });
    $$(".menu button").forEach(function (b) {
      if (b.dataset.view === view) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
    });
    if (view === "eval") renderEval();
    if (view === "saved") renderSaved();
    if (view === "settings") loadSettings();
    try { history.replaceState(null, "", "#" + view); } catch (e) {}
    window.scrollTo(0, 0);
  }
  $$(".menu button").forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.view); }); });

  /* ---------- Search setup ---------- */
  var claim = $("#claim");
  claim.value = window.CLAIM;
  function count() { $("#charCount").textContent = claim.value.length.toLocaleString() + " characters"; }
  claim.addEventListener("input", count);
  count();
  claim.addEventListener("keydown", function (e) { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); });
  $("#runBtn").addEventListener("click", run);

  function opts() {
    return {
      decompose: $("#optDecompose").checked, hyde: $("#optHyde").checked,
      hybrid: $("#optHybrid").checked, rerank: $("#optRerank").checked, k: +$("#optK").value
    };
  }

  function renderElements(o, els) {
    var host = $("#elements");
    if (!els) {
      host.innerHTML = '<p style="color:var(--ink3);font-size:13.5px">Run a search to see how the claim is split into elements.</p>';
      return;
    }
    if (els.length === 1 && els[0].whole_claim) {
      host.innerHTML = '<div class="el"><span class="chip">All</span><span>Whole claim searched as a single query. Turn on decomposition to search each element.</span></div>';
      return;
    }
    host.innerHTML = els.map(function (e) {
      return '<div class="el"><span class="chip e' + ((e.n - 1) % 3 + 1) + '">E' + e.n + "</span><span>" + esc(e.text) + "</span></div>";
    }).join("");
  }

  /* ---------- Demo ranking (used only when the API is not reachable) ---------- */
  function compute(o) {
    var list = PATENTS.map(function (p) {
      var c = Object.assign({}, p);
      c.rrf = 1 / (RRF_K + p.dense) + 1 / (RRF_K + p.bm25);
      return c;
    });
    list.slice().sort(function (a, b) { return b.rrf - a.rrf; }).forEach(function (p, i) { p.rrfRank = i + 1; });
    list.slice().sort(function (a, b) { return b.score - a.score; }).forEach(function (p, i) { p.rerankRank = i + 1; });
    var pool = list.filter(function (p) {
      if (!o.decompose && NEEDS_DECOMP[p.id]) return false;
      if (!o.hyde && NEEDS_HYDE[p.id]) return false;
      return true;
    });
    var mode = !o.hybrid ? "dense" : (o.rerank ? "rerank" : "rrf");
    pool.sort(function (a, b) {
      if (mode === "dense") return a.dense - b.dense;
      if (mode === "rrf") return b.rrf - a.rrf;
      return b.score - a.score;
    });
    pool.forEach(function (p) {
      p.scoreKind = mode;
      p.missed = p.dense > 10 && mode !== "dense";
      if (mode === "rrf") p.score = p.rrf * 100;
      if (mode === "dense") p.score = Math.max(0.2, 0.93 - p.dense * 0.011);
    });
    return {
      all: pool, shown: pool.slice(0, o.k), degraded: [], searchId: null,
      elements: o.decompose ? ELEMENTS : [{ n: 1, text: claim.value, whole_claim: true }],
      meta: null
    };
  }

  /* ---------- Live API result -> view model ---------- */
  function adapt(data, o) {
    var items = data.results.map(function (r) {
      return {
        id: r.id, title: r.title, assignee: r.assignee, year: r.year, cpc: r.cpc, el: r.el,
        dense: r.dense, bm25: r.bm25, rrfRank: r.rrf_rank, rerankRank: r.rerank_rank,
        score: r.score, scoreKind: r.score_kind, loc: r.loc, passage: r.passage, why: r.why,
        missed: r.missed_by_vector
      };
    });
    return { all: items, shown: items, degraded: data.degraded_stages || [], searchId: data.search_id,
             elements: data.elements, meta: data };
  }

  function scoreView(p) {
    if (p.scoreKind === "rerank") return { v: p.score.toFixed(2), l: "relevance" };
    if (p.scoreKind === "rrf") return { v: p.score.toFixed(2), l: "fusion score" };
    return { v: p.score.toFixed(2), l: "similarity" };
  }

  /* ---------- Progress ---------- */
  function buildProgress() {
    var prog = $("#prog");
    prog.classList.add("on");
    prog.innerHTML = STAGES.map(function (s, i) {
      return '<div class="pstep" data-i="' + i + '">' + s.n + "</div>";
    }).join("");
  }
  function setStep(i, state) {
    var el = $$(".pstep")[i];
    if (!el || i < 0) return;
    var name = STAGES[i].n;
    if (state === "running") { el.className = "pstep run"; el.textContent = name; }
    else if (state === "done") { el.className = "pstep done"; el.innerHTML = icon("check") + name; }
    else if (state === "degraded") { el.className = "pstep"; el.innerHTML = name + " (fallback)"; }
    else { el.className = "pstep"; el.textContent = name + " (off)"; }
  }
  function stageIndex(key) {
    for (var i = 0; i < STAGES.length; i++) if (STAGES[i].key === key) return i;
    return -1;
  }

  /* ---------- Run ---------- */
  function run() {
    if (running) return;
    var o = opts();
    if (claim.value.trim().length < 40) {
      toast("Paste a full claim first. It needs at least 40 characters.");
      claim.focus();
      return;
    }
    running = true;
    $("#runBtn").setAttribute("aria-busy", "true");
    $("#results").innerHTML = "";
    buildProgress();
    var t0 = performance.now();
    var job = live ? runLive(o) : runDemo(o);
    job.then(function (res) {
      finish(o, res, (performance.now() - t0) / 1000);
    }).catch(function (err) {
      running = false;
      $("#runBtn").removeAttribute("aria-busy");
      $("#prog").classList.remove("on");
      showError(err);
    });
  }

  function runLive(o) {
    var options = { decompose: o.decompose, hyde: o.hyde, hybrid: o.hybrid, rerank: o.rerank };
    return PriorlyAPI.searchStream(claim.value.trim(), options, o.k, function (ev) {
      setStep(stageIndex(ev.stage), ev.status);
    }).then(function (data) { return adapt(data, o); });
  }

  function runDemo(o) {
    return new Promise(function (resolve) {
      var i = 0;
      (function tick() {
        if (i > 0) setStep(i - 1, STAGES[i - 1].on(o) ? "done" : "skipped");
        if (i >= STAGES.length) { resolve(compute(o)); return; }
        setStep(i, "running");
        var wait = STAGES[i].on(o) ? 420 : 120;
        i++;
        setTimeout(tick, wait);
      })();
    });
  }

  function showError(err) {
    var msg = (err && err.message) || "Something went wrong. Try again.";
    $("#results").innerHTML = '<div class="empty err" role="alert" style="margin-top:16px"><h3>Search didn\'t finish</h3><p>' +
      esc(msg) + '</p><p style="margin-top:14px"><button class="btn" id="retryBtn">Retry search</button></p></div>';
    $("#retryBtn").addEventListener("click", run);
  }

  function finish(o, res, secs) {
    running = false;
    $("#runBtn").removeAttribute("aria-busy");
    lastRun = { o: o, res: res, claim: claim.value };
    renderElements(o, res.elements);
    renderResults(res, o);
    renderCoverage(res);
    renderRunInfo(res, o, secs);
    setTimeout(function () { $("#prog").classList.remove("on"); }, 900);
    var first = $("#results .r");
    if (first) first.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function chips(p) {
    var n = Math.max(3, p.elN || 0);
    return Array.apply(null, Array(n)).map(function (_, i) {
      var num = i + 1;
      return '<span class="chip' + (p.el.indexOf(num) > -1 ? " e" + ((num - 1) % 3 + 1) : "") + '">E' + num + "</span>";
    }).join("");
  }

  function renderResults(res, o) {
    var els = res.elements.length;
    res.all.forEach(function (p) { p.elN = els > 1 ? els : 0; });
    var parts = [];
    parts.push(o.hybrid ? "keyword + vector" : "vector only");
    if (o.rerank) parts.push("rerank");
    var head = '<div class="resbar"><h3>Prior art</h3><span>Top ' + res.shown.length + (res.meta ? "" : " of " + res.all.length) +
      " &middot; " + parts.join(", ") + '</span><button class="btn sm" id="saveReport"><span>' + icon("bookmark") + "</span>Save report</button></div>";
    if (res.degraded.length) {
      head = '<div class="notice" role="status"><b>Degraded run.</b> These stages fell back: ' + esc(res.degraded.join(", ")) +
        ". Results are still ranked, but quality may be lower.</div>" + head;
    }
    if (!res.shown.length) {
      $("#results").innerHTML = head + '<div class="empty"><h3>No prior art found</h3><p>Try a longer claim, or turn on more stages.</p></div>';
      $("#saveReport").addEventListener("click", saveReport);
      return;
    }
    $("#results").innerHTML = head + res.shown.map(function (p, i) {
      var sc = scoreView(p);
      var kwOnly = !!p.missed && o.hybrid;
      return '<article class="r">' +
        '<div class="rank">' + (i + 1) + "</div>" +
        '<div><h4 style="font-size:16px;font-weight:600;letter-spacing:-.01em">' + esc(p.title) + "</h4>" +
        '<div style="font-size:13px;color:var(--ink3);margin-top:2px">' + esc(p.id) + " &middot; " + esc(p.assignee) + " &middot; " + p.year + " &middot; " + esc(p.cpc) + "</div>" +
        '<div class="head">' + (p.elN ? chips(p) : "") + (kwOnly ? '<span class="chip flag">Missed by vector search</span>' : "") + "</div>" +
        '<p class="quote">' + p.passage + ' <span style="font-family:var(--sans);font-size:12px;color:var(--ink3)">' + esc(p.loc) + "</span></p>" +
        '<p class="why">' + esc(p.why) + "</p></div>" +
        '<div class="side-col"><div class="score">' + sc.v + "<small>" + sc.l + "</small></div>" +
        '<div class="actions-row">' +
        '<button class="icon-btn' + (shortlist[p.id] ? " on" : "") + '" data-short="' + esc(p.id) + '" aria-label="Shortlist ' + esc(p.id) + '" aria-pressed="' + (!!shortlist[p.id]) + '">' + icon("bookmark") + "</button>" +
        '<button class="icon-btn" data-trace="' + esc(p.id) + '" aria-label="Open trace for ' + esc(p.id) + '">' + icon("trace") + "</button></div></div>" +
        "</article>";
    }).join("");
    $("#saveReport").addEventListener("click", saveReport);
  }

  function renderCoverage(res) {
    var n = res.shown.length;
    if (res.elements.length === 1 && res.elements[0].whole_claim) {
      $("#coverage").innerHTML = '<p style="color:var(--ink3);font-size:13.5px">Coverage is reported when the claim is decomposed.</p>';
      return;
    }
    $("#coverage").innerHTML = res.elements.map(function (e) {
      var c = res.shown.filter(function (p) { return p.el.indexOf(e.n) > -1; }).length;
      return '<div class="cov"><span class="chip e' + ((e.n - 1) % 3 + 1) + '">E' + e.n + '</span><div class="track"><i style="width:' + Math.round(c / Math.max(n, 1) * 100) + '%"></i></div><span>' + c + "/" + n + "</span></div>";
    }).join("") + '<p style="font-size:12.5px;color:var(--ink3);margin-top:8px">Results that cover each element.</p>';
  }

  function renderRunInfo(res, o, secs) {
    var kw = res.shown.filter(function (p) { return p.missed; }).length;
    var m = res.meta;
    var rows = [
      ["Retrieval", o.hybrid ? "BM25 + vector" : "Vector only"],
      ["Fusion", o.hybrid ? "RRF, k=" + (m ? m.retrieval.rrf_k : RRF_K) : "None"],
      ["Reranker", o.rerank ? (m ? m.retrieval.rerank_backend : "BGE cross-encoder") : "Off"],
      ["Found only by keyword", String(kw)]
    ];
    if (m) {
      rows.push(["Candidates considered", String(m.candidates_considered)]);
      rows.push(["Server time", (m.timings_ms.total / 1000).toFixed(2) + " s"]);
    } else {
      rows.push(["Demo run time", secs.toFixed(1) + " s"]);
    }
    $("#runInfo").innerHTML = rows.map(function (r) {
      return '<div class="kv"><span>' + r[0] + "</span><b>" + esc(r[1]) + "</b></div>";
    }).join("");
  }

  /* ---------- Results interactions ---------- */
  $("#results").addEventListener("click", function (e) {
    var t = e.target.closest("[data-trace]");
    var s = e.target.closest("[data-short]");
    if (t) openTrace(t.dataset.trace);
    if (s) {
      var id = s.dataset.short;
      shortlist[id] = !shortlist[id];
      s.classList.toggle("on", shortlist[id]);
      s.setAttribute("aria-pressed", String(!!shortlist[id]));
      toast(shortlist[id] ? "Added to shortlist" : "Removed from shortlist");
    }
  });

  function rankText(r) { return r == null ? "not ranked" : "#" + r; }
  function rankBar(r) { return r == null ? 0 : Math.max(4, Math.round(100 - Math.min(r, 100) + 4)); }

  function openTrace(id) {
    var p = lastRun.res.all.filter(function (x) { return x.id === id; })[0];
    var rows = [["Vector search", p.dense], ["Keyword (BM25)", p.bm25], ["After fusion", p.rrfRank], ["After rerank", p.rerankRank]];
    $("#drawer").innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">' +
      '<div><div style="font-size:13px;color:var(--ink3)">' + esc(p.id) + '</div><h2 style="font-size:20px;letter-spacing:-.02em;margin-top:2px">' + esc(p.title) + "</h2></div>" +
      '<button class="icon-btn" id="closeDrawer" aria-label="Close trace">' + icon("x") + "</button></div>" +
      (p.elN ? '<div style="margin-top:14px;display:flex;gap:6px">' + chips(p) + "</div>" : "") +
      '<h3 style="font-size:14px;margin-top:22px">Rank at each stage</h3>' +
      '<div class="trace">' + rows.map(function (r) {
        return '<div class="tr"><span>' + r[0] + '</span><div class="track"><i style="--w:' + rankBar(r[1]) + '%"></i></div><b style="width:auto;min-width:40px">' + rankText(r[1]) + "</b></div>";
      }).join("") + "</div>" +
      '<h3 style="font-size:14px;margin-top:24px">Matching passage</h3>' +
      '<p class="quote" style="font-family:var(--serif);font-size:15px;line-height:1.7;color:var(--ink2);margin-top:8px;padding-left:14px;border-left:2px solid var(--line2)">' + p.passage + "</p>" +
      '<p style="font-size:13px;color:var(--ink3);margin-top:6px">' + esc(p.loc) + "</p>" +
      '<h3 style="font-size:14px;margin-top:24px">Why it ranked here</h3>' +
      '<p style="color:var(--ink2);margin-top:6px">' + esc(p.why) + "</p>";
    $("#drawer").classList.add("on");
    $("#drawer").setAttribute("aria-hidden", "false");
    $("#scrim").classList.add("on");
    $("#closeDrawer").focus();
    $("#closeDrawer").addEventListener("click", closeDrawer);
  }
  function closeDrawer() {
    $("#drawer").classList.remove("on");
    $("#drawer").setAttribute("aria-hidden", "true");
    $("#scrim").classList.remove("on");
  }
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeDrawer(); });

  /* ---------- Saved reports ---------- */
  var local = [];
  try { local = JSON.parse(localStorage.getItem("priorly-saved") || "[]"); } catch (e) {}

  function titleFromClaim(text) {
    var t = text.replace(/\s+/g, " ").trim();
    if (t.length <= 70) return t;
    var cut = t.slice(0, 70), at = cut.lastIndexOf(" ");
    return (at > 40 ? cut.slice(0, at) : cut) + "...";
  }

  function saveReport() {
    if (!lastRun) return;
    var ids = Object.keys(shortlist).filter(function (k) { return shortlist[k]; });
    var title = titleFromClaim(lastRun.claim);
    if (live && lastRun.res.searchId) {
      PriorlyAPI.saveReport(lastRun.res.searchId, title, ids).then(function () {
        toast("Report saved");
        refreshSavedCount();
      }).catch(function (e) { toast(e.message); });
      return;
    }
    var best = lastRun.res.shown[0];
    local.unshift({ title: title, date: new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }),
                    n: lastRun.res.shown.length, best: best ? best.score : 0, short: ids.length });
    try { localStorage.setItem("priorly-saved", JSON.stringify(local)); } catch (e) {}
    refreshSavedCount();
    toast("Report saved");
  }

  function refreshSavedCount() {
    if (live) {
      PriorlyAPI.listReports().then(function (list) { savedRemote = list; $("#savedCount").textContent = list.length; });
    } else {
      $("#savedCount").textContent = local.concat(SAVED_SEED).length;
    }
  }

  function renderSaved() {
    var host = $("#savedList");
    function draw(items) {
      if (!items.length) {
        host.innerHTML = '<div class="empty"><h3>Save your first report</h3><p>Run a search, shortlist what matters, and choose Save report.</p></div>';
        return;
      }
      host.innerHTML = items.map(function (s) {
        var link = s.report_id ? ' <a class="btn sm" href="' + PriorlyAPI.exportUrl(s.report_id) + '">Export CSV</a> <a class="btn sm" href="' + PriorlyAPI.pdfUrl(s.report_id) + '">Export PDF</a>' : "";
        return '<div class="item"><div><b>' + esc(s.title) + "</b><small>" + esc(s.date) + " &middot; " + s.n + " results" +
          (s.short ? " &middot; " + s.short + " shortlisted" : "") + '</small></div><span style="display:flex;gap:8px;align-items:center"><span class="chip">Best ' +
          Number(s.best).toFixed(2) + "</span>" + link + "</span></div>";
      }).join("");
    }
    if (live) {
      PriorlyAPI.listReports().then(function (list) { savedRemote = list; draw(list); }).catch(function (e) {
        host.innerHTML = '<div class="empty err"><h3>Couldn\'t load reports</h3><p>' + esc(e.message) + "</p></div>";
      });
    } else {
      draw(local.concat(SAVED_SEED));
    }
  }

  /* ---------- Benchmark ---------- */
  function renderEval() {
    if (live) {
      PriorlyAPI.benchmark().then(function (r) {
        curEval = { stages: r.stages, byType: r.by_type, count: r.query_count, note: r.note, sample: r.sample, version: r.dataset_version, k: r.k || 10 };
        drawEval();
      }).catch(function (e) {
        $("#metrics").innerHTML = "";
        $("#chartHost").innerHTML = '<div class="empty err"><h3>No benchmark results yet</h3><p>' + esc(e.message) + "</p></div>";
        $("#typeTable").innerHTML = ""; $("#ablation").innerHTML = "";
      });
    } else {
      curEval = { stages: EVAL.stages, byType: EVAL.byType, count: 200, sample: true, version: "demo", k: 10,
                  note: "Sample numbers for the demo. Run the evaluation harness for real ones." };
      drawEval();
    }
  }

  function drawEval() {
    var ev = curEval, st = ev.stages, first = st[0], last = st[st.length - 1];
    var pts = function (a, b) {
      var d = Math.round((a - b) * 100);
      return '<span class="' + (d < 0 ? "neg" : "") + '">' + (d >= 0 ? "+" : "") + d + " pts vs dense only</span>";
    };
    $("#evalSub").textContent = "How much each retrieval layer adds, measured on " + ev.count + " queries with examiner-cited ground truth.";
    $("#metrics").innerHTML =
      '<div class="metric"><small>Recall@' + ev.k + "</small><b>" + last.recall.toFixed(2) + "</b>" + pts(last.recall, first.recall) + "</div>" +
      '<div class="metric"><small>Precision@' + ev.k + "</small><b>" + last.precision.toFixed(2) + "</b>" + pts(last.precision, first.precision) + "</div>" +
      '<div class="metric"><small>MRR</small><b>' + last.mrr.toFixed(2) + "</b><span>" + pts(last.mrr, first.mrr) + "</span></div>" +
      '<div class="metric"><small>Queries evaluated</small><b>' + ev.count + '</b><span style="color:var(--ink3)">Dataset ' + esc(ev.version) + "</span></div>";
    renderChart();
    $("#typeTable").innerHTML = '<thead><tr><th>Query type</th><th class="num">Queries</th><th class="num">Dense</th><th class="num">Hybrid</th><th class="num">Full pipeline</th><th class="num">Lift</th></tr></thead><tbody>' +
      ev.byType.map(function (r) {
        var lift = Math.round((r.full - r.dense) * 100);
        return "<tr><td>" + esc(r.type) + '</td><td class="num">' + r.n + '</td><td class="num' + (r.dense < 0.2 ? " bad" : "") + '">' + r.dense.toFixed(2) + '</td><td class="num">' + r.hybrid.toFixed(2) + '</td><td class="num"><b>' + r.full.toFixed(2) + '</b></td><td class="num ' + (lift >= 0 ? "good" : "bad") + '">' + (lift >= 0 ? "+" : "") + lift + " pts</td></tr>";
      }).join("") + "</tbody>";
    var names = ["BM25 and RRF fusion", "Cross-encoder rerank", "Claim decomposition and HyDE"];
    $("#ablation").innerHTML = names.map(function (n, i) {
      var d = Math.round((st[i + 1].recall - st[i].recall) * 100);
      return '<div class="item"><div><b>' + n + "</b><small>Recall@" + ev.k + " moved from " + st[i].recall.toFixed(2) + " to " + st[i + 1].recall.toFixed(2) +
        '</small></div><span class="' + (d >= 0 ? "good" : "bad") + '">' + (d >= 0 ? "+" : "") + d + " pts</span></div>";
    }).join("");
    $("#evalNote").textContent = ev.note;
  }

  function renderChart() {
    var ev = curEval;
    var W = 760, H = 300, L = 40, B = 44, T = 20, gw = (W - L - 10) / ev.stages.length;
    var series = [["recall", "Recall@" + ev.k, "var(--brand)"], ["precision", "Precision@" + ev.k, "var(--ok)"], ["mrr", "MRR", "var(--warn)"]];
    var bw = 26, gap = 6, ph = H - B - T;
    var svg = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Grouped bar chart of recall, precision and MRR for each pipeline stage">';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) {
      var y = T + ph - g * ph;
      svg += '<line class="gl" x1="' + L + '" x2="' + (W - 10) + '" y1="' + y + '" y2="' + y + '"/><text x="' + (L - 8) + '" y="' + (y + 4) + '" text-anchor="end">' + g.toFixed(2) + "</text>";
    });
    ev.stages.forEach(function (s, i) {
      var gx = L + i * gw + gw / 2 - (3 * bw + 2 * gap) / 2;
      series.forEach(function (se, j) {
        var v = s[se[0]], h = v * ph, x = gx + j * (bw + gap), y = T + ph - h;
        svg += '<rect x="' + x + '" y="' + y + '" width="' + bw + '" height="' + h + '" rx="4" style="fill:' + se[2] + '"/>';
        svg += '<text x="' + (x + bw / 2) + '" y="' + (y - 6) + '" text-anchor="middle" style="font-size:11px">' + v.toFixed(2) + "</text>";
      });
      svg += '<text x="' + (L + i * gw + gw / 2) + '" y="' + (H - 16) + '" text-anchor="middle">' + esc(s.name) + "</text>";
    });
    svg += "</svg>";
    var legend = '<div style="display:flex;gap:18px;margin-top:10px;font-size:13px;color:var(--ink2)">' + series.map(function (se) {
      return '<span style="display:inline-flex;align-items:center;gap:6px"><i style="width:10px;height:10px;border-radius:3px;background:' + se[2] + ';display:inline-block"></i>' + se[1] + "</span>";
    }).join("") + "</div>";
    $("#chartHost").innerHTML = svg + legend;
  }

  $("#exportBtn").addEventListener("click", function () {
    if (!curEval) { toast("Open the benchmark first."); return; }
    var csv = "stage,recall,precision,mrr\n" + curEval.stages.map(function (s) {
      return ['"' + s.name + '"', s.recall, s.precision, s.mrr].join(",");
    }).join("\n");
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "priorly-benchmark.csv";
    a.click();
    toast("Benchmark exported");
  });

  /* ---------- Settings ---------- */
  var FIELD = { emb: "embedding_model", rr: "reranker", llm: "llm_model", rrf: "rrf_k", cand: "candidates", depth: "rerank_depth" };
  function loadSettings() {
    if (!live) { $("#settingsNote").textContent = "Demo data: settings are not saved until the API is running."; return; }
    PriorlyAPI.getSettings().then(function (s) {
      var form = $("#settingsForm");
      Object.keys(FIELD).forEach(function (name) {
        var input = form.elements[name], value = s[FIELD[name]];
        if (!input || value == null) return;
        if (input.tagName === "SELECT" && !$$("option", input).some(function (o) { return o.value === String(value) || o.text === String(value); })) {
          input.insertAdjacentHTML("afterbegin", '<option value="' + esc(value) + '">' + esc(value) + "</option>");
        }
        input.value = value;
      });
      $("#settingsNote").textContent = "Active backends: vector " + s.backends.dense + ", rerank " + s.backends.rerank + ", LLM " + s.backends.llm +
        ". Model changes apply after the API restarts. RRF constant, candidates, and rerank depth apply immediately.";
    }).catch(function (e) { $("#settingsNote").textContent = e.message; });
  }
  $("#settingsForm").addEventListener("submit", function (e) {
    e.preventDefault();
    if (!live) { toast("Demo data: start the API to save settings."); return; }
    var f = e.target.elements;
    PriorlyAPI.putSettings({
      embedding_model: f.emb.value, reranker: f.rr.value, llm_model: f.llm.value,
      rrf_k: +f.rrf.value, candidates: +f.cand.value, rerank_depth: +f.depth.value
    }).then(function () { toast("Settings saved"); }).catch(function (err) { toast(err.message); });
  });

  $("#eraseBtn").addEventListener("click", function () {
    if (!live) { toast("Demo data: there is nothing stored on a server to erase."); return; }
    if (!window.confirm("Erase every stored search, saved report and cached LLM output? This cannot be undone.")) return;
    PriorlyAPI.eraseAll().then(function (r) {
      var d = r.deleted || {};
      $("#eraseNote").textContent = "Erased " + (d.searches || 0) + " searches, " + (d.reports || 0) + " reports and " + (d.llm_cache || 0) + " cached LLM outputs.";
      toast("Stored data erased");
      refreshSavedCount();
    }).catch(function (err) { toast(err.message); });
  });

  /* ---------- Initial state ---------- */
  renderElements(opts(), null);
  $("#results").innerHTML = '<div class="empty" style="margin-top:16px"><h3>Find your first prior art</h3><p>The sample claim is loaded. Run the search to see ranked results with matching passages.</p></div>';
  PriorlyAPI.health().then(setMode, function () { setMode(null); }).then(function () {
    refreshSavedCount();
    var h = (location.hash || "").replace("#", "");
    if (["search", "eval", "saved", "settings"].indexOf(h) > -1) show(h);
  });
})();
