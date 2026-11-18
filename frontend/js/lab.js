/* Experiments, strategy playground, retrieval inspector and observability.
   Everything shown here comes from the API (eval/grid.py output, /api/playground, /api/traces, /api/metrics).
   Nothing in this file contains a measured number. */
(function () {
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var API = window.PriorlyAPI;
  var METRICS = [
    ["ndcg@10", "nDCG@10"], ["recall@10", "Recall@10"], ["recall@50", "Recall@50"], ["precision@10", "Precision@10"],
    ["mrr", "MRR"], ["map", "MAP"], ["hit@10", "Hit rate@10"]
  ];
  var RET = { bm25: "BM25", dense: "Dense", hybrid: "Hybrid" };
  var STAGES = ["decompose", "hyde", "retrieve", "fuse", "rerank", "explain"];

  var grid = null, metric = "ndcg@10", selected = null, sortKey = null, sortDir = -1;
  var defaults = null, queries = null, traces = [], timer = null;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function f3(v) { return v == null ? "-" : Number(v).toFixed(3); }
  function label(m) { return (METRICS.filter(function (x) { return x[0] === m; })[0] || [m, m])[1]; }
  function ms(v) { return v == null ? "-" : (v >= 1000 ? (v / 1000).toFixed(1) + " s" : Math.round(v) + " ms"); }
  function err(host, e) {
    host.innerHTML = '<div class="card"><p style="color:var(--ink2)">' + esc(e && e.message || "Something went wrong.") + "</p></div>";
  }
  function chip(t, cls) { return '<span class="chip ' + (cls || "") + '">' + esc(t) + "</span>"; }
  function head(title, sub, right) {
    return '<div class="top"><div><h1>' + esc(title) + "</h1><p>" + esc(sub) + "</p></div>" + (right || "") + "</div>";
  }

  /* ------------------------------------------------------------ default configuration */
  function optionText(o) {
    var parts = [o.dense === false ? "BM25 only" : (o.hybrid === false ? "vector only" : "BM25 + vector (RRF)")];
    if (o.prf) parts.push("query expansion");
    if (o.decompose) parts.push("decomposition");
    if (o.hyde) parts.push("HyDE");
    if (o.rerank) parts.push("rerank");
    return parts.join(" + ");
  }

  function applyOptions(o) {
    var map = { optDecompose: "decompose", optHyde: "hyde", optHybrid: "hybrid", optRerank: "rerank", optDense: "dense", optPrf: "prf" };
    Object.keys(map).forEach(function (id) {
      var el = $("#" + id);
      if (el && o[map[id]] !== undefined) el.checked = !!o[map[id]];
    });
  }

  function loadDefaults() {
    return API.get("/api/default-config").then(function (d) {
      defaults = d;
      var chipHost = $("#defaultChip");
      if (!d.available) {
        if (chipHost) chipHost.innerHTML = "No validated default yet. Run the comparison (python -m eval.grid) to choose one from measured results.";
        return d;
      }
      // Keep search toggles at their HTML defaults; apply benchmark options only on user reset.
      if (chipHost) {
        chipHost.innerHTML = '<span class="chip flag">Validated default</span><span>' + esc(optionText(d.pipeline_options)) +
          " &middot; nDCG@10 " + f3(d.config.metrics["ndcg@10"].mean) + ", Recall@10 " + f3(d.config.metrics["recall@10"].mean) +
          '</span><button class="btn ghost sm" type="button" id="resetDefault">Reset to default</button>' +
          (d.chunking_matches_live ? "" : '<span style="flex:1 0 100%;font-size:12.5px;color:var(--ink3)">Best chunking is ' + esc(d.config.options.chunking) +
            ", live index uses " + esc(d.live.chunking) + ". Set CHUNKING=" + esc(d.config.options.chunking) + " and restart to use it.</span>");
        var b = $("#resetDefault");
        if (b) b.onclick = function () { applyOptions(Object.assign({}, d.pipeline_options, { prf: true })); };
      }
      return d;
    }, function () { return null; });
  }

  /* ------------------------------------------------------------ experiments */
  function combos() { return grid ? grid.combos : []; }
  function byId(id) { return combos().filter(function (c) { return c.id === id; })[0]; }
  function bestId() { return grid && grid.best ? grid.best.combo_id : null; }

  function renderLab() {
    var host = $("#v-lab");
    host.innerHTML = head("Experiments", "Every chunking, retrieval, query-processing and rerank combination, measured on the same real queries.",
      '<button class="btn" id="pdfBtn"><span data-i="download"></span>Research report (PDF)</button>') + '<div id="labBody"></div>';
    if (window.hydrateIcons) window.hydrateIcons(host);
    $("#pdfBtn").onclick = function () { API.downloadFile("/api/experiments/report.pdf", "priorly-research-report.pdf").catch(function (e) { alert(e.message); }); };
    var body = $("#labBody");
    body.innerHTML = '<div class="card">Loading results&hellip;</div>';
    Promise.all([API.get("/api/experiments"), loadDefaults()]).then(function (r) {
      if (r[0].available === false) {
        body.innerHTML = '<div class="card"><h3>No comparison has been run yet</h3><p style="color:var(--ink2)">Run <code>python -m eval.grid --queries data/real/queries.json --n 40</code> in the backend folder. It writes <code>grid.json</code>, and this page fills in with the measured results.</p></div>';
        return;
      }
      grid = r[0];
      if (!selected || !byId(selected)) selected = bestId();
      drawLab();
    }, function (e) { err(body, e); });
  }

  function drawLab() {
    var m = grid.meta, body = $("#labBody"), best = grid.best, d = defaults;
    var offline = m.dense_backend !== "bge";
    var html = "";
    if (offline) {
      html += '<div class="warnbox" style="margin-bottom:16px">This run used the offline stand-in for dense retrieval (' + esc(m.dense_backend) + ") and reranker (" + esc(m.rerank_backend) +
        "), not the BGE models. It validates the method. Run <code>python -m eval.grid</code> with DENSE_BACKEND=bge and RERANK_BACKEND=bge for the final numbers.</div>";
    }
    if (!m.complete) html += '<div class="warnbox" style="margin-bottom:16px">This run is still in progress or was interrupted. The table shows the combinations finished so far.</div>';
    html += '<div class="card"><h3>Setup</h3><div class="kvs">' +
      chip(m.queries + " queries") + chip(Object.keys(m.query_types).map(function (k) { return m.query_types[k] + " " + k; }).join(", ")) +
      chip("pool " + m.pool_patents.toLocaleString() + " of " + m.corpus_patents.toLocaleString() + " patents") +
      chip(combos().length + " combinations") + chip("split " + m.split) + chip("dense " + m.dense_backend) + chip("rerank " + m.rerank_backend) +
      chip("LLM " + m.llm_backend + ", temperature " + m.llm_temperature) + chip("RRF k=" + m.rrf_k) + chip(m.created.slice(0, 10)) +
      '</div><p class="note">Ground truth is examiner-cited prior art only, an incomplete record of relevance, so absolute scores are a lower bound. Compare strategies with each other. The query patent is excluded from its own results and only earlier patents may be returned.</p></div>';

    if (best) {
      html += '<div class="card"><h3>Default configuration</h3><p><b>' + esc(best.label) + "</b></p><div class=\"kvs\">" +
        chip("nDCG@10 " + f3(best.metrics["ndcg@10"].mean), "flag") + chip("Recall@10 " + f3(best.metrics["recall@10"].mean)) +
        chip("MRR " + f3(best.metrics["mrr"].mean)) + chip("Precision@10 " + f3(best.metrics["precision@10"].mean)) + "</div>" +
        '<p class="note">Selection rule: ' + esc(best.rule) + ". The search page loads this configuration on every refresh. " +
        (d && d.available ? (d.chunking_matches_live ? "The live index already uses this chunking."
          : "The live index uses " + esc(d.live.chunking) + " chunking; set CHUNKING=" + esc(best.options.chunking) + " and restart to use the best one.") : "") + "</p></div>";
    }

    html += '<div class="card"><div class="row-ctl"><h3 style="margin:0">Chunking &times; retrieval</h3><span style="flex:1"></span><label>Metric <select class="inp" id="metricSel">' +
      METRICS.map(function (x) { return '<option value="' + x[0] + '"' + (x[0] === metric ? " selected" : "") + ">" + x[1] + "</option>"; }).join("") +
      '</select></label></div><div class="scroll-x" id="heat"></div><p class="note">Darker is better. Click a cell to inspect it below. The green outline marks the best cell.</p></div>';

    html += '<div class="card"><h3>What each choice did, and whether it is real</h3><p class="note" style="margin:-6px 0 12px">Each finding is a paired comparison on the same queries. If the confidence interval spans zero, the difference may be noise.</p><div class="lab-grid">' +
      (grid.findings || []).map(function (f) {
        var v = f.comparison.verdict, cls = v === "better" ? "better" : v === "worse" ? "worse" : "none";
        return '<div class="finding"><b>' + esc(f.title) + ' <span class="verdict ' + cls + '">' + esc(v) + "</span></b>" + esc(f.text) + "</div>";
      }).join("") + "</div></div>";

    html += '<div class="card"><h3>All combinations</h3><div class="scroll-x"><table class="tbl" id="board"></table></div></div>';
    html += '<div class="card"><h3 id="barTitle"></h3><div id="bars"></div></div>';
    html += '<div id="detail"></div>';
    if (grid.skipped && grid.skipped.length) {
      html += '<div class="card"><h3>Not run</h3>' + grid.skipped.map(function (s) { return "<p class=\"note\" style=\"margin:4px 0\">" + esc(s.label) + ": " + esc(s.reason) + "</p>"; }).join("") + "</div>";
    }
    body.innerHTML = html;
    $("#metricSel").onchange = function () { metric = this.value; sortKey = null; drawHeat(); drawBoard(); drawBars(); };
    drawHeat(); drawBoard(); drawBars(); drawDetail();
  }

  function drawHeat() {
    var chunkers = Object.keys(grid.meta.chunkers), cells = {}, vals = [];
    combos().filter(function (c) { return c.stage === "A"; }).forEach(function (c) { cells[c.chunking + "|" + c.retrieval] = c; vals.push(c.metrics[metric].mean); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), best = vals.length ? hi : null;
    var h = '<table class="heat"><tr><th></th>' + Object.keys(RET).map(function (r) { return "<th>" + RET[r] + "</th>"; }).join("") + "</tr>";
    chunkers.forEach(function (ch) {
      h += "<tr><th>" + esc(grid.meta.chunkers[ch].label) + "<small style=\"display:block;font-weight:400;color:var(--ink3)\">" + esc(grid.meta.chunkers[ch].description) + "</small></th>";
      Object.keys(RET).forEach(function (r) {
        var c = cells[ch + "|" + r];
        if (!c) { h += "<td>-</td>"; return; }
        var v = c.metrics[metric].mean, t = hi === lo ? 50 : Math.round(15 + 70 * (v - lo) / (hi - lo));
        h += '<td data-id="' + esc(c.id) + '" class="' + (v === best ? "best " : "") + (c.id === selected ? "sel" : "") + '" style="background:color-mix(in srgb,var(--brand) ' + t + '%,var(--surface))">' +
          f3(v) + "<small>" + f3(c.metrics[metric].ci[0]) + " to " + f3(c.metrics[metric].ci[1]) + "</small></td>";
      });
      h += "</tr>";
    });
    $("#heat").innerHTML = h + "</table>";
    $$("#heat td[data-id]").forEach(function (td) { td.onclick = function () { selected = td.dataset.id; drawHeat(); drawBoard(); drawDetail(); $("#detail").scrollIntoView({ behavior: "smooth", block: "nearest" }); }; });
  }

  function sorted() {
    var key = sortKey || metric;
    function val(c) { return key === "latency" ? c.latency_ms.p95 : key === "label" ? c.label : c.metrics[key].mean; }
    var isMetric = key !== "latency" && key !== "label";
    return combos().slice().sort(function (a, b) {
      var av = val(a), bv = val(b), cmp = av < bv ? -1 : av > bv ? 1 : 0;
      cmp = isMetric ? -cmp : cmp;  // best first: high metrics, low latency, A to Z
      return sortDir === -1 ? cmp : -cmp;
    });
  }

  function drawBoard() {
    var cols = ["recall@10", "recall@50", "precision@10", "mrr", "ndcg@10", "map", "hit@10"];
    var h = "<thead><tr><th class=\"sort\" data-k=\"label\">Combination</th>" + cols.map(function (k) { return '<th class="num sort" data-k="' + k + '">' + label(k) + (k === metric && !sortKey ? " &darr;" : "") + "</th>"; }).join("") +
      '<th class="num">p50</th><th class="num sort" data-k="latency">p95</th><th class="num">vs baseline</th></tr></thead><tbody>';
    sorted().forEach(function (c) {
      var vb = c.vs_baseline && c.vs_baseline["ndcg@10"];
      h += '<tr class="click ' + (c.id === selected ? "sel" : "") + '" data-id="' + esc(c.id) + '"><td>' + esc(c.label) + (c.id === bestId() ? ' <span class="chip flag">default</span>' : "") + "</td>" +
        cols.map(function (k) { return '<td class="num">' + f3(c.metrics[k].mean) + "</td>"; }).join("") +
        '<td class="num">' + ms(c.latency_ms.p50) + '</td><td class="num">' + ms(c.latency_ms.p95) + '</td><td class="num">' +
        (vb ? '<span class="verdict ' + (vb.verdict === "better" ? "better" : vb.verdict === "worse" ? "worse" : "none") + '">' + (vb.diff >= 0 ? "+" : "") + vb.diff.toFixed(3) + "</span>" : "baseline") + "</td></tr>";
    });
    $("#board").innerHTML = h + "</tbody>";
    $$("#board th.sort").forEach(function (th) { th.onclick = function () { if (sortKey === th.dataset.k) sortDir *= -1; sortKey = th.dataset.k; drawBoard(); }; });
    $$("#board tr.click").forEach(function (tr) { tr.onclick = function () { selected = tr.dataset.id; drawBoard(); drawHeat(); drawDetail(); $("#detail").scrollIntoView({ behavior: "smooth", block: "nearest" }); }; });
  }

  function drawBars() {
    $("#barTitle").textContent = label(metric) + " by combination, with 95% confidence intervals";
    var list = combos().slice().sort(function (a, b) { return b.metrics[metric].mean - a.metrics[metric].mean; });
    var max = Math.max.apply(null, list.map(function (c) { return c.metrics[metric].ci[1]; })) || 1, W = 900, L = 330, R = 60, rowH = 22;
    var svg = '<svg viewBox="0 0 ' + W + " " + (list.length * rowH + 8) + '" style="width:100%;height:auto" role="img" aria-label="' + esc(label(metric)) + ' by combination">';
    list.forEach(function (c, i) {
      var y = i * rowH + 4, m = c.metrics[metric], x = function (v) { return L + (W - L - R) * v / max; };
      var txt = c.label.length > 56 ? c.label.slice(0, 55) + "…" : c.label;
      svg += '<text x="' + (L - 8) + '" y="' + (y + 13) + '" text-anchor="end" font-size="11" fill="var(--ink2)">' + esc(txt) + "</text>" +
        '<rect x="' + L + '" y="' + (y + 3) + '" width="' + Math.max(1, x(m.mean) - L) + '" height="12" rx="3" fill="' + (c.id === bestId() ? "var(--ok)" : "var(--brand)") + '"/>' +
        '<line x1="' + x(m.ci[0]) + '" x2="' + x(m.ci[1]) + '" y1="' + (y + 9) + '" y2="' + (y + 9) + '" stroke="var(--ink)" stroke-width="1.3"/>' +
        '<text x="' + (x(m.ci[1]) + 6) + '" y="' + (y + 13) + '" font-size="11" fill="var(--ink)">' + f3(m.mean) + "</text>";
    });
    $("#bars").innerHTML = svg + "</svg>";
  }

  function drawDetail() {
    var c = byId(selected), host = $("#detail");
    if (!c) { host.innerHTML = ""; return; }
    var types = Object.keys(c.by_type).map(function (t) {
      var x = c.by_type[t];
      return "<tr><td>" + esc(t) + '</td><td class="num">' + x.queries + '</td><td class="num">' + f3(x["recall@10"]) + '</td><td class="num">' + f3(x.mrr) + '</td><td class="num">' + f3(x["ndcg@10"]) + "</td></tr>";
    }).join("");
    var stages = Object.keys(c.stage_ms_mean).map(function (k) { return chip(k + " " + ms(c.stage_ms_mean[k])); }).join("");
    var work = Object.keys(c.work).map(function (k) { return chip(k.replace(/_/g, " ") + " " + c.work[k]); }).join("");
    var rows = c.per_query.slice().sort(function (a, b) { return (a.first_hit_rank || 999) - (b.first_hit_rank || 999); }).map(function (r) {
      return "<tr><td>" + esc(r.id) + "</td><td>" + esc(r.type) + '</td><td class="num">' + r.relevant + '</td><td class="num">' + (r.first_hit_rank || "not found") +
        '</td><td class="num">' + f3(r["recall@10"]) + '</td><td class="num">' + f3(r.mrr) + '</td><td class="num">' + f3(r["ndcg@10"]) + '</td><td class="num">' + ms(r.latency_ms) + "</td></tr>";
    }).join("");
    host.innerHTML = '<div class="card"><h3>' + esc(c.label) + '</h3><div class="kvs">' + chip("stage " + c.stage) + chip("p50 " + ms(c.latency_ms.p50)) + chip("p95 " + ms(c.latency_ms.p95)) + stages + work + "</div>" +
      '<h3 style="margin-top:18px">By query type</h3><table class="tbl"><thead><tr><th>Type</th><th class="num">Queries</th><th class="num">Recall@10</th><th class="num">MRR</th><th class="num">nDCG@10</th></tr></thead><tbody>' + types + "</tbody></table>" +
      '<h3 style="margin-top:18px">Every query</h3><div class="scroll-x" style="max-height:340px;overflow:auto"><table class="tbl"><thead><tr><th>Query</th><th>Type</th><th class="num">Cited</th><th class="num">First hit rank</th><th class="num">Recall@10</th><th class="num">MRR</th><th class="num">nDCG@10</th><th class="num">Time</th></tr></thead><tbody>' + rows + "</tbody></table></div></div>";
  }

  /* ------------------------------------------------------------ strategy playground */
  var PRESETS = {
    bm25: ["BM25 only", { decompose: false, hyde: false, hybrid: true, dense: false, prf: false, rerank: false }],
    dense: ["Vector only", { decompose: false, hyde: false, hybrid: false, dense: true, prf: false, rerank: false }],
    hybrid: ["Hybrid (BM25 + vector, RRF)", { decompose: false, hyde: false, hybrid: true, dense: true, prf: false, rerank: false }],
    rerank: ["Hybrid + rerank", { decompose: false, hyde: false, hybrid: true, dense: true, prf: false, rerank: true }],
    decompose: ["Hybrid + decomposition", { decompose: true, hyde: false, hybrid: true, dense: true, prf: false, rerank: false }],
    prf: ["Hybrid + query expansion", { decompose: false, hyde: false, hybrid: true, dense: true, prf: true, rerank: false }],
    full: ["Everything on", { decompose: true, hyde: true, hybrid: true, dense: true, prf: false, rerank: true }]
  };
  var TOGGLES = [["decompose", "Decompose claim"], ["hyde", "HyDE (uses Groq)"], ["hybrid", "Keyword search + RRF"], ["dense", "Vector search"], ["prf", "Query expansion (PRF)"], ["rerank", "Cross-encoder rerank"]];

  function renderPlayground() {
    var host = $("#v-playground");
    host.innerHTML = head("Strategy playground", "Run the same claim through different strategies side by side. For benchmark claims, each result is checked against the examiner's citations.");
    host.insertAdjacentHTML("beforeend", '<div class="card"><div class="row-ctl"><label>Claim <select class="inp" id="pgQuery"><option value="">Loading&hellip;</option></select></label></div>' +
      '<textarea id="pgClaim" spellcheck="false" style="width:100%;min-height:110px"></textarea><p class="note" id="pgNote"></p>' +
      '<div class="pg-cols" id="pgCfg"></div><div class="row-ctl" style="margin-top:14px"><button class="btn primary" id="pgRun">Run strategies</button><span id="pgStatus" class="note" style="margin:0"></span></div></div><div id="pgOut"></div>');
    var chosen = [["bm25"], ["hybrid"], ["rerank"]];
    var cfgHost = $("#pgCfg");
    chosen.forEach(function (c, i) {
      var opts = Object.keys(PRESETS).map(function (k) { return '<option value="' + k + '"' + (k === c[0] ? " selected" : "") + ">" + PRESETS[k][0] + "</option>"; }).join("");
      var extra = defaults && defaults.available ? '<option value="default">Validated default</option>' : "";
      cfgHost.insertAdjacentHTML("beforeend", '<div class="pg-col" data-n="' + i + '"><label>Strategy ' + (i + 1) + ' <select class="inp pgPreset">' + extra + opts + "</select></label>" +
        TOGGLES.map(function (t) { return '<label class="tg"><input type="checkbox" data-k="' + t[0] + '"><span class="sw"></span>' + t[1] + "</label>"; }).join("") + "</div>");
    });
    function setPreset(col, key) {
      var o = key === "default" ? defaults.pipeline_options : PRESETS[key][1];
      TOGGLES.forEach(function (t) { $('[data-k="' + t[0] + '"]', col).checked = !!o[t[0]]; });
    }
    $$(".pg-col", cfgHost).forEach(function (col) {
      var sel = $(".pgPreset", col);
      sel.onchange = function () { setPreset(col, sel.value); };
      setPreset(col, sel.value);
    });
    function loadQueries() {
      var done = function (list) {
        queries = list;
        $("#pgQuery").innerHTML = '<option value="">Paste my own claim (no ground truth)</option>' + list.map(function (q) {
          return '<option value="' + esc(q.id) + '">' + esc(q.id + " | " + q.type + " | " + q.relevant + " cited | " + q.claim.slice(0, 60)) + "</option>";
        }).join("");
        if (list.length) { $("#pgQuery").value = list[0].id; pick(); }
      };
      if (queries) return done(queries);
      API.get("/api/queries?limit=80").then(function (r) { done(r.queries); }, function () { done([]); });
    }
    function pick() {
      var q = (queries || []).filter(function (x) { return x.id === $("#pgQuery").value; })[0];
      $("#pgClaim").value = q ? q.claim : $("#pgClaim").value;
      $("#pgClaim").readOnly = !!q;
      $("#pgNote").textContent = q ? "Benchmark claim from patent " + q.patent_id + ", " + q.year + ". The query patent is excluded from its own results and only earlier patents may be returned." : "Your own claim has no ground truth, so only the rankings and timings are shown.";
    }
    $("#pgQuery").onchange = pick;
    loadQueries();
    $("#pgRun").onclick = function () {
      var configs = $$(".pg-col", cfgHost).map(function (col) {
        var o = {};
        TOGGLES.forEach(function (t) { o[t[0]] = $('[data-k="' + t[0] + '"]', col).checked; });
        var name = $(".pgPreset", col).selectedOptions[0].textContent;
        return { name: name, options: o };
      });
      var qid = $("#pgQuery").value, body = { configs: configs };
      if (qid) body.query_id = qid; else body.claim = $("#pgClaim").value;
      $("#pgRun").disabled = true; $("#pgStatus").textContent = "Running " + configs.length + " strategies. Stages that call the LLM or reranker take a while on CPU.";
      API.post("/api/playground", body).then(function (r) { drawPlayground(r); $("#pgStatus").textContent = ""; }, function (e) { $("#pgStatus").textContent = e.message; })
        .then(function () { $("#pgRun").disabled = false; });
    };
  }

  function drawPlayground(r) {
    var host = $("#pgOut"), all = {};
    r.runs.forEach(function (run) { run.results.forEach(function (x) { all[x.id] = (all[x.id] || 0) + 1; }); });
    host.innerHTML = '<div class="pg-cols">' + r.runs.map(function (run) {
      var m = run.metrics;
      return '<div class="pg-col"><h3>' + esc(run.name) + "</h3>" +
        (m ? '<div class="metric-row"><span>Recall@10 ' + f3(m["recall@10"]) + "</span><span>Precision@10 " + f3(m["precision@10"]) + "</span><span>MRR " + f3(m.mrr) + "</span><span>nDCG@10 " + f3(m["ndcg@10"]) + "</span><span>Hit " + f3(m["hit@10"]) + "</span></div>" : "") +
        '<div class="metric-row"><span>' + ms(run.latency_ms) + "</span>" + Object.keys(run.timings_ms).filter(function (k) { return k !== "total" && run.timings_ms[k] > 0; }).map(function (k) { return "<span>" + k + " " + ms(run.timings_ms[k]) + "</span>"; }).join("") + "</div>" +
        (run.degraded.length ? '<div class="warnbox">Fell back for: ' + esc(run.degraded.join(", ")) + "</div>" : "") +
        run.results.map(function (x) {
          return '<div class="pg-hit ' + (x.relevant ? "rel" : "") + '"><span class="rk">' + x.rank + "</span><span style=\"flex:1\"><b>" + esc(x.id) + "</b> " + esc(x.year) + "<br>" + esc(x.title) +
            (x.relevant ? '<br><span class="verdict better">examiner-cited</span>' : "") + "</span></span>" +
            (all[x.id] === r.runs.length ? "" : '<span class="chip" title="Not returned by every strategy">unique</span>') + "</div>";
        }).join("") + "</div>";
    }).join("") + "</div>";
  }

  /* ------------------------------------------------------------ retrieval inspector */
  var inspectId = null;

  function renderInspector() {
    var host = $("#v-inspector");
    host.innerHTML = head("Retrieval inspector", "See how one search moved through the pipeline: what each retriever ranked, how fusion combined them, and what the reranker changed.") +
      '<div class="card"><div class="row-ctl"><label>Search <select class="inp" id="insSel"></select></label><button class="btn sm" id="insReload">Refresh list</button></div><div id="insBody"></div></div>';
    $("#insReload").onclick = loadInspectorList;
    $("#insSel").onchange = function () { inspectId = this.value; loadInspect(); };
    loadInspectorList();
  }

  function loadInspectorList() {
    API.get("/api/traces?limit=40").then(function (r) {
      traces = r.traces;
      $("#insSel").innerHTML = traces.length ? traces.map(function (t) {
        return '<option value="' + esc(t.search_id) + '">' + esc(t.created.slice(11, 19) + " | " + t.claim.slice(0, 70)) + "</option>";
      }).join("") : '<option value="">No searches yet</option>';
      if (!traces.length) { $("#insBody").innerHTML = '<p class="note">Run a search first, then come back here.</p>'; return; }
      if (!inspectId || !traces.some(function (t) { return t.search_id === inspectId; })) inspectId = traces[0].search_id;
      $("#insSel").value = inspectId;
      loadInspect();
    }, function (e) { err($("#insBody"), e); });
  }

  function stackBar(timings) {
    var keys = STAGES.filter(function (k) { return timings[k] > 0; }), total = keys.reduce(function (s, k) { return s + timings[k]; }, 0) || 1;
    return '<div class="stack" title="' + esc(keys.map(function (k) { return k + " " + ms(timings[k]); }).join(", ")) + '">' + keys.map(function (k) {
      return '<i class="s' + STAGES.indexOf(k) + '" style="width:' + (100 * timings[k] / total) + '%"></i>';
    }).join("") + "</div>";
  }
  function legend() {
    return '<div class="kvs">' + STAGES.map(function (k, i) { return '<span class="chip"><i class="s' + i + '" style="display:inline-block;width:10px;height:10px;border-radius:3px"></i>' + k + "</span>"; }).join("") + "</div>";
  }

  function loadInspect() {
    var host = $("#insBody");
    if (!inspectId) return;
    host.innerHTML = "<p class=\"note\">Loading&hellip;</p>";
    API.get("/api/search/" + encodeURIComponent(inspectId)).then(function (s) {
      var r = s.response, o = r.options, t = r.timings_ms, w = r.work || {};
      var rows = r.results.map(function (x, i) {
        var move = x.rrf_rank && x.rerank_rank ? x.rrf_rank - x.rerank_rank : 0;
        return "<tr><td class=\"num\">" + (i + 1) + "</td><td><b>" + esc(x.id) + "</b> " + esc(x.year) + "<br><span style=\"color:var(--ink2)\">" + esc(x.title) + "</span></td>" +
          '<td class="num">' + (x.bm25 || "-") + '</td><td class="num">' + (x.dense || "-") + '</td><td class="num">' + (x.rrf_rank || "-") + '</td><td class="num">' + (x.rerank_rank || "-") +
          '</td><td class="num">' + (move ? '<span class="verdict ' + (move > 0 ? "better" : "worse") + '">' + (move > 0 ? "+" : "") + move + "</span>" : "") + "</td><td class=\"num\">" + f3(x.score) +
          "</td><td>" + esc(x.loc) + (x.missed_by_vector ? ' <span class="chip flag">keyword only</span>' : "") + "</td></tr>";
      }).join("");
      host.innerHTML = '<div class="kvs" style="margin:0 0 12px">' + chip(optionText(o), "flag") + chip(r.candidates_considered + " candidates considered") + chip("total " + ms(t.total)) +
        (r.degraded_stages.length ? chip("fell back: " + r.degraded_stages.join(", ")) : "") + "</div>" +
        '<h3>Where the time went</h3>' + stackBar(t) + legend() +
        '<h3 style="margin-top:18px">Work actually done</h3><div class="kvs">' + chip("reranker pairs scored " + (w.rerank_pairs_scored || 0)) + chip("reranker pairs from cache " + (w.rerank_pairs_from_cache || 0)) +
        chip("LLM calls " + (w.llm_calls || 0)) + chip("LLM answers from cache " + (w.llm_from_cache || 0)) + "</div>" +
        '<h3 style="margin-top:18px">Claim elements</h3><div class="kvs">' + (r.elements || []).map(function (e) { return chip(e.n + ". " + e.text.slice(0, 80)); }).join("") + "</div>" +
        '<h3 style="margin-top:18px">Ranks at each stage</h3><p class="note" style="margin:-6px 0 10px">Lower is better. BM25 and vector columns are each retriever\'s own rank. RRF is the fused rank. Rerank is the final cross-encoder rank. Moved shows how far rerank pushed the patent up (+) or down (-).</p>' +
        '<div class="scroll-x"><table class="tbl"><thead><tr><th class="num">#</th><th>Patent</th><th class="num">BM25</th><th class="num">Vector</th><th class="num">RRF</th><th class="num">Rerank</th><th class="num">Moved</th><th class="num">Score</th><th>Best passage</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
    }, function (e) { err(host, e); });
  }

  /* ------------------------------------------------------------ observability */
  function renderObs() {
    var host = $("#v-obs");
    if (!$("#obsBody")) host.innerHTML = head("Observability", "Latency, stage timings, caches, errors and a trace of every recent search. Counters reset when the server restarts.",
      '<button class="btn" id="obsRefresh">Refresh</button>') + '<div id="obsBody"></div>';
    $("#obsRefresh").onclick = drawObs;
    drawObs();
  }

  function drawObs() {
    var body = $("#obsBody");
    Promise.all([API.get("/api/metrics"), API.get("/api/traces?limit=25"), API.get("/api/usage")]).then(function (r) {
      var m = r[0], tr = r[1].traces, usage = r[2], cache = m.llm_cache || {}, lat = m.latency_ms || {};
      var stages = m.stage_p95_ms || {}, maxStage = Math.max.apply(null, Object.keys(stages).map(function (k) { return stages[k] || 0; }).concat([1]));
      body.innerHTML = '<div class="stat-grid">' +
        [["Searches", m.searches], ["Degraded", m.degraded_searches], ["p50 latency", ms(lat.p50)], ["p95 latency", ms(lat.p95)],
        ["LLM cache hits", (cache.hits || 0) + " / " + ((cache.hits || 0) + (cache.misses || 0))], ["Stored searches", (m.stored || {}).searches == null ? "-" : m.stored.searches],
        ["Uptime", Math.round(m.uptime_s / 60) + " min"]].map(function (x) { return '<div class="stat"><b>' + esc(x[1]) + "</b><small>" + esc(x[0]) + "</small></div>"; }).join("") + "</div>" +
        '<div class="card"><h3>Stage p95 latency</h3>' + (Object.keys(stages).length ? Object.keys(stages).map(function (k) {
          return '<div class="hbar"><span>' + esc(k) + '</span><div class="track"><i style="width:' + (100 * (stages[k] || 0) / maxStage) + '%"></i></div><span>' + ms(stages[k]) + "</span></div>";
        }).join("") : '<p class="note">No searches recorded yet.</p>') + "</div>" +
        usageCard(usage) +
        '<div class="card"><h3>Errors by status</h3><div class="kvs">' + (Object.keys(m.errors_by_status || {}).length ? Object.keys(m.errors_by_status).map(function (k) { return chip(k + ": " + m.errors_by_status[k]); }).join("") : chip("none")) + "</div></div>" +
        '<div class="card"><h3>Recent traces</h3>' + legend() + '<div class="scroll-x"><table class="tbl" style="margin-top:10px"><thead><tr><th>Time</th><th>Claim</th><th>Strategy</th><th>Stages</th><th class="num">Total</th><th class="num">Rerank pairs</th><th class="num">LLM calls</th><th class="num">Tokens</th></tr></thead><tbody>' +
        tr.map(function (t) {
          var w = t.work || {};
          return '<tr class="click" data-id="' + esc(t.search_id) + '"><td>' + esc(t.created.slice(11, 19)) + "</td><td>" + esc(t.claim.slice(0, 60)) + "</td><td>" + esc(optionText(t.options)) + "</td><td style=\"min-width:140px\">" + stackBar(t.timings_ms) +
            '</td><td class="num">' + ms(t.timings_ms.total) + '</td><td class="num">' + (w.rerank_pairs_scored || 0) + '</td><td class="num">' + (w.llm_calls || 0) + '</td><td class="num">' + (w.llm_tokens == null ? "-" : fmtN(w.llm_tokens)) + "</td></tr>";
        }).join("") + '</tbody></table></div><p class="note">Click a row to open it in the inspector.</p></div>';
      refreshMeter();
      $$("#obsBody tr.click").forEach(function (row) { row.onclick = function () { inspectId = row.dataset.id; window.__show("inspector"); }; });
    }, function (e) { err(body, e); });
  }

  /* ------------------------------------------------------------ token usage */
  function fmtN(n) { return n == null ? "-" : Number(n).toLocaleString("en-US"); }

  function refreshMeter() {
    var host = $("#tokenMeter");
    if (!host) return;
    API.get("/api/usage").then(function (u) {
      if (u.backend !== "groq") {
        host.innerHTML = "LLM tokens<div style=\"margin-top:6px\">Offline mode, no tokens used</div>";
        return;
      }
      var pct = u.limit_24h ? Math.min(100, 100 * u.used_24h / u.limit_24h) : 0;
      host.innerHTML = "LLM tokens, last 24 h" + '<div class="track"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
        '<div style="margin-top:6px">' + fmtN(u.used_24h) + (u.limit_24h ? " of " + fmtN(u.limit_24h) : "") + " used</div>";
    }, function () { host.innerHTML = "LLM tokens<div style=\"margin-top:6px\">Unavailable</div>"; });
  }

  function usageCard(u) {
    if (u.backend !== "groq") {
      return '<div class="card"><h3>LLM token usage</h3><p class="note" style="margin-top:0">The LLM backend is the offline heuristic, so no tokens are used. Set LLM_BACKEND=groq to track Groq usage.</p></div>';
    }
    var tasks = u.by_task, max = Math.max.apply(null, tasks.map(function (t) { return t.total; }).concat([1]));
    var days = u.by_day, dmax = Math.max.apply(null, days.map(function (d) { return d.tokens; }).concat([1]));
    return '<div class="card"><h3>LLM token usage (' + esc(u.model) + ")</h3>" +
      '<div class="stat-grid">' + [["Used, last 24 h", fmtN(u.used_24h)], ["Daily limit", u.limit_24h ? fmtN(u.limit_24h) : "none"],
        ["Remaining", u.remaining_24h == null ? "-" : fmtN(u.remaining_24h)], ["Groq calls, all time", fmtN(u.calls_total)],
        ["Answers from cache", fmtN(u.cache_hits)], ["Tokens saved (estimate)", "~" + fmtN(u.tokens_saved_estimate)]].map(function (x) {
        return '<div class="stat"><b>' + esc(x[1]) + "</b><small>" + esc(x[0]) + "</small></div>";
      }).join("") + "</div>" +
      '<h3 style="margin-top:18px">By task, last 24 h</h3>' + (tasks.length ? tasks.map(function (t) {
        return '<div class="hbar"><span>' + esc(t.task) + '</span><div class="track"><i style="width:' + (100 * t.total / max) + '%"></i></div><span>' +
          fmtN(t.total) + " tok, " + t.calls + " calls</span></div>";
      }).join("") : '<p class="note">No Groq calls in the last 24 hours.</p>') +
      '<h3 style="margin-top:18px">Last 7 days</h3>' + (days.length ? days.map(function (d) {
        return '<div class="hbar"><span>' + esc(d.day.slice(5)) + '</span><div class="track"><i style="width:' + (100 * d.tokens / dmax) + '%"></i></div><span>' + fmtN(d.tokens) + "</span></div>";
      }).join("") : '<p class="note">Nothing yet.</p>') +
      '<h3 style="margin-top:18px">Recent calls</h3><div class="scroll-x"><table class="tbl"><thead><tr><th>Time</th><th>Task</th><th class="num">Prompt</th><th class="num">Completion</th><th class="num">Total</th><th>Source</th></tr></thead><tbody>' +
      u.recent.map(function (r) {
        return "<tr><td>" + esc(r.time.slice(11, 19)) + "</td><td>" + esc(r.task) + '</td><td class="num">' + (r.cached ? "-" : fmtN(r.prompt)) + '</td><td class="num">' +
          (r.cached ? "-" : fmtN(r.completion)) + '</td><td class="num">' + (r.cached ? "~" : "") + fmtN(r.total) + "</td><td>" + (r.cached ? "cache (estimate, no tokens spent)" : "Groq, " + ms(r.ms)) + "</td></tr>";
      }).join("") + '</tbody></table></div><p class="note">Token counts come from the usage field Groq returns with every answer. Cached answers cost nothing; their size is an estimate (4 characters per token).</p></div>';
  }

  /* ------------------------------------------------------------ safety */
  var POLICIES = [
    ["Input guard", "Every claim is Unicode-normalised, stripped of hidden characters, then checked for instruction overrides, role hijacks, prompt-extraction requests, fake system or chat delimiters, secret requests and attacks in Hindi, Hinglish, Spanish, French, German and Chinese. Matches are refused with HTTP 422."],
    ["Prompt isolation", "The claim, claim elements and patent passages are sent to the model inside data tags, and the system prompt tells the model that anything inside them is data and never an instruction."],
    ["Prompt-leak guard", "A random marker lives only in the system prompts. If a model answer contains it, repeats the prompt wording or shows that it obeyed an injected instruction, the answer is dropped, never cached, and the pipeline uses its fallback."],
    ["Output guard", "Model text is stripped of HTML and links, length-capped, and decomposed elements must share words with the claim, so a hijacked answer cannot reach the results page."],
    ["Secret hygiene", "API keys and bearer tokens are redacted from logs and error messages. Stored attack excerpts are redacted and capped at 120 characters."],
    ["Spend guard", "A rolling 24-hour token limit (DAILY_TOKEN_LIMIT) stops Groq calls before the free quota runs out. Searches continue with the offline fallbacks."]
  ];

  function verdictChip(a) { return chip(a, a === "block" ? "flag" : (a === "flag" ? "e2" : "e1")); }

  function renderSafety() {
    var host = $("#v-safety");
    if (!$("#safeBody")) {
      host.innerHTML = head("Safety guardrails", "Prompt-injection, prompt-leak and secret-exfiltration protection around every LLM call. Everything on this page is measured on this server.",
        '<button class="btn" id="safeRefresh">Refresh</button>') +
        '<div class="card"><h3>What is protected</h3><div class="stat-grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">' +
        POLICIES.map(function (p) { return '<div class="stat"><b style="font-size:14px">' + esc(p[0]) + '</b><small style="display:block;margin-top:6px;line-height:1.5">' + esc(p[1]) + "</small></div>"; }).join("") + "</div></div>" +
        '<div class="card"><h3>Try an attack</h3><p class="note" style="margin-top:0">Nothing here starts a search or calls Groq. The text is only run through the input guard.</p>' +
        '<div class="kvs" id="safePresets"></div><textarea id="safeText" rows="4" style="width:100%;margin-top:12px" placeholder="Paste a claim or an attack"></textarea>' +
        '<div style="margin-top:10px"><button class="btn primary" id="safeRun">Check text</button></div><div id="safeVerdict" style="margin-top:12px"></div></div>' +
        '<div class="card"><h3>Self-test</h3><p class="note" style="margin-top:0">Runs the built-in red-team list and the real benchmark claims through the guard.</p>' +
        '<button class="btn" id="safeTest">Run self-test</button><div id="safeTestOut" style="margin-top:12px"></div></div>' +
        '<div id="safeBody"></div>';
      $("#safeRefresh").onclick = drawSafety;
      $("#safeRun").onclick = function () {
        var t = $("#safeText").value;
        if (!t.trim()) return;
        API.post("/api/guardrails/check", { text: t }).then(function (v) {
          $("#safeVerdict").innerHTML = verdictChip(v.action) + " " + (v.reasons.length ? v.reasons.map(function (r) { return chip(r.rule + " (" + r.category.replace("_", " ") + ")"); }).join(" ") : chip("no attack pattern found")) +
            (v.action === "flag" ? '<p class="note">The text continues with this cleaned version:</p><pre class="note" style="white-space:pre-wrap">' + esc(v.cleaned) + "</pre>" : "") +
            (v.action === "block" ? '<p class="note">A search with this text is refused with HTTP 422 and never reaches the models.</p>' : "");
          drawSafety();
        }, function (e) { err($("#safeVerdict"), e); });
      };
      $("#safeTest").onclick = function () {
        var out = $("#safeTestOut"); out.textContent = "Running...";
        API.get("/api/guardrails/selftest").then(function (r) {
          out.innerHTML = '<div class="stat-grid">' + [["Attacks caught", r.attacks_caught + " of " + r.attacks_total], ["Detection rate", (100 * r.detection_rate).toFixed(1) + "%"],
            ["Legitimate texts checked", fmtN(r.legit_total) + " (" + r.legit_source + ")"], ["False positives", r.false_positives]].map(function (x) {
            return '<div class="stat"><b>' + esc(x[1]) + "</b><small>" + esc(x[0]) + "</small></div>"; }).join("") + "</div>" +
            '<div class="kvs">' + r.attacks.map(function (a) { return chip(a.name + ": " + a.action, a.action === "block" ? "flag" : (a.action === "flag" ? "e2" : "")); }).join("") + "</div>" +
            (r.false_positive_examples.length ? '<p class="note">Wrongly stopped: ' + esc(r.false_positive_examples.join(" | ")) + "</p>" : "") +
            '<p class="note">The red-team list is small and written by us, so this measures regressions, not protection against a determined attacker. Model-side guards (isolation, canary, output checks) cover what pattern matching cannot.</p>';
          $$("#safePresets button").length || null;
          if (!$("#safePresets button")) $("#safePresets").innerHTML = r.attacks.slice(0, 8).map(function (a, i) { return '<button class="btn" data-i="' + i + '">' + esc(a.name) + "</button>"; }).join("");
          $$("#safePresets button").forEach(function (b) { b.onclick = function () { $("#safeText").value = r.attacks[+b.dataset.i].text; }; });
        }, function (e) { err(out, e); });
      };
      API.get("/api/guardrails/selftest").then(function (r) {
        $("#safePresets").innerHTML = r.attacks.slice(0, 8).map(function (a, i) { return '<button class="btn" data-i="' + i + '">' + esc(a.name) + "</button>"; }).join("");
        $$("#safePresets button").forEach(function (b) { b.onclick = function () { $("#safeText").value = r.attacks[+b.dataset.i].text; }; });
      }, function () {});
    }
    drawSafety();
  }

  function drawSafety() {
    var body = $("#safeBody");
    API.get("/api/guardrails/events?limit=25").then(function (e) {
      var c = e.counts || {}, rules = Object.keys(e.by_rule || {});
      body.innerHTML = '<div class="card"><h3>Counters since the database was created</h3><div class="stat-grid">' +
        [["Inputs blocked", c.input_block || 0], ["Inputs cleaned", c.input_flag || 0], ["Model answers rejected", c.output_block || 0]].map(function (x) {
          return '<div class="stat"><b>' + esc(x[1]) + "</b><small>" + esc(x[0]) + "</small></div>"; }).join("") + "</div>" +
        '<div class="kvs">' + (rules.length ? rules.map(function (k) { return chip(k + ": " + e.by_rule[k]); }).join("") : chip("no events yet")) + "</div></div>" +
        '<div class="card"><h3>Recent events</h3>' + (e.recent.length ? '<div class="scroll-x"><table class="tbl"><thead><tr><th>Time</th><th>Where</th><th>Action</th><th>Rules</th><th>Excerpt</th><th>Source</th></tr></thead><tbody>' +
          e.recent.map(function (r) {
            return "<tr><td>" + esc(r.time.slice(11, 19)) + "</td><td>" + esc(r.stage) + "</td><td>" + verdictChip(r.action) + "</td><td>" + esc(r.rules.join(", ")) + "</td><td>" + esc(r.excerpt) + "</td><td>" + esc(r.source) + "</td></tr>";
          }).join("") + "</tbody></table></div>" : '<p class="note" style="margin-top:0">Nothing blocked or cleaned yet.</p>') + "</div>";
    }, function (er) { err(body, er); });
  }

  /* ------------------------------------------------------------ wiring */
  function gridSummary() {
    var host = $("#gridSummary");
    if (!host) return;
    API.get("/api/experiments").then(function (d) {
      if (!d.combos) { host.innerHTML = "<h3>Comparison grid</h3><p class=\"note\" style=\"margin-top:0\">" + esc(d.message || "No comparison run yet.") + "</p>"; return; }
      var m = d.meta, by = {}, best = d.combos.slice().sort(function (a, b) { return b.metrics["ndcg@10"].mean - a.metrics["ndcg@10"].mean; })[0];
      d.combos.forEach(function (c) { by[c.stage] = (by[c.stage] || 0) + 1; });
      var names = { A: "chunking x retrieval", B: "query processing and rerank", C: "rerank on other chunkings", D: "RRF constant", E: "rerank depth" };
      host.innerHTML = '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center"><h3 style="margin:0">Comparison grid: ' + d.combos.length +
        " pipeline combinations on " + m.queries + " real queries</h3>" + '<button class="btn primary" id="gridOpen">Open all comparisons</button></div>' +
        '<div class="kvs">' + Object.keys(by).sort().map(function (k) { return chip(by[k] + " x " + (names[k] || "stage " + k)); }).join("") +
        chip("best nDCG@10 " + f3(best.metrics["ndcg@10"].mean) + ": " + best.label) + "</div>" +
        '<p class="note">Every combination was run on the same queries and scored against examiner citations. Open Experiments for the heatmap, confidence intervals, per-query results and the PDF report.</p>';
      $("#gridOpen").onclick = function () { window.__show("lab"); };
    }, function () { host.style.display = "none"; });
  }

  function onShow(view) {
    if (view === "eval") gridSummary();
    if (timer) { clearInterval(timer); timer = null; }
    if (view === "lab") renderLab();
    if (view === "playground") loadDefaults().then(renderPlayground);
    if (view === "inspector") renderInspector();
    if (view === "obs") { renderObs(); timer = setInterval(drawObs, 10000); }
    if (view === "safety") renderSafety();
  }

  window.Lab = { onShow: onShow, applyDefault: loadDefaults, refreshMeter: refreshMeter };
  window.__show = function (v) { var b = $('.menu button[data-view="' + v + '"]'); if (b) b.click(); };

  /* The validated default is applied to the search toggles on every page load. */
  loadDefaults();
  refreshMeter();
})();
