/* Thin client for the Priorly API (PRD section 10). Every function returns a Promise. */
(function () {
  var base = (window.PRIORLY_API_BASE || "").replace(/\/$/, "");

  function ApiError(status, code, message) {
    this.status = status; this.code = code; this.message = message;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function fail(status, json) {
    var e = (json && json.error) || {};
    return new ApiError(status, e.code || "http_" + status, e.message || "The request failed. Try again.");
  }

  function request(method, path, body) {
    return fetch(base + path, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (json) {
        if (!r.ok) throw fail(r.status, json);
        return json;
      });
    }, function () {
      throw new ApiError(0, "unreachable", "Can't reach the Priorly API. Start the backend and try again.");
    });
  }

  /* POST /api/search/stream: calls onEvent for each stage event, resolves with the final result. */
  function searchStream(claim, options, k, onEvent) {
    return fetch(base + "/api/search/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim: claim, options: options, k: k })
    }).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (j) { throw fail(r.status, j); });
      }
      var reader = r.body.getReader(), decoder = new TextDecoder(), buffer = "", result = null;
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) {
            if (!result) throw new ApiError(0, "stream_closed", "The search ended before it finished. Try again.");
            return result;
          }
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split("\n\n");
          buffer = parts.pop();
          parts.forEach(function (part) {
            if (part.indexOf("data: ") !== 0) return;
            var ev = JSON.parse(part.slice(6));
            if (ev.type === "error") throw fail(500, ev);
            if (ev.type === "result") result = ev.data;
            else if (onEvent) onEvent(ev);
          });
          return pump();
        });
      }
      return pump();
    }, function () {
      throw new ApiError(0, "unreachable", "Can't reach the Priorly API. Start the backend and try again.");
    });
  }

  window.PriorlyAPI = {
    ApiError: ApiError,
    health: function () { return request("GET", "/api/health"); },
    searchStream: searchStream,
    benchmark: function () { return request("GET", "/api/benchmark"); },
    listReports: function () { return request("GET", "/api/reports").then(function (j) { return j.reports; }); },
    saveReport: function (searchId, title, shortlist) {
      return request("POST", "/api/reports", { search_id: searchId, title: title, shortlist: shortlist });
    },
    exportUrl: function (reportId) { return base + "/api/reports/" + encodeURIComponent(reportId) + "/export.csv"; },
    pdfUrl: function (reportId) { return base + "/api/reports/" + encodeURIComponent(reportId) + "/export.pdf"; },
    eraseAll: function () { return request("DELETE", "/api/data?confirm=yes"); },
    getSettings: function () { return request("GET", "/api/settings"); },
    putSettings: function (values) { return request("PUT", "/api/settings", values); }
  };
})();
