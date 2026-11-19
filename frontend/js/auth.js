(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var gate = $("authGate"), app = document.querySelector(".app");
  var loginForm = $("loginForm"), registerForm = $("registerForm");
  var isRegister = false, dashboardLoaded = false, busy = false;

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("priorly-theme", theme); } catch (ignore) {}
    var dark = theme === "dark";
    $("authThemeLabel").textContent = dark ? "Use light theme" : "Use dark theme";
    $("authThemeBtn").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  function toggleTheme() {
    setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  }
  function message(text) { $("authMessage").textContent = text || ""; }
  function setBusy(value) {
    busy = value;
    [$("loginSubmit"), $("registerSubmit"), $("authSwitch")].forEach(function (el) { el.disabled = value; });
    $("loginSubmit").textContent = value && !isRegister ? "Signing in…" : "Sign in";
    $("registerSubmit").textContent = value && isRegister ? "Creating account…" : "Create account";
  }
  function showGate(register) {
    isRegister = !!register;
    document.body.classList.remove("auth-pending");
    gate.hidden = false;
    app.classList.add("auth-hidden");
    app.setAttribute("aria-hidden", "true");
    loginForm.hidden = isRegister;
    registerForm.hidden = !isRegister;
    $("authSwitchRow").hidden = false;
    $("authTitle").textContent = isRegister ? "Create your account" : "Welcome back";
    $("authSubtitle").textContent = isRegister ? "Use your invitation code to join Priorly." : "Sign in to continue to your Priorly workspace.";
    $("authSwitchText").textContent = isRegister ? "Already have an account?" : "New to Priorly?";
    $("authSwitch").textContent = isRegister ? "Sign in" : "Create an account";
    message("");
    window.setTimeout(function () { (isRegister ? $("registerUsername") : $("loginUsername")).focus(); }, 0);
  }
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src; script.onload = resolve; script.onerror = reject;
      document.body.appendChild(script);
    });
  }
  function enterWorkspace(user) {
    gate.hidden = true;
    app.classList.remove("auth-hidden");
    app.removeAttribute("aria-hidden");
    document.body.classList.remove("auth-pending");
    $("currentUsername").textContent = user.username + "'s workspace";
    $("userAvatar").textContent = user.username.slice(0, 1).toUpperCase();
    if (!dashboardLoaded) {
      dashboardLoaded = true;
      loadScript("js/app.js").then(function () { return loadScript("js/lab.js"); }).catch(function () {
        dashboardLoaded = false;
        app.classList.add("auth-hidden"); app.setAttribute("aria-hidden", "true");
        showGate(false); message("The workspace could not load. Refresh the page and try again.");
      });
    }
  }
  function authError(error) {
    if (error && error.status === 429) return "Too many login attempts. Please wait before trying again.";
    return error && error.message ? error.message : "Something went wrong. Please try again.";
  }

  $("authSwitch").addEventListener("click", function () { if (!busy) showGate(!isRegister); });
  $("authThemeBtn").addEventListener("click", toggleTheme);
  loginForm.addEventListener("submit", function (event) {
    event.preventDefault(); if (busy) return;
    if (!loginForm.reportValidity()) return;
    setBusy(true); message("");
    PriorlyAPI.login($("loginUsername").value.trim(), $("loginPassword").value)
      .then(function (result) { enterWorkspace(result.user); })
      .catch(function (error) { message(authError(error)); $("loginPassword").focus(); })
      .finally(function () { setBusy(false); });
  });
  registerForm.addEventListener("submit", function (event) {
    event.preventDefault(); if (busy) return;
    if (!registerForm.reportValidity()) return;
    var username = $("registerUsername").value.trim();
    if (!/^[A-Za-z0-9._-]{3,32}$/.test(username)) { message("Use 3–32 letters, numbers, dots, underscores or hyphens."); return; }
    if ($("registerPassword").value.length < 12) { message("Password must contain at least 12 characters."); return; }
    setBusy(true); message("");
    PriorlyAPI.signup(username, $("registerPassword").value)
      .then(function (result) { enterWorkspace(result.user); })
      .catch(function (error) { message(authError(error)); })
      .finally(function () { setBusy(false); });
  });

  // Logout control is delegated because it lives in the authenticated workspace.
  document.addEventListener("click", function (event) {
    if (!event.target.closest || !event.target.closest("#logoutBtn")) return;
    if (busy) return;
    setBusy(true);
    PriorlyAPI.logout().then(function () {
      app.classList.add("auth-hidden"); app.setAttribute("aria-hidden", "true");
      showGate(false); loginForm.reset(); registerForm.reset();
    }).catch(function (error) {
      if (window.toast) window.toast(authError(error));
      else window.alert(authError(error));
    }).finally(function () { setBusy(false); });
  });

  function showProfileNote(id, text, isError) {
    var el = $(id); if (!el) return;
    el.textContent = text || "";
    el.style.color = isError ? "var(--danger, #c44)" : "";
  }
  function loadProfile() {
    showProfileNote("profileNote", "Loading profile…", false);
    PriorlyAPI.getProfile().then(function (result) {
      var p = result.profile || {};
      $("profileEmail").value = p.email || "";
      $("profileName").value = p.display_name || "";
      $("profileTopic").value = p.learning_topic || "";
      $("memoryEnabled").checked = p.memory_enabled !== false;
      showProfileNote("profileNote", "", false);
    }).catch(function (error) { showProfileNote("profileNote", authError(error), true); });
  }
  document.addEventListener("click", function (event) {
    var nav = event.target.closest && event.target.closest('[data-view="profile"]');
    if (nav) loadProfile();
  });
  $("profileForm").addEventListener("submit", function (event) {
    event.preventDefault();
    PriorlyAPI.saveProfile({ email: $("profileEmail").value.trim(), display_name: $("profileName").value.trim(),
      learning_topic: $("profileTopic").value.trim(), memory_enabled: $("memoryEnabled").checked })
      .then(function () { showProfileNote("profileNote", "Changes saved.", false); })
      .catch(function (error) { showProfileNote("profileNote", authError(error), true); });
  });
  $("passwordForm").addEventListener("submit", function (event) {
    event.preventDefault();
    var next = $("newPassword").value;
    if (next.length < 12) { showProfileNote("passwordNote", "New password must contain at least 12 characters.", true); return; }
    PriorlyAPI.changePassword($("currentPassword").value, next).then(function () {
      showProfileNote("passwordNote", "Password changed. Please sign in again.", false);
      app.classList.add("auth-hidden"); app.setAttribute("aria-hidden", "true");
      showGate(false); loginForm.reset(); registerForm.reset();
    }).catch(function (error) { showProfileNote("passwordNote", authError(error), true); });
  });
  $("logoutEverywhereBtn").addEventListener("click", function () {
    if (!window.confirm("Sign out of every device where you are logged in?")) return;
    PriorlyAPI.logoutEverywhere().then(function () {
      app.classList.add("auth-hidden"); app.setAttribute("aria-hidden", "true");
      showGate(false); loginForm.reset(); registerForm.reset();
    }).catch(function (error) { showProfileNote("profileNote", authError(error), true); });
  });

  var currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  setTheme(currentTheme);
  // A stale or expired cookie is never enough to reveal the workspace.
  PriorlyAPI.currentUser().then(function (result) { enterWorkspace(result.user); })
    .catch(function () { showGate(false); });
})();
