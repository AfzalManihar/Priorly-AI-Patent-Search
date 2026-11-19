# Phase B — Frontend Authentication

## Included
- Session check on app load; the workspace remains hidden until `/api/auth/me` succeeds.
- Login and open registration screens backed by the Phase A auth endpoints; signup requires only a username and password.
- Logout with the Phase A CSRF token.
- Logged-in username and logout control in the workspace sidebar (also visible on mobile).
- Login/register theme toggle using the existing `priorly-theme` preference.
- Keyboard-operable forms, visible focus styles, labels, validation and live error messages.
- Same-origin cookies; mutation requests send `X-CSRF-Token` from the readable CSRF cookie.
- Remote browser sessions bypass the legacy access-code header after successful login, so a browser does not need the invitation/access secret exposed in JavaScript.

## Files
- `frontend/app.html`
- `frontend/css/styles.css`
- `frontend/js/api.js`
- `frontend/js/auth.js` (new)
- `backend/app/main.py`
- `backend/tests/test_auth.py`

## Apply
Extract this patch ZIP into the project root and allow it to replace the listed files. It does not include `.env`, data, Qdrant storage, backups, or the existing search application JavaScript. Keep the Phase A authentication files and the rest of the project unchanged.

## Verify
Stop Uvicorn, then run from the project root:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest backend/tests/test_auth.py -q
& ".\.venv\Scripts\python.exe" -m pytest -q
```

After tests pass, start the backend and open `/app.html`. Register using only a username and password (at least 12 characters), confirm the existing dashboard opens, refresh to confirm the session is restored, then log out and confirm only the login/register screen remains.

## Security note

Public signup means anyone who can reach the server can create an account. The current application data (searches, reports, settings and usage) is not yet isolated per user; do not expose this build as a public multi-user service until Phase C data ownership/isolation is implemented.
