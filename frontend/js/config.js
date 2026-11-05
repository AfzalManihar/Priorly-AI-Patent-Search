/* Where the Priorly API lives.
 * "" means same origin, which is right when you run `uvicorn app.main:app` (it serves this frontend too).
 * If you host the frontend elsewhere, set the API address, for example:
 *   window.PRIORLY_API_BASE = "https://priorly-api.example.com";
 */
window.PRIORLY_API_BASE = "";
