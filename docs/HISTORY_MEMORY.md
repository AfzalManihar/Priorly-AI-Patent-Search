# History, similar searches and personal memory

* Every private endpoint calls `owner_id()` first: anonymous callers get 401. Searches, reports, exports, traces, history, notes and recall are filtered by `user_id` in SQL.
* Global controls (`PUT /api/settings`, `GET /api/guardrails/events`) are admin only (the first account).
* Records saved before login existed (`user_id` NULL) are visible to nobody. An administrator can assign them to one account with `python -m app.admin claim-legacy USERNAME --yes`.
* Similar past searches: local TF-IDF cosine (same tokenizer as BM25) over the user's last 300 claims. No model call, no tokens. It does not reuse BGE embeddings.
* Memory notes: OFF by default (accounts that existed before consent was added are switched off once). Notes are screened by the input guardrail, capped at 50 notes x 1000 characters, shown with `textContent`, and are never added to any LLM prompt.
* Turning memory off hides notes but keeps them; "Delete all notes" or deleting the account removes them.
