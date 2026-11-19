"""SQLite storage: search log, saved reports, settings and the LLM cache (PRD section 14)."""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .tokenizer import tokenize

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
  id TEXT PRIMARY KEY, created TEXT NOT NULL, request TEXT NOT NULL, response TEXT NOT NULL, user_id TEXT);
CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY, created TEXT NOT NULL, title TEXT NOT NULL, search_id TEXT,
  shortlist TEXT NOT NULL, n INTEGER NOT NULL, best REAL NOT NULL, user_id TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS guardrail_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, stage TEXT NOT NULL, action TEXT NOT NULL,
  rules TEXT NOT NULL, excerpt TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, task TEXT NOT NULL, model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL,
  cached INTEGER NOT NULL, ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_salt BLOB NOT NULL, password_hash BLOB NOT NULL,
  created_at TEXT NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0,
  email TEXT NOT NULL DEFAULT '', display_name TEXT NOT NULL DEFAULT '',
  learning_topic TEXT NOT NULL DEFAULT '', memory_enabled INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  csrf_hash TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS memory_notes (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  text TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_notes(user_id, updated);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        with self.lock:
            self.db.executescript(SCHEMA)
            # Forward-compatible migration for databases created before profile settings.
            cols = {r[1] for r in self.db.execute("PRAGMA table_info(users)")}
            for name, definition in (("email", "TEXT NOT NULL DEFAULT ''"), ("display_name", "TEXT NOT NULL DEFAULT ''"), ("learning_topic", "TEXT NOT NULL DEFAULT ''"), ("memory_enabled", "INTEGER NOT NULL DEFAULT 0")):
                if name not in cols:
                    self.db.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")
            search_cols = {r[1] for r in self.db.execute("PRAGMA table_info(searches)")}
            if "user_id" not in search_cols:
                self.db.execute("ALTER TABLE searches ADD COLUMN user_id TEXT")
            report_cols = {r[1] for r in self.db.execute("PRAGMA table_info(reports)")}
            if "user_id" not in report_cols:
                self.db.execute("ALTER TABLE reports ADD COLUMN user_id TEXT")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_searches_user_created ON searches(user_id, created)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_reports_user_created ON reports(user_id, created)")
            # Memory is opt-in. Accounts created before consent existed never gave it, so switch it off once.
            done = self.db.execute("SELECT 1 FROM settings WHERE key='memory_default_off_v1'").fetchone()
            if not done:
                self.db.execute("UPDATE users SET memory_enabled=0")
                self.db.execute("INSERT OR REPLACE INTO settings VALUES ('memory_default_off_v1', '1')")
            self.db.commit()

    # authentication --------------------------------------------------------------
    def create_user(self, username: str, salt: bytes, password_hash: bytes) -> dict | None:
        """Create a user. The first account is the initial admin. Returns None on duplicate username."""
        user_id = "u_" + secrets.token_hex(16)
        created = _now()
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                count = self.db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                self.db.execute(
                    "INSERT INTO users (id, username, password_salt, password_hash, created_at, is_admin) "
                    "VALUES (?,?,?,?,?,?)",
                    (user_id, username, salt, password_hash, created, int(count == 0)),
                )
                self.db.commit()
            except sqlite3.IntegrityError:
                self.db.rollback()
                return None
        return {"id": user_id, "username": username, "created_at": created, "is_admin": count == 0}

    def user_by_username(self, username: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
        return dict(row) if row else None

    def user_by_id(self, user_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT id, username, created_at, is_admin FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def set_user_password(self, username: str, salt: bytes, password_hash: bytes) -> bool:
        with self.lock:
            cur = self.db.execute(
                "UPDATE users SET password_salt=?, password_hash=? WHERE username=? COLLATE NOCASE",
                (salt, password_hash, username),
            )
            if cur.rowcount == 1:
                self.db.execute(
                    "DELETE FROM sessions WHERE user_id=(SELECT id FROM users WHERE username=? COLLATE NOCASE)",
                    (username,),
                )
            self.db.commit()
            return cur.rowcount == 1

    def create_session(self, token_hash: str, user_id: str, csrf_hash: str,
                       created_at: float, expires_at: float) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO sessions (token_hash,user_id,csrf_hash,created_at,expires_at) VALUES (?,?,?,?,?)",
                (token_hash, user_id, csrf_hash, created_at, expires_at),
            )
            self.db.execute("DELETE FROM sessions WHERE expires_at<=?", (created_at,))
            self.db.commit()

    def get_session(self, token_hash: str, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        with self.lock:
            row = self.db.execute(
                "SELECT s.token_hash,s.user_id,s.csrf_hash,s.created_at,s.expires_at,"
                "u.username,u.created_at AS user_created_at,u.is_admin "
                "FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.token_hash=? AND s.expires_at>?",
                (token_hash, now),
            ).fetchone()
            if not row:
                self.db.execute("DELETE FROM sessions WHERE token_hash=? OR expires_at<=?", (token_hash, now))
                self.db.commit()
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            self.db.commit()

    def profile_by_id(self, user_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT id,username,email,display_name,learning_topic,memory_enabled,created_at FROM users WHERE id=?", (user_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["memory_enabled"] = bool(result["memory_enabled"])
        return result

    def update_profile(self, user_id: str, email: str, display_name: str, learning_topic: str, memory_enabled: bool) -> dict | None:
        with self.lock:
            self.db.execute("UPDATE users SET email=?,display_name=?,learning_topic=?,memory_enabled=? WHERE id=?",
                            (email, display_name, learning_topic, int(memory_enabled), user_id))
            self.db.commit()
        return self.profile_by_id(user_id)

    def update_password(self, user_id: str, salt: bytes, password_hash: bytes) -> None:
        with self.lock:
            self.db.execute("UPDATE users SET password_salt=?,password_hash=? WHERE id=?", (salt, password_hash, user_id))
            self.db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            self.db.commit()

    def delete_user_sessions(self, user_id: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            self.db.commit()

    def delete_user(self, username: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM users WHERE username=? COLLATE NOCASE", (username,))
            self.db.commit()
            return cur.rowcount == 1

    # search log -----------------------------------------------------------------
    def save_search(self, search_id: str, request: dict, response: dict,
                    user_id: str | None = None) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO searches (id,created,request,response,user_id) VALUES (?,?,?,?,?)",
                (search_id, _now(), json.dumps(request), json.dumps(response), user_id),
            )
            self.db.commit()

    def recent_searches(self, limit: int = 30, user_id: str | None = None) -> list[dict]:
        with self.lock:
            if user_id is None:
                rows = self.db.execute(
                    "SELECT * FROM searches WHERE user_id IS NULL ORDER BY created DESC, rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT * FROM searches WHERE user_id=? ORDER BY created DESC, rowid DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        out = []
        for row in rows:
            req, res = json.loads(row["request"]), json.loads(row["response"])
            out.append({
                "search_id": row["id"], "created": row["created"],
                "claim": req.get("claim", "")[:160],
                "options": res.get("options", req.get("options", {})),
                "timings_ms": res.get("timings_ms", {}), "work": res.get("work", {}),
                "degraded": res.get("degraded_stages", []),
                "results": len(res.get("results", [])),
                "candidates": res.get("candidates_considered"),
            })
        return out

    def get_search(self, search_id: str, user_id: str | None = None) -> dict | None:
        with self.lock:
            if user_id is None:
                row = self.db.execute(
                    "SELECT * FROM searches WHERE id=? AND user_id IS NULL", (search_id,)
                ).fetchone()
            else:
                row = self.db.execute(
                    "SELECT * FROM searches WHERE id=? AND user_id=?", (search_id, user_id)
                ).fetchone()
        if not row:
            return None
        return {
            "search_id": row["id"], "created": row["created"],
            "request": json.loads(row["request"]), "response": json.loads(row["response"]),
        }

    def latest_search(self, user_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM searches WHERE user_id=? ORDER BY created DESC, rowid DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "search_id": row["id"], "created": row["created"],
            "request": json.loads(row["request"]), "response": json.loads(row["response"]),
        }

    # reports --------------------------------------------------------------------
    def save_report(self, report_id: str, title: str, search_id: str | None,
                    shortlist: list[str], n: int, best: float,
                    user_id: str | None = None) -> dict:
        created = _now()
        with self.lock:
            self.db.execute(
                "INSERT INTO reports (id,created,title,search_id,shortlist,n,best,user_id) VALUES (?,?,?,?,?,?,?,?)",
                (report_id, created, title, search_id, json.dumps(shortlist), n, best, user_id),
            )
            self.db.commit()
        return {"report_id": report_id, "created": created}

    def list_reports(self, user_id: str | None = None) -> list[dict]:
        with self.lock:
            if user_id is None:
                rows = self.db.execute(
                    "SELECT * FROM reports WHERE user_id IS NULL ORDER BY created DESC, rowid DESC"
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT * FROM reports WHERE user_id=? ORDER BY created DESC, rowid DESC",
                    (user_id,),
                ).fetchall()
        return [{
            "report_id": r["id"], "title": r["title"], "date": r["created"][:10],
            "n": r["n"], "best": r["best"], "short": len(json.loads(r["shortlist"])),
            "search_id": r["search_id"], "shortlist": json.loads(r["shortlist"]),
        } for r in rows]

    def delete_report(self, report_id: str, user_id: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM reports WHERE id=? AND user_id=?", (report_id, user_id))
            self.db.commit()
            return cur.rowcount == 1

    def get_report(self, report_id: str, user_id: str | None = None) -> dict | None:
        with self.lock:
            if user_id is None:
                r = self.db.execute(
                    "SELECT * FROM reports WHERE id=? AND user_id IS NULL", (report_id,)
                ).fetchone()
            else:
                r = self.db.execute(
                    "SELECT * FROM reports WHERE id=? AND user_id=?", (report_id, user_id)
                ).fetchone()
        if not r:
            return None
        return {
            "report_id": r["id"], "title": r["title"], "created": r["created"],
            "search_id": r["search_id"], "shortlist": json.loads(r["shortlist"]),
            "n": r["n"], "best": r["best"],
        }

    # settings -------------------------------------------------------------------
    def get_settings(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def put_settings(self, values: dict) -> None:
        with self.lock:
            for key, value in values.items():
                self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))
            self.db.commit()

    # LLM cache ------------------------------------------------------------------
    def cache_get(self, key: str) -> str | None:
        with self.lock:
            row = self.db.execute("SELECT value FROM llm_cache WHERE key=?", (key,)).fetchone()
            if row:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
        return row["value"] if row else None

    def cache_put(self, key: str, value: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO llm_cache VALUES (?,?)", (key, value))
            self.db.commit()

    # guardrail events -------------------------------------------------------------
    def log_guardrail(self, stage: str, action: str, rules: list[str], excerpt: str, source: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO guardrail_events (ts, stage, action, rules, excerpt, source) VALUES (?,?,?,?,?,?)",
                            (time.time(), stage, action, json.dumps(rules), excerpt[:120], source))
            self.db.commit()

    def guardrail_summary(self, limit: int = 30) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT * FROM guardrail_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            counts = self.db.execute("SELECT stage, action, COUNT(*) n FROM guardrail_events GROUP BY stage, action").fetchall()
            by_rule: dict[str, int] = {}
            for r in self.db.execute("SELECT rules FROM guardrail_events").fetchall():
                for rule in json.loads(r["rules"]):
                    by_rule[rule] = by_rule.get(rule, 0) + 1
        return {"counts": {f"{c['stage']}_{c['action']}": c["n"] for c in counts}, "by_rule": by_rule,
                "recent": [{"time": datetime.fromtimestamp(r["ts"], timezone.utc).isoformat(timespec="seconds"),
                            "stage": r["stage"], "action": r["action"], "rules": json.loads(r["rules"]),
                            "excerpt": r["excerpt"], "source": r["source"]} for r in rows]}

    # LLM token usage --------------------------------------------------------------
    def log_usage(self, task: str, model: str, prompt: int, completion: int, total: int, cached: bool, ms: int) -> None:
        with self.lock:
            self.db.execute("INSERT INTO llm_usage (ts, task, model, prompt_tokens, completion_tokens, total_tokens, cached, ms)"
                            " VALUES (?,?,?,?,?,?,?,?)", (time.time(), task, model, prompt, completion, total, int(cached), ms))
            self.db.commit()

    def tokens_since(self, seconds: float) -> int:
        with self.lock:
            return self.db.execute("SELECT COALESCE(SUM(total_tokens),0) FROM llm_usage WHERE cached=0 AND ts>=?",
                                   (time.time() - seconds,)).fetchone()[0]

    def tokens_total(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COALESCE(SUM(total_tokens),0) FROM llm_usage WHERE cached=0").fetchone()[0]

    def usage_summary(self, recent: int = 15) -> dict:
        day = time.time() - 86400
        with self.lock:
            by_task = self.db.execute(
                "SELECT task, COUNT(*) calls, SUM(prompt_tokens) p, SUM(completion_tokens) c, SUM(total_tokens) t FROM llm_usage"
                " WHERE cached=0 AND ts>=? GROUP BY task ORDER BY t DESC", (day,)).fetchall()
            by_day = self.db.execute(
                "SELECT date(ts,'unixepoch') d, SUM(total_tokens) t, COUNT(*) calls FROM llm_usage WHERE cached=0 AND ts>=?"
                " GROUP BY d ORDER BY d", (time.time() - 7 * 86400,)).fetchall()
            saved = self.db.execute("SELECT COUNT(*) n, COALESCE(SUM(total_tokens),0) t FROM llm_usage WHERE cached=1").fetchone()
            rows = self.db.execute("SELECT * FROM llm_usage ORDER BY id DESC LIMIT ?", (recent,)).fetchall()
            allc = self.db.execute("SELECT COUNT(*) FROM llm_usage WHERE cached=0").fetchone()[0]
        return {
            "by_task": [{"task": r["task"], "calls": r["calls"], "prompt": r["p"], "completion": r["c"], "total": r["t"]} for r in by_task],
            "by_day": [{"day": r["d"], "tokens": r["t"], "calls": r["calls"]} for r in by_day],
            "cache_hits": saved["n"], "tokens_saved_estimate": saved["t"], "calls_total": allc,
            "recent": [{"time": datetime.fromtimestamp(r["ts"], timezone.utc).isoformat(timespec="seconds"), "task": r["task"],
                        "model": r["model"], "prompt": r["prompt_tokens"], "completion": r["completion_tokens"],
                        "total": r["total_tokens"], "cached": bool(r["cached"]), "ms": r["ms"]} for r in rows]}

    # deletion and statistics (PRD NFR-08, NFR-09) -----------------------------------
    def delete_search(self, search_id: str, user_id: str | None = None) -> dict | None:
        with self.lock:
            if user_id is None:
                found = self.db.execute(
                    "SELECT 1 FROM searches WHERE id=? AND user_id IS NULL", (search_id,)
                ).fetchone()
                if not found:
                    return None
                reports = self.db.execute(
                    "DELETE FROM reports WHERE search_id=? AND user_id IS NULL", (search_id,)
                ).rowcount
                self.db.execute(
                    "DELETE FROM searches WHERE id=? AND user_id IS NULL", (search_id,)
                )
            else:
                found = self.db.execute(
                    "SELECT 1 FROM searches WHERE id=? AND user_id=?", (search_id, user_id)
                ).fetchone()
                if not found:
                    return None
                reports = self.db.execute(
                    "DELETE FROM reports WHERE search_id=? AND user_id=?", (search_id, user_id)
                ).rowcount
                self.db.execute(
                    "DELETE FROM searches WHERE id=? AND user_id=?", (search_id, user_id)
                )
            self.db.commit()
        return {"searches": 1, "reports": reports}

    def delete_all(self, user_id: str | None = None) -> dict:
        with self.lock:
            if user_id is not None:
                searches = self.db.execute(
                    "SELECT COUNT(*) FROM searches WHERE user_id=?", (user_id,)
                ).fetchone()[0]
                reports = self.db.execute(
                    "SELECT COUNT(*) FROM reports WHERE user_id=?", (user_id,)
                ).fetchone()[0]
                self.db.execute("DELETE FROM reports WHERE user_id=?", (user_id,))
                self.db.execute("DELETE FROM searches WHERE user_id=?", (user_id,))
                self.db.commit()
                return {"searches": searches, "reports": reports, "llm_cache": 0}

            counts = {
                name: self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for name, table in (
                    ("searches", "searches"), ("reports", "reports"), ("llm_cache", "llm_cache")
                )
            }
            for table in ("searches", "reports", "llm_cache", "guardrail_events"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.commit()
        return counts

    def counts(self, user_id: str | None = None) -> dict:
        with self.lock:
            if user_id is None:
                return {
                    "searches": self.db.execute(
                        "SELECT COUNT(*) FROM searches WHERE user_id IS NULL"
                    ).fetchone()[0],
                    "reports": self.db.execute(
                        "SELECT COUNT(*) FROM reports WHERE user_id IS NULL"
                    ).fetchone()[0],
                }
            return {
                "searches": self.db.execute(
                    "SELECT COUNT(*) FROM searches WHERE user_id=?", (user_id,)
                ).fetchone()[0],
                "reports": self.db.execute(
                    "SELECT COUNT(*) FROM reports WHERE user_id=?", (user_id,)
                ).fetchone()[0],
            }

    def cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        return {"hits": self.cache_hits, "misses": self.cache_misses,
                "hit_rate": round(self.cache_hits / total, 4) if total else None}


# ----------------------------------------------------------------- history, similar searches, memory notes
MAX_NOTES_PER_USER = 50
MAX_NOTE_CHARS = 1000
SIMILAR_SCAN = 300


def _idf_vectors(docs: list[list[str]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    from collections import Counter
    from math import log

    df: Counter = Counter()
    for toks in docs:
        df.update(set(toks))
    n = len(docs)
    idf = {t: log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    vecs = []
    for toks in docs:
        tf = Counter(toks)
        vecs.append({t: (1 + log(c)) * idf[t] for t, c in tf.items()})
    return idf, vecs


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(t, 0.0) for t, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def rank_by_similarity(query: str, items: list[tuple[str, str]], limit: int, min_score: float) -> list[tuple[str, float]]:
    """TF-IDF cosine of `query` against (key, text) pairs. Local and deterministic: no model call, no tokens."""
    if not items:
        return []
    docs = [tokenize(t) for _, t in items] + [tokenize(query)]
    _, vecs = _idf_vectors(docs)
    q = vecs[-1]
    scored = [(items[i][0], round(_cosine(q, vecs[i]), 4)) for i in range(len(items))]
    scored = [x for x in scored if x[1] >= min_score]
    scored.sort(key=lambda x: -x[1])
    return scored[:limit]


class _StoreMemory:
    """Mixed into Store. Every query is scoped by user_id; there is no unscoped read."""

    def history_page(self, user_id: str, limit: int = 20, offset: int = 0, q: str = "") -> dict:
        like = f"%{q.strip().lower()}%" if q.strip() else None
        with self.lock:
            if like:
                total = self.db.execute("SELECT COUNT(*) FROM searches WHERE user_id=? AND lower(request) LIKE ?", (user_id, like)).fetchone()[0]
                rows = self.db.execute("SELECT * FROM searches WHERE user_id=? AND lower(request) LIKE ? ORDER BY created DESC, rowid DESC LIMIT ? OFFSET ?",
                                       (user_id, like, limit, offset)).fetchall()
            else:
                total = self.db.execute("SELECT COUNT(*) FROM searches WHERE user_id=?", (user_id,)).fetchone()[0]
                rows = self.db.execute("SELECT * FROM searches WHERE user_id=? ORDER BY created DESC, rowid DESC LIMIT ? OFFSET ?",
                                       (user_id, limit, offset)).fetchall()
        items = []
        for r in rows:
            req, res = json.loads(r["request"]), json.loads(r["response"])
            items.append({"search_id": r["id"], "created": r["created"], "claim": req.get("claim", "")[:240],
                          "results": len(res.get("results", [])), "k": req.get("k")})
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def similar_searches(self, user_id: str, claim: str, limit: int = 5, exclude_id: str | None = None,
                         min_score: float = 0.2) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id, created, request FROM searches WHERE user_id=? ORDER BY created DESC, rowid DESC LIMIT ?",
                                   (user_id, SIMILAR_SCAN)).fetchall()
        meta, items = {}, []
        for r in rows:
            if r["id"] == exclude_id:
                continue
            text = json.loads(r["request"]).get("claim", "")
            meta[r["id"]] = (r["created"], text)
            items.append((r["id"], text))
        ranked = rank_by_similarity(claim, items, limit, min_score)
        return [{"search_id": sid, "score": score, "created": meta[sid][0], "claim": meta[sid][1][:240]} for sid, score in ranked]

    # notes -----------------------------------------------------------------------
    def list_notes(self, user_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id,text,created,updated FROM memory_notes WHERE user_id=? ORDER BY updated DESC, rowid DESC", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_note(self, user_id: str, text: str) -> dict | None:
        """None when the per-user cap is reached."""
        note_id, now = "n_" + secrets.token_hex(8), _now()
        with self.lock:
            if self.db.execute("SELECT COUNT(*) FROM memory_notes WHERE user_id=?", (user_id,)).fetchone()[0] >= MAX_NOTES_PER_USER:
                return None
            self.db.execute("INSERT INTO memory_notes (id,user_id,text,created,updated) VALUES (?,?,?,?,?)", (note_id, user_id, text, now, now))
            self.db.commit()
        return {"id": note_id, "text": text, "created": now, "updated": now}

    def update_note(self, user_id: str, note_id: str, text: str) -> dict | None:
        now = _now()
        with self.lock:
            cur = self.db.execute("UPDATE memory_notes SET text=?, updated=? WHERE id=? AND user_id=?", (text, now, note_id, user_id))
            self.db.commit()
            if cur.rowcount != 1:
                return None
            row = self.db.execute("SELECT id,text,created,updated FROM memory_notes WHERE id=? AND user_id=?", (note_id, user_id)).fetchone()
        return dict(row)

    def delete_note(self, user_id: str, note_id: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM memory_notes WHERE id=? AND user_id=?", (note_id, user_id))
            self.db.commit()
            return cur.rowcount == 1

    def delete_all_notes(self, user_id: str) -> int:
        with self.lock:
            cur = self.db.execute("DELETE FROM memory_notes WHERE user_id=?", (user_id,))
            self.db.commit()
            return cur.rowcount

    def relevant_notes(self, user_id: str, claim: str, limit: int = 3, min_score: float = 0.1) -> list[dict]:
        notes = self.list_notes(user_id)
        ranked = rank_by_similarity(claim, [(n["id"], n["text"]) for n in notes], limit, min_score)
        by_id = {n["id"]: n for n in notes}
        return [{**by_id[nid], "score": score} for nid, score in ranked]

    def claim_legacy(self, user_id: str) -> dict:
        """Admin CLI only: give searches and reports that have no owner to one user."""
        with self.lock:
            s = self.db.execute("UPDATE searches SET user_id=? WHERE user_id IS NULL", (user_id,)).rowcount
            r = self.db.execute("UPDATE reports SET user_id=? WHERE user_id IS NULL", (user_id,)).rowcount
            self.db.commit()
        return {"searches": s, "reports": r}


for _name in [n for n in vars(_StoreMemory) if not n.startswith("__")]:
    setattr(Store, _name, vars(_StoreMemory)[_name])
