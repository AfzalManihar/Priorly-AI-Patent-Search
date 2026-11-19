"""SQLite storage: search log, saved reports, settings and the LLM cache (PRD section 14)."""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

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
  learning_topic TEXT NOT NULL DEFAULT '', memory_enabled INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  csrf_hash TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL);
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
            for name, definition in (("email", "TEXT NOT NULL DEFAULT ''"), ("display_name", "TEXT NOT NULL DEFAULT ''"), ("learning_topic", "TEXT NOT NULL DEFAULT ''"), ("memory_enabled", "INTEGER NOT NULL DEFAULT 1")):
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
