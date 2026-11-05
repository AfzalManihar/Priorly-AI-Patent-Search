"""SQLite storage: search log, saved reports, settings and the LLM cache (PRD section 14)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
  id TEXT PRIMARY KEY, created TEXT NOT NULL, request TEXT NOT NULL, response TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY, created TEXT NOT NULL, title TEXT NOT NULL, search_id TEXT,
  shortlist TEXT NOT NULL, n INTEGER NOT NULL, best REAL NOT NULL);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        with self.lock:
            self.db.executescript(SCHEMA)

    # search log -----------------------------------------------------------------
    def save_search(self, search_id: str, request: dict, response: dict) -> None:
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO searches VALUES (?,?,?,?)",
                (search_id, _now(), json.dumps(request), json.dumps(response)),
            )
            self.db.commit()

    def get_search(self, search_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM searches WHERE id=?", (search_id,)).fetchone()
        if not row:
            return None
        return {"search_id": row["id"], "created": row["created"],
                "request": json.loads(row["request"]), "response": json.loads(row["response"])}

    # reports --------------------------------------------------------------------
    def save_report(self, report_id: str, title: str, search_id: str | None,
                    shortlist: list[str], n: int, best: float) -> dict:
        created = _now()
        with self.lock:
            self.db.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?)",
                            (report_id, created, title, search_id, json.dumps(shortlist), n, best))
            self.db.commit()
        return {"report_id": report_id, "created": created}

    def list_reports(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM reports ORDER BY created DESC, rowid DESC").fetchall()
        return [{"report_id": r["id"], "title": r["title"], "date": r["created"][:10],
                 "n": r["n"], "best": r["best"], "short": len(json.loads(r["shortlist"])),
                 "search_id": r["search_id"], "shortlist": json.loads(r["shortlist"])} for r in rows]

    def get_report(self, report_id: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        if not r:
            return None
        return {"report_id": r["id"], "title": r["title"], "created": r["created"], "search_id": r["search_id"],
                "shortlist": json.loads(r["shortlist"]), "n": r["n"], "best": r["best"]}

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

    # deletion and statistics (PRD NFR-08, NFR-09) -----------------------------------
    def delete_search(self, search_id: str) -> dict | None:
        """Remove one stored search and every report built from it. None if it does not exist."""
        with self.lock:
            found = self.db.execute("SELECT 1 FROM searches WHERE id=?", (search_id,)).fetchone()
            if not found:
                return None
            reports = self.db.execute("DELETE FROM reports WHERE search_id=?", (search_id,)).rowcount
            self.db.execute("DELETE FROM searches WHERE id=?", (search_id,))
            self.db.commit()
        return {"searches": 1, "reports": reports}

    def delete_all(self) -> dict:
        """Remove every stored claim: searches, reports and cached LLM text. Settings are kept."""
        with self.lock:
            counts = {name: self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                      for name, table in (("searches", "searches"), ("reports", "reports"), ("llm_cache", "llm_cache"))}
            for table in ("searches", "reports", "llm_cache"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.commit()
        return counts

    def counts(self) -> dict:
        with self.lock:
            return {"searches": self.db.execute("SELECT COUNT(*) FROM searches").fetchone()[0],
                    "reports": self.db.execute("SELECT COUNT(*) FROM reports").fetchone()[0]}

    def cache_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        return {"hits": self.cache_hits, "misses": self.cache_misses,
                "hit_rate": round(self.cache_hits / total, 4) if total else None}
