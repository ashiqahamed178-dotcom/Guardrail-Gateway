"""SQLite log of every guard decision. Stores a hash of the text by default, not the text."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    route TEXT NOT NULL,
    role TEXT NOT NULL,
    detector TEXT NOT NULL,
    score REAL NOT NULL,
    blocked INTEGER NOT NULL,
    reason TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    text_ref TEXT NOT NULL
)
"""


class EventStore:
    def __init__(self, path: str = "guard.db", log_text: bool = False):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.log_text = log_text
        self.lock = threading.Lock()

    def add(self, route: str, role: str, verdict, text: str) -> None:
        ref = text[:200] if self.log_text else "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]
        with self.lock:
            self.conn.execute(
                "INSERT INTO events (ts, route, role, detector, score, blocked, reason, latency_ms, text_ref) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), route, role, verdict.detector, verdict.score, int(verdict.blocked),
                 verdict.reason, verdict.latency_ms, ref),
            )
            self.conn.commit()

    def stats(self) -> dict:
        with self.lock:
            rows = self.conn.execute("SELECT blocked, latency_ms FROM events").fetchall()
            recent = self.conn.execute(
                "SELECT ts, route, role, detector, score, blocked, reason, latency_ms, text_ref "
                "FROM events ORDER BY id DESC LIMIT 20").fetchall()
            reasons = self.conn.execute(
                "SELECT reason, COUNT(*) FROM events WHERE blocked = 1 GROUP BY detector, reason "
                "ORDER BY COUNT(*) DESC LIMIT 10").fetchall()
        lat = sorted(r[1] for r in rows)
        keys = ["ts", "route", "role", "detector", "score", "blocked", "reason", "latency_ms", "text_ref"]
        return {
            "checked": len(rows),
            "blocked": sum(r[0] for r in rows),
            "latency_ms_p50": _pct(lat, 50),
            "latency_ms_p95": _pct(lat, 95),
            "top_block_reasons": [{"reason": r, "count": c} for r, c in reasons],
            "recent": [dict(zip(keys, r)) for r in recent],
        }


def _pct(values: list[float], p: int) -> float | None:
    if not values:
        return None
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]
