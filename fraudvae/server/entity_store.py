"""v5: the server's memory of the transactions it has scored, keyed by (source, customer id),
so a caller that identifies the customer gets the history-conditioned score without having
to send the history every time. SQLite in the app home; one row per scored transaction
with the model-space encoding (standardised numerics, mask, categorical codes) so the
history window can be rebuilt without re-encoding.

This is the smallest thing that behaves like a feature store: append after scoring, read
the last N before scoring. It is not a transaction log for the bank; it is per-run
(the encoding depends on the run's standardisation stats), so switching runs starts a
new table.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

# rows kept per (source, customer): the history window is at most a few dozen rows, so the
# oldest beyond this are never read again and only make the file grow
MAX_ROWS_PER_CUSTOMER = int(os.environ.get("FRAUDVAE_HISTORY_MAX_ROWS", "1000"))


def _table_name(run_name: str) -> str:
    """a table per run. the readable part is the run name with anything but letters and digits
    turned into underscores; a short hash of the exact name follows, so two runs that sanitise
    to the same word (v5-h200-e and v5_h200_e) do not share a history"""
    readable = "".join(c if c.isalnum() else "_" for c in run_name)
    return f"h_{readable}_{hashlib.sha1(run_name.encode()).hexdigest()[:8]}"


class EntityStore:
    def __init__(self, path: Path, run_name: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.table = _table_name(run_name)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        # readers and the one writer no longer block each other, and a crash mid-write leaves a
        # consistent file
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate_unhashed(run_name)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (source TEXT, customer TEXT, ts REAL, "
                          "num BLOB, mask BLOB, cat BLOB, added REAL)")
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_k ON {self.table} (source, customer, ts)")
        # v5: one json profile per (source, customer): the behaviour enrolment (embeddings per modality)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table}_profile (source TEXT, customer TEXT, profile TEXT, "
                          f"sessions INTEGER, updated REAL, PRIMARY KEY (source, customer))")
        self.conn.commit()

    def _migrate_unhashed(self, run_name: str):
        """a store written before the hash suffix existed has its tables under the bare sanitised
        name; carry them over once rather than start the customer's history from nothing"""
        old = "h_" + "".join(c if c.isalnum() else "_" for c in run_name)
        if old == self.table:
            return
        have = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if old in have and self.table not in have:
            self.conn.execute(f"ALTER TABLE {old} RENAME TO {self.table}")
            if f"{old}_profile" in have:
                self.conn.execute(f"ALTER TABLE {old}_profile RENAME TO {self.table}_profile")
            self.conn.commit()

    def append(self, source: str, customer: str, ts: float, num: np.ndarray, mask: np.ndarray, cat: np.ndarray):
        with self._lock:
            self.conn.execute(f"INSERT INTO {self.table} VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (source or "", str(customer), float(ts), np.asarray(num, np.float32).tobytes(),
                               np.asarray(mask, np.float32).tobytes(), np.asarray(cat, np.int64).tobytes(), time.time()))
            # keep the newest MAX_ROWS_PER_CUSTOMER rows of this customer: the window never reads further back
            self.conn.execute(
                f"DELETE FROM {self.table} WHERE source = ? AND customer = ? AND rowid NOT IN "
                f"(SELECT rowid FROM {self.table} WHERE source = ? AND customer = ? ORDER BY ts DESC, rowid DESC LIMIT ?)",
                (source or "", str(customer), source or "", str(customer), MAX_ROWS_PER_CUSTOMER))
            self.conn.commit()

    def last(self, source: str, customer: str, before_ts: float, n: int):
        """the n most recent rows strictly before before_ts, oldest first:
        list of (ts, num, mask, cat) numpy tuples"""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT ts, num, mask, cat FROM {self.table} WHERE source = ? AND customer = ? AND ts < ? "
                "ORDER BY ts DESC LIMIT ?", (source or "", str(customer), float(before_ts), int(n))).fetchall()
        out = []
        for ts, num, mask, cat in reversed(rows):
            out.append((float(ts), np.frombuffer(num, np.float32), np.frombuffer(mask, np.float32), np.frombuffer(cat, np.int64)))
        return out

    def count(self, source: str, customer: str) -> int:
        with self._lock:
            return int(self.conn.execute(f"SELECT COUNT(*) FROM {self.table} WHERE source = ? AND customer = ?",
                                         (source or "", str(customer))).fetchone()[0])

    def forget(self, source: str, customer: str) -> int:
        with self._lock:
            cur = self.conn.execute(f"DELETE FROM {self.table} WHERE source = ? AND customer = ?", (source or "", str(customer)))
            self.conn.execute(f"DELETE FROM {self.table}_profile WHERE source = ? AND customer = ?", (source or "", str(customer)))
            self.conn.commit()
            return int(cur.rowcount)

    def profile(self, source: str, customer: str) -> tuple[dict | None, int]:
        """(behaviour enrolment or None, sessions it was built from)"""
        import json

        with self._lock:
            row = self.conn.execute(f"SELECT profile, sessions FROM {self.table}_profile WHERE source = ? AND customer = ?",
                                    (source or "", str(customer))).fetchone()
        return (json.loads(row[0]), int(row[1])) if row else (None, 0)

    def set_profile(self, source: str, customer: str, profile: dict, sessions: int):
        import json

        with self._lock:
            self.conn.execute(f"INSERT OR REPLACE INTO {self.table}_profile VALUES (?, ?, ?, ?, ?)",
                              (source or "", str(customer), json.dumps(profile), int(sessions), time.time()))
            self.conn.commit()

    def close(self):
        """an evicted generation's store is closed, not left to the garbage collector"""
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
