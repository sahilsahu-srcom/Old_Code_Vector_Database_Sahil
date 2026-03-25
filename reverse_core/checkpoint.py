"""
File Summary:
SQLite WAL checkpoint for HAUP v3.0 Reverse Pipeline. Crash-safe progress tracking with chunk status management.

====================================================================
Startup
====================================================================

SQLiteCheckpoint()
||
├── __init__()  [Function] -------------------------------> Set db_path, threading lock, call _init_db()
│
├── _init_db()  [Function] -------------------------------> Initialize database schema
│       │
│       ├── PRAGMA journal_mode=WAL ----------------------> Enable WAL mode for crash safety
│       │
│       ├── CREATE TABLE chunks --------------------------> Chunk status tracking table
│       │
│       └── CREATE TABLE job_meta ------------------------> Key-value metadata storage
│
├── _connect()  [Function] -------------------------------> Open SQLite connection with Row factory
│
├── _upsert_status()  [Function] -------------------------> Generic chunk status upsert helper
│
├── _fetchone()  [Function] ------------------------------> Execute query and return single row
│
├── _fetchall()  [Function] ------------------------------> Execute query and return all rows
│
├── is_done()  [Function] --------------------------------> Check if chunk status is 'done'
│
├── mark_running()  [Function] ---------------------------> Upsert chunk status to 'running'
│
├── mark_done()  [Function] ------------------------------> Upsert chunk status to 'done' with rows_done
│
├── mark_failed()  [Function] ----------------------------> Upsert chunk status to 'failed'
│
├── get_resume_summary()  [Function] ---------------------> GROUP BY status for done/running/failed/pending
│       │
│       └── Returns ResumeSummary  [Class → Object] ------> Dataclass with done, running, failed, pending, total
│
├── get_failed_chunks()  [Function] ----------------------> Return list of all failed chunk_ids
│
├── reset_running_to_pending()  [Function] ---------------> UPDATE 'running' → 'pending' for crash recovery
│
├── save_meta()  [Function] ------------------------------> Upsert key-value pair into job_meta
│
├── get_meta()  [Function] -------------------------------> Fetch value by key from job_meta
│
├── total_rows_written()  [Function] ---------------------> SUM(rows_done) for all 'done' chunks
│
└── close()  [Function] ----------------------------------> No-op connection cleanup placeholder

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


"""================= Startup class ResumeSummary ================="""
@dataclass
class ResumeSummary:
    done:    int
    running: int
    failed:  int
    pending: int
    total:   int

    def __str__(self) -> str:
        return (
            f"done={self.done}  running={self.running}  "
            f"failed={self.failed}  pending={self.pending}  total={self.total}"
        )
"""================= End class ResumeSummary ================="""


"""================= Startup class SQLiteCheckpoint ================="""
class SQLiteCheckpoint:

    """================= Startup method __init__ ================="""
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        import threading
        self._lock = threading.Lock()
        self._init_db()
    """================= End method __init__ ================="""

    """================= Startup method _init_db ================="""
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id   INTEGER PRIMARY KEY,
                    status     TEXT    NOT NULL DEFAULT 'pending',
                    rowid_col  TEXT,
                    source     TEXT,
                    rows_done  INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS job_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            conn.commit()
    """================= End method _init_db ================="""

    """================= Startup method _connect ================="""
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    """================= End method _connect ================="""

    """================= Startup method _upsert_status ================="""
    def _upsert_status(self, chunk_id: int, status: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO chunks (chunk_id, status, updated_at)
                       VALUES (?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         status     = excluded.status,
                         updated_at = CURRENT_TIMESTAMP""",
                    (chunk_id, status),
                )
                conn.commit()
    """================= End method _upsert_status ================="""

    """================= Startup method _fetchone ================="""
    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()
    """================= End method _fetchone ================="""

    """================= Startup method _fetchall ================="""
    def _fetchall(self, sql: str, params: tuple = ()) -> list:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()
    """================= End method _fetchall ================="""

    """================= Startup method is_done ================="""
    def is_done(self, chunk_id: int) -> bool:
        with self._lock:
            row = self._fetchone(
                "SELECT status FROM chunks WHERE chunk_id = ?", (chunk_id,)
            )
            return row is not None and row[0] == "done"
    """================= End method is_done ================="""

    """================= Startup method mark_running ================="""
    def mark_running(self, chunk_id: int) -> None:
        self._upsert_status(chunk_id, "running")
    """================= End method mark_running ================="""

    """================= Startup method mark_done ================="""
    def mark_done(self, chunk_id: int, rows_done: int = 0) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO chunks (chunk_id, status, rows_done, updated_at)
                       VALUES (?, 'done', ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         status = 'done',
                         rows_done = excluded.rows_done,
                         updated_at = CURRENT_TIMESTAMP""",
                    (chunk_id, rows_done),
                )
                conn.commit()
    """================= End method mark_done ================="""

    """================= Startup method mark_failed ================="""
    def mark_failed(self, chunk_id: int) -> None:
        self._upsert_status(chunk_id, "failed")

    def retry_failed_chunks(self) -> int:
        """Reset failed chunks to allow retry. Returns number of chunks reset."""
        with self._lock:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE status = 'failed'")
            failed_count = cursor.fetchone()[0]
            
            if failed_count > 0:
                cursor.execute("DELETE FROM chunks WHERE status = 'failed'")
                conn.commit()
            
            conn.close()
            return failed_count
    """================= End method mark_failed ================="""

    """================= Startup method get_resume_summary ================="""
    def get_resume_summary(self) -> ResumeSummary:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM chunks GROUP BY status"
                ).fetchall()
        counts  = {r[0]: r[1] for r in rows}
        done    = counts.get("done",    0)
        running = counts.get("running", 0)
        failed  = counts.get("failed",  0)
        pending = counts.get("pending", 0)
        total   = done + running + failed + pending
        return ResumeSummary(done, running, failed, pending, total)
    """================= End method get_resume_summary ================="""

    """================= Startup method get_failed_chunks ================="""
    def get_failed_chunks(self) -> list[int]:
        with self._lock:
            rows = self._fetchall(
                "SELECT chunk_id FROM chunks WHERE status = 'failed' ORDER BY chunk_id"
            )
        return [r[0] for r in rows]
    """================= End method get_failed_chunks ================="""

    """================= Startup method reset_running_to_pending ================="""
    def reset_running_to_pending(self) -> int:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "UPDATE chunks SET status = 'pending' WHERE status = 'running'"
                )
                conn.commit()
                return cursor.rowcount
    """================= End method reset_running_to_pending ================="""

    """================= Startup method save_meta ================="""
    def save_meta(self, key: str, value: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO job_meta (key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
                conn.commit()
    """================= End method save_meta ================="""

    """================= Startup method get_meta ================="""
    def get_meta(self, key: str) -> Optional[str]:
        row = self._fetchone("SELECT value FROM job_meta WHERE key = ?", (key,))
        return row[0] if row else None
    """================= End method get_meta ================="""

    """================= Startup method total_rows_written ================="""
    def total_rows_written(self) -> int:
        row = self._fetchone("SELECT COALESCE(SUM(rows_done), 0) FROM chunks WHERE status = 'done'")
        return row[0] if row else 0
    """================= End method total_rows_written ================="""

    """================= Startup method close ================="""
    def close(self) -> None:
        pass
    """================= End method close ================="""

"""================= End class SQLiteCheckpoint ================="""


Checkpoint = SQLiteCheckpoint


"""
====================================================================
How to Run
====================================================================

Import and use in reverse pipeline:
    from reverse_core.checkpoint import SQLiteCheckpoint

    checkpoint = SQLiteCheckpoint("./reverse_job.db")

Mark chunk states:
    checkpoint.mark_running(chunk_id=0)
    checkpoint.mark_done(chunk_id=0, rows_done=500)
    checkpoint.mark_failed(chunk_id=1)

Resume check:
    summary = checkpoint.get_resume_summary()
    print(summary)
    # done=1  running=0  failed=1  pending=0  total=2

Crash recovery (reset 'running' → 'pending' on restart):
    recovered = checkpoint.reset_running_to_pending()
    print(f"Recovered {recovered} crashed chunks")

Get all failed chunks for retry:
    failed_ids = checkpoint.get_failed_chunks()

Save and retrieve metadata:
    checkpoint.save_meta("collection", "haup_vectors")
    value = checkpoint.get_meta("collection")

Get total rows written across all done chunks:
    total = checkpoint.total_rows_written()
    print(f"Total rows extracted: {total}")
"""
