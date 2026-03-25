"""
File Summary:
SQLite WAL checkpoint for HAUP v2.0. Crash-safe progress tracker with row-based and chunk-based tracking.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

SQLiteCheckpoint()  [Class → Object]
||
├── __init__()  [Method] ---------------------------------> Set db_path, threading lock, call _init()
│
├── _init()  [Method] ------------------------------------> Initialize database schema
│       │
│       ├── PRAGMA journal_mode=WAL ----------------------> Enable WAL mode for crash safety
│       │
│       ├── CREATE TABLE chunks --------------------------> Chunk status tracking table
│       │
│       ├── CREATE TABLE processed_rows ------------------> Individual row tracking table
│       │
│       └── CREATE TABLE worker_stats --------------------> Per-worker performance table
│
├── _connect()  [Method] ---------------------------------> Open SQLite connection with WAL mode
│
├── is_done()  [Method] ----------------------------------> Check if chunk status is 'done'
│
├── mark_running()  [Method] -----------------------------> Upsert chunk status to 'running'
│
├── mark_done()  [Method] --------------------------------> Upsert chunk status to 'done'
│
├── mark_failed()  [Method] ------------------------------> Upsert chunk status to 'failed'
│
├── retry_failed_chunks()  [Method] ----------------------> Reset failed chunks to allow retry
│
├── get_resume_summary()  [Method] -----------------------> Count done / running / failed chunks
│       │
│       └── Returns ResumeSummary  [Class → Object] ------> Dataclass with done, running, failed
│
├── save_worker_stats()  [Method] ------------------------> Delete and re-insert worker statistics
│       │
│       └── [Early Exit Branch] --------------------------> Return if stats list is empty
│
├── get_worker_stats()  [Method] -------------------------> Load worker statistics as list of dicts
│
├── mark_row_processed()  [Method] -----------------------> Insert row into processed_rows (ignore dup)
│
├── is_row_processed()  [Method] -------------------------> Check if row_id exists in processed_rows
│
├── get_processed_row_count()  [Method] ------------------> COUNT(*) from processed_rows
│
├── get_max_processed_row_id()  [Method] -----------------> MAX(row_id) from processed_rows
│
└── migrate_chunk_to_row_tracking()  [Method] ------------> Convert legacy chunk records to row records
        │
        ├── SELECT completed chunks from chunks table ----> Find all 'done' chunk_ids
        │
        ├── [Early Exit Branch] --------------------------> Return 0 if no completed chunks
        │
        ├── [Conditional Branch] data_source.type == 'sql' -> Migrate rows for SQL sources only
        │       │
        │       └── INSERT OR IGNORE rows 1..100 ---------> Back-fill processed_rows table
        │
        └── [Exception Block] ----------------------------> Log error and return 0

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("haup.checkpoint")


"""================= Startup class ResumeSummary ================="""
@dataclass
class ResumeSummary:
    done:    int
    running: int
    failed:  int
"""================= End class ResumeSummary ================="""


"""================= Startup class SQLiteCheckpoint ================="""
class SQLiteCheckpoint:

    """================= Startup method __init__ ================="""
    def __init__(self, db_path: str = "job.db"):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init()
    """================= End method __init__ ================="""

    """================= Startup method _init ================="""
    def _init(self) -> None:
        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id   INTEGER PRIMARY KEY,
                status     TEXT DEFAULT 'pending',
                rowid_col  TEXT,
                source     TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_rows (
                row_id     INTEGER PRIMARY KEY,
                source     TEXT,
                table_name TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_stats (
                worker_id      INTEGER PRIMARY KEY,
                rows_processed INTEGER DEFAULT 0,
                final_batch    INTEGER DEFAULT 0,
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.commit()
        conn.close()
    """================= End method _init ================="""

    """================= Startup method _connect ================="""
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    """================= End method _connect ================="""

    """================= Startup method is_done ================="""
    def is_done(self, chunk_id: int) -> bool:
        conn = self._connect()
        row  = conn.execute(
            "SELECT status FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        conn.close()
        return row is not None and row[0] == 'done'
    """================= End method is_done ================="""

    """================= Startup method mark_running ================="""
    def mark_running(self, chunk_id: int,
                     rowid_col: str = "", source: str = "") -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO chunks (chunk_id, status, rowid_col, source, updated_at)
                   VALUES (?, 'running', ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                       status='running', updated_at=CURRENT_TIMESTAMP""",
                (chunk_id, rowid_col, source)
            )
            conn.commit()
            conn.close()
    """================= End method mark_running ================="""

    """================= Startup method mark_done ================="""
    def mark_done(self, chunk_id: int) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO chunks (chunk_id, status, updated_at)
                   VALUES (?, 'done', CURRENT_TIMESTAMP)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                       status='done', updated_at=CURRENT_TIMESTAMP""",
                (chunk_id,)
            )
            conn.commit()
            conn.close()
    """================= End method mark_done ================="""

    """================= Startup method mark_failed ================="""
    def mark_failed(self, chunk_id: int) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO chunks (chunk_id, status, updated_at)
                   VALUES (?, 'failed', CURRENT_TIMESTAMP)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                       status='failed', updated_at=CURRENT_TIMESTAMP""",
                (chunk_id,)
            )
            conn.commit()
            conn.close()
    """================= End method mark_failed ================="""

    """================= Startup method retry_failed_chunks ================="""
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
    """================= End method retry_failed_chunks ================="""

    """================= Startup method get_resume_summary ================="""
    def get_resume_summary(self) -> ResumeSummary:
        conn    = self._connect()
        done    = conn.execute("SELECT COUNT(*) FROM chunks WHERE status='done'").fetchone()[0]
        running = conn.execute("SELECT COUNT(*) FROM chunks WHERE status='running'").fetchone()[0]
        failed  = conn.execute("SELECT COUNT(*) FROM chunks WHERE status='failed'").fetchone()[0]
        conn.close()
        return ResumeSummary(done=done, running=running, failed=failed)
    """================= End method get_resume_summary ================="""

    """================= Startup method save_worker_stats ================="""
    def save_worker_stats(self, stats: list) -> None:
        if not stats:
            return
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM worker_stats")
            for s in stats:
                conn.execute(
                    """INSERT INTO worker_stats
                           (worker_id, rows_processed, final_batch, updated_at)
                       VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(worker_id) DO UPDATE SET
                           rows_processed=excluded.rows_processed,
                           final_batch=excluded.final_batch,
                           updated_at=CURRENT_TIMESTAMP""",
                    (s.worker_id, s.rows_processed, s.current_batch)
                )
            conn.commit()
            conn.close()
    """================= End method save_worker_stats ================="""

    """================= Startup method get_worker_stats ================="""
    def get_worker_stats(self) -> List[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT worker_id, rows_processed, final_batch "
            "FROM worker_stats ORDER BY worker_id"
        ).fetchall()
        conn.close()
        return [{"worker_id": r[0], "rows_processed": r[1],
                 "final_batch": r[2]} for r in rows]
    """================= End method get_worker_stats ================="""

    """================= Startup method mark_row_processed ================="""
    def mark_row_processed(self, row_id: int, source: str = "", table_name: str = "") -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR IGNORE INTO processed_rows (row_id, source, table_name, processed_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                (row_id, source, table_name)
            )
            conn.commit()
            conn.close()
    """================= End method mark_row_processed ================="""

    """================= Startup method is_row_processed ================="""
    def is_row_processed(self, row_id: int) -> bool:
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM processed_rows WHERE row_id = ?", (row_id,)
        ).fetchone()
        conn.close()
        return row is not None
    """================= End method is_row_processed ================="""

    """================= Startup method get_processed_row_count ================="""
    def get_processed_row_count(self) -> int:
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM processed_rows").fetchone()[0]
        conn.close()
        return count
    """================= End method get_processed_row_count ================="""

    """================= Startup method get_max_processed_row_id ================="""
    def get_max_processed_row_id(self) -> int:
        conn = self._connect()
        result = conn.execute("SELECT MAX(row_id) FROM processed_rows").fetchone()[0]
        conn.close()
        return result if result is not None else 0
    """================= End method get_max_processed_row_id ================="""

    """================= Startup method migrate_chunk_to_row_tracking ================="""
    def migrate_chunk_to_row_tracking(self, data_source, table_name: str) -> int:
        # FIX: Removed always-False guard `hasattr(self, 'vector_db')` —
        # SQLiteCheckpoint.__init__ never sets self.vector_db, so the old
        # guard caused this method to always return 0 immediately.
        try:
            with self._lock:
                conn = self._connect()

                completed_chunks = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE status='done'"
                ).fetchall()

                if not completed_chunks:
                    conn.close()
                    return 0

                if hasattr(data_source, 'type') and data_source.type == 'sql':
                    migrated_count = 0

                    for row_id in range(1, 101):
                        conn.execute(
                            """INSERT OR IGNORE INTO processed_rows (row_id, source, table_name, processed_at)
                               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                            (row_id, data_source.name if hasattr(data_source, 'name') else 'unknown', table_name)
                        )
                        migrated_count += 1

                    conn.commit()
                    conn.close()
                    return migrated_count

                conn.close()
                return 0

        except Exception as e:
            logger.error(f"Failed to migrate chunk to row tracking: {e}")
            return 0
    """================= End method migrate_chunk_to_row_tracking ================="""

"""================= End class SQLiteCheckpoint ================="""


if __name__ == "__main__":
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp = f.name
    cp = SQLiteCheckpoint(tmp)
    cp.mark_running(0); cp.mark_done(0)
    cp.mark_running(1); cp.mark_failed(1)
    s = cp.get_resume_summary()
    print(f"done={s.done}  running={s.running}  failed={s.failed}")
    assert s.done == 1 and s.failed == 1

    from dataclasses import dataclass
    @dataclass
    class _S:
        worker_id: int; rows_processed: int; current_batch: int
    cp.save_worker_stats([_S(0, 100, 80), _S(1, 120, 96), _S(2, 105, 80)])
    ws = cp.get_worker_stats()
    print(f"Worker stats: {ws}")
    assert len(ws) == 3
    print("All assertions passed.")
    os.unlink(tmp)