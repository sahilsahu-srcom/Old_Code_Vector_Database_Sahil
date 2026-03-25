"""
File Summary:
Query analytics and usage tracking for HAUP RAG engine. Stores every query event
in a local SQLite database. Tracks latency, cache hits, error rates, top query
patterns, and hourly volume. No external analytics service required.

====================================================================
                     Startup
====================================================================

Analytics()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Store db_path, init logger, call _init_db()
│
├── _init_db()  [Function] -------------------------------> CREATE TABLE query_events + indexes
│       │
│       ├── CREATE TABLE query_events --------------------> Main analytics event store
│       ├── CREATE INDEX idx_qe_timestamp ----------------> Fast time-range queries
│       └── CREATE INDEX idx_qe_session ------------------> Fast session lookups
│
├── _conn()  [Function] ----------------------------------> Open SQLite connection with WAL mode
│
├── record()  [Function] ---------------------------------> Persist one QueryEvent to database
│       │
│       ├── INSERT INTO query_events ---------------------> Store all event fields
│       │
│       ├── [Conditional Branch] latency > 10s -----------> Log slow query warning
│       │
│       └── [Exception Block] ----------------------------> Silently swallow — never breaks pipeline
│
├── summary()  [Function] --------------------------------> Aggregated stats for last N hours
│       │
│       ├── SELECT COUNT / AVG / MAX / SUM ---------------> Aggregate query metrics
│       │
│       └── [Early Exit Branch] no rows ------------------> Return minimal dict
│
├── p95_latency()  [Function] ----------------------------> P95 latency for non-cached queries
│       │
│       ├── SELECT latency_ms ORDER BY latency_ms --------> Sorted latency values
│       │
│       └── Index at 95th percentile ---------------------> Return rounded value
│
├── top_queries()  [Function] ----------------------------> Most frequent query patterns
│       │
│       └── GROUP BY LOWER(TRIM(query)) ORDER BY count ---> Deduplicated query ranking
│
├── error_log()  [Function] ------------------------------> Recent errors for debugging
│       │
│       └── WHERE error IS NOT NULL ORDER BY timestamp ---> Latest error entries
│
├── hourly_volume()  [Function] --------------------------> Query count per hour for load analysis
│       │
│       └── GROUP BY hour bucket -------------------------> Time-series volume data
│
├── warm_cache_candidates()  [Function] ------------------> Queries seen 3+ times with no cache hits
│       │
│       └── HAVING COUNT >= min_count AND cache_hits = 0 -> Cache warming candidates
│
└── purge_old()  [Function] ------------------------------> Delete analytics older than N days
        │
        ├── DELETE WHERE timestamp < cutoff --------------> Remove stale records
        └── VACUUM ---------------------------------------> Reclaim disk space

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_core import logger as log


"""================= Startup class QueryEvent ================="""
@dataclass
class QueryEvent:
    session_id:     str
    query:          str
    answer_length:  int
    retrieved_rows: int
    latency_ms:     float
    cache_hit:      bool
    llm_backend:    str
    llm_model:      str
    error:          Optional[str]       = None
    warnings:       Optional[List[str]] = None
    timestamp:      float               = 0.0

    """================= Startup method __post_init__ ================="""
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
    """================= End method __post_init__ ================="""

"""================= End class QueryEvent ================="""


"""================= Startup class Analytics ================="""
class Analytics:

    _SLOW_QUERY_THRESHOLD_MS = 10_000   # 10 seconds

    """================= Startup method __init__ ================="""
    def __init__(self, db_path: str = "./rag_analytics.db"):
        self._db_path = db_path
        self._log     = log.get("analytics")
        self._init_db()
    """================= End method __init__ ================="""

    """================= Startup method _conn ================="""
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    """================= End method _conn ================="""

    """================= Startup method _init_db ================="""
    def _init_db(self) -> None:
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_events (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT    NOT NULL,
                query          TEXT    NOT NULL,
                answer_length  INTEGER NOT NULL DEFAULT 0,
                retrieved_rows INTEGER NOT NULL DEFAULT 0,
                latency_ms     REAL    NOT NULL DEFAULT 0,
                cache_hit      INTEGER NOT NULL DEFAULT 0,
                llm_backend    TEXT    NOT NULL DEFAULT '',
                llm_model      TEXT    NOT NULL DEFAULT '',
                error          TEXT,
                warnings       TEXT,
                timestamp      REAL    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qe_timestamp ON query_events(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qe_session ON query_events(session_id)"
        )
        conn.commit()
        conn.close()
    """================= End method _init_db ================="""

    """================= Startup method record ================="""
    def record(self, event: QueryEvent) -> None:
        """Persist a query event. Fire-and-forget — never raises."""
        try:
            conn = self._conn()
            conn.execute(
                """INSERT INTO query_events
                   (session_id, query, answer_length, retrieved_rows,
                    latency_ms, cache_hit, llm_backend, llm_model,
                    error, warnings, timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.session_id,
                    event.query[:500],    # truncate long queries
                    event.answer_length,
                    event.retrieved_rows,
                    event.latency_ms,
                    int(event.cache_hit),
                    event.llm_backend,
                    event.llm_model,
                    event.error,
                    json.dumps(event.warnings) if event.warnings else None,
                    event.timestamp,
                ),
            )
            conn.commit()
            conn.close()

            if event.latency_ms > self._SLOW_QUERY_THRESHOLD_MS:
                self._log.warning(
                    "SLOW QUERY %.0fms session=%s query=%r",
                    event.latency_ms, event.session_id, event.query[:80],
                )
        except Exception as exc:
            # Analytics must never break the main pipeline
            self._log.debug("Analytics record failed: %s", exc)
    """================= End method record ================="""

    """================= Startup method summary ================="""
    def summary(self, last_n_hours: int = 24) -> Dict[str, Any]:
        """High-level stats for the last N hours."""
        since = time.time() - (last_n_hours * 3600)
        conn  = self._conn()
        row   = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(cache_hit) as cache_hits,
                AVG(latency_ms) as avg_latency_ms,
                MAX(latency_ms) as max_latency_ms,
                AVG(retrieved_rows) as avg_retrieved,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
               FROM query_events WHERE timestamp >= ?""",
            (since,),
        ).fetchone()
        conn.close()

        if not row or row[0] == 0:
            return {"period_hours": last_n_hours, "total_queries": 0}

        total      = row[0] or 0
        cache_hits = row[1] or 0
        return {
            "period_hours":      last_n_hours,
            "total_queries":     total,
            "cache_hit_rate":    round(cache_hits / max(total, 1), 3),
            "avg_latency_ms":    round(row[2] or 0, 1),
            "max_latency_ms":    round(row[3] or 0, 1),
            "avg_retrieved_rows": round(row[4] or 0, 1),
            "error_rate":        round((row[5] or 0) / max(total, 1), 3),
            "errors":            row[5] or 0,
        }
    """================= End method summary ================="""

    """================= Startup method p95_latency ================="""
    def p95_latency(self, last_n_hours: int = 24) -> float:
        """Calculate P95 latency over the last N hours."""
        since = time.time() - (last_n_hours * 3600)
        conn  = self._conn()
        rows  = conn.execute(
            "SELECT latency_ms FROM query_events "
            "WHERE timestamp >= ? AND cache_hit = 0 ORDER BY latency_ms",
            (since,),
        ).fetchall()
        conn.close()
        if not rows:
            return 0.0
        idx = int(len(rows) * 0.95)
        return round(rows[min(idx, len(rows) - 1)][0], 1)
    """================= End method p95_latency ================="""

    """================= Startup method top_queries ================="""
    def top_queries(self, limit: int = 20) -> List[Dict]:
        """Most frequent query patterns (useful for cache warming)."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT query, COUNT(*) as count, AVG(latency_ms) as avg_ms,
                      SUM(cache_hit) as cache_hits
               FROM query_events
               GROUP BY LOWER(TRIM(query))
               ORDER BY count DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {"query": r[0], "count": r[1], "avg_ms": round(r[2], 0), "cache_hits": r[3]}
            for r in rows
        ]
    """================= End method top_queries ================="""

    """================= Startup method error_log ================="""
    def error_log(self, limit: int = 50) -> List[Dict]:
        """Recent errors for debugging."""
        conn = self._conn()
        rows = conn.execute(
            """SELECT session_id, query, error, timestamp
               FROM query_events
               WHERE error IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {
                "session_id": r[0],
                "query":      r[1],
                "error":      r[2],
                "timestamp":  r[3],
            }
            for r in rows
        ]
    """================= End method error_log ================="""

    """================= Startup method hourly_volume ================="""
    def hourly_volume(self, last_n_hours: int = 48) -> List[Dict]:
        """Query volume per hour — useful for load pattern analysis."""
        since = time.time() - (last_n_hours * 3600)
        conn  = self._conn()
        rows  = conn.execute(
            """SELECT
                CAST(timestamp / 3600 AS INTEGER) * 3600 as hour_ts,
                COUNT(*) as queries,
                SUM(cache_hit) as cache_hits,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
               FROM query_events
               WHERE timestamp >= ?
               GROUP BY hour_ts
               ORDER BY hour_ts""",
            (since,),
        ).fetchall()
        conn.close()
        return [
            {
                "hour":       r[0],
                "queries":    r[1],
                "cache_hits": r[2],
                "errors":     r[3],
            }
            for r in rows
        ]
    """================= End method hourly_volume ================="""

    """================= Startup method warm_cache_candidates ================="""
    def warm_cache_candidates(self, min_count: int = 3) -> List[str]:
        """
        Return queries seen 3+ times that are NOT cached.
        Use this to pre-warm the cache on startup.
        """
        conn = self._conn()
        rows = conn.execute(
            """SELECT query
               FROM query_events
               GROUP BY LOWER(TRIM(query))
               HAVING COUNT(*) >= ? AND SUM(cache_hit) = 0
               ORDER BY COUNT(*) DESC
               LIMIT 50""",
            (min_count,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    """================= End method warm_cache_candidates ================="""

    """================= Startup method purge_old ================="""
    def purge_old(self, keep_days: int = 30) -> int:
        """Remove analytics older than N days. Returns deleted count."""
        cutoff = time.time() - (keep_days * 86400)
        conn   = self._conn()
        cur    = conn.execute(
            "DELETE FROM query_events WHERE timestamp < ?", (cutoff,)
        )
        conn.commit()
        deleted = cur.rowcount
        conn.execute("VACUUM")
        conn.commit()
        conn.close()
        return deleted
    """================= End method purge_old ================="""

"""================= End class Analytics ================="""
