"""
File Summary:
Background maintenance daemon for HAUP RAG engine. Runs alongside the RAG engine
in a single daemon thread. Handles session cleanup, cache warming, health monitoring,
and analytics purging on configurable intervals. Starts automatically on engine init
and shuts down cleanly on process exit.

====================================================================
               Startup
====================================================================

BackgroundWorker()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Init job list, thread, stop event, logger
│
├── add_job()  [Function] --------------------------------> Register a maintenance job, return self
│
├── start()  [Function] ----------------------------------> Launch daemon thread
│       │
│       └── _run()  [Function] ---------------------------> Cooperative scheduler loop
│               │
│               ├── stop_event.wait(_TICK=60s) -----------> Sleep until next tick
│               │
│               └── Loop: check each job deadline --------> _execute() if interval elapsed
│                       │
│                       └── _execute()  [Function] -------> Run job function with timing
│                               │
│                               ├── job.fn() -------------> Call registered job function
│                               ├── job.run_count += 1 ---> Track successful runs
│                               │
│                               └── [Exception Block] ----> Log error, increment error_count
│
├── stop()  [Function] -----------------------------------> Set stop event and join thread
│
└── status()  [Function] ---------------------------------> Return job status list with next_run_in_seconds

build_for_engine()  [Function] ---------------------------> Wire BackgroundWorker to RAGEngine instance
||
├── BackgroundWorker()  [Class → Object] -----------------> Create worker instance
│
├── _session_cleanup()  [Function] -----------------------> Job 1: purge expired sessions every 5 min
│       │
│       └── engine._conversation.cleanup_expired() ------> Remove stale sessions
│
├── _cache_warmer()  [Function] --------------------------> Job 2: pre-warm cache every 15 min
│       │
│       ├── engine._analytics.warm_cache_candidates() ---> Get uncached frequent queries
│       ├── engine.new_session() -------------------------> Temp session for warming
│       ├── engine.ask() ---------------------------------> Generate and cache responses
│       └── engine._conversation.delete() ----------------> Remove temp session
│
├── _health_monitor()  [Function] -----------------------> Job 3: health check every 1 min
│       │
│       ├── engine.health_check() -----------------------> Check LLM and ChromaDB status
│       └── [Conditional Branch] unhealthy components ---> Log warning per component
│
├── _analytics_purger()  [Function] ---------------------> Job 4: trim old analytics every 24 hours
│       │
│       └── engine._analytics.purge_old(keep_days=30) ---> Delete stale records
│
└── worker.start() --------------------------------------> Launch background thread

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from rag_core import logger as log


"""================= Startup class Job ================="""
@dataclass
class Job:
    name:             str
    fn:               Callable[[], None]
    interval_seconds: int
    last_run:         float          = 0.0
    run_count:        int            = 0
    error_count:      int            = 0
    last_error:       Optional[str]  = None
"""================= End class Job ================="""


"""================= Startup class BackgroundWorker ================="""
class BackgroundWorker:
    """
    Simple cooperative scheduler running jobs in a daemon thread.
    Tick interval is 60s — jobs with a longer interval only run
    when their deadline has passed.
    """

    _TICK = 60   # check every minute

    """================= Startup method __init__ ================="""
    def __init__(self):
        self._jobs:       List[Job]                    = []
        self._thread:     Optional[threading.Thread]   = None
        self._stop_event  = threading.Event()
        self._log         = log.get("background")
    """================= End method __init__ ================="""

    """================= Startup method add_job ================="""
    def add_job(
        self, name: str, fn: Callable[[], None], interval_seconds: int
    ) -> "BackgroundWorker":
        """Register a maintenance job. Returns self for chaining."""
        self._jobs.append(Job(name=name, fn=fn, interval_seconds=interval_seconds))
        return self
    """================= End method add_job ================="""

    """================= Startup method start ================="""
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target = self._run,
            name   = "haup-rag-bg-worker",
            daemon = True,
        )
        self._thread.start()
        self._log.info("Background worker started (%d jobs)", len(self._jobs))
    """================= End method start ================="""

    """================= Startup method stop ================="""
    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._log.info("Background worker stopped")
    """================= End method stop ================="""

    """================= Startup method status ================="""
    def status(self) -> List[Dict]:
        return [
            {
                "name":                 j.name,
                "interval_seconds":     j.interval_seconds,
                "run_count":            j.run_count,
                "error_count":          j.error_count,
                "last_run":             j.last_run,
                "last_error":           j.last_error,
                "next_run_in_seconds":  max(
                    0, j.interval_seconds - (time.time() - j.last_run)
                ),
            }
            for j in self._jobs
        ]
    """================= End method status ================="""

    """================= Startup method _run ================="""
    def _run(self) -> None:
        while not self._stop_event.wait(self._TICK):
            now = time.time()
            for job in self._jobs:
                if now - job.last_run >= job.interval_seconds:
                    self._execute(job)
    """================= End method _run ================="""

    """================= Startup method _execute ================="""
    def _execute(self, job: Job) -> None:
        self._log.debug("Running job: %s", job.name)
        t0 = time.perf_counter()
        try:
            job.fn()
            elapsed       = (time.perf_counter() - t0) * 1000
            job.run_count += 1
            job.last_run   = time.time()
            self._log.debug("Job '%s' completed in %.0fms", job.name, elapsed)
        except Exception as exc:
            job.error_count += 1
            job.last_error   = str(exc)
            self._log.error("Job '%s' failed: %s", job.name, exc)
    """================= End method _execute ================="""

"""================= End class BackgroundWorker ================="""


"""================= Startup function build_for_engine ================="""
def build_for_engine(engine) -> BackgroundWorker:
    """
    Build and start a BackgroundWorker pre-wired to a RAGEngine instance.

    Called automatically by RAGEngine.__init__ if background tasks
    are enabled in config.

    Jobs registered:
      • session_cleanup    every  5 min
      • cache_warmer       every 15 min
      • health_monitor     every  1 min
      • analytics_purger   every 24 hours
    """
    worker = BackgroundWorker()

    """================= Startup function _session_cleanup ================="""
    def _session_cleanup():
        n = engine._conversation.cleanup_expired()
        if n:
            log.get("background").info("Cleaned up %d expired sessions", n)
    """================= End function _session_cleanup ================="""

    worker.add_job("session_cleanup", _session_cleanup, interval_seconds=300)

    """================= Startup function _cache_warmer ================="""
    def _cache_warmer():
        if not hasattr(engine, "_analytics"):
            return
        candidates = engine._analytics.warm_cache_candidates(min_count=3)
        if not candidates:
            return
        log.get("background").info("Cache warming %d candidates", len(candidates))
        sid = engine.new_session({"source": "cache_warmer"})
        for query in candidates[:5]:   # warm at most 5 per cycle
            try:
                if engine._cache.get(query) is None:
                    engine.ask(query, session_id=sid, use_cache=False)
            except Exception as exc:
                log.get("background").debug("Cache warm failed for %r: %s", query, exc)
        engine._conversation.delete(sid)
    """================= End function _cache_warmer ================="""

    worker.add_job("cache_warmer", _cache_warmer, interval_seconds=900)

    """================= Startup function _health_monitor ================="""
    def _health_monitor():
        status = engine.health_check()
        if not status.get("llm_healthy"):
            log.get("background").warning(
                "Health check: LLM backend unhealthy — %s",
                status.get("llm_error", "unknown"),
            )
        if not status.get("chroma_healthy"):
            log.get("background").warning("Health check: ChromaDB unhealthy")
    """================= End function _health_monitor ================="""

    worker.add_job("health_monitor", _health_monitor, interval_seconds=60)

    """================= Startup function _analytics_purger ================="""
    def _analytics_purger():
        if not hasattr(engine, "_analytics"):
            return
        n = engine._analytics.purge_old(keep_days=30)
        if n:
            log.get("background").info("Purged %d old analytics rows", n)
    """================= End function _analytics_purger ================="""

    worker.add_job("analytics_purger", _analytics_purger, interval_seconds=86400)

    worker.start()
    return worker
"""================= End function build_for_engine ================="""
