"""
File Summary:
Structured logging and optional JSONL trace file for the HAUP RAG engine.
Every event (query, retrieval, LLM call, cache hit, error) is recorded with a
consistent schema so traces are machine-readable. Configurable verbosity per
event type.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

logger
||
├── setup()  [Function] ----------------------------------> Configure logging level, handlers, trace file
│       │
│       ├── logging.basicConfig() -----------------------> Set level, format, stream to stdout
│       │
│       └── [Conditional Branch] trace_file set ---------> Open JSONL trace file for appending
│
├── _emit_trace()  [Function] ----------------------------> Write one JSONL record to trace file
│       │
│       └── [Early Exit Branch] _TRACE_FILE is None -----> Skip silently if no trace file
│
├── get()  [Function] ------------------------------------> Return named child logger under haup.rag
│
├── log_query()  [Function] ------------------------------> Log session query and expansions
│       │
│       ├── [Conditional Branch] _LOG_QUERIES -----------> Log at INFO level if enabled
│       │
│       └── _emit_trace("query") ------------------------> Write to JSONL trace
│
├── log_retrieval()  [Function] --------------------------> Log vector retrieval count and latency
│       │
│       └── _emit_trace("retrieval") ---------------------> Write to JSONL trace
│
├── log_cache()  [Function] ------------------------------> Log cache HIT or MISS
│       │
│       └── _emit_trace("cache") -------------------------> Write to JSONL trace
│
├── log_llm_call()  [Function] ---------------------------> Log LLM backend, model, latency, tokens
│       │
│       └── _emit_trace("llm_call") ----------------------> Write to JSONL trace
│
└── log_error()  [Function] ------------------------------> Log error at stage with ERROR level
        │
        └── _emit_trace("error") -------------------------> Write to JSONL trace

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


_TRACE_FILE: Optional[Any] = None   # open file handle or None
_LOG_QUERIES: bool = True
_LOG_ROWS:    bool = False
_LOG_PROMPTS: bool = False


"""================= Startup function setup ================="""
def setup(
    level:              str           = "INFO",
    log_queries:        bool          = True,
    log_retrieved_rows: bool          = False,
    log_llm_prompts:    bool          = False,
    trace_file:         Optional[str] = None,
) -> None:
    """Call once at startup."""
    global _TRACE_FILE, _LOG_QUERIES, _LOG_ROWS, _LOG_PROMPTS

    _LOG_QUERIES = log_queries
    _LOG_ROWS    = log_retrieved_rows
    _LOG_PROMPTS = log_llm_prompts

    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
        stream  = sys.stdout,
    )

    if trace_file:
        p = Path(trace_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        _TRACE_FILE = p.open("a", encoding="utf-8")
"""================= End function setup ================="""


"""================= Startup function _emit_trace ================="""
def _emit_trace(event: str, payload: Dict[str, Any]) -> None:
    if _TRACE_FILE is None:
        return
    record = {"ts": time.time(), "event": event, **payload}
    _TRACE_FILE.write(json.dumps(record, default=str) + "\n")
    _TRACE_FILE.flush()
"""================= End function _emit_trace ================="""


_root = logging.getLogger("haup.rag")


"""================= Startup function get ================="""
def get(name: str) -> logging.Logger:
    return _root.getChild(name)
"""================= End function get ================="""


"""================= Startup function log_query ================="""
def log_query(session_id: str, query: str, expanded: list[str]) -> None:
    log = get("query")
    if _LOG_QUERIES:
        log.info("session=%s query=%r expansions=%d", session_id, query[:120], len(expanded))
    _emit_trace("query", {"session_id": session_id, "query": query, "expanded": expanded})
"""================= End function log_query ================="""


"""================= Startup function log_retrieval ================="""
def log_retrieval(session_id: str, n_results: int, latency_ms: float) -> None:
    get("retrieval").info(
        "session=%s retrieved=%d latency=%.0fms", session_id, n_results, latency_ms
    )
    _emit_trace("retrieval", {"session_id": session_id, "n_results": n_results, "latency_ms": latency_ms})
"""================= End function log_retrieval ================="""


"""================= Startup function log_cache ================="""
def log_cache(session_id: str, hit: bool, query: str) -> None:
    status = "HIT" if hit else "MISS"
    get("cache").debug("session=%s %s query=%r", session_id, status, query[:80])
    _emit_trace("cache", {"session_id": session_id, "hit": hit})
"""================= End function log_cache ================="""


"""================= Startup function log_llm_call ================="""
def log_llm_call(session_id: str, backend: str, model: str, latency_ms: float,
                 prompt_tokens: int, completion_tokens: int) -> None:
    get("llm").info(
        "session=%s backend=%s model=%s latency=%.0fms tokens=%d+%d",
        session_id, backend, model, latency_ms, prompt_tokens, completion_tokens,
    )
    _emit_trace("llm_call", {
        "session_id":         session_id,
        "backend":            backend,
        "model":              model,
        "latency_ms":         latency_ms,
        "prompt_tokens":      prompt_tokens,
        "completion_tokens":  completion_tokens,
    })
"""================= End function log_llm_call ================="""


"""================= Startup function log_error ================="""
def log_error(session_id: str, stage: str, error: str) -> None:
    get("error").error("session=%s stage=%s error=%s", session_id, stage, error)
    _emit_trace("error", {"session_id": session_id, "stage": stage, "error": error})
"""================= End function log_error ================="""