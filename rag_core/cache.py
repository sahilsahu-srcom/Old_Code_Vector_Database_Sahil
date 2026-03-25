"""
File Summary:
Semantic response cache with exact-match (O(1)) and semantic similarity lookup,
supporting LRU eviction, TTL expiry, and optional SQLite persistence.

====================================================================
CLASS → FUNCTION → OBJECT TREE
====================================================================

ResponseCache  [Class → Object]
||
├── __init__(cfg, db_path)
│       ├── Initialize in-memory store (OrderedDict)
│       ├── Setup thread lock
│       └── _init_db() (optional, if SQLite enabled)
│
├── get(query, embedding, session_id)  [CORE READ]
│       │
│       ├── _key() ------------------------------> Generate hash key
│       ├── _exact_get() ------------------------> O(1) lookup
│       │
│       ├── _semantic_get() (if embedding) -----> Cosine similarity search
│       │
│       └── Return cached response / None
│
├── set(query, response, embedding)  [CORE WRITE]
│       │
│       ├── _key()
│       ├── _evict_expired()
│       ├── LRU eviction (popitem)
│       ├── Store entry (response, ts, embedding)
│       └── _persist() (optional SQLite)
│
├── clear()
│       ├── Clear in-memory store
│       └── Delete SQLite entries (if enabled)
│
├── stats()
│       └── Return cache metadata
│
├── _key()
├── _exact_get()
├── _semantic_get()
├── _evict_expired()
│
├── _init_db() ------------------------------> Create SQLite table
└── _persist() ------------------------------> Save cache entry

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Optional

import numpy as np

from rag_core import logger as log
from rag_core.config import CacheConfig


"""================= Startup class ResponseCache ================="""
class ResponseCache:

    """================= Startup function __init__ ================="""
    def __init__(self, cfg: CacheConfig, db_path: Optional[str] = None):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._log = log.get("cache")

        self._store: OrderedDict[str, tuple[str, float, Optional[np.ndarray]]] = OrderedDict()

        self._db_path = db_path
        if db_path:
            self._init_db()
    """================= End function __init__ ================="""

    # ── Public API ─────────────────────────────────────────────────────────

    """================= Startup function get ================="""
    def get(
        self,
        query: str,
        embedding: Optional[np.ndarray] = None,
        session_id: str = "",
    ) -> Optional[str]:
        """
        Flow:
            Generate key → Exact match lookup
            → If miss, semantic search → Return result / None
        """
        if not self._cfg.enabled:
            return None

        key = self._key(query)

        result = self._exact_get(key)
        if result is not None:
            log.log_cache(session_id, True, query)
            return result

        if embedding is not None:
            result = self._semantic_get(embedding)
            if result is not None:
                log.log_cache(session_id, True, query)
                return result

        log.log_cache(session_id, False, query)
        return None
    """================= End function get ================="""

    """================= Startup function set ================="""
    def set(
        self,
        query: str,
        response: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Flow:
            Generate key → Evict expired → Apply LRU
            → Store entry → Persist (optional)
        """
        if not self._cfg.enabled:
            return

        key = self._key(query)

        with self._lock:
            self._evict_expired()

            if len(self._store) >= self._cfg.max_entries:
                self._store.popitem(last=False)

            self._store[key] = (response, time.time(), embedding)
            self._store.move_to_end(key)

        if self._db_path:
            self._persist(key, query, response)
    """================= End function set ================="""

    """================= Startup function clear ================="""
    def clear(self) -> None:
        with self._lock:
            self._store.clear()

        if self._db_path:
            conn = sqlite3.connect(self._db_path)
            conn.execute("DELETE FROM response_cache")
            conn.commit()
            conn.close()
    """================= End function clear ================="""

    """================= Startup function stats ================="""
    def stats(self) -> dict:
        with self._lock:
            total = len(self._store)

        return {
            "entries": total,
            "max_entries": self._cfg.max_entries,
            "enabled": self._cfg.enabled,
        }
    """================= End function stats ================="""

    # ── Internal ───────────────────────────────────────────────────────────

    """================= Startup function _key ================="""
    def _key(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    """================= End function _key ================="""

    """================= Startup function _exact_get ================="""
    def _exact_get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._store:
                return None

            response, ts, emb = self._store[key]

            if time.time() - ts > self._cfg.ttl_seconds:
                del self._store[key]
                return None

            self._store.move_to_end(key)
            return response
    """================= End function _exact_get ================="""

    """================= Startup function _semantic_get ================="""
    def _semantic_get(self, query_embedding: np.ndarray) -> Optional[str]:
        threshold = self._cfg.similarity_threshold

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        best_sim = 0.0
        best_response: Optional[str] = None

        with self._lock:
            now = time.time()

            for key, (response, ts, emb) in list(self._store.items()):
                if emb is None:
                    continue

                if now - ts > self._cfg.ttl_seconds:
                    continue

                e_norm = emb / (np.linalg.norm(emb) + 1e-10)
                sim = float(np.dot(q_norm, e_norm))

                if sim > best_sim:
                    best_sim = sim
                    if sim >= threshold:
                        best_response = response

        return best_response if best_sim >= threshold else None
    """================= End function _semantic_get ================="""

    """================= Startup function _evict_expired ================="""
    def _evict_expired(self) -> None:
        cutoff = time.time() - self._cfg.ttl_seconds
        expired = [k for k, (_, ts, _) in self._store.items() if ts < cutoff]

        for k in expired:
            del self._store[k]
    """================= End function _evict_expired ================="""

    # ── SQLite persistence ─────────────────────────────────────────────────

    """================= Startup function _init_db ================="""
    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key   TEXT PRIMARY KEY,
                query       TEXT,
                response    TEXT,
                created_at  REAL
            )
        """)
        conn.commit()
        conn.close()
    """================= End function _init_db ================="""

    """================= Startup function _persist ================="""
    def _persist(self, key: str, query: str, response: str) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT OR REPLACE INTO response_cache VALUES(?,?,?,?)",
                (key, query, response, time.time()),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            self._log.debug("Cache persist failed: %s", exc)
    """================= End function _persist ================="""

"""================= End class ResponseCache ================="""