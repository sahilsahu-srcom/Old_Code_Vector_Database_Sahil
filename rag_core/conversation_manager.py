"""
File Summary:
Manages multi-turn conversation sessions with in-memory LRU cache and
optional SQLite persistence, supporting TTL-based expiry and thread safety.

====================================================================
CLASS → FUNCTION → OBJECT TREE
====================================================================

Turn  [Dataclass → Object] ------------------------------> Represents a single conversation message
||
└── Fields
        ├── role        ---------------------------------> Sender of message (user/assistant)
        ├── content     ---------------------------------> Message text content
        ├── timestamp   ---------------------------------> Time when message was created
        └── citations   ---------------------------------> Optional references for assistant replies

--------------------------------------------------------------------

Session  [Dataclass → Object] ---------------------------> Maintains full conversation state
||
├── add_user(content) -----------------------------------> Append user turn → update last_active
│       └── Append user turn → update last_active
│
├── add_assistant(content, citations) -------------------> Append assistant turn with citations
│       └── Append assistant turn → update last_active
│
├── to_messages(max_turns) ------------------------------> Convert turns → Message list (LLM input)
│       └── Convert turns → Message list (LLM input)
│
└── Fields
        ├── session_id   --------------------------------> Unique identifier for session
        ├── turns[]      --------------------------------> Ordered list of conversation turns
        ├── created_at   --------------------------------> Session creation timestamp
        ├── last_active  --------------------------------> Last interaction time
        └── metadata     --------------------------------> Additional session information

--------------------------------------------------------------------

ConversationManager  [Class → Object] -------------------> Controls session lifecycle, cache, and DB
||
├── __init__(cfg, db_path) ------------------------------> Initialize LRU cache, lock, and DB setup
│       ├── Init LRU cache
│       ├── Setup lock
│       └── _init_db() (optional)
│
├── new_session(metadata) -------------------------------> Create Session → store via _put()
│       └── Create Session → _put()
│
├── get(session_id) -------------------------------------> Retrieve session (cache → DB → expiry check)
│       ├── _get_from_cache()
│       ├── _load_from_db() (fallback)
│       ├── _is_expired()
│       └── delete() if expired
│
├── save(session) ---------------------------------------> Persist session changes
│       └── _put()
│
├── delete(session_id) ---------------------------------> Remove session from cache and database
│       ├── Remove from cache
│       └── _delete_from_db()
│
├── list_sessions() ------------------------------------> Return list of active sessions
│       ├── Return from cache
│       └── OR _list_from_db()
│
├── cleanup_expired() ----------------------------------> Remove expired sessions (TTL cleanup)
│       ├── Remove expired (DB)
│       └── Remove expired (cache)
│
├── _put(session) --------------------------------------> Insert into cache + trim + LRU eviction + persist
│       ├── Trim history
│       ├── LRU eviction
│       ├── Store in cache
│       └── _save_to_db()
│
├── _get_from_cache() ----------------------------------> Fetch session from in-memory cache
├── _is_expired() --------------------------------------> Check if session exceeded TTL
│
├── SQLite Layer ---------------------------------------> Handles persistent session storage
│       ├── _conn() ------------------------------------> Create SQLite connection (WAL mode)
│       ├── _init_db() ---------------------------------> Initialize sessions table
│       ├── _save_to_db() ------------------------------> Insert/update session in DB
│       ├── _load_from_db() ----------------------------> Load session and rebuild objects
│       ├── _delete_from_db() --------------------------> Delete session from DB
│       └── _list_from_db() ----------------------------> Fetch session summaries from DB


====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag_core import logger as log
from rag_core.config import ConversationConfig
from rag_core.llm_client import Message


# ─── Session data model ─────────────────────────────────────────────

"""================= Startup class Turn ================="""
@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    citations: Optional[List[Dict]] = None
"""================= End class Turn ================="""


"""================= Startup class Session ================="""
@dataclass
class Session:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    """================= Startup function add_user ================="""
    def add_user(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content))
        self.last_active = time.time()
    """================= End function add_user ================="""

    """================= Startup function add_assistant ================="""
    def add_assistant(self, content: str, citations: Optional[List[Dict]] = None) -> None:
        self.turns.append(Turn(role="assistant", content=content, citations=citations))
        self.last_active = time.time()
    """================= End function add_assistant ================="""

    """================= Startup function to_messages ================="""
    def to_messages(self, max_turns: int) -> List[Message]:
        recent = self.turns[-(max_turns * 2):]
        return [Message(role=t.role, content=t.content) for t in recent]
    """================= End function to_messages ================="""

"""================= End class Session ================="""


# ─── Manager ───────────────────────────────────────────────────────

"""================= Startup class ConversationManager ================="""
class ConversationManager:

    _CACHE_SIZE = 128

    """================= Startup function __init__ ================="""
    def __init__(self, cfg: ConversationConfig, db_path: str):
        self._cfg = cfg
        self._db_path = db_path
        self._lock = threading.Lock()
        self._log = log.get("conversation")

        self._cache: OrderedDict[str, Session] = OrderedDict()

        if cfg.persist_sessions:
            self._init_db()
    """================= End function __init__ ================="""

    # ── Public API ─────────────────────────────────────────────

    """================= Startup function new_session ================="""
    def new_session(self, metadata: Optional[Dict] = None) -> Session:
        session = Session(
            session_id=str(uuid.uuid4()),
            metadata=metadata or {},
        )
        self._put(session)
        self._log.info("New session: %s", session.session_id)
        return session
    """================= End function new_session ================="""

    """================= Startup function get ================="""
    def get(self, session_id: str) -> Optional[Session]:
        session = self._get_from_cache(session_id)

        if session is None and self._cfg.persist_sessions:
            session = self._load_from_db(session_id)

        if session is None:
            return None

        if self._is_expired(session):
            self.delete(session_id)
            return None

        return session
    """================= End function get ================="""

    """================= Startup function save ================="""
    def save(self, session: Session) -> None:
        self._put(session)
    """================= End function save ================="""

    """================= Startup function delete ================="""
    def delete(self, session_id: str) -> None:
        with self._lock:
            self._cache.pop(session_id, None)

        if self._cfg.persist_sessions:
            self._delete_from_db(session_id)
    """================= End function delete ================="""

    """================= Startup function list_sessions ================="""
    def list_sessions(self) -> List[Dict]:
        if not self._cfg.persist_sessions:
            with self._lock:
                return [
                    {
                        "session_id": s.session_id,
                        "turns": len(s.turns),
                        "last_active": s.last_active,
                    }
                    for s in self._cache.values()
                ]
        return self._list_from_db()
    """================= End function list_sessions ================="""

    """================= Startup function cleanup_expired ================="""
    def cleanup_expired(self) -> int:
        cutoff = time.time() - self._cfg.session_ttl_seconds
        removed = 0

        if self._cfg.persist_sessions:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM sessions WHERE last_active < ?", (cutoff,)
            )
            conn.commit()
            removed = cur.rowcount
            conn.close()

        with self._lock:
            expired = [sid for sid, s in self._cache.items() if s.last_active < cutoff]
            for sid in expired:
                del self._cache[sid]
            removed += len(expired)

        if removed:
            self._log.info("Cleaned up %d expired sessions", removed)

        return removed
    """================= End function cleanup_expired ================="""

    # ── Internal ─────────────────────────────────────────────

    """================= Startup function _is_expired ================="""
    def _is_expired(self, session: Session) -> bool:
        return (time.time() - session.last_active) > self._cfg.session_ttl_seconds
    """================= End function _is_expired ================="""

    """================= Startup function _put ================="""
    def _put(self, session: Session) -> None:
        with self._lock:
            max_msgs = self._cfg.max_history_turns * 2
            if len(session.turns) > max_msgs:
                session.turns = session.turns[-max_msgs:]

            if len(self._cache) >= self._CACHE_SIZE:
                self._cache.popitem(last=False)

            self._cache[session.session_id] = session
            self._cache.move_to_end(session.session_id)

        if self._cfg.persist_sessions:
            self._save_to_db(session)
    """================= End function _put ================="""

    """================= Startup function _get_from_cache ================="""
    def _get_from_cache(self, session_id: str) -> Optional[Session]:
        with self._lock:
            if session_id in self._cache:
                self._cache.move_to_end(session_id)
                return self._cache[session_id]
        return None
    """================= End function _get_from_cache ================="""

    # ── SQLite persistence ─────────────────────────────────────

    """================= Startup function _conn ================="""
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    """================= End function _conn ================="""

    """================= Startup function _init_db ================="""
    def _init_db(self) -> None:
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                turns_json  TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  REAL NOT NULL,
                last_active REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    """================= End function _init_db ================="""

    """================= Startup function _save_to_db ================="""
    def _save_to_db(self, session: Session) -> None:
        turns_data = [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
                "citations": t.citations,
            }
            for t in session.turns
        ]

        conn = self._conn()
        conn.execute(
            """INSERT INTO sessions(session_id, turns_json, metadata, created_at, last_active)
               VALUES(?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                 turns_json=excluded.turns_json,
                 last_active=excluded.last_active""",
            (
                session.session_id,
                json.dumps(turns_data, default=str),
                json.dumps(session.metadata, default=str),
                session.created_at,
                session.last_active,
            ),
        )
        conn.commit()
        conn.close()
    """================= End function _save_to_db ================="""

    """================= Startup function _load_from_db ================="""
    def _load_from_db(self, session_id: str) -> Optional[Session]:
        conn = self._conn()
        row = conn.execute(
            "SELECT turns_json, metadata, created_at, last_active FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        turns_data = json.loads(row[0])

        turns = [
            Turn(
                role=t["role"],
                content=t["content"],
                timestamp=t.get("timestamp", 0.0),
                citations=t.get("citations"),
            )
            for t in turns_data
        ]

        return Session(
            session_id=session_id,
            turns=turns,
            created_at=row[2],
            last_active=row[3],
            metadata=json.loads(row[1]),
        )
    """================= End function _load_from_db ================="""

    """================= Startup function _delete_from_db ================="""
    def _delete_from_db(self, session_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        conn.commit()
        conn.close()
    """================= End function _delete_from_db ================="""

    """================= Startup function _list_from_db ================="""
    def _list_from_db(self) -> List[Dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT session_id, last_active FROM sessions ORDER BY last_active DESC"
        ).fetchall()
        conn.close()

        return [{"session_id": r[0], "last_active": r[1]} for r in rows]
    """================= End function _list_from_db ================="""

"""================= End class ConversationManager ================="""