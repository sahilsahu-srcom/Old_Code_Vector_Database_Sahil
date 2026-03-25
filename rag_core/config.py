"""
File Summary:
Central configuration for the HAUP RAG engine. All tunables live here —
no magic numbers scattered across modules. Covers LLM backends, retrieval,
context, conversation, cache, guardrails, observability, and HAUP integration paths.

====================================================================
Startup
====================================================================

RAGConfig  [Class]
||
├── OllamaConfig  [Class] --------------------------------> Local Ollama LLM server settings
│
├── OpenAIConfig  [Class] --------------------------------> OpenAI cloud API settings
│
├── AnthropicConfig  [Class] -----------------------------> Anthropic Claude API settings
│
├── RetrievalConfig  [Class] -----------------------------> Vector search and ranking settings
│       │
│       ├── top_k ----------------------------------------> Initial ChromaDB candidate count
│       ├── rerank_top_n ---------------------------------> Final context size after reranking
│       ├── similarity_threshold -------------------------> Relevance filtering cutoff
│       └── enable_query_expansion ----------------------> Multi-query rewriting toggle
│
├── ContextConfig  [Class] -------------------------------> Response formatting and token budget
│       │
│       ├── max_context_tokens ---------------------------> Token budget for retrieved rows
│       ├── row_format -----------------------------------> markdown_table / json / key_value
│       └── truncate_long_values -------------------------> Chars per cell before truncation
│
├── ConversationConfig  [Class] --------------------------> Session and history management
│       │
│       ├── max_history_turns ----------------------------> User/assistant pairs kept in memory
│       ├── session_ttl_seconds --------------------------> Idle timeout before session expires
│       └── persist_sessions -----------------------------> Write sessions to SQLite toggle
│
├── CacheConfig  [Class] ---------------------------------> Response cache settings
│       │
│       ├── ttl_seconds ----------------------------------> Cache entry lifetime
│       ├── max_entries ----------------------------------> Max cached responses
│       └── similarity_threshold -------------------------> Cosine sim for cache hit detection
│
├── GuardrailsConfig  [Class] ----------------------------> Safety, security, and rate limiting
│       │
│       ├── __post_init__()  [Function] ------------------> Default blocked_keywords to []
│       ├── rate_limit_enabled / max_queries_per_minute --> Abuse prevention
│       ├── injection_detection --------------------------> Prompt injection scanning
│       ├── pii_detection --------------------------------> Privacy protection
│       └── hallucination_check --------------------------> Response accuracy validation
│
├── ObservabilityConfig  [Class] -------------------------> Logging and tracing settings
│       │
│       ├── log_level ------------------------------------> WARNING / INFO / DEBUG
│       ├── log_queries / log_retrieved_rows -------------> Verbose request logging
│       └── trace_file -----------------------------------> JSONL trace output path
│
└── RAGConfig  [Class] -----------------------------------> Master config dataclass
        │
        ├── llm_backend ----------------------------------> Active backend selection
        ├── ollama / openai / anthropic ------------------> Backend sub-configs
        ├── retrieval / context / conversation -----------> Pipeline sub-configs
        ├── cache / observability / guardrails -----------> System sub-configs
        ├── chroma_path / collection_name ----------------> HAUP vector DB paths
        ├── source_type / source_host / source_table -----> Original data source config
        ├── source_connection_string ---------------------> Full Neon DSN (overrides host/port/user)
        │
        └── from_env()  [Function] ----------------------> Build config from environment variables
                │
                ├── DB_TYPE / SOURCE_TYPE ----------------> Both accepted for source_type
                ├── NEON_CONNECTION_STRING ---------------> Populates source_connection_string
                └── os.getenv() overrides ----------------> Every field overridable via env

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


# ─────────────────────────────────────────────
#  LLM Backend
# ─────────────────────────────────────────────
LLMBackend = Literal["ollama", "openai", "anthropic"]


"""================= Startup class OllamaConfig ================="""
@dataclass
class OllamaConfig:
    base_url:   str = "http://localhost:11434"
    model:      str = "deepseek-v3.1:671b-cloud"
    timeout:    int = 120          # seconds per request
    keep_alive: str = "10m"        # how long Ollama keeps model in VRAM
"""================= End class OllamaConfig ================="""


"""================= Startup class OpenAIConfig ================="""
@dataclass
class OpenAIConfig:
    api_key:  str           = ""        # read from env if empty
    model:    str           = "gpt-4o-mini"
    base_url: Optional[str] = None      # supports Azure / proxy endpoints
    timeout:  int           = 60
"""================= End class OpenAIConfig ================="""


"""================= Startup class AnthropicConfig ================="""
@dataclass
class AnthropicConfig:
    api_key: str = ""
    model:   str = "claude-3-5-haiku-20241022"
    timeout: int = 60
"""================= End class AnthropicConfig ================="""


# ─────────────────────────────────────────────
#  Retrieval
# ─────────────────────────────────────────────

"""================= Startup class RetrievalConfig ================="""
@dataclass
class RetrievalConfig:
    top_k:                  int   = 8       # candidates fetched from ChromaDB
    rerank_top_n:           int   = 5       # kept after reranking
    similarity_threshold:   float = 0.30
    enable_query_expansion: bool  = True
    expansion_variations:   int   = 3       # how many rewritten queries to combine
    max_context_rows:       int   = 20      # hard cap before context window overflow
"""================= End class RetrievalConfig ================="""


# ─────────────────────────────────────────────
#  Context Builder
# ─────────────────────────────────────────────

"""================= Startup class ContextConfig ================="""
@dataclass
class ContextConfig:
    max_context_tokens:     int                                        = 6000
    include_schema_summary: bool                                       = True
    row_format:             Literal["markdown_table", "json", "key_value"] = "markdown_table"
    truncate_long_values:   int                                        = 200
"""================= End class ContextConfig ================="""


# ─────────────────────────────────────────────
#  Conversation
# ─────────────────────────────────────────────

"""================= Startup class ConversationConfig ================="""
@dataclass
class ConversationConfig:
    max_history_turns:   int  = 10      # number of user/assistant pairs kept
    session_ttl_seconds: int  = 3600    # 1 hour idle → session expires
    persist_sessions:    bool = True    # write sessions to SQLite
"""================= End class ConversationConfig ================="""


# ─────────────────────────────────────────────
#  Cache
# ─────────────────────────────────────────────

"""================= Startup class CacheConfig ================="""
@dataclass
class CacheConfig:
    enabled:              bool  = True
    ttl_seconds:          int   = 300     # 5 min — data changes slowly
    max_entries:          int   = 500
    similarity_threshold: float = 0.95   # cosine sim to consider a cache hit
"""================= End class CacheConfig ================="""


# ─────────────────────────────────────────────
#  Guardrails
# ─────────────────────────────────────────────

"""================= Startup class GuardrailsConfig ================="""
@dataclass
class GuardrailsConfig:
    max_query_length:       int  = 1000
    min_query_length:       int  = 2
    rate_limit_enabled:     bool = True
    max_queries_per_minute: int  = 30
    injection_detection:    bool = True
    block_injections:       bool = True
    pii_detection:          bool = True
    pii_redact_in_query:    bool = False
    pii_redact_in_response: bool = False
    blocked_keywords:       list = None
    hallucination_check:    bool = True

    """================= Startup method __post_init__ ================="""
    def __post_init__(self):
        if self.blocked_keywords is None:
            self.blocked_keywords = []
    """================= End method __post_init__ ================="""

"""================= End class GuardrailsConfig ================="""


# ─────────────────────────────────────────────
#  Logging / Observability
# ─────────────────────────────────────────────

"""================= Startup class ObservabilityConfig ================="""
@dataclass
class ObservabilityConfig:
    log_level:           str           = "WARNING"
    log_queries:         bool          = False
    log_retrieved_rows:  bool          = False
    log_llm_prompts:     bool          = False
    trace_file:          Optional[str] = None    # write JSONL trace if set
"""================= End class ObservabilityConfig ================="""


# ─────────────────────────────────────────────
#  Master Config
# ─────────────────────────────────────────────

"""================= Startup class RAGConfig ================="""
@dataclass
class RAGConfig:
    # Which LLM backend to use
    llm_backend: LLMBackend = "ollama"

    # Backend configs
    ollama:    OllamaConfig    = field(default_factory=OllamaConfig)
    openai:    OpenAIConfig    = field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)

    # Pipeline configs
    retrieval:     RetrievalConfig     = field(default_factory=RetrievalConfig)
    context:       ContextConfig       = field(default_factory=ContextConfig)
    conversation:  ConversationConfig  = field(default_factory=ConversationConfig)
    cache:         CacheConfig         = field(default_factory=CacheConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # HAUP integration paths
    chroma_path:     str = "./chroma_db"
    collection_name: str = "haup_vectors"
    checkpoint_db:   str = "./haup_checkpoint.db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Session persistence DB
    session_db: str = "./rag_sessions.db"

    # Source DB (for reverse row lookup)
    source_type:        Literal["mysql", "postgresql", "sqlite", "none"] = "postgresql"
    source_host:        str = "localhost"
    source_port:        int = 5432
    source_user:        str = "postgres"
    source_password:    str = ""
    source_database:    str = "Vector"
    source_table:       str = "Users"
    source_primary_key: str = "id"
    # Full DSN — when set, retriever uses this directly and ignores
    # source_host / source_port / source_user / source_password / source_database.
    # Populated automatically from NEON_CONNECTION_STRING env var.
    source_connection_string: str = ""

    """================= Startup method from_env ================="""
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """
        Build config from environment variables so nothing sensitive
        lives in source code. Every field can be overridden via env.

        Source-type resolution order (first match wins):
          1. SOURCE_TYPE  (explicit override)
          2. DB_TYPE      (your Neon .env key)
          3. dataclass default ("postgresql")

        Connection-string resolution order:
          1. NEON_CONNECTION_STRING  (full DSN — recommended for Neon)
          2. Individual SOURCE_HOST / SOURCE_PORT / SOURCE_USER / … vars
        """
        cfg = cls()

        # LLM backend
        cfg.llm_backend = os.getenv("RAG_LLM_BACKEND", cfg.llm_backend)  # type: ignore

        # Ollama
        cfg.ollama.base_url = os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url)
        cfg.ollama.model    = os.getenv("OLLAMA_MODEL",    cfg.ollama.model)

        # OpenAI
        cfg.openai.api_key  = os.getenv("OPENAI_API_KEY",  cfg.openai.api_key)
        cfg.openai.model    = os.getenv("OPENAI_MODEL",    cfg.openai.model)
        cfg.openai.base_url = os.getenv("OPENAI_BASE_URL", cfg.openai.base_url)

        # Anthropic
        cfg.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY", cfg.anthropic.api_key)
        cfg.anthropic.model   = os.getenv("ANTHROPIC_MODEL",   cfg.anthropic.model)

        # ── Source DB type ────────────────────────────────────────────────────
        # Accept both SOURCE_TYPE (explicit) and DB_TYPE (Neon .env convention).
        # SOURCE_TYPE takes precedence if both are set.
        source_type = (
            os.getenv("SOURCE_TYPE")
            or os.getenv("DB_TYPE")
            or cfg.source_type
        )
        cfg.source_type = source_type  # type: ignore

        # ── Connection string (Neon DSN) ──────────────────────────────────────
        # NEON_CONNECTION_STRING carries the full DSN including sslmode and
        # channel_binding, so nothing extra needs to be configured.
        # When present it takes priority; individual host/port/user vars are
        # still read so the config object is fully populated for logging/debug.
        neon_dsn = os.getenv("NEON_CONNECTION_STRING", "")
        if neon_dsn:
            cfg.source_connection_string = neon_dsn

        # Individual source DB parameters (used when no full DSN is set)
        cfg.source_host     = os.getenv("SOURCE_HOST",     cfg.source_host)
        cfg.source_port     = int(os.getenv("SOURCE_PORT", str(cfg.source_port)))
        cfg.source_user     = os.getenv("SOURCE_USER",     cfg.source_user)
        cfg.source_password = os.getenv("SOURCE_PASSWORD", cfg.source_password)
        cfg.source_database = os.getenv("SOURCE_DATABASE", cfg.source_database)
        cfg.source_table    = os.getenv("PG_TABLE",        cfg.source_table)

        # Paths
        cfg.chroma_path     = os.getenv("CHROMA_PATH",     cfg.chroma_path)
        cfg.collection_name = os.getenv("COLLECTION_NAME", cfg.collection_name)
        cfg.checkpoint_db   = os.getenv("CHECKPOINT_DB",   cfg.checkpoint_db)

        return cfg
    """================= End method from_env ================="""

"""================= End class RAGConfig ================="""