"""
File Summary:
Central orchestrator for the HAUP RAG engine. Coordinates the full query pipeline
from input guardrails through vector retrieval, reranking, context building, LLM
generation, output guardrails, caching, analytics, and conversation persistence.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

RAGEngine()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Initialise all pipeline components
│       │
│       ├── log.setup() ---------------------------------> Configure structured logging
│       ├── Retriever()  [Class → Object] ----------------> ChromaDB vector search
│       ├── build_llm_client()  [Function] ---------------> Ollama / OpenAI / Anthropic client
│       ├── QueryRewriter()  [Class → Object] ------------> Multi-query expansion
│       ├── Reranker()  [Class → Object] -----------------> Cross-encoder reranking
│       │       └── [Exception Block] --------------------> Fall back to PassthroughReranker
│       ├── ResponseCache()  [Class → Object] ------------> SQLite-backed response cache
│       ├── ConversationManager()  [Class → Object] ------> Session and history persistence
│       ├── ContextBuilder()  [Class → Object] -----------> Format retrieved rows for prompt
│       ├── PromptBuilder()  [Class → Object] ------------> Assemble LLM message list
│       ├── Guardrails()  [Class → Object] ---------------> Input/output safety checks
│       ├── Analytics()  [Class → Object] ----------------> SQLite query event tracking
│       └── build_for_engine()  [Function] ---------------> Launch background maintenance worker
│               └── [Exception Block] --------------------> Warn and continue without bg worker
│
├── new_session()  [Function] ----------------------------> Create new conversation session
│
├── ask()  [Function] ------------------------------------> Full blocking RAG query pipeline
│       │
│       ├── Step 1: _guardrails.check_input() ------------> Validate and optionally modify query
│       │       └── [Early Exit Branch] not allowed ------> Return _blocked_response()
│       │
│       ├── Step 2: _cache.get() ------------------------> Return cached answer if hit
│       │       └── [Early Exit Branch] cache hit --------> Save to session, return RAGResponse
│       │
│       ├── Step 3: _rewriter.expand() -------------------> Generate expanded query variants
│       │
│       ├── Step 4: _retriever.retrieve() ----------------> Fetch top-k vectors from ChromaDB
│       │
│       ├── Step 5: _reranker.rerank() -------------------> Re-score and filter to top-n rows
│       │
│       ├── Step 6: _context_builder.build() -------------> Format rows into context string
│       │
│       ├── Step 7: _prompt_builder.build() --------------> Assemble system + history + user messages
│       │
│       ├── Step 8: _llm.chat() --------------------------> Generate answer from LLM
│       │       └── [Exception Block] --------------------> Log error, set fallback answer
│       │
│       ├── Step 9: _guardrails.check_output() -----------> Validate and optionally modify answer
│       │
│       ├── Step 10: _cache.set() + session save ---------> Persist answer and conversation turn
│       │
│       └── _record()  [Function] -----------------------> Write QueryEvent to analytics
│
├── ask_stream()  [Function] -----------------------------> Streaming RAG query pipeline
│       │
│       ├── _guardrails.check_input() --------------------> Validate input
│       │       └── [Early Exit Branch] not allowed ------> Yield [Blocked] message and return
│       │
│       ├── Expand → Retrieve → Rerank → Build -----------> Same as ask() steps 3-7
│       │
│       ├── _llm.chat(stream=True) -----------------------> Yield tokens one by one
│       │
│       └── Save cache + session + analytics -------------> Persist after stream completes
│
├── get_session_history()  [Function] -------------------> Return conversation turns for session
│       └── [Conditional Branch] session not found ------> Return None
│
├── health_check()  [Function] --------------------------> Check LLM, ChromaDB, reranker, cache
│
├── analytics_summary()  [Function] ---------------------> Delegate to Analytics.summary()
│
├── shutdown()  [Function] -------------------------------> Stop background worker, log shutdown
│
├── _get_or_create_session()  [Function] ----------------> Get existing or create new session
│
├── _blocked_response()  [Function] ---------------------> Build RAGResponse for blocked queries
│
└── _record()  [Function] -------------------------------> Record QueryEvent to Analytics

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from rag_core import logger as log
from rag_core.analytics import Analytics, QueryEvent
from rag_core.background_worker import BackgroundWorker, build_for_engine
from rag_core.cache import ResponseCache
from rag_core.config import GuardrailsConfig, RAGConfig
from rag_core.context_builder import ContextBuilder, build_schema_summary
from rag_core.conversation_manager import ConversationManager, Session
from rag_core.guardrails import Guardrails
from rag_core.llm_client import BaseLLMClient, build_llm_client
from rag_core.prompt_builder import PromptBuilder
from rag_core.query_rewriter import QueryRewriter
from rag_core.reranker import PassthroughReranker, Reranker
from rag_core.retriever import RetrievalResult, Retriever


"""================= Startup class RAGResponse ================="""
@dataclass
class RAGResponse:
    answer:              str
    session_id:          str
    citations:           List[Dict]
    retrieved_rows:      int
    latency_ms:          float
    cache_hit:           bool
    expanded_queries:    List[str]
    source_db_available: bool
    reranked:            bool       = False
    guard_warnings:      List[str]  = field(default_factory=list)
    metadata:            Dict       = field(default_factory=dict)
"""================= End class RAGResponse ================="""


"""================= Startup class RAGEngine ================="""
class RAGEngine:
    """
    Production RAG engine for HAUP v2.0.

    Quickstart:
        cfg = RAGConfig.from_env()
        engine = RAGEngine(cfg)
        sid = engine.new_session()
        response = engine.ask("find active users from India", sid)
        print(response.answer)
    """

    """================= Startup method __init__ ================="""
    def __init__(self, cfg: RAGConfig):
        self._cfg = cfg
        self._log = log.get("engine")

        log.setup(
            level              = cfg.observability.log_level,
            log_queries        = cfg.observability.log_queries,
            log_retrieved_rows = cfg.observability.log_retrieved_rows,
            log_llm_prompts    = cfg.observability.log_llm_prompts,
            trace_file         = cfg.observability.trace_file,
        )

        self._log.info("Initialising RAG engine (backend=%s)", cfg.llm_backend)
        t0 = time.perf_counter()

        self._retriever      = Retriever(cfg)
        self._llm: BaseLLMClient = build_llm_client(cfg)
        self._rewriter       = QueryRewriter(
            llm_client     = None,
            max_variations = cfg.retrieval.expansion_variations,
        )

        try:
            self._reranker: Reranker = Reranker(top_n=cfg.retrieval.rerank_top_n, enabled=True)
        except Exception:
            self._log.warning("Reranker unavailable, using passthrough")
            self._reranker = PassthroughReranker(top_n=cfg.retrieval.rerank_top_n)

        self._cache        = ResponseCache(cfg.cache, db_path=cfg.session_db)
        self._conversation = ConversationManager(cfg.conversation, db_path=cfg.session_db)

        schema_summary        = build_schema_summary(cfg.checkpoint_db)
        self._context_builder = ContextBuilder(cfg.context)
        self._prompt_builder  = PromptBuilder(schema_summary)
        self._guardrails      = Guardrails(GuardrailsConfig())
        self._analytics       = Analytics(
            db_path=cfg.session_db.replace(".db", "_analytics.db")
        )

        self._bg_worker: Optional[BackgroundWorker] = None
        try:
            self._bg_worker = build_for_engine(self)
        except Exception as exc:
            self._log.warning("Background worker failed: %s", exc)

        elapsed = (time.perf_counter() - t0) * 1000
        self._log.info(
            "RAG engine ready %.0fms | reranker=%s | backend=%s",
            elapsed,
            "cross-encoder" if self._reranker.is_available() else "passthrough",
            cfg.llm_backend,
        )
    """================= End method __init__ ================="""

    """================= Startup method new_session ================="""
    def new_session(self, metadata: Optional[Dict] = None) -> str:
        return self._conversation.new_session(metadata).session_id
    """================= End method new_session ================="""

    """================= Startup method ask ================="""
    def ask(
        self,
        question:   str,
        session_id: Optional[str] = None,
        *,
        use_cache:  bool = True,
    ) -> RAGResponse:
        t0             = time.perf_counter()
        guard_warnings: List[str]    = []
        error:          Optional[str] = None

        session = self._get_or_create_session(session_id)
        sid     = session.session_id

        # Step 1: Input guard
        guard_in = self._guardrails.check_input(question, session_id=sid)
        if not guard_in.allowed:
            return self._blocked_response(guard_in.block_reason or "Blocked", sid, t0)
        guard_warnings.extend(guard_in.warnings)
        effective_query = guard_in.modified_query or question

        # Step 2: Cache
        if use_cache:
            cached = self._cache.get(effective_query, session_id=sid)
            if cached is not None:
                session.add_user(question)
                session.add_assistant(cached)
                self._conversation.save(session)
                resp = RAGResponse(
                    answer               = cached,
                    session_id           = sid,
                    citations            = [],
                    retrieved_rows       = 0,
                    latency_ms           = (time.perf_counter() - t0) * 1000,
                    cache_hit            = True,
                    expanded_queries     = [question],
                    source_db_available  = False,
                    guard_warnings       = guard_warnings,
                )
                self._record(resp, question, sid, None)
                return resp

        # Step 3: Query expansion
        expanded = (self._rewriter.expand(effective_query)
                    if self._cfg.retrieval.enable_query_expansion
                    else [effective_query])
        log.log_query(sid, question, expanded)

        # Step 4: Retrieval
        retrieval: RetrievalResult = self._retriever.retrieve(expanded, session_id=sid)

        # Step 5: Reranking
        reranked_rows = self._reranker.rerank(query=effective_query, rows=retrieval.rows)
        did_rerank    = self._reranker.is_available() and bool(retrieval.rows)

        # Step 6: Context building
        context, citations = self._context_builder.build(reranked_rows)
        has_results        = bool(reranked_rows)

        # Step 7: Prompt assembly
        history  = session.to_messages(self._cfg.conversation.max_history_turns)
        messages = self._prompt_builder.build(
            question    = effective_query,
            context     = context,
            history     = history,
            has_results = has_results,
        )

        # Step 8: LLM call
        try:
            answer = self._llm.chat(
                messages, max_tokens=1024, temperature=0.2, session_id=sid,
            )
        except Exception as exc:
            error  = str(exc)
            log.log_error(sid, "llm_call", error)
            answer = f"Error generating response: {exc}"

        # Step 9: Output guard
        if not error:
            guard_out = self._guardrails.check_output(
                answer, [r.document for r in reranked_rows], session_id=sid,
            )
            guard_warnings.extend(guard_out.warnings)
            if guard_out.modified_response:
                answer = guard_out.modified_response

        # Step 10: Cache + persist
        if not error:
            self._cache.set(effective_query, answer)
        session.add_user(question)
        session.add_assistant(answer, citations=citations)
        self._conversation.save(session)

        resp = RAGResponse(
            answer               = str(answer),
            session_id           = sid,
            citations            = citations,
            retrieved_rows       = len(reranked_rows),
            latency_ms           = (time.perf_counter() - t0) * 1000,
            cache_hit            = False,
            expanded_queries     = expanded,
            source_db_available  = retrieval.source_db_available,
            reranked             = did_rerank,
            guard_warnings       = guard_warnings,
        )
        self._record(resp, question, sid, error)
        return resp
    """================= End method ask ================="""

    """================= Startup method ask_stream ================="""
    def ask_stream(
        self,
        question:   str,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        t0      = time.perf_counter()
        session = self._get_or_create_session(session_id)
        sid     = session.session_id

        guard_in = self._guardrails.check_input(question, session_id=sid)
        if not guard_in.allowed:
            yield f"[Blocked: {guard_in.block_reason}]"
            return

        effective_query = guard_in.modified_query or question
        expanded = (self._rewriter.expand(effective_query)
                    if self._cfg.retrieval.enable_query_expansion
                    else [effective_query])
        log.log_query(sid, question, expanded)

        retrieval     = self._retriever.retrieve(expanded, session_id=sid)
        reranked_rows = self._reranker.rerank(effective_query, retrieval.rows)
        context, citations = self._context_builder.build(reranked_rows)
        history  = session.to_messages(self._cfg.conversation.max_history_turns)
        messages = self._prompt_builder.build(
            question    = effective_query,
            context     = context,
            history     = history,
            has_results = bool(reranked_rows),
        )

        full_answer: List[str] = []
        for token in self._llm.chat(
            messages, max_tokens=1024, temperature=0.2, stream=True, session_id=sid,
        ):
            full_answer.append(token)
            yield token

        answer = "".join(full_answer)
        self._cache.set(effective_query, answer)
        session.add_user(question)
        session.add_assistant(answer, citations=citations)
        self._conversation.save(session)
        self._analytics.record(QueryEvent(
            session_id     = sid,
            query          = question,
            answer_length  = len(answer),
            retrieved_rows = len(reranked_rows),
            latency_ms     = (time.perf_counter() - t0) * 1000,
            cache_hit      = False,
            llm_backend    = self._cfg.llm_backend,
            llm_model      = self._llm.model,
        ))
    """================= End method ask_stream ================="""

    """================= Startup method get_session_history ================="""
    def get_session_history(self, session_id: str) -> Optional[List[Dict]]:
        session = self._conversation.get(session_id)
        if not session:
            return None
        return [
            {"role": t.role, "content": t.content,
             "timestamp": t.timestamp, "citations": t.citations}
            for t in session.turns
        ]
    """================= End method get_session_history ================="""

    """================= Startup method health_check ================="""
    def health_check(self) -> Dict:
        status = {
            "llm_backend":         self._cfg.llm_backend,
            "llm_model":           self._llm.model,
            "llm_healthy":         False,
            "chroma_healthy":      False,
            "reranker_available":  self._reranker.is_available(),
            "cache_stats":         self._cache.stats(),
        }
        try:
            status["llm_healthy"] = self._llm.health_check()
        except Exception as e:
            status["llm_error"] = str(e)
        try:
            n = self._retriever._collection.count()
            status["chroma_healthy"] = True
            status["vector_count"]   = n
        except Exception as e:
            status["chroma_error"] = str(e)
        if self._bg_worker:
            status["background_jobs"] = self._bg_worker.status()
        return status
    """================= End method health_check ================="""

    """================= Startup method analytics_summary ================="""
    def analytics_summary(self, hours: int = 24) -> Dict:
        return self._analytics.summary(last_n_hours=hours)
    """================= End method analytics_summary ================="""

    """================= Startup method shutdown ================="""
    def shutdown(self) -> None:
        if self._bg_worker:
            self._bg_worker.stop()
        self._log.info("RAG engine shut down")
    """================= End method shutdown ================="""

    """================= Startup method _get_or_create_session ================="""
    def _get_or_create_session(self, session_id: Optional[str]) -> Session:
        if session_id:
            session = self._conversation.get(session_id)
            if session:
                return session
        return self._conversation.new_session()
    """================= End method _get_or_create_session ================="""

    """================= Startup method _blocked_response ================="""
    def _blocked_response(self, reason: str, session_id: str, t0: float) -> RAGResponse:
        return RAGResponse(
            answer               = f"Your request could not be processed: {reason}.",
            session_id           = session_id,
            citations            = [],
            retrieved_rows       = 0,
            latency_ms           = (time.perf_counter() - t0) * 1000,
            cache_hit            = False,
            expanded_queries     = [],
            source_db_available  = False,
            guard_warnings       = [reason],
        )
    """================= End method _blocked_response ================="""

    """================= Startup method _record ================="""
    def _record(self, resp: RAGResponse, question: str, sid: str, error: Optional[str]) -> None:
        self._analytics.record(QueryEvent(
            session_id     = sid,
            query          = question,
            answer_length  = len(resp.answer),
            retrieved_rows = resp.retrieved_rows,
            latency_ms     = resp.latency_ms,
            cache_hit      = resp.cache_hit,
            llm_backend    = self._cfg.llm_backend,
            llm_model      = self._llm.model,
            error          = error,
            warnings       = resp.guard_warnings or None,
        ))
    """================= End method _record ================="""

"""================= End class RAGEngine ================="""