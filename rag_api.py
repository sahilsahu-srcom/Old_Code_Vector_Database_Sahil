"""
File Summary:
FastAPI REST API wrapper around the HAUP RAG engine. Exposes conversational AI
endpoints for session management, question answering, SSE streaming, health checks,
cache control, and analytics over HAUP structured vector data.

====================================================================
Startup
====================================================================

app  [FastAPI Instance]
||
├── CORSMiddleware  [Middleware] ---------------------------> Allow cross-origin requests
│
├── get_engine()  [Function] ------------------------------> Lazy-init global RAGEngine instance
│       │
│       ├── RAGConfig.from_env()  [Function] --------------> Load config from environment
│       └── RAGEngine()  [Class → Object] -----------------> Initialize RAG engine
│
├── startup()  [Function] ---------------------------------> Pre-warm engine on app startup
│       │
│       └── loop.run_in_executor()  [Function] ------------> Background engine warm-up
│
├── Session Endpoints
│       │
│       ├── create_session()  [Function] ------------------> POST /sessions
│       │       └── engine.new_session()  [Function] ------> Create and return session_id
│       │
│       ├── delete_session()  [Function] ------------------> DELETE /sessions/{id}
│       │       └── engine._conversation.delete() ---------> Remove session from store
│       │
│       ├── get_history()  [Function] ---------------------> GET /sessions/{id}/history
│       │       ├── engine.get_session_history() ----------> Fetch conversation turns
│       │       └── [Conditional Branch] -----------------> Session found?
│       │               ├── Found  -----------------------> Return List[HistoryTurn]
│       │               └── Not found  -------------------> Raise HTTP 404
│       │
│       └── list_sessions()  [Function] -------------------> GET /sessions
│               └── engine._conversation.list_sessions() --> Return active session list
│
├── RAG Endpoints
│       │
│       ├── ask()  [Function] -----------------------------> POST /sessions/{id}/ask
│       │       ├── engine.ask()  [Function] --------------> Blocking RAG query
│       │       ├── [Exception Block] --------------------> Raise HTTP 500 on failure
│       │       └── Return AskResponse --------------------> answer, citations, latency, cache_hit
│       │
│       └── ask_stream()  [Function] ----------------------> GET /sessions/{id}/stream
│               ├── engine.ask_stream()  [Function] -------> Token-by-token generator
│               ├── generate()  [Function] ----------------> SSE event formatter
│               │       ├── yield "data: <token>\n\n" -----> Stream each token
│               │       ├── [Exception Block] ------------> yield "[ERROR]" on failure
│               │       └── yield "data: [DONE]\n\n" ------> Signal stream end
│               └── StreamingResponse() -------------------> Return SSE response
│
├── System Endpoints
│       │
│       ├── health()  [Function] -------------------------> GET /health
│       │       ├── engine.health_check()  [Function] ----> Check LLM + ChromaDB status
│       │       ├── [Conditional Branch] -----------------> llm_healthy AND chroma_healthy?
│       │       │       ├── True  -----------------------> status = "healthy"
│       │       │       └── False  ----------------------> status = "degraded"
│       │       └── [Exception Block] -------------------> status = "unhealthy"
│       │
│       ├── cache_stats()  [Function] --------------------> GET /cache/stats
│       │       └── engine._cache.stats()  [Function] ----> Return cache hit/miss metrics
│       │
│       ├── clear_cache()  [Function] --------------------> DELETE /cache
│       │       └── engine._cache.clear()  [Function] ----> Flush response cache
│       │
│       └── background_status()  [Function] --------------> GET /background/status
│               ├── [Conditional Branch] -----------------> bg_worker running?
│               │       ├── None  -----------------------> Return not_running status
│               │       └── Running  --------------------> Return job status list
│               └── engine._bg_worker.status()  [Function] -> Active job details
│
└── Analytics Endpoints
        │
        ├── analytics_summary()  [Function] --------------> GET /analytics/summary
        │       └── engine.analytics_summary()  [Function] -> Usage stats for last N hours
        │
        ├── analytics_top_queries()  [Function] ----------> GET /analytics/top-queries
        │       └── engine._analytics.top_queries() ------> Most frequent query patterns
        │
        ├── analytics_errors()  [Function] ---------------> GET /analytics/errors
        │       └── engine._analytics.error_log() --------> Recent error entries
        │
        └── analytics_p95()  [Function] ------------------> GET /analytics/p95-latency
                └── engine._analytics.p95_latency() ------> P95 latency for last N hours

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Path, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI not installed. Run:\n"
        "  pip install fastapi uvicorn"
    )

from rag_core.config import RAGConfig
from rag_core.rag_engine import RAGEngine, RAGResponse


# ─── Progress reporting ───────────────────────────────────────────────────────

"""================= Startup function progress_update ================="""
def progress_update(step: str, substep: str, status: str = "⏳", details: str = ""):
    """Real-time progress update with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"[{timestamp}] [{status}] {step} → {substep} {details}")
"""================= End function progress_update ================="""


# ─── Engine init ──────────────────────────────────────────────────────────────
# Defined before lifespan so the lifespan context can reference it directly.

# Global engine instance (lazy init on first request)
_engine: Optional[RAGEngine] = None


"""================= Startup function get_engine ================="""
def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        progress_update("ENGINE", "Initialization", "⏳", "Creating RAG engine from environment config...")
        progress_update("ENGINE.1", "Config Loading", "⏳", "Loading RAGConfig from environment...")
        config = RAGConfig.from_env()
        progress_update("ENGINE.1", "Config Loading", "✅", "Configuration loaded")
        progress_update("ENGINE.2", "Component Setup", "⏳", "Initializing retriever, LLM, cache...")
        _engine = RAGEngine(config)
        progress_update("ENGINE.2", "Component Setup", "✅", "All components initialized")
        progress_update("ENGINE", "Initialization", "✅", "RAG engine initialized successfully")
    return _engine
"""================= End function get_engine ================="""


# ─── App setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    progress_update("API", "Startup", "⏳", "Pre-warming RAG engine...")
    progress_update("API.1", "Event Loop", "⏳", "Getting asyncio event loop...")
    loop = asyncio.get_event_loop()
    progress_update("API.1", "Event Loop", "✅", "Event loop ready")
    progress_update("API.2", "Engine Warmup", "⏳", "Pre-loading RAG engine in background...")
    loop.run_in_executor(None, get_engine)
    progress_update("API.2", "Engine Warmup", "✅", "Engine warmup initiated")
    progress_update("API", "Startup", "✅", "API server ready")
    yield
    # Shutdown
    progress_update("API", "Shutdown", "✅", "API server shutting down gracefully")

app = FastAPI(
    title="HAUP RAG API",
    description="Conversational AI interface for HAUP structured data",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

"""================= Startup class CreateSessionRequest ================="""
class CreateSessionRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional session metadata")
"""================= End class CreateSessionRequest ================="""


"""================= Startup class CreateSessionResponse ================="""
class CreateSessionResponse(BaseModel):
    session_id: str
"""================= End class CreateSessionResponse ================="""


"""================= Startup class AskRequest ================="""
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    use_cache: bool = True
"""================= End class AskRequest ================="""


"""================= Startup class Citation ================="""
class Citation(BaseModel):
    index: int
    rowid: str
    similarity: float
    source: str
"""================= End class Citation ================="""


"""================= Startup class AskResponse ================="""
class AskResponse(BaseModel):
    answer: str
    session_id: str
    citations: List[Citation]
    retrieved_rows: int
    latency_ms: float
    cache_hit: bool
    expanded_queries: List[str]
    source_db_available: bool
"""================= End class AskResponse ================="""


"""================= Startup class HealthResponse ================="""
class HealthResponse(BaseModel):
    status: str
    details: Dict[str, Any]
"""================= End class HealthResponse ================="""


"""================= Startup class HistoryTurn ================="""
class HistoryTurn(BaseModel):
    role: str
    content: str
    timestamp: float
    citations: Optional[List[Dict]] = None
"""================= End class HistoryTurn ================="""


# ─── Session endpoints ────────────────────────────────────────────────────────

"""================= Startup function root ================="""
@app.get("/", tags=["System"])
def root():
    """Root endpoint - API information"""
    return {
        "name": "HAUP RAG API",
        "version": "2.0.0",
        "description": "Conversational AI interface for HAUP structured data",
        "docs": "/docs",
        "health": "/health"
    }
"""================= End function root ================="""


"""================= Startup function create_session ================="""
@app.post("/sessions", response_model=CreateSessionResponse, status_code=201,
          tags=["Sessions"])
def create_session(req: CreateSessionRequest = CreateSessionRequest()):
    """Create a new conversation session. Returns a session_id."""
    progress_update("SESSION", "Creation", "⏳", "Creating new conversation session...")
    progress_update("SESSION.1", "Engine Access", "⏳", "Getting RAG engine instance...")
    engine = get_engine()
    progress_update("SESSION.1", "Engine Access", "✅", "Engine ready")
    progress_update("SESSION.2", "ID Generation", "⏳", "Generating unique session ID...")
    sid = engine.new_session(req.metadata)
    progress_update("SESSION.2", "ID Generation", "✅", f"Session ID: {sid[:8]}...")
    progress_update("SESSION", "Creation", "✅", f"Session created successfully")
    return CreateSessionResponse(session_id=sid)
"""================= End function create_session ================="""


"""================= Startup function delete_session ================="""
@app.delete("/sessions/{session_id}", status_code=204, tags=["Sessions"])
def delete_session(session_id: str = Path(...)):
    """Delete a session and its conversation history."""
    engine = get_engine()
    engine._conversation.delete(session_id)
"""================= End function delete_session ================="""


"""================= Startup function get_history ================="""
@app.get("/sessions/{session_id}/history", response_model=List[HistoryTurn],
         tags=["Sessions"])
def get_history(session_id: str = Path(...)):
    """Return the conversation history for a session."""
    engine = get_engine()
    history = engine.get_session_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return [HistoryTurn(**turn) for turn in history]
"""================= End function get_history ================="""


"""================= Startup function list_sessions ================="""
@app.get("/sessions", tags=["Sessions"])
def list_sessions():
    """List active sessions (admin endpoint)."""
    engine = get_engine()
    return engine._conversation.list_sessions()
"""================= End function list_sessions ================="""


# ─── Ask endpoints ────────────────────────────────────────────────────────────

"""================= Startup function ask ================="""
@app.post("/sessions/{session_id}/ask", response_model=AskResponse, tags=["RAG"])
def ask(session_id: str, req: AskRequest):
    """
    Ask a question in a session.
    Returns the complete answer after all retrieval and LLM generation.
    """
    progress_update("RAG", "Query Processing", "⏳", f"Processing question: {req.question[:50]}...")
    progress_update("RAG.1", "Engine Access", "⏳", "Getting RAG engine instance...")
    engine = get_engine()
    progress_update("RAG.1", "Engine Access", "✅", "Engine ready")

    try:
        progress_update("RAG.2", "Cache Check", "⏳", f"Checking cache (enabled: {req.use_cache})...")
        progress_update("RAG.3", "Query Rewrite", "⏳", "Expanding query for better retrieval...")
        progress_update("RAG.4", "Vector Retrieval", "⏳", "Searching ChromaDB for relevant vectors...")
        progress_update("RAG.5", "Context Building", "⏳", "Building context from retrieved rows...")
        progress_update("RAG.6", "LLM Generation", "⏳", "Generating response from LLM...")
        resp: RAGResponse = engine.ask(req.question, session_id, use_cache=req.use_cache)
        progress_update("RAG.6", "LLM Generation", "✅", f"Response generated")
        progress_update("RAG.7", "Citation Formatting", "✅", f"Retrieved {resp.retrieved_rows} rows")
        progress_update("RAG", "Query Processing", "✅", f"Response generated in {resp.latency_ms:.0f}ms (cache: {resp.cache_hit})")
    except Exception as exc:
        progress_update("RAG", "Query Processing", "❌", f"Failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=resp.answer,
        session_id=resp.session_id,
        citations=[Citation(**c) for c in resp.citations],
        retrieved_rows=resp.retrieved_rows,
        latency_ms=resp.latency_ms,
        cache_hit=resp.cache_hit,
        expanded_queries=resp.expanded_queries,
        source_db_available=resp.source_db_available,
    )
"""================= End function ask ================="""


"""================= Startup function ask_stream ================="""
@app.get("/sessions/{session_id}/stream", tags=["RAG"])
def ask_stream(
    session_id: str,
    question: str = Query(..., min_length=1, max_length=2000),
):
    """
    Ask a question with Server-Sent Events streaming.
    The response is streamed token-by-token as `data: <token>` SSE events.
    A final `data: [DONE]` event signals completion.

    Client example:
        const es = new EventSource(`/sessions/${sid}/stream?question=...`);
        es.onmessage = e => { if (e.data === '[DONE]') es.close(); else write(e.data); }
    """
    progress_update("STREAM", "Setup", "⏳", f"Starting stream for: {question[:50]}...")
    progress_update("STREAM.1", "Engine Access", "⏳", "Getting RAG engine instance...")
    engine = get_engine()
    progress_update("STREAM.1", "Engine Access", "✅", "Engine ready")
    progress_update("STREAM.2", "SSE Preparation", "✅", "Server-Sent Events stream configured")

    """================= Startup function generate ================="""
    def generate():
        try:
            progress_update("STREAM.3", "Query Rewrite", "⏳", "Expanding query...")
            progress_update("STREAM.4", "Vector Retrieval", "⏳", "Searching ChromaDB...")
            progress_update("STREAM.5", "Context Building", "⏳", "Building context...")
            progress_update("STREAM.6", "Token Streaming", "🔄", "Streaming tokens from LLM...")
            for token in engine.ask_stream(question, session_id=session_id):
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
            progress_update("STREAM.6", "Token Streaming", "✅", "All tokens streamed")
            progress_update("STREAM", "Generation", "✅", "Stream completed successfully")
        except Exception as exc:
            progress_update("STREAM", "Generation", "❌", f"Stream failed: {exc}")
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    """================= End function generate ================="""

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
"""================= End function ask_stream ================="""


# ─── Utility endpoints ────────────────────────────────────────────────────────

"""================= Startup function health ================="""
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check the health of all system components."""
    progress_update("HEALTH", "Check", "⏳", "Running system health check...")
    try:
        progress_update("HEALTH.1", "Engine Access", "⏳", "Getting RAG engine...")
        engine = get_engine()
        progress_update("HEALTH.1", "Engine Access", "✅", "Engine ready")
        progress_update("HEALTH.2", "Component Test", "⏳", "Testing LLM and ChromaDB...")
        details = engine.health_check()
        progress_update("HEALTH.2", "Component Test", "✅", "Components tested")
        status = "healthy" if details.get("llm_healthy") and details.get("chroma_healthy") \
            else "degraded"
        progress_update("HEALTH", "Check", "✅", f"Status: {status}")
    except Exception as exc:
        progress_update("HEALTH", "Check", "❌", f"Health check failed: {exc}")
        return HealthResponse(status="unhealthy", details={"error": str(exc)})
    return HealthResponse(status=status, details=details)
"""================= End function health ================="""


"""================= Startup function cache_stats ================="""
@app.get("/cache/stats", tags=["System"])
def cache_stats():
    """Return cache statistics."""
    engine = get_engine()
    return engine._cache.stats()
"""================= End function cache_stats ================="""


"""================= Startup function clear_cache ================="""
@app.delete("/cache", status_code=204, tags=["System"])
def clear_cache():
    """Clear the response cache."""
    engine = get_engine()
    engine._cache.clear()
"""================= End function clear_cache ================="""


"""================= Startup function background_status ================="""
@app.get("/background/status", tags=["System"])
def background_status():
    """Status of background maintenance jobs."""
    engine = get_engine()
    if engine._bg_worker is None:
        return {"status": "not_running", "jobs": []}
    return {"status": "running", "jobs": engine._bg_worker.status()}
"""================= End function background_status ================="""


# ─── Analytics endpoints ──────────────────────────────────────────────────────

"""================= Startup function analytics_summary ================="""
@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary(hours: int = Query(24, ge=1, le=720)):
    """High-level usage stats for the last N hours."""
    engine = get_engine()
    return engine.analytics_summary(hours=hours)
"""================= End function analytics_summary ================="""


"""================= Startup function analytics_top_queries ================="""
@app.get("/analytics/top-queries", tags=["Analytics"])
def analytics_top_queries(limit: int = Query(20, ge=1, le=100)):
    """Most frequent query patterns — useful for cache warming."""
    engine = get_engine()
    return engine._analytics.top_queries(limit=limit)
"""================= End function analytics_top_queries ================="""


"""================= Startup function analytics_errors ================="""
@app.get("/analytics/errors", tags=["Analytics"])
def analytics_errors(limit: int = Query(50, ge=1, le=200)):
    """Recent error log for debugging."""
    engine = get_engine()
    return engine._analytics.error_log(limit=limit)
"""================= End function analytics_errors ================="""


"""================= Startup function analytics_p95 ================="""
@app.get("/analytics/p95-latency", tags=["Analytics"])
def analytics_p95(hours: int = Query(24, ge=1, le=720)):
    """P95 latency for non-cached queries."""
    engine = get_engine()
    return {"p95_latency_ms": engine._analytics.p95_latency(last_n_hours=hours)}
"""================= End function analytics_p95 ================="""


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    progress_update("MAIN", "Server Start", "⏳", "Starting HAUP RAG API server...")
    progress_update("MAIN.1", "Dependencies", "⏳", "Importing uvicorn...")
    import uvicorn
    progress_update("MAIN.1", "Dependencies", "✅", "Uvicorn imported")

    progress_update("MAIN.2", "Configuration", "⏳", "Reading server configuration...")
    port = int(os.getenv("PORT", "8080"))
    progress_update("MAIN.2", "Configuration", "✅", f"Server will run on 127.0.0.1:{port}")

    progress_update("MAIN.3", "Uvicorn Launch", "⏳", "Launching uvicorn server...")
    progress_update("MAIN.3.1", "Worker Setup", "✅", "Single worker mode (engine not multi-process safe)")
    progress_update("MAIN.3.2", "Server Binding", "⏳", f"Binding to 127.0.0.1:{port}...")
    uvicorn.run(
        "rag_api:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        workers=1,   # single worker — engine is not multi-process safe
        log_level="info",
    )


"""
====================================================================
How to Run
====================================================================

Install dependencies:
    pip install fastapi uvicorn

Start the API server:
    uvicorn rag_api:app --host 0.0.0.0 --port 8080 --reload

Run directly:
    python rag_api.py

Run with custom port:
    PORT=9090 python rag_api.py

Interactive API docs (after server is running):
    http://localhost:8080/docs      ← Swagger UI
    http://localhost:8080/redoc     ← ReDoc UI

Example usage:

  Create a session:
    curl -X POST http://localhost:8080/sessions

  Ask a question (blocking):
    curl -X POST http://localhost:8080/sessions/{session_id}/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "Show me users from Delhi", "use_cache": true}'

  Ask with SSE streaming:
    curl http://localhost:8080/sessions/{session_id}/stream?question=Who+are+the+top+users

  Get conversation history:
    curl http://localhost:8080/sessions/{session_id}/history

  Delete a session:
    curl -X DELETE http://localhost:8080/sessions/{session_id}

  Health check:
    curl http://localhost:8080/health

  Cache statistics:
    curl http://localhost:8080/cache/stats

  Clear cache:
    curl -X DELETE http://localhost:8080/cache

  Analytics summary (last 24 hours):
    curl http://localhost:8080/analytics/summary?hours=24

  Top queries:
    curl http://localhost:8080/analytics/top-queries?limit=10

  P95 latency:
    curl http://localhost:8080/analytics/p95-latency?hours=24
"""