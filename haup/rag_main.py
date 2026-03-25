"""
File Summary:
Interactive CLI for the HAUP RAG engine. Provides a Rich terminal UI with session management,
streaming output, citation display, and single-shot query mode backed by a configurable RAGEngine.

====================================================================
Startup
====================================================================

main()
||
├── parse_args()  [Function] ------------------------------> Parse CLI arguments
│
├── build_engine()  [Function] ----------------------------> Initialise RAGEngine from config
│       │
│       ├── RAGConfig.from_env()  [Class → Object] --------> Load base config from environment
│       └── RAGEngine()  [Class → Object] ------------------> Build engine with merged config
│
├── [Conditional Branch] args.query -----------------------> Single-shot or interactive mode?
│       │
│       ├── run_single_query()  [Function] -----------------> Single non-interactive query
│       │       │
│       │       ├── engine.new_session()  [Function] -------> Create session ID
│       │       │
│       │       ├── [Conditional Branch] stream=True -------> Streaming or full response?
│       │       │       ├── display_stream()  [Function] ---> Stream tokens live to terminal
│       │       │       └── display_response()  [Function] -> Render full RAGResponse panel
│       │       │
│       │       └── engine.ask()  [Function] ---------------> Fetch full response (no-stream path)
│       │
│       └── run_interactive()  [Function] ------------------> Interactive REPL loop
│               │
│               ├── engine.new_session()  [Function] -------> Create initial session ID
│               │
│               └── REPL Loop -----------------------------> Read → Dispatch → Repeat
│                       │
│                       ├── /quit / /exit -----------------> Exit loop
│                       ├── /help -------------------------> Print HELP_TEXT
│                       ├── /new --------------------------> engine.new_session()
│                       ├── /history ---------------------> display_history()  [Function]
│                       ├── /health ----------------------> display_health()   [Function]
│                       ├── /clear -----------------------> console.clear()
│                       │
│                       └── Normal question
│                               │
│                               ├── display_stream()  [Function] -------> stream=True path
│                               └── display_response()  [Function] -----> stream=False path
│
└── [Exception Block] -------------------------------------> sys.exit(1) on engine init failure

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rag_core.config import RAGConfig
from rag_core.rag_engine import RAGEngine, RAGResponse


console = Console()


"""================= Startup function progress_update ================="""
def progress_update(step: str, substep: str, status: str = "⏳", details: str = ""):
    """Real-time progress update with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")

    status_color = "green" if status == "✅" else \
                   "red" if status == "❌" else \
                   "yellow" if status == "⚠️" else "cyan"

    console.print(f"[dim]{timestamp}[/] [{status_color}]{status}[/] [bold]{step}[/] → [cyan]{substep}[/] {details}")
"""================= End function progress_update ================="""


"""================= Startup function parse_args ================="""
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HAUP RAG — Conversational interface for your structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["ollama", "openai", "anthropic"],
                   default=None, help="LLM backend (default: ollama)")
    p.add_argument("--model", default=None, help="Model name for chosen backend")
    p.add_argument("--query", "-q", default=None, help="Single non-interactive query")
    p.add_argument("--top-k", type=int, default=None, help="Max rows to retrieve")
    p.add_argument("--stream", action="store_true", default=True,
                   help="Stream tokens as they arrive (default: True)")
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--chroma-path", default=None)
    p.add_argument("--collection", default=None)
    p.add_argument("--checkpoint-db", default=None)
    # postgresql added for Neon (reads NEON_CONNECTION_STRING from env)
    p.add_argument("--source-type", choices=["mysql", "sqlite", "postgresql", "none"],
                   default=None,
                   help="Source DB type. Use 'postgresql' for Neon.")
    p.add_argument("--source-db", default=None,
                   help="PostgreSQL/Neon database name, MySQL database name, or SQLite path")
    p.add_argument("--source-host", default=None)
    p.add_argument("--source-user", default=None)
    p.add_argument("--source-password", default=None)
    p.add_argument("--source-table", default=None)
    return p.parse_args()
"""================= End function parse_args ================="""


"""================= Startup function build_engine ================="""
def build_engine(args: argparse.Namespace) -> RAGEngine:
    progress_update("ENGINE", "Configuration", "⏳", "Loading RAG configuration...")
    progress_update("ENGINE.1", "Environment Config", "⏳", "Reading environment variables...")
    cfg = RAGConfig.from_env()
    progress_update("ENGINE.1", "Environment Config", "✅", "Base configuration loaded")

    progress_update("ENGINE.2", "Arguments", "⏳", "Processing command line arguments...")
    if args.backend:
        cfg.llm_backend = args.backend  # type: ignore
        progress_update("ENGINE.2.1", "Backend Override", "✅", f"Backend set to {args.backend}")
    if args.model:
        backend = cfg.llm_backend
        if backend == "ollama":      cfg.ollama.model = args.model
        elif backend == "openai":    cfg.openai.model = args.model
        elif backend == "anthropic": cfg.anthropic.model = args.model
        progress_update("ENGINE.2.2", "Model Override", "✅", f"Model set to {args.model}")
    if args.top_k:
        cfg.retrieval.top_k = args.top_k
        progress_update("ENGINE.2.3", "Retrieval Config", "✅", f"Top-K set to {args.top_k}")
    if args.chroma_path:
        cfg.chroma_path = args.chroma_path
    if args.collection:
        cfg.collection_name = args.collection
    if args.checkpoint_db:
        cfg.checkpoint_db = args.checkpoint_db
    if args.source_type:
        cfg.source_type = args.source_type  # type: ignore
    if args.source_db:
        cfg.source_database = args.source_db
    if args.source_host:
        cfg.source_host = args.source_host
    if args.source_user:
        cfg.source_user = args.source_user
    if args.source_password:
        cfg.source_password = args.source_password
    if args.source_table:
        cfg.source_table = args.source_table
    progress_update("ENGINE.2", "Arguments", "✅", "Arguments processed successfully")

    progress_update("ENGINE.3", "Initialization", "⏳", "Creating RAG engine...")
    progress_update("ENGINE.3.1", "Component Setup", "⏳", "Initializing retriever, LLM client, cache...")
    engine = RAGEngine(cfg)
    progress_update("ENGINE.3.1", "Component Setup", "✅", "All components initialized")
    progress_update("ENGINE.3", "Initialization", "✅", "RAG engine created successfully")
    return engine
"""================= End function build_engine ================="""


"""================= Startup function display_response ================="""
def display_response(response: RAGResponse) -> None:
    """Render a complete RAGResponse to the terminal."""
    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="[bold green]Answer[/bold green]",
        border_style="green",
    ))

    if response.citations:
        relevant_citations = [c for c in response.citations if c["similarity"] > 0.1]

        if relevant_citations:
            t = Table(title="Sources", show_header=True, header_style="bold cyan",
                      box=None, padding=(0, 1))
            t.add_column("#", style="dim", width=4)
            t.add_column("Row ID", width=10)
            t.add_column("Similarity", width=15)
            t.add_column("Source", width=20)

            for c in relevant_citations:
                bar = "█" * int(c["similarity"] * 10) + "░" * (10 - int(c["similarity"] * 10))
                t.add_row(
                    str(c["index"]),
                    str(c["rowid"]),
                    f"{bar} {c['similarity']:.2f}",
                    c.get("source", "Users"),
                )
            console.print(t)

    meta = (
        f"[dim]{response.retrieved_rows} rows retrieved · "
        f"{response.latency_ms:.0f}ms · "
        f"session {response.session_id[:8]}…[/dim]"
    )
    if not response.source_db_available:
        meta += "  [yellow]⚠ Source DB unavailable[/yellow]"
    console.print(meta)
"""================= End function display_response ================="""


"""================= Startup function display_stream ================="""
def display_stream(engine: RAGEngine, question: str, session_id: str) -> None:
    """Stream tokens with live display."""
    console.print()
    console.print("[bold green]Answer[/bold green]")
    console.print("─" * 60)
    try:
        for token in engine.ask_stream(question, session_id=session_id):
            console.print(token, end="", highlight=False)
        console.print()
        console.print("─" * 60)
        console.print(f"[dim]session {session_id[:8]}…[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
"""================= End function display_stream ================="""


"""================= Startup function display_history ================="""
def display_history(engine: RAGEngine, session_id: str) -> None:
    history = engine.get_session_history(session_id)
    if not history:
        console.print("[yellow]No history found.[/yellow]")
        return
    for turn in history:
        prefix = "[bold cyan]You:[/bold cyan]" if turn["role"] == "user" \
            else "[bold green]Assistant:[/bold green]"
        console.print(f"\n{prefix}")
        console.print(Markdown(turn["content"]))
"""================= End function display_history ================="""


"""================= Startup function display_health ================="""
def display_health(engine: RAGEngine) -> None:
    status = engine.health_check()
    t = Table(title="System Health", show_header=False, box=None)
    t.add_column("Key", style="bold")
    t.add_column("Value")
    for k, v in status.items():
        colour = "green" if str(v) in ("True", "true") else \
                 "red" if str(v) in ("False", "false") else "white"
        t.add_row(k, f"[{colour}]{v}[/{colour}]")
    console.print(t)
"""================= End function display_health ================="""


"""================= Startup function run_single_query ================="""
def run_single_query(engine: RAGEngine, question: str, stream: bool) -> None:
    progress_update("QUERY", "Session Creation", "⏳", "Creating new session...")
    progress_update("QUERY.1", "Session ID Generation", "⏳", "Generating unique session identifier...")
    session_id = engine.new_session()
    progress_update("QUERY.1", "Session ID Generation", "✅", f"Session ID: {session_id[:8]}...")
    progress_update("QUERY", "Session Creation", "✅", f"Session created successfully")

    if stream:
        progress_update("QUERY.2", "Streaming", "⏳", "Starting streaming response...")
        progress_update("QUERY.2.1", "Retrieval", "⏳", "Retrieving relevant vectors...")
        progress_update("QUERY.2.2", "LLM Generation", "⏳", "Streaming tokens from LLM...")
        display_stream(engine, question, session_id)
        progress_update("QUERY.2.2", "LLM Generation", "✅", "Token streaming completed")
        progress_update("QUERY.2", "Streaming", "✅", "Streaming completed")
    else:
        progress_update("QUERY.2", "Processing", "⏳", "Processing query...")
        progress_update("QUERY.2.1", "Retrieval", "⏳", "Retrieving relevant vectors...")
        progress_update("QUERY.2.2", "LLM Generation", "⏳", "Generating response...")
        with console.status("[bold green]Thinking…[/bold green]"):
            response = engine.ask(question, session_id)
        progress_update("QUERY.2.2", "LLM Generation", "✅", f"Response generated")
        progress_update("QUERY.2.3", "Citation Building", "✅", f"Found {response.retrieved_rows} relevant rows")
        progress_update("QUERY.2", "Processing", "✅", f"Query processed in {response.latency_ms:.0f}ms")
        display_response(response)
"""================= End function run_single_query ================="""


HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/new[/cyan]       — Start a new conversation session
  [cyan]/history[/cyan]   — Show current session history
  [cyan]/health[/cyan]    — Check system component health
  [cyan]/clear[/cyan]     — Clear screen
  [cyan]/quit[/cyan]      — Exit
  [dim]Anything else is sent as a question to the RAG engine.[/dim]
"""


"""================= Startup function run_interactive ================="""
def run_interactive(engine: RAGEngine, stream: bool) -> None:
    progress_update("INTERACTIVE", "Session Setup", "⏳", "Starting interactive session...")
    progress_update("INTERACTIVE.1", "Session Creation", "⏳", "Generating session ID...")
    session_id = engine.new_session()
    progress_update("INTERACTIVE.1", "Session Creation", "✅", f"Session ID: {session_id[:8]}...")
    progress_update("INTERACTIVE.2", "UI Initialization", "✅", "Terminal UI ready")
    progress_update("INTERACTIVE", "Session Setup", "✅", f"Interactive session ready")

    console.print(Panel(
        "[bold]HAUP RAG[/bold] — Conversational data analyst\n"
        f"Session: [dim]{session_id[:8]}…[/dim]\n"
        "Type [cyan]/help[/cyan] for commands.",
        border_style="blue",
    ))

    while True:
        try:
            raw = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            progress_update("INTERACTIVE", "Session End", "✅", "Session terminated by user")
            console.print("\n[yellow]Goodbye.[/yellow]")
            break

        if not raw:
            continue

        if raw.lower() in ("/quit", "/exit", "/q"):
            progress_update("INTERACTIVE", "Session End", "✅", "Session ended gracefully")
            console.print("[yellow]Goodbye.[/yellow]")
            break
        elif raw.lower() in ("/help", "/?"):
            console.print(HELP_TEXT)
        elif raw.lower() == "/new":
            progress_update("INTERACTIVE", "New Session", "⏳", "Creating new session...")
            progress_update("INTERACTIVE.NEW.1", "Session Reset", "⏳", "Clearing conversation history...")
            session_id = engine.new_session()
            progress_update("INTERACTIVE.NEW.1", "Session Reset", "✅", "History cleared")
            progress_update("INTERACTIVE", "New Session", "✅", f"New session: {session_id[:8]}...")
            console.print(f"[green]New session: {session_id[:8]}…[/green]")
        elif raw.lower() == "/history":
            progress_update("INTERACTIVE", "History", "⏳", "Fetching session history...")
            progress_update("INTERACTIVE.HIST.1", "History Retrieval", "⏳", f"Loading conversation for {session_id[:8]}...")
            display_history(engine, session_id)
            progress_update("INTERACTIVE.HIST.1", "History Retrieval", "✅", "History loaded")
            progress_update("INTERACTIVE", "History", "✅", "History displayed")
        elif raw.lower() == "/health":
            progress_update("INTERACTIVE", "Health Check", "⏳", "Checking system health...")
            progress_update("INTERACTIVE.HEALTH.1", "Component Check", "⏳", "Testing LLM and ChromaDB...")
            display_health(engine)
            progress_update("INTERACTIVE.HEALTH.1", "Component Check", "✅", "All components checked")
            progress_update("INTERACTIVE", "Health Check", "✅", "Health check completed")
        elif raw.lower() == "/clear":
            console.clear()
        else:
            progress_update("INTERACTIVE", "Query Processing", "⏳", f"Processing: {raw[:50]}...")
            progress_update("INTERACTIVE.Q.1", "Query Rewrite", "⏳", "Expanding query...")
            progress_update("INTERACTIVE.Q.2", "Vector Retrieval", "⏳", "Searching ChromaDB...")
            progress_update("INTERACTIVE.Q.3", "Context Building", "⏳", "Building context from results...")
            if stream:
                progress_update("INTERACTIVE.Q.4", "LLM Streaming", "⏳", "Streaming response tokens...")
                display_stream(engine, raw, session_id)
                progress_update("INTERACTIVE.Q.4", "LLM Streaming", "✅", "Stream completed")
                progress_update("INTERACTIVE", "Query Processing", "✅", "Streaming response completed")
            else:
                progress_update("INTERACTIVE.Q.4", "LLM Generation", "⏳", "Generating response...")
                with console.status("[bold green]Thinking…[/bold green]"):
                    response = engine.ask(raw, session_id)
                progress_update("INTERACTIVE.Q.4", "LLM Generation", "✅", f"Response generated")
                progress_update("INTERACTIVE.Q.5", "Citation Formatting", "✅", f"{response.retrieved_rows} sources retrieved")
                progress_update("INTERACTIVE", "Query Processing", "✅", f"Response generated in {response.latency_ms:.0f}ms")
                display_response(response)
"""================= End function run_interactive ================="""


"""================= Startup function main ================="""
def main() -> None:
    progress_update("STARTUP", "Initialization", "⏳", "Starting HAUP RAG system...")
    progress_update("STARTUP.1", "Argument Parsing", "⏳", "Parsing command line arguments...")
    args = parse_args()
    progress_update("STARTUP.1", "Argument Parsing", "✅", "Arguments parsed successfully")

    progress_update("STARTUP.2", "Logging Setup", "⏳", "Configuring logging levels...")
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("haup").setLevel(logging.WARNING)
    progress_update("STARTUP.2", "Logging Setup", "✅", "Logging configured")

    if not args.query:
        console.print("[dim]Loading RAG engine…[/dim]")

    progress_update("STARTUP.3", "Engine Build", "⏳", "Building RAG engine...")
    try:
        engine = build_engine(args)
        progress_update("STARTUP.3", "Engine Build", "✅", "RAG engine ready")
    except Exception as exc:
        progress_update("STARTUP.3", "Engine Build", "❌", f"Failed: {exc}")
        console.print(f"[bold red]Failed to initialise engine:[/bold red] {exc}")
        sys.exit(1)

    progress_update("STARTUP.4", "Mode Selection", "⏳", "Determining execution mode...")
    if args.query:
        progress_update("STARTUP.4", "Mode Selection", "✅", "Single query mode")
        run_single_query(engine, args.query, stream=args.stream)
    else:
        progress_update("STARTUP.4", "Mode Selection", "✅", "Interactive mode")
        run_interactive(engine, stream=args.stream)
"""================= End function main ================="""


if __name__ == "__main__":
    main()