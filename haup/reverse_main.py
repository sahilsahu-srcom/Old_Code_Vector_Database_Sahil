"""
File Summary:
Main entry point of HAUP v3.0 Reverse Pipeline. Controls the full vector-to-PostgreSQL/Excel
extraction workflow.

run(args)
||
├── init_vector_db() -----------------------------------> chromadb.PersistentClient()
│
├── detect_hardware() ----------------------------------> CPU / RAM / GPU / VRAM
│
├── load_collection_stats() ----------------------------> get_collection_stats()
│       └── [If empty] --------------------------------> exit
│
├── load_schema() --------------------------------------> schema_loader.load()
│
├── infer_types() --------------------------------------> schema_loader.infer_sql_types()
│
├── load_constraints() ---------------------------------> read_postgresql_constraints()
│       └── [If fail] ---------------------------------> continue
│
├── reconcile_schema() ---------------------------------> strategy + types + constraints
│
├── init_checkpoint() ----------------------------------> ReverseCheckpoint()
│       ├── reset_running_to_pending()
│       ├── get_resume_summary()
│       └── [If complete] -----------------------------> exit
│
├── init_target() --------------------------------------> excel / postgresql
│
├── start_writer() -------------------------------------> ReverseWriter.start()
│
├── start_monitor() ------------------------------------> ReverseMonitor.start()
│
├── spawn_workers() ------------------------------------> launch processes
│
├── orchestrate_pipeline() -----------------------------> stream → process → store
│       │
│       ├── stream vectors
│       ├── checkpoint.is_done()
│       ├── checkpoint.mark_running()
│       ├── work_q.put()
│       ├── workers process
│       ├── result_q.put()
│       ├── writer writes
│       └── checkpoint updates
│       │
│       └── [If interrupt] -----------------------------> graceful teardown
│
├── shutdown_workers() ---------------------------------> stop processes
│
├── writer.stop() --------------------------------------> stop writer thread
│
├── monitor.stop() -------------------------------------> stop monitor thread
│
├── collect_final_stats() ------------------------------> checkpoint.get_resume_summary()
│
└── display_and_export_results() ----------------------> display + txt export

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

import logging
import math
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION - Edit these values to run without arguments
# ═══════════════════════════════════════════════════════════════════

COLLECTION_NAME = "haup_vectors"
CHROMA_PATH     = "./chroma_db"
CHECKPOINT_DB   = "haup_checkpoint.db"   # Created by forward pipeline (main.py)
REVERSE_JOB_DB  = "reverse_job.db"       # Tracks reverse pipeline chunk progress

# Neon/PostgreSQL connection (source and target are the same server)
DEFAULT_CONNECTION_STRING = os.getenv("NEON_CONNECTION_STRING", "")
DEFAULT_SOURCE_TABLE      = "users"
DEFAULT_TARGET_TABLE      = "users_extracted"

PRESERVE_CONSTRAINTS = True

EXCEL_CONFIG = {
    "path": "./extracted.xlsx",
}

# ═══════════════════════════════════════════════════════════════════

try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.table   import Table
    from rich.markup  import escape
    console = Console()
except ImportError:
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    console  = Console()
    Panel    = lambda text, **kwargs: text
    Table    = None
    escape   = str

logging.basicConfig(
    level   = logging.ERROR,
    format  = "%(message)s",
    datefmt = "%H:%M:%S",
)

for _n in [
    "httpx", "httpcore", "huggingface_hub", "chromadb",
    "chromadb.telemetry.product.posthog",
    "reverse_core.vect_batch_reader", "reverse_core.schema_loader",
    "reverse_core.constraint_reader", "reverse_core.schema_reconciler",
    "reverse_core.reverse_writer",    "reverse_core.reverse_worker_pool",
]:
    logging.getLogger(_n).setLevel(logging.ERROR)

logger = logging.getLogger("haup.reverse_main")


# ═══════════════════════════════════════════════════════════════════
# WORKER CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WorkerConfig:
    num_workers: int
    chunk_size:  int  = 1000
    cpu_cores:   int  = 1
    total_ram_gb: float = 0.0
    device:      str  = "cpu"


# ═══════════════════════════════════════════════════════════════════
# REVERSE CHECKPOINT
# Tracks per-chunk status so the run is fully resumable after a crash.
# Stored in a dedicated SQLite file separate from the forward pipeline DB.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResumeSummary:
    done:    int
    failed:  int
    running: int
    total:   int


class ReverseCheckpoint:
    """Crash-safe chunk progress tracker backed by a local SQLite WAL file."""

    def __init__(self, db_path: str, total_chunks: int = 0) -> None:
        self._path         = db_path
        self._total_chunks = total_chunks
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_progress (
                chunk_id   INTEGER PRIMARY KEY,
                status     TEXT    NOT NULL,
                rows_done  INTEGER DEFAULT 0,
                updated_at TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

    def is_done(self, chunk_id: int) -> bool:
        conn = self._connect()
        row  = conn.execute(
            "SELECT 1 FROM chunk_progress WHERE chunk_id = ? AND status = 'done'",
            (chunk_id,),
        ).fetchone()
        conn.close()
        return row is not None

    def mark_running(self, chunk_id: int) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO chunk_progress (chunk_id, status)
            VALUES (?, 'running')
            ON CONFLICT(chunk_id) DO UPDATE
              SET status     = 'running',
                  updated_at = datetime('now')
        """, (chunk_id,))
        conn.commit()
        conn.close()

    def mark_done(self, chunk_id: int, rows_done: int = 0) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO chunk_progress (chunk_id, status, rows_done)
            VALUES (?, 'done', ?)
            ON CONFLICT(chunk_id) DO UPDATE
              SET status     = 'done',
                  rows_done  = excluded.rows_done,
                  updated_at = datetime('now')
        """, (chunk_id, rows_done))
        conn.commit()
        conn.close()

    def mark_failed(self, chunk_id: int) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO chunk_progress (chunk_id, status)
            VALUES (?, 'failed')
            ON CONFLICT(chunk_id) DO UPDATE
              SET status     = 'failed',
                  updated_at = datetime('now')
        """, (chunk_id,))
        conn.commit()
        conn.close()

    def reset_running_to_pending(self) -> int:
        """Convert any 'running' rows left from a prior crash back to 'pending' (deleted)."""
        conn = self._connect()
        cur  = conn.execute(
            "DELETE FROM chunk_progress WHERE status = 'running'"
        )
        recovered = cur.rowcount
        conn.commit()
        conn.close()
        return recovered

    def retry_failed_chunks(self) -> int:
        """Reset all 'failed' chunks so they will be retried on the next run."""
        conn = self._connect()
        cur  = conn.execute(
            "DELETE FROM chunk_progress WHERE status = 'failed'"
        )
        retried = cur.rowcount
        conn.commit()
        conn.close()
        return retried

    def get_resume_summary(self) -> ResumeSummary:
        conn    = self._connect()
        done    = conn.execute("SELECT COUNT(*) FROM chunk_progress WHERE status = 'done'").fetchone()[0]
        failed  = conn.execute("SELECT COUNT(*) FROM chunk_progress WHERE status = 'failed'").fetchone()[0]
        running = conn.execute("SELECT COUNT(*) FROM chunk_progress WHERE status = 'running'").fetchone()[0]
        conn.close()
        return ResumeSummary(done=done, failed=failed, running=running, total=self._total_chunks)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

"""================= Startup function progress_update ================="""
def progress_update(step: str, substep: str, status: str = "⏳", details: str = "") -> None:
    from datetime import datetime
    timestamp    = datetime.now().strftime("%H:%M:%S")
    status_color = (
        "green"  if status == "✅" else
        "red"    if status == "❌" else
        "yellow" if status == "⚠️" else
        "cyan"
    )
    console.print(
        f"[dim]{timestamp}[/] [{status_color}]{status}[/] "
        f"[bold]{step}[/] → [cyan]{substep}[/] {escape(details)}"
    )
"""================= End function progress_update ================="""


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

"""================= Startup function run ================="""
def run(args: argparse.Namespace) -> int:
    progress_update("STARTUP", "System Initialization", "⏳", "Starting HAUP v3.0 Reverse Pipeline...")

    console.print(Panel(
        "[bold white]HAUP v3.0[/]  ─  [cyan]Vector DB → PostgreSQL/Excel Reverse Extractor[/]\n"
        "[dim]Neon PostgreSQL Edition  │  Constraint Preservation  │  Cost: $0.00[/]\n"
        "[dim]Resumable exports with checkpoint tracking.[/]",
        border_style = "bright_blue",
        padding      = (0, 1),
    ))

    # ── Step 1: ChromaDB ─────────────────────────────────────────
    progress_update("STEP 1", "Dependencies", "⏳", "Loading ChromaDB...")
    try:
        import chromadb
        progress_update("STEP 1", "Dependencies", "✅", "ChromaDB loaded")
    except ImportError:
        progress_update("STEP 1", "Dependencies", "❌", "ChromaDB not installed")
        console.print("[red]ERROR:[/] pip install chromadb")
        return 1

    # ── Step 2: Core modules ─────────────────────────────────────
    progress_update("STEP 2", "Core Modules", "⏳", "Importing reverse pipeline modules...")
    from reverse_core import vect_batch_reader, schema_loader, schema_reconciler
    from reverse_core.reverse_writer      import ReverseWriter, make_sql_target, make_excel_target
    from reverse_core.monitor             import ReverseMonitor
    from reverse_core.reverse_worker_pool import spawn_workers, shutdown_workers
    from reverse_core.constraint_reader   import read_postgresql_constraints
    progress_update("STEP 2", "Core Modules", "✅", "All modules imported")

    # ── Step 3: Hardware detection ───────────────────────────────
    progress_update("STEP 3", "Hardware Detection", "⏳", "Detecting system capabilities...")
    try:
        from reverse_core import hardware_detector
        config = hardware_detector.detect()
    except Exception:
        import multiprocessing as _mp
        config = WorkerConfig(
            num_workers  = max(1, _mp.cpu_count() - 1),
            chunk_size   = 1000,
            cpu_cores    = _mp.cpu_count(),
            total_ram_gb = 0.0,
            device       = "cpu",
        )
    progress_update("STEP 3", "Hardware Detection", "✅",
                    f"{config.num_workers} workers, chunk_size={config.chunk_size}")

    # ── Step 4: ChromaDB connection ──────────────────────────────
    progress_update("STEP 4", "ChromaDB Connection", "⏳", "Connecting to vector database...")
    console.print("\n[bold bright_blue]Data Source[/]  " + "─" * 45)
    console.print(f"  [bold]Collection[/]              [cyan]{args.collection}[/]")
    console.print(f"  [bold]ChromaDB Path[/]           [cyan]{args.chroma_path}[/]")

    client = chromadb.PersistentClient(path=args.chroma_path)
    progress_update("STEP 4", "ChromaDB Connection", "✅", "Connected")

    # ── Step 5: Collection stats ─────────────────────────────────
    progress_update("STEP 5", "Collection Stats", "⏳", "Analysing collection...")
    stats = vect_batch_reader.get_collection_stats(client, args.collection)
    if stats.total_entries == 0:
        progress_update("STEP 5", "Collection Stats", "❌", "Collection is empty or missing")
        console.print(f"[red]ERROR:[/] Collection '{args.collection}' is empty or does not exist.")
        return 1
    progress_update("STEP 5", "Collection Stats", "✅", f"{stats.total_entries:,} entries found")

    if not stats.has_documents:
        progress_update("STEP 5", "Document Check", "❌", "No documents stored in collection")
        console.print(
            "[red]ERROR:[/] Collection has no 'documents' field.\n"
            "  Re-run the forward pipeline with documents stored in vector_writer.py."
        )
        return 1

    console.print("\n[bold bright_blue]Hardware[/]  " + "─" * 48)
    console.print(f"  [bold]CPU cores[/]               [cyan]{config.cpu_cores}[/]")
    console.print(f"  [bold]Total RAM[/]               [cyan]{config.total_ram_gb:.2f} GB[/]")
    console.print(f"  [bold]Device[/]                  [cyan]{config.device}[/]")
    console.print(f"  [bold]Workers[/]                 [cyan]{config.num_workers}[/]")

    console.print("\n[bold bright_blue]Data Stats[/]  " + "─" * 47)
    console.print(f"  [bold]Total rows[/]              [cyan]{stats.total_entries:,}[/]")
    console.print(f"  [bold]Has documents[/]           [cyan]{stats.has_documents}[/]")

    # ── Step 6: Schema loading ───────────────────────────────────
    progress_update("STEP 6", "Schema Loading", "⏳", "Loading saved schema strategy...")
    strategy = schema_loader.load(args.checkpoint)
    console.print(
        f"  [bold]Columns[/]                 [cyan]{len(strategy.all_cols)}[/]  →  "
        f"{', '.join(strategy.all_cols[:5])}{', ...' if len(strategy.all_cols) > 5 else ''}"
    )
    progress_update("STEP 6", "Schema Loading", "✅", "Schema strategy loaded")

    # ── Step 7: Type inference ───────────────────────────────────
    progress_update("STEP 7", "Type Inference", "⏳", "Analysing sample data for SQL types...")
    source_conn = None
    if args.pg_connection_string and args.source_table:
        try:
            import psycopg2
            source_conn = psycopg2.connect(args.pg_connection_string)
            progress_update("STEP 7", "Type Inference", "⏳",
                            f"Using source table '{args.source_table}' on Neon")
        except Exception as exc:
            progress_update("STEP 7", "Type Inference", "⚠️",
                            f"Source connect failed ({str(exc)[:80]}) — using document inference")
            source_conn = None

    sample_chunk = next(
        vect_batch_reader.stream_chunks(client, args.collection, chunk_size=200),
        None,
    )
    sample_docs = sample_chunk.docs if sample_chunk else []
    type_map    = schema_loader.infer_sql_types(
        strategy,
        sample_docs,
        source_conn  = source_conn,
        source_table = args.source_table,
    )
    progress_update("STEP 7", "Type Inference", "✅", "Column types inferred")

    # ── Step 8: Constraints ──────────────────────────────────────
    constraints = None
    if args.preserve_constraints and source_conn and args.source_table:
        progress_update("STEP 8", "Constraints", "⏳", "Loading original table constraints...")
        console.print("\n[bold bright_blue]Constraint Preservation[/]  " + "─" * 32)
        console.print(f"  [bold]Source table[/]            [cyan]{args.source_table}[/]")
        try:
            constraints = read_postgresql_constraints(
                conn       = source_conn,
                table_name = args.source_table,
            )
            progress_update("STEP 8", "Constraints", "✅",
                            f"Loaded {len(constraints.columns)} column constraints")
            console.print(f"  [bold]Constraints loaded[/]      [green]✓[/] {len(constraints.columns)} columns")
        except Exception as exc:
            progress_update("STEP 8", "Constraints", "⚠️", f"Failed: {str(exc)[:80]}")
            console.print(f"  [yellow]WARNING:[/] Could not read constraints: {escape(str(exc))}")
    else:
        progress_update("STEP 8", "Constraints", "⏳", "Skipped (--preserve-constraints not set or no source)")

    if source_conn:
        try:
            source_conn.close()
        except Exception:
            pass

    # ── Step 9: Schema reconciliation ───────────────────────────
    progress_update("STEP 9", "Schema Reconciliation", "⏳", "Merging strategy, types, and constraints...")
    schema             = schema_reconciler.reconcile(strategy, type_map)
    schema.constraints = constraints
    progress_update("STEP 9", "Schema Reconciliation", "✅", "Schema reconciled")

    # ── Step 10: Checkpoint ──────────────────────────────────────
    if stats.total_entries <= config.chunk_size:
        effective_chunk    = max(1, math.ceil(stats.total_entries / config.num_workers))
        config.chunk_size  = effective_chunk

    total_chunks = vect_batch_reader.total_chunks(stats, config.chunk_size)

    job_db_path = args.job_db or str(Path(args.checkpoint).parent / REVERSE_JOB_DB)
    checkpoint  = ReverseCheckpoint(job_db_path, total_chunks)

    recovered = checkpoint.reset_running_to_pending()
    if recovered:
        console.print(f"\n[yellow]Crash recovery:[/] reset {recovered} 'running' chunks to pending")

    summary = checkpoint.get_resume_summary()
    if summary.failed > 0:
        progress_update("STEP 10", "Failed Chunk Retry", "⏳",
                        f"Retrying {summary.failed} failed chunks...")
        retried = checkpoint.retry_failed_chunks()
        if retried:
            summary = checkpoint.get_resume_summary()
            progress_update("STEP 10", "Failed Chunk Retry", "✅",
                            f"Reset {retried} failed chunks for retry")

    if summary.done >= total_chunks > 0:
        console.print(f"\n[green]✓[/] All {total_chunks} chunks already done. Nothing to do.")
        return 0

    console.print("\n[bold bright_blue]Resume Check[/]  " + "─" * 44)
    console.print(f"  [bold]Chunks done[/]             [cyan]{summary.done}[/] / [cyan]{total_chunks}[/]")
    console.print(f"  [bold]Chunks failed[/]           [cyan]{summary.failed}[/]")
    console.print(f"  [bold]Chunk size[/]              [cyan]{config.chunk_size:,} rows[/]")
    console.print(f"  [bold]Active workers[/]          [cyan]{config.num_workers}[/]")

    # ── Step 11: Build output target ─────────────────────────────
    console.print("\n[bold bright_blue]Output Target[/]  " + "─" * 43)
    if args.output_excel:
        target = make_excel_target(args.output_excel, total_rows=stats.total_entries)
        console.print(f"  [bold]Format[/]                  [cyan]Excel[/]")
        console.print(f"  [bold]Path[/]                    [cyan]{args.output_excel}[/]")
    else:
        table_name = args.output_table or f"{args.collection}_extracted"
        target     = make_sql_target(
            table_name = table_name,
            pg_config  = {"dsn": args.pg_connection_string},
        )
        console.print(f"  [bold]Format[/]                  [cyan]PostgreSQL / Neon[/]")
        console.print(f"  [bold]Table[/]                   [cyan]{table_name}[/]")

    # ── Step 12: Wire queues, writer, monitor, workers ───────────
    work_q   = mp.Queue(maxsize=20)
    result_q = mp.Queue()
    stats_q  = mp.Queue()

    writer = ReverseWriter(result_q, checkpoint, schema, target)
    writer.start()

    monitor = ReverseMonitor(
        stats_q         = stats_q,
        checkpoint      = checkpoint,
        writer          = writer,
        total_chunks    = total_chunks,
        collection_name = args.collection,
    )
    monitor.start()

    active_workers     = min(config.num_workers, total_chunks)
    config.num_workers = active_workers

    console.print("\n[bold bright_blue]Pipeline[/]  " + "─" * 48)
    console.print(f"  [bold]Workers[/]                 Spawning {active_workers} processes")

    processes  = spawn_workers(config, strategy, work_q, result_q, stats_q)
    start_time = time.time()

    # ── Step 13: Stream chunks into the work queue ───────────────
    try:
        for chunk in vect_batch_reader.stream_chunks(
            client, args.collection, config.chunk_size
        ):
            if checkpoint.is_done(chunk.chunk_id):
                continue
            checkpoint.mark_running(chunk.chunk_id)
            work_q.put(chunk)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/] — shutting down cleanly...")

    finally:
        shutdown_workers(processes, work_q, active_workers)
        writer.stop()
        monitor.stop()

    # ── Step 14: Final summary ───────────────────────────────────
    final   = checkpoint.get_resume_summary()
    elapsed = int(time.time() - start_time)
    total_processed = final.done + final.failed
    success_rate    = (final.done / total_processed * 100) if total_processed > 0 else 0

    console.print("\n" + "=" * 60)
    console.print("HAUP v3.0  Extraction Complete")
    console.print(f"  Collection   : {args.collection}")
    console.print(f"  Rows written : {writer.total_rows_written:,}")
    console.print(f"  Parse fails  : {writer.total_parse_fails:,}")
    console.print(f"  Elapsed      : {elapsed}s")
    console.print(f"  Cost         : $0.00")
    console.print("=" * 60 + "\n")

    status_color = "green" if final.failed == 0 else "yellow"
    status_icon  = "✅"    if final.failed == 0 else "⚠️"

    console.print(Panel(
        f"[bold {status_color}]{status_icon}  Pipeline Complete[/bold {status_color}]\n\n"
        f"  [bold]Chunks done   :[/bold]  [cyan]{final.done}[/cyan]\n"
        f"  [bold]Chunks failed :[/bold]  "
        f"[{'red' if final.failed > 0 else 'dim'}]{final.failed}"
        f"[/{'red' if final.failed > 0 else 'dim'}]\n"
        f"  [bold]Rows extracted:[/bold]  [cyan]{writer.total_rows_written:,}[/cyan]\n"
        f"  [bold]Success rate  :[/bold]  [cyan]{success_rate:.1f}%[/cyan]\n"
        f"  [bold]Cost          :[/bold]  [bold green]$0.00[/bold green]\n",
        title        = "[bold white]HAUP v3.0[/bold white]",
        border_style = status_color,
    ))

    if final.failed:
        console.print(Panel(
            f"[yellow]⚠️  {final.failed} chunks failed during processing.[/yellow]\n\n"
            "[dim]• Re-run this command to retry failed chunks\n"
            "• Check logs above for specific error details[/dim]",
            title        = "[bold yellow]Retry Recommended[/bold yellow]",
            border_style = "yellow",
        ))

    if writer.total_rows_written > 0 and not args.output_excel:
        _display_and_export_results(args.pg_connection_string, target.table_name)

    return 0
"""================= End function run ================="""


"""================= Startup function _display_and_export_results ================="""
def _display_and_export_results(pg_connection_string: str, table_name: str) -> None:
    """
    Fetch a 10-row sample from the extracted PostgreSQL table, render it in the
    terminal, then write a full plain-text export file alongside the script.
    """
    try:
        import psycopg2
        import psycopg2.extras

        conn   = psycopg2.connect(pg_connection_string)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        total_count = cursor.fetchone()[0]

        if total_count == 0:
            console.print("\n[dim]No rows to display.[/dim]")
            return

        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 1')
        columns = [desc[0] for desc in cursor.description]

        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 10')
        sample_rows = [dict(r) for r in cursor.fetchall()]

        # ── Terminal table ────────────────────────────────────────
        if Table is not None:
            tbl = Table(
                title        = f"Sample Output (10 of {total_count:,} rows)",
                show_header  = True,
                header_style = "bold magenta",
                border_style = "bright_blue",
            )
            for col in columns:
                if col in ("__extract_id", "__orig_rowid__"):
                    tbl.add_column(col, style="dim",   width=12)
                elif "id"    in col.lower(): tbl.add_column(col, style="cyan",   width=12)
                elif "email" in col.lower(): tbl.add_column(col, style="blue",   width=25)
                elif "name"  in col.lower(): tbl.add_column(col, style="green",  width=20)
                elif "phone" in col.lower(): tbl.add_column(col, style="yellow", width=15)
                else:                        tbl.add_column(col, style="white")

            for row in sample_rows:
                tbl.add_row(*[
                    str(row[col]) if row[col] is not None else "[dim]NULL[/dim]"
                    for col in columns
                ])
            console.print("\n")
            console.print(tbl)
            console.print("\n")

        # ── Full txt export ───────────────────────────────────────
        cursor.execute(f'SELECT * FROM "{table_name}"')
        all_rows = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        conn.close()

        output_file = Path(".") / f"{table_name}_export.txt"
        col_widths  = {col: len(col) for col in columns}
        for row in all_rows:
            for col in columns:
                col_widths[col] = max(col_widths[col], len(str(row[col]) if row[col] is not None else "NULL"))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("  HAUP v3.0 — Extracted Data Export\n")
            f.write(f"  Table      : {table_name}\n")
            f.write(f"  Total Rows : {total_count:,}\n")
            f.write(f"  Columns    : {len(columns)}\n")
            f.write("=" * 100 + "\n\n")

            header = " | ".join(col.ljust(col_widths[col]) for col in columns)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for row in all_rows:
                f.write(" | ".join(
                    str(row[col]).ljust(col_widths[col]) if row[col] is not None
                    else "NULL".ljust(col_widths[col])
                    for col in columns
                ) + "\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write(f"End of export — {total_count:,} rows total\n")

        console.print(Panel(
            f"[green]✓[/green] Export saved : [cyan bold]{output_file}[/cyan bold]\n"
            f"[green]✓[/green] Rows exported: [bold white]{total_count:,}[/bold white]\n"
            f"[green]✓[/green] Columns      : [bold white]{len(columns)}[/bold white]\n"
            f"[green]✓[/green] File size    : [dim]{output_file.stat().st_size / 1024:.1f} KB[/dim]",
            title        = "[bold white]PostgreSQL Export Complete[/bold white]",
            border_style = "green",
            padding      = (1, 2),
        ))

    except Exception as exc:
        console.print(f"\n[yellow]Warning:[/yellow] Could not display/export results: {escape(str(exc))}\n")
"""================= End function _display_and_export_results ================="""


"""================= Startup function build_parser ================="""
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description     = "HAUP v3.0 — Vector DB to PostgreSQL/Excel reverse extraction\n"
                          "Run without arguments to use defaults from top of file",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # ChromaDB
    p.add_argument("--collection",  default=COLLECTION_NAME,
                   help=f"ChromaDB collection name (default: {COLLECTION_NAME})")
    p.add_argument("--chroma-path", default=CHROMA_PATH,
                   help=f"Path to ChromaDB folder (default: {CHROMA_PATH})")
    p.add_argument("--checkpoint",  default=CHECKPOINT_DB,
                   help=f"Path to forward-pipeline checkpoint DB (default: {CHECKPOINT_DB})")

    # Output
    out = p.add_mutually_exclusive_group(required=False)
    out.add_argument("--output-excel", metavar="PATH",
                     help="Write output to Excel file (e.g. extracted.xlsx)")
    # Default output is PostgreSQL (no flag needed)

    p.add_argument("--output-table", default=DEFAULT_TARGET_TABLE,
                   help=f"Output table name (default: {DEFAULT_TARGET_TABLE})")

    # PostgreSQL / Neon
    p.add_argument("--pg-connection-string", default=DEFAULT_CONNECTION_STRING,
                   help="PostgreSQL/Neon DSN (uses NEON_CONNECTION_STRING env if not set)")
    p.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE,
                   help=f"Source table for type/constraint inference (default: {DEFAULT_SOURCE_TABLE})")

    # Constraint preservation
    p.add_argument("--preserve-constraints", action="store_true", default=PRESERVE_CONSTRAINTS,
                   help="Preserve UNIQUE/NOT NULL constraints from original table")

    # Misc
    p.add_argument("--job-db",    default="",
                   help="Path to reverse-pipeline progress SQLite file")
    p.add_argument("--log-level", default="ERROR",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p
"""================= End function build_parser ================="""


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if not args.output_excel and not args.pg_connection_string:
        parser.error(
            "PostgreSQL connection string required.\n"
            "Set NEON_CONNECTION_STRING env var or use --pg-connection-string."
        )

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    sys.exit(run(args))