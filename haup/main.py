"""
File Summary:
Main entry point of HAUP v2.0 with REAL-TIME PROGRESS TRACKING.
Shows live updates as execution flows through each component: A → B → C → D

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

main()
||
├── [STEP 1] get_data_source()  [Function] -----------> Configure Neon (PostgreSQL) connection
│       │
│       ├── [1.1] PostgreSQL Driver Check -----------> Verify psycopg2 is available
│       ├── [1.2] Connection Test -------------------> Connect to Neon
│       ├── [1.3] SqlSource Creation ----------------> Create data source object
│       └── [1.4] Table Validation ------------------> Test table access
│
├── [STEP 2] HardwareDetector()  [Class → Object] ----> Detect system hardware
│       │
│       ├── [2.1] CPU Detection ---------------------> Physical/logical cores
│       ├── [2.2] RAM Detection ---------------------> Total system memory
│       ├── [2.3] GPU Detection ---------------------> CUDA availability & VRAM
│       └── [2.4] Config Calculation ----------------> Workers, chunk size, batch size
│
├── [STEP 3] Data Statistics ---------------------> Analyze source data
│       │
│       ├── [3.1] Stream Reader Creation ------------> SQL reader initialization
│       ├── [3.2] File Stats Collection -------------> Row count, column analysis
│       └── [3.3] Chunk Calculation ----------------> Optimal chunk sizing
│
├── [STEP 4] Checkpoint System -------------------> Resume capability setup
│       │
│       ├── [4.1] SQLite Checkpoint Init -----------> Progress tracking database
│       ├── [4.2] Resume Summary Check --------------> Previous run analysis
│       ├── [4.3] Row Tracking Migration ------------> Legacy checkpoint conversion
│       └── [4.4] Early Exit Check -----------------> Skip if already complete
│
├── [STEP 5] Schema Analysis ---------------------> Column classification
│       │
│       ├── [5.1] Sample Data Extraction -----------> First chunk analysis
│       ├── [5.2] Column Categorization ------------> Semantic/numeric/date/skip
│       └── [5.3] Template Generation ---------------> Text serialization format
│
├── [STEP 6] Pipeline Initialization -------------> Core components setup
│       │
│       ├── [6.1] Queue Creation -------------------> Work/result/stats queues
│       ├── [6.2] Worker Pool Spawning -------------> Multiprocess workers
│       ├── [6.3] Vector Database Init --------------> ChromaDB connection
│       ├── [6.4] Vector Writer Start ---------------> Background storage thread
│       └── [6.5] Monitor Start --------------------> Progress tracking thread
│
├── [STEP 7] Data Processing Loop ----------------> Main execution
│       │
│       ├── [7.1] Orchestrator Start ---------------> Pipeline controller
│       ├── [7.2] Chunk Streaming ------------------> Data source reading
│       ├── [7.3] Worker Processing ----------------> Embedding generation
│       ├── [7.4] Vector Storage -------------------> ChromaDB writes
│       └── [7.5] Progress Updates -----------------> Real-time monitoring
│
└── [STEP 8] Cleanup & Results -------------------> Final statistics
        │
        ├── [8.1] Worker Shutdown ------------------> Graceful process termination
        ├── [8.2] Thread Cleanup -------------------> Stop background threads
        ├── [8.3] Stats Collection -----------------> Final performance metrics
        └── [8.4] Results Display ------------------> Summary and completion

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

import multiprocessing
import logging
import sys
import math
import os
import time
import json
import threading
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path


# Fix Windows console encoding for Rich library
if os.name == 'nt':  # Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# FIX: Removed duplicate logging.basicConfig (DEBUG-level call was immediately
# overridden by the ERROR-level call below it, making it dead code).
logging.basicConfig(level=logging.ERROR, format="%(message)s")
for _n in ["httpx", "httpcore", "huggingface_hub", "huggingface_hub.utils._http",
           "sentence_transformers", "sentence_transformers.SentenceTransformer",
           "transformers", "filelock", "urllib3", "requests", "hf_transfer",
           "torch", "PIL", "tqdm"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

# Enable debug logging for vector writer and ChromaDB
logging.getLogger("haup.vector_writer").setLevel(logging.DEBUG)
logging.getLogger("chromadb").setLevel(logging.DEBUG)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",  "error")
os.environ.setdefault("HF_HUB_VERBOSITY",        "error")

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box as rbox
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text

console = Console()

from forward_core.hardware_detector       import HardwareDetector
from forward_core.stream_reader           import SQLStreamReader
from forward_core.schema_analyzer         import SchemaAnalyzer
from forward_core.worker_pool_manager     import WorkerPoolManager
from forward_core.vector_writer           import VectorWriter
from forward_core.checkpoint_queue_bridge import SQLiteCheckpoint
from forward_core.monitor                 import Monitor
# FIX: Removed unused ExcelSource and ExcelStreamReader imports — pipeline
# is Neon (PostgreSQL) only; neither class is referenced anywhere in main().
from forward_core.orchestrator import SqlSource, Orchestrator


"""================= Startup function progress_update ================="""
def progress_update(step: str, substep: str, status: str = "⏳", details: str = ""):
    """Real-time progress update with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_color = {
        "⏳": "yellow",
        "✅": "green",
        "❌": "red",
        "🔄": "blue",
        "⚡": "cyan"
    }.get(status, "white")

    console.print(f"[dim]{timestamp}[/] [{status_color}]{status}[/] [bold]{step}[/] → [cyan]{substep}[/] {details}")
"""================= End function progress_update ================="""


"""================= Startup function get_data_source ================="""
def get_data_source():
    """
    Configure Neon (PostgreSQL) database connection.
    Requires NEON_CONNECTION_STRING or individual PG_* environment variables.
    """
    progress_update("STEP 1", "Database Connection", "⏳", "Initializing Neon (PostgreSQL) connection...")

    try:
        conn_string = os.getenv("NEON_CONNECTION_STRING")

        if not conn_string:
            # Build a connection string from individual PG_* vars
            host     = os.getenv("PG_HOST", "localhost")
            port     = os.getenv("PG_PORT", "5432")
            user     = os.getenv("PG_USER", "postgres")
            password = os.getenv("PG_PASSWORD", "")
            database = os.getenv("PG_DATABASE", "Vector")
            conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            progress_update("STEP 1.1", "Connection String", "⏳", "Built from PG_* environment variables...")
        else:
            progress_update("STEP 1.1", "Connection String", "✅", "Loaded from NEON_CONNECTION_STRING")

        progress_update("STEP 1.2", "Connection Test", "⏳", "Verifying connection to Neon...")
        import psycopg2
        conn = psycopg2.connect(conn_string)
        progress_update("STEP 1.2", "Connection Test", "✅", "Connected to Neon (PostgreSQL) successfully")

        progress_update("STEP 1.3", "Source Creation", "⏳", "Creating SqlSource object...")
        source = SqlSource(
            connection = conn,
            table             = os.getenv("PG_TABLE", "users"),
            primary_key       = "id",
            name              = "Srcom-soft",
        )
        progress_update("STEP 1.3", "Source Creation", "✅", f"SqlSource created for table '{source.table}'")

        progress_update("STEP 1.4", "Table Validation", "⏳", "Verifying table access...")
        verify_conn = psycopg2.connect(conn_string)
        cursor = verify_conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{source.table}" LIMIT 1')
        cursor.fetchone()
        cursor.close()
        verify_conn.close()
        progress_update("STEP 1.4", "Table Validation", "✅", "Table access verified")

        return source

    except ImportError:
        progress_update("STEP 1.1", "PostgreSQL Driver", "❌", "psycopg2 not installed")
        console.print("[red]ERROR: psycopg2 is required. Install with: pip install psycopg2-binary[/]")
        raise
    except Exception as e:
        progress_update("STEP 1", "Database Connection", "❌", f"Failed: {str(e)}")
        raise
"""================= End function get_data_source ================="""


"""================= Startup function init_vector_db ================="""
def init_vector_db(collection_name: str = "haup_vectors"):
    progress_update("STEP 6.3", "Vector Database", "⏳", "Initializing ChromaDB...")

    try:
        import chromadb
        progress_update("STEP 6.3.1", "ChromaDB Import", "✅", "ChromaDB library loaded")

        progress_update("STEP 6.3.2", "Client Creation", "⏳", "Creating persistent client...")
        client = chromadb.PersistentClient(path="./chroma_db")
        progress_update("STEP 6.3.2", "Client Creation", "✅", "Persistent client created")

        progress_update("STEP 6.3.3", "Collection Setup", "⏳", f"Getting/creating collection '{collection_name}'...")
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 16,
                "hnsw:num_threads": 4,
                "hnsw:sync_threshold": 2000
            }
        )
        progress_update("STEP 6.3.3", "Collection Setup", "✅", f"Collection '{collection_name}' ready (optimized HNSW)")

        return collection

    except ImportError:
        progress_update("STEP 6.3", "Vector Database", "❌", "ChromaDB not available, using stub")
        return _StubVectorDB()
"""================= End function init_vector_db ================="""

"""================= Startup class _StubVectorDB ================="""
class _StubVectorDB:

    """================= Startup function upsert ================="""
    def upsert(self, ids, embeddings, metadatas):
        pass
    """================= End function upsert ================="""

    """================= Startup function query ================="""
    def query(self, query_embeddings, n_results, include):
        return {'ids': [[]], 'metadatas': [[]], 'distances': [[]]}
    """================= End function query ================="""

    """================= Startup function delete ================="""
    def delete(self, ids):
        pass
    """================= End function delete ================="""

"""================= End class _StubVectorDB ================="""


"""================= Startup function _kv ================="""
def _kv(label: str, value: str, val_style: str = "cyan") -> None:
    console.print(f"  [bold]{label:<24}[/][{val_style}]{value}[/]")
"""================= End function _kv ================="""


"""================= Startup function _section ================="""
def _section(title: str) -> None:
    console.print(f"\n[bold bright_blue]{title}[/]  "
                  f"[dim]{'─' * max(0, 46 - len(title))}[/]")
"""================= End function _section ================="""


"""================= Startup function _hw_table ================="""
def _hw_table(cfg) -> None:
    tbl = Table(box=rbox.SIMPLE, show_header=False, expand=False, padding=(0, 1))
    tbl.add_column("k", style="bold dim", width=22)
    tbl.add_column("v", style="cyan")
    tbl.add_row("Physical CPU cores",  str(cfg.cpu_physical))
    tbl.add_row("Logical cores (HT)",  str(cfg.cpu_logical))
    tbl.add_row("Total RAM",           f"{cfg.total_ram_gb:.2f} GB")
    tbl.add_row("GPU",
                f"CUDA  {cfg.gpu_vram_gb:.1f} GB VRAM"
                if cfg.gpu_available else "Not available (CPU mode)")
    tbl.add_row("", "")
    tbl.add_row("Workers spawned",    str(cfg.num_workers))
    tbl.add_row("Device",             cfg.device)
    tbl.add_row("Initial batch size", str(cfg.initial_batch))
    console.print(tbl)
"""================= End function _hw_table ================="""


"""================= Startup function _schema_table ================="""
def _schema_table(strategy) -> None:
    tbl = Table(box=rbox.SIMPLE, show_header=True,
                header_style="bold magenta", expand=False, padding=(0, 1))
    tbl.add_column("Category",  style="bold",  width=14)
    tbl.add_column("Columns",   style="cyan")
    tbl.add_column("Action",    style="dim",   width=24)
    tbl.add_row("RowID",      strategy.rowid_col,                               "primary link-back key")
    tbl.add_row("Semantic",   ", ".join(strategy.semantic_cols) or "—",         "embedded as text")
    tbl.add_row("Numeric",    ", ".join(strategy.numeric_cols)  or "—",         "embedded with label")
    tbl.add_row("Date/Time",  ", ".join(strategy.date_cols)     or "—",         "stored as metadata")
    tbl.add_row("ID/Meta",    ", ".join(strategy.id_cols)       or "—",         "stored as metadata")
    tbl.add_row("Skipped",    ", ".join(strategy.skip_cols)     or "—",         "excluded entirely")
    console.print(tbl)
    console.print(f"  [dim]Template :[/]  [yellow]{strategy.template}[/]")
"""================= End function _schema_table ================="""


"""================= Startup function _worker_stats_table ================="""
def _worker_stats_table(worker_stats: list, title: str = "Worker Stats") -> None:
    if not worker_stats:
        console.print("  [dim]No worker stats saved yet.[/]")
        return

    _section(title)

    n_workers  = len(worker_stats)
    total_rows = sum(ws["rows_processed"] for ws in worker_stats)

    console.print(
        f"  [bold white]{n_workers} workers[/] processed "
        f"[bold cyan]{total_rows:,} rows[/] in total"
    )

    tbl = Table(box=rbox.ROUNDED, show_header=True,
                header_style="bold dim", expand=False, padding=(0, 2))
    tbl.add_column("#",             style="bold cyan",   width=4,  justify="center")
    tbl.add_column("Rows processed",style="white",       width=16, justify="right")
    tbl.add_column("Share %",       style="yellow",      width=10, justify="right")
    tbl.add_column("Rows bar",      width=22)
    tbl.add_column("Final batch",   style="green",       width=13, justify="right")

    for ws in sorted(worker_stats, key=lambda x: x["worker_id"]):
        wid   = ws["worker_id"]
        rows  = ws["rows_processed"]
        batch = ws["final_batch"]
        share = (rows / total_rows * 100) if total_rows else 0

        filled = max(1, int(share / 100 * 20))
        bar    = f"[cyan]{'█' * filled}[/][dim]{'░' * (20 - filled)}[/]"

        tbl.add_row(
            str(wid),
            f"{rows:,}",
            f"{share:.1f}%",
            bar,
            str(batch),
        )

    console.print(tbl)
    console.print(
        f"  [dim]Avg rows/worker: "
        f"[bold]{total_rows // n_workers if n_workers else 0:,}[/]   │   "
        f"Batch size grows automatically as VRAM allows.[/]\n"
    )
"""================= End function _worker_stats_table ================="""


"""================= Startup function _drain_stats_q ================="""
def _drain_stats_q(stats_q) -> list:
    import queue
    latest = {}
    while True:
        try:
            stat = stats_q.get_nowait()
            latest[stat.worker_id] = stat
        except Exception:
            break
    return list(latest.values())
"""================= End function _drain_stats_q ================="""


"""================= Startup function start_cdc_if_enabled ================="""
def start_cdc_if_enabled(config_path: str, vector_db, model, strategy, source):
    """Start CDC listener if enabled in config"""
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            cdc_config = json.load(f)

        if not cdc_config.get('enabled', False):
            progress_update("CDC", "Status", "⏭️", "CDC disabled in config (cdc_config.json)")
            return None

        progress_update("CDC", "Initialization", "⏳", "CDC enabled - checking dependencies...")

        try:
            import kafka
            progress_update("CDC", "Dependencies", "✅", "kafka-python installed")
        except ImportError:
            progress_update("CDC", "Dependencies", "❌", "kafka-python not installed")
            console.print("[yellow]⚠️  CDC enabled but kafka-python not installed.[/]")
            console.print("[dim]Install with: pip install kafka-python[/]")
            return None

        from forward_core.cdc_listener import CDCListener

        progress_update("CDC", "Configuration", "⏳", f"Connecting to Kafka: {cdc_config['kafka_broker']}")

        cdc_listener = CDCListener(
            kafka_broker=cdc_config['kafka_broker'],
            topic=cdc_config['kafka_topic'],
            vector_db=vector_db,
            model=model,
            strategy=strategy,
            poll_timeout_ms=cdc_config.get('poll_timeout_ms', 500)
        )

        cdc_thread = threading.Thread(
            target=cdc_listener.start_cdc_consumer,
            name="haup-cdc-listener",
            daemon=True
        )
        cdc_thread.start()

        progress_update("CDC", "Status", "✅", f"CDC listener started on topic: {cdc_config['kafka_topic']}")
        console.print(Panel(
            f"[bold green]✅ CDC Real-Time Sync Active[/]\n\n"
            f"  [bold]Kafka Broker:[/] {cdc_config['kafka_broker']}\n"
            f"  [bold]Topic:[/] {cdc_config['kafka_topic']}\n"
            f"  [bold]Mode:[/] Real-time change capture\n\n"
            f"  [dim]Database changes will be automatically synced to ChromaDB.[/]\n"
            f"  [dim]Press Ctrl+C to stop CDC listener.[/]",
            border_style="cyan",
            expand=False
        ))

        return cdc_listener

    except Exception as e:
        progress_update("CDC", "Error", "❌", f"Failed to start CDC: {str(e)}")
        console.print(f"[red]CDC startup failed: {e}[/]")
        return None
"""================= End function start_cdc_if_enabled ================="""


"""================= Startup function main ================="""
def main():
    import sys

    # Handle reset flag
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        progress_update("RESET", "Checkpoint Cleanup", "⏳", "Removing checkpoint files...")
        files_to_remove = ['job.db', 'haup_checkpoint.db']
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                progress_update("RESET", "File Removal", "✅", f"Removed {file}")
        progress_update("RESET", "Checkpoint Cleanup", "✅", "Reset complete")
        console.print("[green]Checkpoint reset complete. Run again without --reset flag.[/]")
        return

    # Header
    console.print(Panel(
        "[bold white]HAUP v2.0  ─  Hybrid Adaptive Unified Pipeline[/]\n"
        "[dim]Neon (PostgreSQL) Edition  │  RowID Reverse Lookup  │  Cost: $0.00[/]\n"
        "[dim]Run [bold]search.py[/bold] for semantic search after ingestion.[/]",
        border_style="bright_blue", expand=False,
    ))

    console.print("\n[bold bright_blue]🚀 REAL-TIME EXECUTION FLOW[/]")
    console.print("[dim]Following the data pipeline step by step...[/]\n")

    # STEP 1: Data Source Configuration
    _section("Data Source")
    source = get_data_source()
    _kv("Mode",   "POSTGRESQL (NEON)")
    _kv("Target", source.table)

    # STEP 2: Hardware Detection
    _section("Hardware Detection")
    progress_update("STEP 2", "Hardware Analysis", "⏳", "Detecting system capabilities...")

    progress_update("STEP 2.1", "CPU Detection", "⏳", "Scanning CPU cores...")
    progress_update("STEP 2.2", "RAM Detection", "⏳", "Checking memory...")
    progress_update("STEP 2.3", "GPU Detection", "⏳", "Looking for CUDA devices...")

    config = HardwareDetector().detect()

    progress_update("STEP 2.4", "Config Calculation", "✅", f"Optimal: {config.num_workers} workers, {config.chunk_size} chunk size")
    _hw_table(config)

    # STEP 3: Data Statistics
    _section("Data Stats")
    progress_update("STEP 3", "Data Analysis", "⏳", "Analyzing source data...")

    progress_update("STEP 3.1", "Reader Creation", "⏳", "Creating stream reader...")
    stats_reader = SQLStreamReader(source.conn, source.table, source.primary_key)
    progress_update("STEP 3.1", "Reader Creation", "✅", "Stream reader initialized")

    progress_update("STEP 3.2", "Stats Collection", "⏳", "Counting rows and columns...")
    stats = stats_reader.get_file_stats()
    progress_update("STEP 3.2", "Stats Collection", "✅", f"Found {stats.total_rows:,} rows, {len(stats.columns)} columns")

    progress_update("STEP 3.3", "Chunk Calculation", "⏳", "Calculating optimal chunk size...")
    if stats.total_rows <= config.chunk_size:
        effective_chunk = max(1, math.ceil(stats.total_rows / config.num_workers))
    else:
        effective_chunk = config.chunk_size

    total_chunks  = math.ceil(stats.total_rows / effective_chunk) if effective_chunk else 1
    active_workers = min(config.num_workers, total_chunks)
    progress_update("STEP 3.3", "Chunk Calculation", "✅", f"Chunks: {total_chunks}, Active workers: {active_workers}")

    _kv("Total rows",    f"{stats.total_rows:,}")
    _kv("Chunk size",    f"{effective_chunk:,} rows  "
                          f"[dim](max configured: {config.chunk_size:,})[/]", "cyan")
    _kv("Total chunks",  f"{total_chunks:,}")
    _kv("Active workers", f"{active_workers}  [dim](capped to chunk count)[/]"
                           if active_workers < config.num_workers
                           else str(active_workers), "cyan")
    _kv("Columns",
        f"{len(stats.columns)}  →  [dim]{', '.join(stats.columns)}[/]", "white")

    # STEP 4: Checkpoint System
    _section("Resume Check")
    progress_update("STEP 4", "Checkpoint System", "⏳", "Initializing progress tracking...")

    progress_update("STEP 4.1", "SQLite Init", "⏳", "Creating checkpoint database...")
    checkpoint = SQLiteCheckpoint('job.db')
    progress_update("STEP 4.1", "SQLite Init", "✅", "Checkpoint database ready")

    progress_update("STEP 4.2", "Resume Analysis", "⏳", "Checking previous progress...")
    summary = checkpoint.get_resume_summary()

    if summary.failed > 0:
        progress_update("STEP 4.2", "Failed Chunk Retry", "⏳", f"Retrying {summary.failed} failed chunks...")
        retried_count = checkpoint.retry_failed_chunks()
        if retried_count > 0:
            summary = checkpoint.get_resume_summary()
            progress_update("STEP 4.2", "Failed Chunk Retry", "✅", f"Reset {retried_count} failed chunks for retry")

    progress_update("STEP 4.2", "Resume Analysis", "✅", f"Found {summary.done} completed, {summary.failed} failed chunks")

    progress_update("STEP 4.3", "Row Tracking", "⏳", "Checking row-level progress...")
    processed_rows = checkpoint.get_processed_row_count()
    if processed_rows == 0 and summary.done > 0:
        progress_update("STEP 4.3", "Migration", "⏳", "Migrating legacy checkpoints...")
        migrated = checkpoint.migrate_chunk_to_row_tracking(source, source.table)
        if migrated > 0:
            processed_rows = checkpoint.get_processed_row_count()
            progress_update("STEP 4.3", "Migration", "✅", f"Migrated {migrated} rows to new tracking")
            _kv("Migration", f"Migrated {migrated} rows to new tracking system", "yellow")

    rows_remaining = max(0, stats.total_rows - processed_rows)
    progress_update("STEP 4.3", "Row Tracking", "✅", f"Processed: {processed_rows:,}, Remaining: {rows_remaining:,}")

    _kv("Rows processed", f"{processed_rows:,} / {stats.total_rows:,}",
        "green" if processed_rows > 0 else "dim")
    _kv("Rows remaining", f"{rows_remaining:,}",
        "yellow" if rows_remaining > 0 else "dim")
    _kv("Chunks done",   f"{summary.done} / {total_chunks}",
        "green" if summary.done > 0 else "dim")
    _kv("Chunks failed", str(summary.failed),
        "red" if summary.failed > 0 else "dim")

    # STEP 4.4: Early Exit Check
    if rows_remaining == 0 and summary.failed == 0:
        progress_update("STEP 4.4", "Early Exit", "✅", "All rows already processed - skipping pipeline")

        saved_stats = checkpoint.get_worker_stats()
        _worker_stats_table(saved_stats, title="Worker Stats  (last run)")

        console.print()
        console.print(Panel(
            "[bold green]✅  All rows already processed.[/]\n\n"
            "[dim]• To search: run [bold]python search.py[/bold]\n"
            "• To re-embed: delete [bold]job.db[/bold] "
            "and [bold]haup_checkpoint.db[/bold][/]",
            border_style="green", expand=False,
        ))
        return

    # STEP 5: Schema Analysis
    _section("Schema Analysis")
    progress_update("STEP 5", "Schema Analysis", "⏳", "Analyzing column types...")

    progress_update("STEP 5.1", "Sample Extraction", "⏳", "Reading first chunk for analysis...")
    first_reader = SQLStreamReader(source.conn, source.table, source.primary_key)
    first_chunk  = next(first_reader.stream_chunks(config.chunk_size), None)
    if first_chunk is None:
        progress_update("STEP 5.1", "Sample Extraction", "❌", "Data source is empty")
        console.print("[bold red]ERROR:[/] Data source is empty.")
        sys.exit(1)
    progress_update("STEP 5.1", "Sample Extraction", "✅", f"Extracted {len(first_chunk.data)} sample rows")

    progress_update("STEP 5.2", "Column Classification", "⏳", "Categorizing columns...")
    strategy = SchemaAnalyzer().analyze(first_chunk.data, stats.columns)
    progress_update("STEP 5.2", "Column Classification", "✅",
                    f"Semantic: {len(strategy.semantic_cols)}, Numeric: {len(strategy.numeric_cols)}")

    progress_update("STEP 5.3", "Template Generation", "✅", "Text serialization template created")
    _schema_table(strategy)

    # STEP 6: Pipeline Initialization
    _section("Pipeline")
    progress_update("STEP 6", "Pipeline Setup", "⏳", "Initializing core components...")

    progress_update("STEP 6.1", "Queue Creation", "⏳", "Creating inter-process queues...")
    work_q   = multiprocessing.Queue(maxsize=20)
    result_q = multiprocessing.Queue()
    stats_q  = multiprocessing.Queue()
    progress_update("STEP 6.1", "Queue Creation", "✅", "Work, result, and stats queues ready")

    config.num_workers = active_workers

    _kv("Workers", f"Spawning {active_workers} processes  [dim](1 per chunk)[/]")
    _kv("Model",   "all-MiniLM-L6-v2  (loading, first run downloads ~80 MB…)")
    console.print()

    progress_update("STEP 6.2", "Worker Pool", "⏳", f"Spawning {active_workers} worker processes...")
    processes = WorkerPoolManager().spawn_workers(
        config, strategy, work_q, result_q, stats_q)

    actual_active_workers = len(processes)
    if actual_active_workers != active_workers:
        console.print(f"[yellow]Note: Adjusted worker count from {active_workers} to {actual_active_workers} for platform compatibility[/]")
        active_workers = actual_active_workers

    progress_update("STEP 6.2", "Worker Pool", "✅", f"{len(processes)} workers spawned successfully")

    vector_db = init_vector_db()

    progress_update("STEP 6.4", "Vector Writer", "⏳", "Starting background storage thread...")
    writer = VectorWriter(
        result_q=result_q, checkpoint=checkpoint, vector_db=vector_db,
        data_source_name=source.table, table_name=source.table, strategy=strategy,
    ).start_thread()
    progress_update("STEP 6.4", "Vector Writer", "✅", "Background writer thread started")

    progress_update("STEP 6.5", "Monitor", "⏳", "Starting progress monitor...")
    monitor = Monitor(
        stats_q=stats_q, checkpoint=checkpoint,
        writer_ref=writer, total_chunks=total_chunks,
    ).start_thread()
    progress_update("STEP 6.5", "Monitor", "✅", "Progress monitor active")

    config.chunk_size = effective_chunk

    # STEP 7: Data Processing Loop
    console.print(f"\n[bold bright_blue]⚡ PROCESSING PIPELINE ACTIVE[/]")
    progress_update("STEP 7", "Data Processing", "🔄", "Starting main execution loop...")

    progress_update("STEP 7.1", "Orchestrator", "⏳", "Initializing pipeline controller...")

    try:
        progress_update("STEP 7.2", "Chunk Streaming", "🔄", "Beginning data stream processing...")
        Orchestrator().run(
            config=config, data_source=source, strategy=strategy,
            processes=processes, work_queue=work_q,
            result_queue=result_q, checkpoint=checkpoint,
        )
        progress_update("STEP 7", "Data Processing", "✅", "Pipeline execution completed successfully")

    except Exception as e:
        progress_update("STEP 7", "Data Processing", "❌", f"Pipeline failed: {str(e)}")
        console.print(f"[red]ERROR: Pipeline failed during execution: {e}[/red]")
        try:
            progress_update("STEP 7", "Emergency Shutdown", "⏳", "Shutting down workers...")
            WorkerPoolManager().shutdown(work_q, timeout=10)
            progress_update("STEP 7", "Emergency Shutdown", "✅", "Workers shut down")
        except:
            pass

    # STEP 8: Cleanup & Results
    console.print(f"\n[bold bright_blue]🏁 PIPELINE CLEANUP[/]")
    progress_update("STEP 8", "Cleanup", "⏳", "Shutting down components...")

    progress_update("STEP 8.1", "Writer Shutdown", "⏳", "Stopping vector writer...")
    writer.stop()
    progress_update("STEP 8.1", "Writer Shutdown", "✅", "Vector writer stopped")

    progress_update("STEP 8.2", "Monitor Shutdown", "⏳", "Stopping progress monitor...")
    monitor.stop()
    progress_update("STEP 8.2", "Monitor Shutdown", "✅", "Monitor stopped")

    progress_update("STEP 8.3", "Stats Collection", "⏳", "Collecting final statistics...")
    worker_stats_list = monitor.get_final_worker_stats()
    checkpoint.save_worker_stats(worker_stats_list)

    time.sleep(0.5)

    progress_update("STEP 8.3", "Stats Collection", "✅", "Statistics saved")

    _worker_stats_table(
        checkpoint.get_worker_stats(),
        title="Worker Stats  (this run)"
    )

    progress_update("STEP 8.4", "Final Summary", "⏳", "Generating completion report...")

    final = checkpoint.get_resume_summary()
    final_processed_rows = checkpoint.get_processed_row_count()

    success_rate = (final_processed_rows / stats.total_rows * 100) if stats.total_rows > 0 else 0

    progress_update("STEP 8.4", "Final Summary", "✅",
                    f"Pipeline complete: {final.done} chunks done, {final.failed} failed, {final_processed_rows:,} rows embedded ({success_rate:.1f}%)")

    console.print()
    console.print(Panel(
        f"[bold green]✅  Pipeline Complete[/]\n\n"
        f"  [bold]Chunks done   :[/]  [cyan]{final.done}[/]\n"
        f"  [bold]Chunks failed :[/]  "
        f"{'[red]' + str(final.failed) + '[/red]' if final.failed else '[dim]0[/dim]'}\n"
        f"  [bold]Rows embedded :[/]  [cyan]{final_processed_rows:,}[/] [dim]({success_rate:.1f}%)[/]\n"
        f"  [bold]Cost          :[/]  [bold green]$0.00[/]\n\n"
        f"  [dim]Run [bold]python search.py[/bold] to query your data.[/]",
        title="[bold white]HAUP v2.0[/]",
        border_style="green", expand=False,
    ))

    console.print(f"\n[bold green]🎉 EXECUTION COMPLETE[/] - All steps finished successfully!")

    # STEP 9: Optional Graph Build
    _section("Graph Build (Optional)")
    progress_update("STEP 9", "Graph Check", "⏳", "Checking graph configuration...")

    if os.path.exists('graph_config.json'):
        with open('graph_config.json', 'r') as f:
            graph_cfg = json.load(f)

        if graph_cfg.get('graph_build', {}).get('auto_start_after_forward', False):
            progress_update("STEP 9.1", "Graph Build", "⏳", "Auto-starting graph build pipeline...")
            console.print(Panel(
                "[bold cyan]🔗 Starting Graph Build Pipeline[/]\n\n"
                "[dim]Building knowledge graph from embeddings...[/]\n"
                "[dim]This will create nodes and edges in Neo4j.[/]",
                border_style="cyan",
                expand=False
            ))

            try:
                from graph_core.graph_orchestrator import GraphOrchestrator

                graph_orch = GraphOrchestrator(config_path='graph_config.json')

                db_config = {
                    "connection_string": os.getenv("NEON_CONNECTION_STRING"),
                    "host":     os.getenv("PG_HOST", "localhost"),
                    "port":     int(os.getenv("PG_PORT", "5432")),
                    "user":     os.getenv("PG_USER", "postgres"),
                    "password": os.getenv("PG_PASSWORD", ""),
                    "database": os.getenv("PG_DATABASE", "Vector"),
                    "table":    source.table,
                    "primary_key": "id",
                }

                graph_orch.initialize(
                    db_config=db_config,
                    db_type="postgresql",
                    chroma_path="./chroma_db",
                    collection_name="haup_vectors"
                )

                graph_orch.run()
                graph_orch.close()

                progress_update("STEP 9.1", "Graph Build", "✅", "Graph build completed successfully")

                console.print(Panel(
                    "[bold green]✅ Graph Build Complete[/]\n\n"
                    "[dim]• Query graph via Neo4j Browser\n"
                    "• Use graph-enhanced RAG in rag_api.py\n"
                    "• Explore with graph query API[/]",
                    border_style="green",
                    expand=False
                ))

            except ImportError:
                progress_update("STEP 9.1", "Graph Build", "❌", "neo4j package not installed")
                console.print("[yellow]⚠️  Graph build enabled but neo4j package not installed.[/]")
                console.print("[dim]Install with: pip install neo4j[/]")
            except Exception as e:
                progress_update("STEP 9.1", "Graph Build", "❌", f"Failed: {str(e)}")
                console.print(f"[red]Graph build failed: {e}[/]")
        else:
            progress_update("STEP 9", "Graph Check", "⏭️", "Graph auto-start disabled in config")
            console.print("[dim]Graph build disabled. Run [bold]python graph_main.py[/bold] to build manually.[/]")
    else:
        progress_update("STEP 9", "Graph Check", "⏭️", "No graph_config.json found")
        console.print("[dim]No graph_config.json found. Graph features not configured.[/]")

    # STEP 10: Optional CDC Real-Time Sync
    _section("CDC Real-Time Sync (Optional)")
    progress_update("STEP 10", "CDC Check", "⏳", "Checking CDC configuration...")

    cdc_model = None
    if os.path.exists('cdc_config.json'):
        with open('cdc_config.json', 'r') as f:
            cdc_cfg = json.load(f)

        if cdc_cfg.get('enabled', False):
            progress_update("STEP 10.1", "Model Loading", "⏳", "Loading embedding model for CDC...")
            try:
                from sentence_transformers import SentenceTransformer
                cdc_model = SentenceTransformer(config.model_name, device=config.device)
                progress_update("STEP 10.1", "Model Loading", "✅", "Model loaded for CDC")
            except Exception as e:
                progress_update("STEP 10.1", "Model Loading", "❌", f"Failed: {str(e)}")

    cdc_listener = start_cdc_if_enabled(
        'cdc_config.json',
        vector_db=vector_db,
        model=cdc_model,
        strategy=strategy,
        source=source
    )

    if cdc_listener is not None:
        console.print("\n[bold cyan]CDC listener is running in background...[/]")
        console.print("[dim]Press Ctrl+C to stop and exit.[/]\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            progress_update("CDC", "Shutdown", "⏳", "Stopping CDC listener...")
            cdc_listener.stop()
            progress_update("CDC", "Shutdown", "✅", "CDC listener stopped")
            console.print("\n[green]CDC listener stopped gracefully.[/]")
"""================= End function main ================="""

if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()