"""
File Summary:
Live terminal dashboard using Rich for reverse extraction pipeline.
Shows extraction progress, per-worker stats, parse fail rate, and resource usage.
Falls back to plain print() if Rich is not installed.

====================================================================
Startup
====================================================================

ReverseMonitor()  [Class → Object]
||
├── start()  [Function] ----------------------------------> Launch background monitor thread
│       │
│       └── _run()  [Function] ---------------------------> Dispatch to rich or plain mode
│               │
│               ├── _run_rich()  [Function] ---------------> Rich Live dashboard loop
│               │       │
│               │       ├── _drain_stats_q()  [Function] -> Collect worker stats from queue
│               │       ├── _update_state()  [Function] --> Sync rows_written and parse_fails
│               │       └── _build_rich_panel()  [Function] -> Render progress panel
│               │               ├── Progress bar ---------> Visual chunk completion bar
│               │               ├── ETA calculation ------> _estimate_eta() time remaining
│               │               ├── Worker stats table ---> Per-worker parsed/fails
│               │               └── CPU / RAM display ----> psutil resource usage
│               │
│               └── _run_plain()  [Function] -------------> Plain text fallback loop
│                       │
│                       ├── _drain_stats_q()  [Function] -> Collect worker stats from queue
│                       ├── _update_state()  [Function] --> Sync rows_written and parse_fails
│                       └── _plain_print()  [Function] ---> Print one-line progress to stdout
│
├── stop()  [Function] -----------------------------------> Set stop event, join thread
│       │
│       └── _print_final_summary()  [Function] ----------> Print completion stats to stdout
│
└── get_final_worker_stats()  [Function] -----------------> Return worker_stats dict snapshot

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


"""================= Startup class MonitorState ================="""
@dataclass
class MonitorState:
    total_chunks:    int   = 0
    rows_written:    int   = 0
    parse_fails:     int   = 0
    start_time:      float = field(default_factory=time.time)
    worker_stats:    dict  = field(default_factory=dict)
    collection_name: str   = ""
"""================= End class MonitorState ================="""


"""================= Startup class ReverseMonitor ================="""
class ReverseMonitor:

    """================= Startup method __init__ ================="""
    def __init__(
        self,
        stats_q:         queue.Queue,
        checkpoint,
        writer,
        total_chunks:    int,
        collection_name: str = "",
    ):
        self.stats_q    = stats_q
        self.checkpoint = checkpoint
        self.writer     = writer
        self.state      = MonitorState(
            total_chunks    = total_chunks,
            start_time      = time.time(),
            collection_name = collection_name,
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._has_rich = _check_rich()
    """================= End method __init__ ================="""

    """================= Startup method start ================="""
    def start(self) -> None:
        self._thread = threading.Thread(
            target = self._run,
            name   = "haup-monitor",
            daemon = True,
        )
        self._thread.start()
    """================= End method start ================="""

    """================= Startup method stop ================="""
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._print_final_summary()
    """================= End method stop ================="""

    """================= Startup method get_final_worker_stats ================="""
    def get_final_worker_stats(self) -> dict:
        return dict(self.state.worker_stats)
    """================= End method get_final_worker_stats ================="""

    """================= Startup method _run ================="""
    def _run(self) -> None:
        if self._has_rich:
            self._run_rich()
        else:
            self._run_plain()
    """================= End method _run ================="""

    """================= Startup method _run_rich ================="""
    def _run_rich(self) -> None:
        from rich.console import Console
        from rich.live    import Live
        from rich.table   import Table
        from rich.panel   import Panel
        from rich.text    import Text

        console = Console()
        with Live(console=console, refresh_per_second=2) as live:
            while not self._stop_event.is_set():
                self._drain_stats_q()
                self._update_state()
                live.update(self._build_rich_panel(Table, Panel, Text))
                time.sleep(0.5)
    """================= End method _run_rich ================="""

    """================= Startup method _run_plain ================="""
    def _run_plain(self) -> None:
        tick = 0
        while not self._stop_event.is_set():
            self._drain_stats_q()
            self._update_state()
            if tick % 10 == 0:
                self._plain_print()
            tick += 1
            time.sleep(0.5)
    """================= End method _run_plain ================="""

    """================= Startup method _drain_stats_q ================="""
    def _drain_stats_q(self) -> None:
        while True:
            try:
                stat = self.stats_q.get_nowait()
                self.state.worker_stats[stat.worker_id] = stat
            except queue.Empty:
                break
    """================= End method _drain_stats_q ================="""

    """================= Startup method _update_state ================="""
    def _update_state(self) -> None:
        self.state.rows_written = self.writer.total_rows_written
        self.state.parse_fails  = self.writer.total_parse_fails
    """================= End method _update_state ================="""

    """================= Startup method _build_rich_panel ================="""
    def _build_rich_panel(self, Table, Panel, Text):
        summary = self.checkpoint.get_resume_summary()
        elapsed = time.time() - self.state.start_time
        eta     = self._estimate_eta(summary.done, elapsed)

        cpu_pct = ram_pct = 0.0
        try:
            import psutil
            cpu_pct = psutil.cpu_percent()
            ram_pct = psutil.virtual_memory().percent
        except ImportError:
            pass

        txt    = Text()
        done   = summary.done
        failed = summary.failed
        total  = self.state.total_chunks

        if total > 0:
            pct    = done / total
            filled = int(pct * 30)
            bar    = f"[green]{'█' * filled}[/][dim]{'░' * (30 - filled)}[/]"
            txt.append("  Progress  ", style="bold")
            txt.append_text(Text.from_markup(
                f"{bar}  {done}/{total}  "
                f"({'[red]' + str(failed) + ' failed[/red]' if failed else '[dim]0 failed[/dim]'})\n"
            ))
        else:
            txt.append(f"  Progress  : {done} chunks done\n", style="bold cyan")

        txt.append_text(Text.from_markup(
            f"  ETA       : [yellow]{eta}[/]   "
            f"Elapsed: [dim]{_fmt_duration(elapsed)}[/]\n"
            f"  Rows in DB: [cyan]{self.state.rows_written:,}[/]\n"
            f"  CPU       : [white]{cpu_pct:.1f}%[/]   "
            f"RAM: [white]{ram_pct:.1f}%[/]\n"
        ))

        ws = self.state.worker_stats

        if ws:
            txt.append("\n  Workers\n", style="bold magenta")
            txt.append(
                f"  {'ID':<10}{'Parsed':>10}{'Fails':>10}\n",
                style="dim",
            )
            txt.append(f"  {'─' * 40}\n", style="dim")
            total_parsed = 0
            for wid in sorted(ws):
                s      = ws[wid]
                parsed = s.rows_parsed
                fails  = s.parse_fails
                total_parsed += parsed
                txt.append_text(Text.from_markup(
                    f"  [cyan]Worker-{wid:<4}[/]"
                    f"[green]{parsed:>10,}[/]"
                    f"[red]{fails:>10,}[/]\n"
                ))
            txt.append(f"  {'─' * 40}\n", style="dim")
            txt.append_text(Text.from_markup(
                f"  [bold]{'TOTAL':<10}[/][bold green]{total_parsed:>10,}[/]"
                f"  [dim]{len(ws)} workers active[/]\n"
            ))
        else:
            txt.append("\n  [dim]Workers: initializing…[/]\n")

        txt.append_text(Text.from_markup(
            "\n  [bold green]Cost: $0.00[/]  [dim](all local)[/]\n"
        ))

        return Panel(
            txt,
            title        = "[bold white]HAUP v3.0 — Reverse Extractor[/]",
            border_style = "bright_blue",
        )
    """================= End method _build_rich_panel ================="""

    """================= Startup method _plain_print ================="""
    def _plain_print(self) -> None:
        summary = self.checkpoint.get_resume_summary()
        elapsed = time.time() - self.state.start_time
        print(
            f"[Monitor]  chunks={summary.done}/{self.state.total_chunks}"
            f"  rows={self.state.rows_written:,}"
            f"  fails={self.state.parse_fails}"
            f"  elapsed={_fmt_duration(elapsed)}"
        )
    """================= End method _plain_print ================="""

    """================= Startup method _print_final_summary ================="""
    def _print_final_summary(self) -> None:
        elapsed = time.time() - self.state.start_time
        print(
            f"\n{'=' * 60}\n"
            f"HAUP v3.0  Extraction Complete\n"
            f"  Collection   : {self.state.collection_name}\n"
            f"  Rows written : {self.state.rows_written:,}\n"
            f"  Parse fails  : {self.state.parse_fails:,}\n"
            f"  Elapsed      : {_fmt_duration(elapsed)}\n"
            f"  Cost         : $0.00\n"
            f"{'=' * 60}\n"
        )
    """================= End method _print_final_summary ================="""

    """================= Startup method _estimate_eta ================="""
    def _estimate_eta(self, done: int, elapsed: float) -> str:
        total = self.state.total_chunks
        if done <= 0 or total <= 0:
            return "calculating..."
        rate      = done / elapsed
        remaining = (total - done) / rate
        return _fmt_duration(remaining)
    """================= End method _estimate_eta ================="""

"""================= End class ReverseMonitor ================="""


"""================= Startup function _check_rich ================="""
def _check_rich() -> bool:
    try:
        import rich
        return True
    except ImportError:
        return False
"""================= End function _check_rich ================="""


"""================= Startup function _fmt_duration ================="""
def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"
"""================= End function _fmt_duration ================="""


"""================= Startup function _pct ================="""
def _pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0
"""================= End function _pct ================="""