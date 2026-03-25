"""
File Summary:
Real-time pipeline monitor for HAUP v2.0. Renders a live Rich dashboard showing
chunk progress, worker stats, CPU/RAM/VRAM usage, ETA, and cost during ingestion.

====================================================================
Startup
====================================================================

Monitor()  [Class → Object]
||
├── __init__()  [Method] ----------------------------------> Initialize queues, checkpoint, writer ref, stop event
│
├── start_thread()  [Method] -----------------------------> Spawn background daemon monitor thread
│       │
│       └── _monitor_thread()  [Method] ------------------> Background thread entry point
│               │
│               ├── Live()  [Class → Object] --------------> Rich live-rendering context
│               │
│               ├── Loop until stop_event set
│               │       │
│               │       ├── _drain_stats_q()  [Method] ---> Consume all pending worker stats from queue
│               │       ├── checkpoint.get_resume_summary() [Function] --> Fetch chunk progress
│               │       ├── writer_ref.total_rows_written  -> Fetch rows written count
│               │       ├── psutil.cpu_percent()  [Function] -----------> CPU usage %
│               │       ├── psutil.virtual_memory()  [Function] ---------> RAM usage %
│               │       ├── _get_vram_info()  [Method] ------------------> GPU VRAM usage (optional)
│               │       └── _build_panel()  [Method] -------------------> Render Rich panel and update Live
│               │
│               └── Final render after stop ─────────────> One last panel update before exit
│
├── stop()  [Method] -------------------------------------> Signal stop event, join thread, drain queue
│       │
│       └── _drain_stats_q()  [Method] -------------------> Final queue drain after thread joins
│
├── get_final_worker_stats()  [Method] -------------------> Return accumulated worker stats list
│
├── _drain_stats_q()  [Method] ---------------------------> Read all stats from queue into _worker_stats dict
│       │
│       └── _AccumulatedStat()  [Class → Object] ---------> Merge new stat with existing accumulated stat
│
├── _build_panel()  [Method] -----------------------------> Build and return Rich Panel with all metrics
│       │
│       ├── Progress bar construction ─────────────────-> Filled/empty block bar from done/total ratio
│       ├── ETA and elapsed time ─────────────────────-> _estimate_eta() call
│       ├── Rows in DB, CPU, RAM, VRAM display ───────-> Live system metrics
│       └── Worker table construction ─────────────────-> Per-worker rows/batch/bar display
│
├── _get_vram_info()  [Method] ---------------------------> Query torch CUDA for VRAM reserved/total
│       │
│       └── [Exception Block] ──────────────────────────> Return None if torch unavailable or no GPU
│
└── _estimate_eta()  [Method] ----------------------------> Calculate ETA string from elapsed and remaining chunks
        │
        ├── done == 0 or total_chunks == 0 ─────────────> Return "calculating…"
        ├── eta_sec < 60 ───────────────────────────────> Return seconds string
        ├── eta_sec < 3600 ─────────────────────────────> Return minutes string
        └── else ───────────────────────────────────────> Return hours string

====================================================================
CLASS / METHOD ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Dict

import psutil
from rich.console import Console
from rich.live    import Live
from rich.panel   import Panel
from rich.text    import Text

logger = logging.getLogger("haup.monitor")


"""================= Startup class Monitor ================="""
class Monitor:

    REFRESH_RATE = 2  # Reduced from 4 to 2 updates per second for stability
    SLEEP_SEC    = 1.0  # Increased from 0.5 to 1.0 second for stability

    """================= Startup method __init__ ================="""
    def __init__(self, stats_q, checkpoint, writer_ref, total_chunks: int = 0):
        self.stats_q        = stats_q
        self.checkpoint     = checkpoint
        self.writer_ref     = writer_ref
        self.total_chunks   = total_chunks
        self._stop_event    = threading.Event()
        self._thread        = None
        self._worker_stats: Dict[int, object] = {}
        self._lock          = threading.Lock()
        self._started       = False  # Prevent multiple starts
    """================= End method __init__ ================="""


    """================= Startup method start_thread ================="""
    def start_thread(self) -> "Monitor":
        if self._started:
            logger.warning("Monitor already started, skipping duplicate start")
            return self
        
        self._started = True
        self._thread = threading.Thread(
            target=self._monitor_thread,
            name="haup-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Monitor thread started successfully")
        return self
    """================= End method start_thread ================="""


    """================= Startup method stop ================="""
    def stop(self) -> None:
        if not self._started:
            logger.warning("Monitor was never started, skipping stop")
            return
            
        logger.info("Stopping monitor thread...")
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                logger.warning("Monitor thread did not stop within timeout")
            else:
                logger.info("Monitor thread stopped successfully")
        
        self._drain_stats_q()
        logger.info("Monitor stopped and stats drained")
    """================= End method stop ================="""


    """================= Startup method get_final_worker_stats ================="""
    def get_final_worker_stats(self) -> list:
        with self._lock:
            return list(self._worker_stats.values())
    """================= End method get_final_worker_stats ================="""


    """================= Startup method _monitor_thread ================="""
    def _monitor_thread(self) -> None:
        try:
            console    = Console(stderr=False, highlight=False)
            start_time = time.time()
            logger.info("Monitor thread running")

            with Live(console=console,
                      refresh_per_second=self.REFRESH_RATE,
                      transient=False) as live:

                iteration = 0
                while not self._stop_event.is_set():
                    try:
                        self._drain_stats_q()

                        summary      = self.checkpoint.get_resume_summary()
                        rows_written = getattr(self.writer_ref, 'total_rows_written', 0)
                        cpu_pct      = psutil.cpu_percent(interval=None)
                        ram_pct      = psutil.virtual_memory().percent
                        vram_info    = self._get_vram_info()

                        live.update(self._build_panel(
                            summary, rows_written, cpu_pct, ram_pct, vram_info, start_time
                        ))
                        
                        iteration += 1
                        if iteration % 10 == 0:  # Log every 10 iterations
                            logger.debug(f"Monitor update #{iteration}: {summary.done}/{self.total_chunks} chunks")
                        
                    except Exception as e:
                        logger.error(f"Error in monitor loop iteration: {e}", exc_info=True)
                    
                    time.sleep(self.SLEEP_SEC)

                # Final update after stop signal
                try:
                    self._drain_stats_q()
                    summary      = self.checkpoint.get_resume_summary()
                    rows_written = getattr(self.writer_ref, 'total_rows_written', 0)
                    live.update(self._build_panel(
                        summary, rows_written,
                        psutil.cpu_percent(interval=None),
                        psutil.virtual_memory().percent,
                        self._get_vram_info(), start_time,
                    ))
                    logger.info("Monitor final update complete")
                except Exception as e:
                    logger.error(f"Error in monitor final update: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Fatal error in monitor thread: {e}", exc_info=True)
        finally:
            logger.info("Monitor thread exiting")
    """================= End method _monitor_thread ================="""


    """================= Startup method _drain_stats_q ================="""
    def _drain_stats_q(self) -> None:
        while True:
            try:
                stat = self.stats_q.get_nowait()
                with self._lock:
                    existing = self._worker_stats.get(stat.worker_id)
                    if existing is not None:
                        stat = _AccumulatedStat(
                            worker_id      = stat.worker_id,
                            rows_processed = existing.rows_processed + stat.rows_processed,
                            current_batch  = stat.current_batch,
                        )
                    self._worker_stats[stat.worker_id] = stat
            except Exception:
                break
    """================= End method _drain_stats_q ================="""


    """================= Startup method _build_panel ================="""
    def _build_panel(self, summary, rows_written, cpu_pct,
                     ram_pct, vram_info, start_time) -> Panel:
        done      = summary.done
        failed    = summary.failed
        total     = self.total_chunks
        eta_str   = self._estimate_eta(done, start_time)
        elapsed   = time.time() - start_time

        txt = Text()

        if total > 0:
            pct    = done / total
            filled = int(pct * 30)
            bar    = f"[green]{'█' * filled}[/][dim]{'░' * (30 - filled)}[/]"
            txt.append(f"  Progress  ", style="bold")
            txt.append_text(Text.from_markup(
                f"{bar}  {done}/{total}  "
                f"({'[red]' + str(failed) + ' failed[/red]' if failed else '[dim]0 failed[/dim]'})\n"
            ))
        else:
            txt.append(f"  Progress  : {done} chunks done\n", style="bold cyan")

        txt.append_text(Text.from_markup(
            f"  ETA       : [yellow]{eta_str}[/]   "
            f"Elapsed: [dim]{elapsed:.0f}s[/]\n"
            f"  Rows in DB: [cyan]{rows_written:,}[/]\n"
            f"  CPU       : [white]{cpu_pct:.1f}%[/]   "
            f"RAM: [white]{ram_pct:.1f}%[/]"
        ))
        if vram_info:
            txt.append_text(Text.from_markup(f"   VRAM: [magenta]{vram_info}[/]"))
        txt.append("\n")

        with self._lock:
            ws = dict(self._worker_stats)

        if ws:
            txt.append("\n  Workers\n", style="bold magenta")
            txt.append(f"  {'ID':<10}{'Rows':>10}{'Batch':>10}  {'Growth':<20}\n",
                       style="dim")
            txt.append(f"  {'─'*50}\n", style="dim")
            total_rows = 0
            for wid in sorted(ws):
                s         = ws[wid]
                rows      = s.rows_processed
                batch     = s.current_batch
                total_rows += rows
                filled    = max(1, int((batch / 512) * 14))
                bar       = "█" * filled + "░" * (14 - filled)
                txt.append_text(Text.from_markup(
                    f"  [cyan]Worker-{wid:<4}[/]"
                    f"[white]{rows:>10,}[/]"
                    f"[green]{batch:>10}[/]"
                    f"  [dim]{bar}[/]\n"
                ))
            txt.append(f"  {'─'*50}\n", style="dim")
            txt.append_text(Text.from_markup(
                f"  [bold]{'TOTAL':<10}[/][bold cyan]{total_rows:>10,}[/]"
                f"  [dim]{len(ws)} workers active[/]\n"
            ))
        else:
            txt.append("\n  [dim]Workers: loading model…[/]\n")

        txt.append_text(Text.from_markup("\n  [bold green]Cost: $0.00[/]  [dim](all local)[/]\n"))

        return Panel(txt,
                     title="[bold white]HAUP v2.0 — Live Pipeline[/]",
                     border_style="bright_blue")
    """================= End method _build_panel ================="""


    """================= Startup method _get_vram_info ================="""
    @staticmethod
    def _get_vram_info() -> Optional[str]:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            total    = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return f"{reserved:.0f}/{total:.0f} MB"
        except Exception:
            return None
    """================= End method _get_vram_info ================="""


    """================= Startup method _estimate_eta ================="""
    def _estimate_eta(self, done: int, start_time: float) -> str:
        if done == 0 or self.total_chunks == 0:
            return "calculating…"
        elapsed   = time.time() - start_time
        remaining = self.total_chunks - done
        eta_sec   = (elapsed / done) * remaining
        if eta_sec < 60:   return f"{eta_sec:.0f}s"
        if eta_sec < 3600: return f"{eta_sec/60:.1f}m"
        return f"{eta_sec/3600:.1f}h"
    """================= End method _estimate_eta ================="""

"""================= End class Monitor ================="""


"""================= Startup class _AccumulatedStat ================="""
class _AccumulatedStat:

    """================= Startup method __init__ ================="""
    def __init__(self, worker_id, rows_processed, current_batch):
        self.worker_id      = worker_id
        self.rows_processed = rows_processed
        self.current_batch  = current_batch
    """================= End method __init__ ================="""

"""================= End class _AccumulatedStat ================="""