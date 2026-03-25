"""
File Summary:
Spawns N independent OS processes that parse Vector DB entries into row dicts.
Each worker runs the heuristic parser independently.
Uses 'spawn' context for clean interpreter per worker (safe with any I/O).

====================================================================
Startup
====================================================================

spawn_workers()  [Function]
||
├── mp.get_context("spawn") -----------------------------> Spawn context for clean process isolation
│
├── Loop: ctx.Process()  [Class → Object] ---------------> Create N worker processes
│       │
│       ├── proc.start() --------------------------------> Launch worker process
│       │
│       ├── [Conditional Branch] proc.is_alive() --------> Check process started successfully
│       │       ├── True  --------------------------------> Append to processes list
│       │       └── False --------------------------------> Log warning, skip worker
│       │
│       └── [Exception Block] ---------------------------> Log warning, continue to next worker
│
├── [Early Exit Branch] no processes started ------------> Raise RuntimeError
│
└── Return processes list -------------------------------> Active worker handles

_worker_main()  [Function]
||
├── Main work loop
│       │
│       ├── work_q.get() --------------------------------> Receive VectChunk (60s timeout)
│       │
│       ├── [Conditional Branch] SHUTDOWN_SIGNAL --------> Re-queue signal and break
│       │
│       ├── Retry loop (max 3 attempts per chunk)
│       │       │
│       │       ├── filter_route()  [Function] ----------> Parse each entry in chunk
│       │       │       │
│       │       │       ├── Item retry loop (max 2 per item)
│       │       │       │       ├── Success --------------> Append row to rows_out
│       │       │       │       └── Failure --------------> Increment parse_fails
│       │       │       │
│       │       │       └── [Exception Block] -----------> _log_worker_error(), retry chunk
│       │       │
│       │       └── result_q.put(ResultPacket) ----------> Send parsed rows to writer
│       │
│       └── stats_q.put(WorkerStat) ---------------------> Send per-chunk stats to monitor
│
└── Cleanup
        └── stats_q.put(WorkerStat) ---------------------> Send final cumulative stats

shutdown_workers()  [Function]
||
├── Loop: work_q.put(SHUTDOWN_SIGNAL) ------------------> Signal all workers to stop
│
├── Loop: proc.join() -----------------------------------> Wait for clean exit
│       │
│       ├── [Conditional Branch] proc.is_alive() --------> Terminate if over timeout
│       │
│       └── [Conditional Branch] still alive ------------> Force kill
│
└── [Exception Block] -----------------------------------> Terminate all remaining processes

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SHUTDOWN_SIGNAL = "__SHUTDOWN__"


"""================= Startup class ResultPacket ================="""
@dataclass
class ResultPacket:
    chunk_id:    int
    rows:        list[dict]
    parse_fails: int  = 0
    has_error:   bool = False
    error_msg:   str  = ""
"""================= End class ResultPacket ================="""


"""================= Startup class WorkerStat ================="""
@dataclass
class WorkerStat:
    worker_id:   int
    rows_parsed: int
    parse_fails: int
"""================= End class WorkerStat ================="""


"""================= Startup function spawn_workers ================="""
def spawn_workers(
    config,
    strategy,
    work_q:   mp.Queue,
    result_q: mp.Queue,
    stats_q:  mp.Queue,
) -> list[mp.Process]:
    import time
    import os

    actual_workers = config.num_workers
    if os.name == "nt":  # Windows: cap at 2 to avoid spawn overhead conflicts
        actual_workers = min(2, config.num_workers)
        logger.info("Windows detected: using %d workers", actual_workers)
    else:
        logger.info("Unix system: using %d workers", actual_workers)

    ctx       = mp.get_context("spawn")
    processes = []

    for i in range(actual_workers):
        try:
            proc = ctx.Process(
                target = _worker_main,
                args   = (i, config, strategy, work_q, result_q, stats_q),
                name   = f"haup-worker-{i}",
                daemon = False,
            )
            proc.start()
            time.sleep(0.5)

            if proc.is_alive():
                processes.append(proc)
                logger.info("[WorkerPool] started worker %d  pid=%d", i, proc.pid)
            else:
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                except Exception:
                    pass
                logger.warning("Failed to start worker %d", i)

        except Exception as e:
            logger.warning("Failed to start worker %d: %s", i, e)
            continue

    if not processes:
        raise RuntimeError("Failed to start any worker processes")

    original_workers   = config.num_workers
    config.num_workers = len(processes)

    if len(processes) < original_workers:
        logger.warning("Started %d workers instead of %d", len(processes), original_workers)

    print(f"[WorkerPool]  {len(processes)} workers started")
    return processes
"""================= End function spawn_workers ================="""


"""================= Startup function shutdown_workers ================="""
def shutdown_workers(
    processes:   list[mp.Process],
    work_q:      mp.Queue,
    num_workers: int,
    timeout:     int = 60,
) -> None:
    import time

    try:
        for _ in range(num_workers):
            try:
                work_q.put(SHUTDOWN_SIGNAL)
            except Exception:
                pass

        start_time = time.time()
        for proc in processes:
            remaining = max(1, timeout - (time.time() - start_time))
            proc.join(timeout=remaining)

            if proc.is_alive():
                logger.warning("[WorkerPool] worker %s still alive; terminating", proc.name)
                try:
                    proc.terminate()
                    proc.join(timeout=5.0)
                except Exception:
                    pass

                if proc.is_alive():
                    try:
                        proc.kill()
                        proc.join(timeout=2.0)
                    except Exception:
                        pass

    except Exception:
        for proc in processes:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
            except Exception:
                pass
"""================= End function shutdown_workers ================="""


"""================= Startup function _worker_main ================="""
def _worker_main(
    worker_id: int,
    config,
    strategy,
    work_q:    mp.Queue,
    result_q:  mp.Queue,
    stats_q:   mp.Queue,
) -> None:
    import logging as _logging
    import time

    worker_logger = _logging.getLogger(f"haup.reverse_worker_{worker_id}")
    worker_logger.setLevel(_logging.ERROR)
    worker_logger.propagate = False

    from reverse_core.text_filter import route as filter_route

    total_parsed = 0
    total_fails  = 0

    while True:
        try:
            chunk = work_q.get(block=True, timeout=60.0)
        except Exception:
            continue

        if chunk == SHUTDOWN_SIGNAL:
            work_q.put(SHUTDOWN_SIGNAL)
            break

        max_retries = 3
        retry_count = 0
        success     = False
        rows_out    = []
        parse_fails = 0

        while retry_count < max_retries and not success:
            rows_out    = []
            parse_fails = 0

            try:
                for i in range(len(chunk.ids)):
                    rowid = chunk.ids[i]
                    meta  = chunk.metas[i] if chunk.metas else {}
                    doc   = chunk.docs[i]  if chunk.docs  else None

                    item_retries = 0
                    item_success = False

                    while item_retries < 2 and not item_success:
                        try:
                            row = filter_route(
                                doc         = doc,
                                meta        = meta,
                                strategy    = strategy,
                                chromadb_id = rowid,
                            )
                            item_success = True
                        except Exception as exc:
                            item_retries += 1
                            if item_retries < 2:
                                time.sleep(0.1 * item_retries)
                            else:
                                row = None
                                worker_logger.warning(
                                    "[Worker %d] parse error rowid=%s after %d retries: %s",
                                    worker_id, rowid, item_retries, exc,
                                )

                    if row is not None:
                        row["__rowid__"] = rowid
                        rows_out.append(row)
                    else:
                        parse_fails += 1

                total_parsed += len(rows_out)
                total_fails  += parse_fails

                result_q.put(ResultPacket(
                    chunk_id    = chunk.chunk_id,
                    rows        = rows_out,
                    parse_fails = parse_fails,
                ))
                success = True

            except Exception as exc:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(0.5 * (2 ** retry_count))
                    worker_logger.warning(
                        "[Worker %d] chunk %d retry %d/%d: %s",
                        worker_id, chunk.chunk_id, retry_count, max_retries, exc,
                    )
                else:
                    _log_worker_error(worker_id, f"chunk {chunk.chunk_id}", exc, result_q, chunk.chunk_id)
                    break

        try:
            stats_q.put(WorkerStat(
                worker_id   = worker_id,
                rows_parsed = len(rows_out) if success else 0,
                parse_fails = parse_fails   if success else len(chunk.ids),
            ))
        except Exception:
            pass

    # Final cumulative stats
    try:
        stats_q.put(WorkerStat(
            worker_id   = worker_id,
            rows_parsed = total_parsed,
            parse_fails = total_fails,
        ))
    except Exception:
        pass
"""================= End function _worker_main ================="""


"""================= Startup function _log_worker_error ================="""
def _log_worker_error(worker_id, phase, exc, result_q, chunk_id: int = -1):
    msg = f"Worker {worker_id} failed in {phase}: {exc}\n{traceback.format_exc()}"
    logger.error(msg)
    if chunk_id >= 0:
        result_q.put(ResultPacket(
            chunk_id  = chunk_id,
            rows      = [],
            has_error = True,
            error_msg = msg,
        ))
"""================= End function _log_worker_error ================="""