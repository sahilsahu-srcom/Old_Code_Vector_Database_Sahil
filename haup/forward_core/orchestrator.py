"""
File Summary:
Orchestrator and work queue for HAUP v2.0. Streams data chunks with back-pressure and crash-resume capability.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

Orchestrator()  [Class → Object]
||
└── run()  [Method] --------------------------------------> Main chunk feed loop
        │
        ├── SQLStreamReader()  [Class → Object] ----------> PostgreSQL paginated stream
        │
        ├── Stream chunks --------------------------------> Lazy iteration over data batches
        │       │
        │       ├── checkpoint.is_row_processed() --------> Skip already processed rows
        │       │
        │       ├── [Conditional Branch] filtered_data empty -> Skip chunk entirely
        │       │
        │       ├── checkpoint.mark_running()  [Function] -> Mark chunk as in-progress
        │       │
        │       └── work_queue.put()  [Function] ---------> Push chunk with back-pressure
        │
        ├── Send SHUTDOWN_SIGNAL --------------------------> Signal each worker to stop
        │       │
        │       └── work_queue.put(SHUTDOWN_SIGNAL) ------> One signal per worker
        │
        └── proc.join()  [Function] ----------------------> Wait for worker processes
                │
                └── [Conditional Branch] proc.is_alive() -> Terminate if timeout exceeded

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import multiprocessing
import logging
from typing import List

# FIX: Removed unused ExcelStreamReader import — pipeline is PostgreSQL only.
from forward_core.stream_reader import SQLStreamReader
from forward_core.worker_pool_manager import SHUTDOWN_SIGNAL

logger = logging.getLogger("haup.orchestrator")


"""================= Startup class SqlSource ================="""
class SqlSource:
    type = 'sql'

    def __init__(self, connection, table: str, primary_key: str = 'id',
                 name: str = "neon_db"):
        self.conn        = connection
        self.table       = table
        self.primary_key = primary_key
        self.name        = name
"""================= End class SqlSource ================="""

# FIX: Removed ExcelSource class — dead code for a PostgreSQL-only pipeline.


"""================= Startup class Orchestrator ================="""
class Orchestrator:

    """================= Startup method run ================="""
    def run(self,
            config,
            data_source,
            strategy,
            processes:    List[multiprocessing.Process],
            work_queue:   multiprocessing.Queue,
            result_queue: multiprocessing.Queue,
            checkpoint) -> None:

        """
        Main feed loop.
        Blocks on work_queue.put() when queue is full (back-pressure).
        """

        # ── Route to correct stream reader ────────────────────────────
        if data_source.type == 'sql':
            reader = SQLStreamReader(
                sql_connection = data_source.conn,
                table_name     = data_source.table,
                primary_key    = data_source.primary_key,
            )
            stream = reader.stream_chunks(config.chunk_size)
        # FIX: Removed 'excel' branch — ExcelSource and ExcelStreamReader
        # are no longer part of this pipeline.
        else:
            raise ValueError(f"Unknown data source type: {data_source.type}")

        # ── Main feed loop ────────────────────────────────────────────
        chunk_count = 0
        for chunk in stream:
            # Filter out already processed rows from the chunk
            filtered_data = []
            for row in chunk.data:
                row_id = row.get(data_source.primary_key)
                if row_id and not checkpoint.is_row_processed(row_id):
                    filtered_data.append(row)

            # Skip chunk if all rows are already processed
            if not filtered_data:
                logger.info(f"Skipping chunk {chunk.chunk_id} (all rows already processed)")
                continue

            # Update chunk with filtered data
            chunk.data = filtered_data

            checkpoint.mark_running(chunk.chunk_id)
            work_queue.put(chunk)    # BLOCKS if queue full (back-pressure)
            chunk_count += 1
            logger.info(f"Sent chunk {chunk.chunk_id} to workers ({len(filtered_data)} rows)")

        logger.info(f"Finished sending {chunk_count} chunks to workers")

        # ── Signal workers to shut down ───────────────────────────────
        logger.info("Signaling workers to shut down")
        for _ in range(config.num_workers):
            work_queue.put(SHUTDOWN_SIGNAL)
        logger.info("Shutdown signals sent")

        # ── Wait for clean worker exit ────────────────────────────────
        logger.info("Waiting for workers to complete")
        for proc in processes:
            logger.info(f"Waiting for worker {proc.name}")
            proc.join(timeout=120)
            if proc.is_alive():
                logger.warning(f"Worker {proc.name} timeout; terminating.")
                proc.terminate()
            logger.info(f"Worker {proc.name} finished")

        logger.info("All workers completed")