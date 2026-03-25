"""
File Summary:
Worker pool manager for HAUP v2.0. Spawns N independent OS processes using the
'spawn' context for true parallelism and CUDA safety. Each worker owns its own
SentenceTransformer model copy and AdaptiveBatchSizer feedback loop.

====================================================================
STARTUP
====================================================================

WorkerPoolManager()
||
├── spawn_workers()  [Function] --------------------------> Create and start N worker processes
│       │
│       ├── multiprocessing.get_context('spawn') ---------> CUDA-safe spawn context
│       │
│       ├── Loop: ctx.Process()  [Class → Object] --------> N independent worker processes
│       │
│       └── proc.start() ---------------------------------> Launch each worker
│
├── shutdown()  [Function] -------------------------------> Graceful worker termination
│       │
│       └── Loop: proc.join() / proc.terminate() ---------> Wait then force-kill if alive
│
└── worker_main()  [Function] ----------------------------> Worker process entry point
        │
        ├── SentenceTransformer()  [Class → Object] ------> Load embedding model in own memory
        │       │
        │       └── [Exception Block] model load fail ----> Send error packets and exit
        │
        ├── AdaptiveBatchSizer()  [Class → Object] -------> Adaptive GPU memory controller
        │
        └── Main work loop
                │
                ├── work_q.get() -------------------------> Receive chunk from queue
                │
                ├── [Conditional Branch] SHUTDOWN_SIGNAL -> Re-queue signal and break
                │
                ├── _serialize_all_rows()  [Function] ----> Convert row dicts to text strings
                │
                ├── batch_sizer.iter()  [Function] -------> Yield mini-batches at current size
                │       │
                │       └── model.encode() ---------------> Generate embedding vectors
                │               │
                │               ├── batch_sizer.feedback("SUCCESS") -> Grow batch if VRAM allows
                │               │
                │               └── [Exception Block] OOM -> Shrink batch, retry with half
                │
                ├── result_q.put(ResultPacket) -----------> Send vectors to writer
                │
                ├── stats_q.put(WorkerStat) --------------> Send metrics to monitor
                │
                └── [Exception Block] chunk error --------> Send error ResultPacket

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import multiprocessing
import logging
from dataclasses import dataclass
from typing import List, Any, Tuple

logger = logging.getLogger("haup.worker")

SHUTDOWN_SIGNAL = "__HAUP_SHUTDOWN__"


"""================= Startup class WorkChunk ================="""
@dataclass
class WorkChunk:
    chunk_id: int
    data:     List[dict]
"""================= End class WorkChunk ================="""


"""================= Startup class ResultPacket ================="""
@dataclass
class ResultPacket:
    chunk_id:  int
    vectors:   List[Tuple[Any, Any]]
    raw_data:  List[dict]
    has_error: bool = False
    source:    str  = ""
    table:     str  = ""
"""================= End class ResultPacket ================="""


"""================= Startup class WorkerStat ================="""
@dataclass
class WorkerStat:
    worker_id:      int
    current_batch:  int
    rows_processed: int
"""================= End class WorkerStat ================="""


"""================= Startup class AdaptiveBatchSizer ================="""
class AdaptiveBatchSizer:
    """
    Dynamically adjusts batch size based on GPU memory usage.
    Grows on success, shrinks on OOM.
    """

    GROW_FACTOR   = 1.25
    SHRINK_FACTOR = 0.5
    VRAM_HEADROOM = 0.80
    MAX_BATCH     = 512

    """================= Startup method __init__ ================="""
    def __init__(self, initial_batch: int):
        self.current = initial_batch
    """================= End method __init__ ================="""

    """================= Startup method feedback ================="""
    def feedback(self, status: str, vram_usage: float = 0.0) -> None:
        """Adjusts batch size based on encode() outcome."""
        if status == "OOM":
            self.current = max(1, int(self.current * self.SHRINK_FACTOR))
        elif status == "SUCCESS" and vram_usage < self.VRAM_HEADROOM:
            self.current = min(self.MAX_BATCH,
                               int(self.current * self.GROW_FACTOR))
    """================= End method feedback ================="""

    """================= Startup method iter ================="""
    def iter(self, texts: List[str], rowids: List[Any]):
        """Yields (mini_batch_texts, mini_batch_rowids) at current size."""
        for start in range(0, len(texts), self.current):
            end = start + self.current
            yield texts[start:end], rowids[start:end]
    """================= End method iter ================="""

"""================= End class AdaptiveBatchSizer ================="""


"""================= Startup function _vram_usage ================="""
# ===========Might be required GPU ========
def _vram_usage() -> float:
    """Returns fraction of GPU VRAM currently in use."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props    = torch.cuda.get_device_properties(0)
        reserved = torch.cuda.memory_reserved(0)
        return reserved / props.total_memory
    except Exception:
        return 0.0
"""================= End function _vram_usage ================="""


"""================= Startup function _serialize_all_rows ================="""
def _serialize_all_rows(rows: List[dict], strategy) -> List[str]:
    """
    Converts row dicts to text strings using schema template.
    """
    texts = []
    for row in rows:
        try:
            text = strategy.template.format_map(row)
        except (KeyError, ValueError):
            parts = [f"{col}: {row[col]}"
                     for col in strategy.semantic_cols + strategy.numeric_cols
                     if col in row and row[col] is not None]
            text = " | ".join(parts)
        texts.append(text)
    return texts
"""================= End function _serialize_all_rows ================="""


"""================= Startup function worker_main ================="""
def worker_main(worker_id: int, config, strategy,
                work_q:   multiprocessing.Queue,
                result_q: multiprocessing.Queue,
                stats_q:  multiprocessing.Queue,
                error_log_q: multiprocessing.Queue = None) -> None:
    """
    Main worker loop running in spawned process.
    Each worker has its own model copy and batch sizer.
    """
    import os
    import sys
    
    # Set up environment for stable operation
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HF_HUB_VERBOSITY'] = 'error'
    
    # Suppress warnings and logs
    import warnings
    warnings.filterwarnings('ignore')
    
    import logging
    logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    
    # Setup worker-specific logger
    worker_logger = logging.getLogger(f"haup.worker_{worker_id}")
    worker_logger.setLevel(logging.DEBUG)
    worker_logger.propagate = False  # Don't propagate to root logger (suppress terminal output)
    
    # Create file handler for this worker
    try:
        handler = logging.FileHandler(f"worker_{worker_id}.log")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
    except Exception as e:
        pass  # Continue even if file logging fails
    
    # FIX: Initialised here (before the model-loading try block) so the
    # finally block can always read it — even when model loading fails and
    # triggers an early return, finally still executes.
    chunks_processed = 0

    try:
        worker_logger.info(f"Worker {worker_id}: Starting up (PID: {os.getpid()})")

        model = None
        try:
            worker_logger.info(f"Worker {worker_id}: Loading sentence transformer model...")
            
            # Import with error handling
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                error_msg = f"Worker {worker_id}: Failed to import SentenceTransformer: {e}"
                worker_logger.error(error_msg)
                if error_log_q:
                    error_log_q.put(error_msg)
                raise
            
            # Load model with Windows-specific settings
            model = SentenceTransformer(
                config.model_name,
                device='cpu',  # Always use CPU for stability
                trust_remote_code=False,
                cache_folder=None  # Use default cache
            )
            
            # Test the model with a simple encoding
            test_result = model.encode(["test"], show_progress_bar=False, device='cpu')
            test_msg = f"Worker {worker_id}: Model loaded and tested successfully (output shape: {test_result.shape})"
            worker_logger.info(test_msg)
            
        except Exception as e:
            error_msg = f"Worker {worker_id}: Failed to load model: {e}"
            worker_logger.error(error_msg)
            import traceback
            tb = traceback.format_exc()
            worker_logger.error(f"Worker {worker_id}: Traceback: {tb}")
            if error_log_q:
                error_log_q.put(f"{error_msg}\n{tb}")
            
            # Send error for all pending chunks and exit
            error_count = 0
            while error_count < 10:  # Limit to prevent infinite loop
                try:
                    chunk = work_q.get(timeout=1.0)
                    if chunk == SHUTDOWN_SIGNAL:
                        work_q.put(SHUTDOWN_SIGNAL)
                        break
                    result_q.put(ResultPacket(
                        chunk_id  = chunk.chunk_id,
                        vectors   = [],
                        raw_data  = chunk.data,
                        has_error = True,
                    ))
                    error_count += 1
                except Exception:
                    break
            return

        batch_sizer = AdaptiveBatchSizer(config.initial_batch)

        ready_msg = f"Worker {worker_id}: Ready to process chunks"
        worker_logger.info(ready_msg)

        while True:
            try:
                chunk = work_q.get(block=True, timeout=60.0)  # Increased timeout
            except Exception:
                timeout_msg = f"Worker {worker_id}: Timeout waiting for work, checking for shutdown..."
                worker_logger.warning(timeout_msg)
                continue

            if chunk == SHUTDOWN_SIGNAL:
                shutdown_msg = f"Worker {worker_id}: Received shutdown signal"
                worker_logger.info(shutdown_msg)
                work_q.put(SHUTDOWN_SIGNAL)
                break

            processing_msg = f"Worker {worker_id}: Processing chunk {chunk.chunk_id} with {len(chunk.data)} rows"
            worker_logger.info(processing_msg)

            try:
                # Validate chunk data
                if not chunk.data:
                    raise ValueError(f"Chunk {chunk.chunk_id} has empty data")
                
                first_row = chunk.data[0]
                first_row_keys = list(first_row.keys())
                
                # Serialize texts
                texts  = _serialize_all_rows(chunk.data, strategy)
                
                # CRITICAL FIX: Validate rowid_col exists in data
                rowid_col = strategy.rowid_col
                if rowid_col not in first_row:
                    available_cols = ", ".join(first_row_keys)
                    error_msg = (f"Worker {worker_id}: rowid_col '{rowid_col}' not found in chunk {chunk.chunk_id} data. "
                               f"Available columns: {available_cols}")
                    worker_logger.error(error_msg)
                    if error_log_q:
                        error_log_q.put(error_msg)
                    raise ValueError(error_msg)
                
                # Extract rowids
                rowids = []
                for idx, row in enumerate(chunk.data):
                    try:
                        rowid = row[rowid_col]
                        rowids.append(rowid)
                    except KeyError as e:
                        error_msg = f"Worker {worker_id}: Row {idx} missing rowid_col '{rowid_col}': {e}"
                        worker_logger.error(error_msg)
                        raise

                serialize_msg = f"Worker {worker_id}: Serialized {len(texts)} texts for chunk {chunk.chunk_id}"
                worker_logger.info(serialize_msg)

                vectors = []

                for batch_num, (mini_batch, mini_ids) in enumerate(batch_sizer.iter(texts, rowids)):
                    retry_texts = mini_batch
                    retry_ids   = mini_ids
                    retry_count = 0
                    max_retries = 3

                    while retry_count < max_retries:
                        try:
                            encode_msg = f"Worker {worker_id}: Encoding batch {batch_num} with {len(retry_texts)} items (retry {retry_count})"
                            worker_logger.debug(encode_msg)
                            # ===========Might be required GPU ========
                            vecs = model.encode(
                                retry_texts,
                                device='cpu',
                                show_progress_bar=False,
                                batch_size=len(retry_texts),
                                normalize_embeddings=False
                            )

                            batch_sizer.feedback("SUCCESS", 0.0)  # No VRAM on CPU
                            vectors.extend(zip(vecs, retry_ids))
                            
                            success_msg = f"Worker {worker_id}: Encoded {len(vecs)} vectors for chunk {chunk.chunk_id}, batch {batch_num}"
                            worker_logger.info(success_msg)
                            break

                        except Exception as exc:
                            retry_count += 1
                            if "memory" in str(exc).lower() and retry_count < max_retries:
                                batch_sizer.feedback("OOM")
                                mid         = max(1, len(retry_texts) // 2)
                                retry_texts = retry_texts[:mid]
                                retry_ids   = retry_ids[:mid]
                                mem_msg = f"Worker {worker_id}: Memory issue, reducing batch to {len(retry_texts)}"
                                worker_logger.warning(mem_msg)
                            else:
                                error_msg = f"Worker {worker_id}: Encoding failed (retry {retry_count}/{max_retries}): {exc}"
                                worker_logger.error(error_msg)
                                import traceback
                                tb = traceback.format_exc()
                                worker_logger.error(tb)
                                if error_log_q:
                                    error_log_q.put(f"{error_msg}\n{tb}")
                                raise

                send_msg = f"Worker {worker_id}: Sending {len(vectors)} vectors for chunk {chunk.chunk_id}"
                worker_logger.info(send_msg)
                
                result_q.put(ResultPacket(
                    chunk_id = chunk.chunk_id,
                    vectors  = vectors,
                    raw_data = chunk.data,
                ))
                
                stats_q.put(WorkerStat(
                    worker_id      = worker_id,
                    current_batch  = batch_sizer.current,
                    rows_processed = len(texts),
                ))
                
                chunks_processed += 1
                complete_msg = f"Worker {worker_id}: Completed chunk {chunk.chunk_id} (total chunks: {chunks_processed})"
                worker_logger.info(complete_msg)

            except Exception as e:
                error_msg = f"Worker {worker_id}: Error processing chunk {chunk.chunk_id}: {e}"
                worker_logger.error(error_msg)
                import traceback
                tb = traceback.format_exc()
                worker_logger.error(f"Worker {worker_id}: Traceback: {tb}")
                if error_log_q:
                    error_log_q.put(f"{error_msg}\n{tb}")
                
                result_q.put(ResultPacket(
                    chunk_id  = chunk.chunk_id,
                    vectors   = [],
                    raw_data  = chunk.data,
                    has_error = True,
                ))

    except Exception as e:
        fatal_msg = f"Worker {worker_id}: Fatal error: {e}"
        worker_logger.error(fatal_msg)
        import traceback
        tb = traceback.format_exc()
        worker_logger.error(f"Worker {worker_id}: Fatal traceback: {tb}")
        if error_log_q:
            error_log_q.put(f"{fatal_msg}\n{tb}")
    finally:
        final_msg = f"Worker {worker_id}: Shutting down (processed {chunks_processed} chunks)"
        worker_logger.info(final_msg)
"""================= End function worker_main ================="""


"""================= Startup class WorkerPoolManager ================="""
class WorkerPoolManager:
    """
    Manages worker process lifecycle.
    Uses 'spawn' context for CUDA safety.
    """

    """================= Startup method __init__ ================="""
    def __init__(self):
        self.processes: List[multiprocessing.Process] = []
    """================= End method __init__ ================="""

    """================= Startup method spawn_workers ================="""
    def spawn_workers(self, config, strategy,
                      work_q:   multiprocessing.Queue,
                      result_q: multiprocessing.Queue,
                      stats_q:  multiprocessing.Queue,
                      error_log_q: multiprocessing.Queue = None) -> List[multiprocessing.Process]:
        """
        Spawns config.num_workers independent processes.
        Returns list of Process objects.
        """
        import os
        
        # Use spawn context for both Windows and Unix for consistency
        # Spawn is safer for complex libraries like sentence-transformers
        ctx = multiprocessing.get_context('spawn')
        
        # Adjust worker count for Windows
        if os.name == 'nt':  # Windows
            # Use 2 workers on Windows for better performance while avoiding conflicts
            actual_workers = min(2, config.num_workers)
            logger.info(f"Windows detected: Using {actual_workers} workers with spawn context")
        else:
            # On Unix systems, use the configured number of workers
            actual_workers = config.num_workers
            logger.info(f"Unix system: Using {actual_workers} workers with spawn context")
        
        # CRITICAL FIX: Update config to reflect actual worker count
        # This ensures the orchestrator sends the correct number of shutdown signals
        config.num_workers = actual_workers
            
        processes = []

        for i in range(actual_workers):
            proc = ctx.Process(
                target = worker_main,
                args   = (i, config, strategy, work_q, result_q, stats_q, error_log_q),
                name   = f"haup-worker-{i}",
            )
            proc.start()
            processes.append(proc)
            logger.info(f"Started worker process {i} (PID: {proc.pid})")

        # Small delay to ensure workers are ready before chunks are sent
        import time
        time.sleep(0.5)

        self.processes = processes
        return processes
    """================= End method spawn_workers ================="""

    """================= Startup method shutdown ================="""
    def shutdown(self, work_q: multiprocessing.Queue, timeout: int = 30) -> None:
        """Gracefully stops all worker processes."""
        work_q.put(SHUTDOWN_SIGNAL)
        for proc in self.processes:
            proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
    """================= End method shutdown ================="""

"""================= End class WorkerPoolManager ================="""


if __name__ == "__main__":
    sizer = AdaptiveBatchSizer(64)
    sizer.feedback("SUCCESS", 0.5)
    print(f"After SUCCESS: {sizer.current}")
    sizer.feedback("OOM")
    print(f"After OOM    : {sizer.current}")