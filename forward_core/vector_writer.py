"""
File Summary:
Async Vector DB writer for HAUP v2.0. Daemon thread that buffers 2000 vectors
before each bulk upsert to ChromaDB. Decouples embedding throughput from DB I/O
latency. Marks checkpoint ONLY after confirmed DB write (crash-safe).

====================================================================
STARTUP
====================================================================

VectorWriter()  [Class → Object]
||
├── __init__()  [Method] ---------------------------------> Initialize queues, checkpoint, vector_db, strategy
│
├── start_thread()  [Method] -----------------------------> Launch daemon writer thread; returns self
│
├── stop()  [Method] -------------------------------------> Signal thread to drain buffer and exit
│
├── _writer_thread()  [Method] ---------------------------> Main writer loop
│       │
│       ├── result_q.get() -------------------------------> Receive embedding packet
│       │
│       ├── [Conditional Branch] packet.has_error --------> Mark chunk failed and skip
│       │
│       ├── build_buffer_entry() -------------------------> Assemble id, embedding, document, metadata
│       │       │
│       │       ├── format_doc_text() --------------------> Use strategy.template.format_map(row)
│       │       │
│       │       └── fallback_join_kv() -------------------> Plain text if template fails
│       │
│       ├── [Conditional Branch] buffer >= BUFFER_SIZE ---> Flush when buffer full
│       │       │
│       │       └── _flush_buffer()  [Method]
│       │
│       └── [Early Exit Branch] stop_event_set + buffer --> Final flush on shutdown
│               │
│               └── _flush_buffer()  [Method]
│
├── _flush()  [Method] -----------------------------------> Single-chunk write
│       │
│       ├── vector_db.upsert() ---------------------------> Write vectors
│       ├── checkpoint.mark_done() -----------------------> Mark chunk complete after write
│       │
│       └── [Exception Block] ----------------------------> Log error, mark chunk failed
│
├── _flush_buffer()  [Method] ---------------------------> Batch write
│       │
│       ├── deduplicate_ids() ----------------------------> Remove duplicate IDs
│       ├── vector_db.upsert() ---------------------------> Write batch to ChromaDB
│       ├── checkpoint.mark_done() -----------------------> Mark chunk_ids done
│       ├── checkpoint.mark_row_processed() --------------> Track each row
│       │
│       └── [Exception Block] ----------------------------> Log error, mark all chunks failed
│
├── _to_list()  [Method] ---------------------------------> Convert numpy array to Python list
│
└── _queue_empty()  [Method] -----------------------------> Safe check if result_q is empty
        │
        └── [Exception Block] ---------------------------> Return True on error

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import threading
import queue
import logging
from typing import List, Any

logger = logging.getLogger("haup.vector_writer")


"""================= Startup class VectorWriter ================="""
class VectorWriter:
    """
    Daemon thread that buffers vectors and writes to ChromaDB.
    Marks checkpoint ONLY after confirmed DB write (crash-safe).
    """

    BUFFER_SIZE = 2000

    """================= Startup method __init__ ================="""
    def __init__(self, result_q: queue.Queue, checkpoint, vector_db,
                 data_source_name: str = "", table_name: str = "", strategy=None):
        self.result_q           = result_q
        self.checkpoint         = checkpoint
        self.vector_db          = vector_db
        self.data_source_name   = data_source_name
        self.table_name         = table_name
        self.strategy           = strategy
        self.total_rows_written = 0
        self._stop_event        = threading.Event()
        self._thread            = None
        self._last_chunk_id     = -1
    """================= End method __init__ ================="""

    """================= Startup method start_thread ================="""
    def start_thread(self) -> "VectorWriter":
        """Launches background writer thread; returns self."""
        self._thread = threading.Thread(
            target = self._writer_thread,
            name   = "haup-vector-writer",
            daemon = True,
        )
        self._thread.start()
        return self
    """================= End method start_thread ================="""

    """================= Startup method stop ================="""
    def stop(self) -> None:
        """Signals thread to drain remaining buffer and exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=60)
    """================= End method stop ================="""

    """================= Startup method _writer_thread ================="""
    def _writer_thread(self) -> None:
        """Main writer loop running in daemon thread."""
        buffer           = []
        chunks_in_buffer = set()

        while not self._stop_event.is_set() or not self._queue_empty():
            try:
                packet = self.result_q.get(timeout=0.5)
            except Exception:
                if buffer and chunks_in_buffer:
                    self._flush_buffer(buffer, chunks_in_buffer)
                    buffer = []
                    chunks_in_buffer = set()
                continue

            self._last_chunk_id = packet.chunk_id

            if packet.has_error:
                logger.error(f"Received error packet for chunk {packet.chunk_id}")
                self.checkpoint.mark_failed(packet.chunk_id)
                continue

            chunks_in_buffer.add(packet.chunk_id)

            for vector, rowid in packet.vectors:
                doc_text = None
                if hasattr(packet, 'raw_data') and packet.raw_data:
                    for row in packet.raw_data:
                        rowid_col = self.strategy.rowid_col if self.strategy else 'id'
                        if row.get(rowid_col) == rowid:
                            try:
                                if self.strategy and hasattr(self.strategy, 'template'):
                                    doc_text = self.strategy.template.format_map(row)
                                else:
                                    parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
                                    doc_text = " | ".join(parts)
                            except (KeyError, ValueError, AttributeError):
                                parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
                                doc_text = " | ".join(parts)
                            break

                buffer.append({
                    'id'       : str(rowid),
                    'embedding': self._to_list(vector),
                    'document' : doc_text,
                    'metadata' : {
                        'rowid'         : rowid,
                        'source'        : packet.source or self.data_source_name,
                        'table_or_sheet': packet.table or self.table_name,
                    }
                })

            if len(buffer) >= self.BUFFER_SIZE:
                self._flush_buffer(buffer, chunks_in_buffer)
                buffer = []
                chunks_in_buffer = set()

        if buffer and chunks_in_buffer:
            self._flush_buffer(buffer, chunks_in_buffer)
    """================= End method _writer_thread ================="""

    """================= Startup method _flush ================="""
    def _flush(self, buffer: list, chunk_id: int) -> None:
        """
        Bulk upsert to vector DB then mark chunk done.
        Order is critical: mark_done ONLY after confirmed DB write.
        """
        try:
            documents = [r.get('document') for r in buffer]

            self.vector_db.upsert(
                ids        = [r['id']        for r in buffer],
                embeddings = [r['embedding'] for r in buffer],
                metadatas  = [r['metadata']  for r in buffer],
                documents  = documents,
            )
            self.checkpoint.mark_done(chunk_id)
            self.total_rows_written += len(buffer)
            logger.debug(f"Flushed {len(buffer)} vectors (chunk {chunk_id})")
        except Exception as exc:
            logger.error(f"Flush failed chunk {chunk_id}: {exc}")
            self.checkpoint.mark_failed(chunk_id)
    """================= End method _flush ================="""

    """================= Startup method _flush_buffer ================="""
    def _flush_buffer(self, buffer: list, chunk_ids: set) -> None:
        """
        Bulk upsert to vector DB then mark all chunks in buffer as done.
        Order is critical: mark_done ONLY after confirmed DB write.
        """
        try:
            # Deduplicate by ID to avoid ChromaDB duplicate ID errors
            seen_ids = set()
            deduplicated_buffer = []
            for item in buffer:
                item_id = item['id']
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    deduplicated_buffer.append(item)
                else:
                    logger.debug(f"Skipping duplicate ID {item_id} in buffer")
            
            if len(deduplicated_buffer) != len(buffer):
                logger.info(f"Deduplicated buffer from {len(buffer)} to {len(deduplicated_buffer)} items")
            
            documents = [r.get('document') for r in deduplicated_buffer]

            logger.debug(f"Attempting to flush {len(deduplicated_buffer)} vectors for chunks {sorted(chunk_ids)}")
            
            self.vector_db.upsert(
                ids        = [r['id']        for r in deduplicated_buffer],
                embeddings = [r['embedding'] for r in deduplicated_buffer],
                metadatas  = [r['metadata']  for r in deduplicated_buffer],
                documents  = documents,
            )
            
            logger.debug(f"Successfully upserted {len(deduplicated_buffer)} vectors to ChromaDB")
            
            for chunk_id in chunk_ids:
                self.checkpoint.mark_done(chunk_id)
                logger.debug(f"Marked chunk {chunk_id} as done")

            for item in deduplicated_buffer:
                row_id = item['metadata']['rowid']
                self.checkpoint.mark_row_processed(
                    row_id,
                    item['metadata']['source'],
                    item['metadata']['table_or_sheet']
                )

            self.total_rows_written += len(deduplicated_buffer)
            logger.info(f"Successfully flushed {len(deduplicated_buffer)} vectors (chunks {sorted(chunk_ids)})")
        except Exception as exc:
            logger.error(f"Flush failed chunks {sorted(chunk_ids)}: {exc}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            for chunk_id in chunk_ids:
                self.checkpoint.mark_failed(chunk_id)
                logger.error(f"Marked chunk {chunk_id} as failed due to flush error")
    """================= End method _flush_buffer ================="""

    """================= Startup method _to_list ================="""
    @staticmethod
    def _to_list(vector) -> list:
        """Converts numpy array to Python list."""
        try:
            return vector.tolist()
        except AttributeError:
            return list(vector)
    """================= End method _to_list ================="""

    """================= Startup method _queue_empty ================="""
    def _queue_empty(self) -> bool:
        try:
            return self.result_q.empty()
        except Exception:
            return True
    """================= End method _queue_empty ================="""

"""================= End class VectorWriter ================="""