"""
File Summary:
Streams all entries from a ChromaDB collection in fixed-size pages.
Yields VectChunk objects compatible with the reverse worker pool.
Never loads the full collection into RAM.

====================================================================
Startup
====================================================================

vect_batch_reader
||
├── CollectionStats  [Class] -----------------------------> Metadata container for collection info
│
├── VectChunk  [Class] -----------------------------------> Single page of vector data
│
├── get_collection_stats()  [Function] -------------------> Open collection and return stats
│       │
│       ├── coll.count() ---------------------------------> Total entry count
│       │
│       ├── [Early Exit Branch] total == 0 ---------------> Return empty CollectionStats
│       │
│       ├── coll.peek(limit=1) ---------------------------> Sample first entry for field detection
│       │
│       └── Returns CollectionStats  [Class → Object] ----> total_entries, has_documents, sample_metadata
│
├── stream_chunks()  [Function] --------------------------> Generator: paginate ChromaDB via LIMIT/OFFSET
│       │
│       ├── coll.get(limit, offset, include) -------------> Fetch one page of entries
│       │
│       ├── [Early Exit Branch] ids empty ----------------> Break when no more entries
│       │
│       ├── Normalise docs list --------------------------> Ensure docs length matches ids length
│       │
│       ├── YIELD VectChunk  [Class → Object] ------------> chunk_id, ids, docs, metas
│       │
│       └── Increment offset + chunk_id ------------------> Advance to next page
│
└── total_chunks()  [Function] ---------------------------> Calculate total chunk count for progress tracking
        │
        └── [Early Exit Branch] total_entries == 0 -------> Return 0

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


"""================= Startup class CollectionStats ================="""
@dataclass
class CollectionStats:
    total_entries:   int
    has_documents:   bool
    sample_metadata: dict
    collection_name: str
"""================= End class CollectionStats ================="""


"""================= Startup class VectChunk ================="""
@dataclass
class VectChunk:
    chunk_id: int
    ids:      list[str]
    docs:     list[Optional[str]]
    metas:    list[dict]
"""================= End class VectChunk ================="""


"""================= Startup function get_collection_stats ================="""
def get_collection_stats(client, collection_name: str) -> CollectionStats:
    """
    Opens the ChromaDB collection and returns stats without reading all data.
    client: chromadb.PersistentClient (or any chromadb client)
    """
    coll  = client.get_collection(name=collection_name)
    total = coll.count()

    if total == 0:
        return CollectionStats(
            total_entries   = 0,
            has_documents   = False,
            sample_metadata = {},
            collection_name = collection_name,
        )

    # Peek at the first entry to learn what fields are stored
    peek  = coll.peek(limit=1)
    docs  = peek.get("documents") or []
    metas = peek.get("metadatas") or [{}]

    has_documents = bool(docs) and docs[0] is not None

    stats = CollectionStats(
        total_entries   = total,
        has_documents   = has_documents,
        sample_metadata = metas[0] if metas else {},
        collection_name = collection_name,
    )

    logger.info(
        "[VectBatchReader] collection=%s  entries=%d  has_documents=%s",
        collection_name, total, has_documents,
    )
    print(
        f"[VectBatchReader]  {collection_name}\n"
        f"  entries      : {total:,}\n"
        f"  has_documents: {has_documents}\n"
    )
    return stats
"""================= End function get_collection_stats ================="""


"""================= Startup function stream_chunks ================="""
def stream_chunks(
    client,
    collection_name:    str,
    chunk_size:         int,
    include_embeddings: bool = False,
) -> Generator[VectChunk, None, None]:
    """
    Generator — yields VectChunk objects one page at a time.
    Uses ChromaDB coll.get(limit, offset) pagination.
    Never loads the full collection into RAM.
    """
    coll = client.get_collection(name=collection_name)

    include_fields = ["metadatas", "documents"]
    if include_embeddings:
        include_fields.append("embeddings")

    offset   = 0
    chunk_id = 0

    while True:
        result = coll.get(
            limit   = chunk_size,
            offset  = offset,
            include = include_fields,
        )

        ids = result.get("ids") or []
        if not ids:
            break

        docs  = result.get("documents") or [None] * len(ids)
        metas = result.get("metadatas") or [{}]   * len(ids)

        # Normalise: ensure docs list is same length as ids
        if len(docs) != len(ids):
            docs = [None] * len(ids)

        yield VectChunk(
            chunk_id = chunk_id,
            ids      = ids,
            docs     = docs,
            metas    = metas,
        )

        offset   += chunk_size
        chunk_id += 1

        logger.debug(
            "[VectBatchReader] yielded chunk %d  offset=%d", chunk_id - 1, offset
        )
"""================= End function stream_chunks ================="""


"""================= Startup function total_chunks ================="""
def total_chunks(stats: CollectionStats, chunk_size: int) -> int:
    """Returns total number of chunks for progress tracking."""
    if stats.total_entries == 0:
        return 0
    return (stats.total_entries + chunk_size - 1) // chunk_size
"""================= End function total_chunks ================="""