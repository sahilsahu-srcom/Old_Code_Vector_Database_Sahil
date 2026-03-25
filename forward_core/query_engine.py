"""
File Summary:
Query engine for HAUP v2.0. Converts natural language queries to vectors and performs similarity search.

====================================================================
Startup
====================================================================

QueryEngine()
||
├── search()  [Function] ---------------------------------> Main search entry point
│       │
│       ├── _load_or_get_cached_model()  [Function] ------> Load or reuse cached model
│       │
│       ├── model.encode()  [Function] -------------------> Embed query into vector
│       │
│       ├── vector_db.query()  [Function] ----------------> Similarity search in ChromaDB
│       │       │
│       │       └── [Exception Block] --------------------> Log error and return empty list
│       │
│       └── _extract_results()  [Function] ---------------> Build SearchResult list from raw output
│
├── search_batch()  [Function] ---------------------------> Batch query processing
│       │
│       ├── _load_or_get_cached_model()  [Function] ------> Load or reuse cached model
│       │
│       ├── model.encode()  [Function] -------------------> Embed all queries at once
│       │
│       └── _extract_results()  [Function] ---------------> Build SearchResult list per query
│
├── _extract_results()  [Function] -----------------------> Extract SearchResult objects from raw results
│
└── _load_or_get_cached_model()  [Function] --------------> Model caching and loading with fallback
        │
        ├── [Conditional Branch] model already cached ----> Return cached model immediately
        │
        ├── SentenceTransformer() local load  ------------> Attempt local cache first
        │       │
        │       └── [Exception Block] --------------------> Warn and fall through to download
        │
        ├── SentenceTransformer() download  --------------> Attempt HuggingFace download
        │       │
        │       └── [Exception Block] --------------------> Log error and try fallback model
        │
        ├── [Conditional Branch] fallback model check ----> Try all-MiniLM-L6-v2 if different model
        │       │
        │       └── [Exception Block] --------------------> Log fallback failure
        │
        └── raise Exception  -----------------------------> No model could be loaded

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("haup.query_engine")

_cached_model      = None
_cached_model_name = ""


"""================= Startup class SearchResult ================="""
@dataclass
class SearchResult:
    rowid:          str | int
    distance:       float
    source:         str
    table_or_sheet: str
"""================= End class SearchResult ================="""


"""================= Startup class QueryEngine ================="""
class QueryEngine:

    """================= Startup method __init__ ================="""
    def __init__(self, vector_db, model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.vector_db  = vector_db
        self.model_name = model_name
        self.device     = device
    """================= End method __init__ ================="""

    """================= Startup method search ================="""
    def search(self, user_query: str, top_k: int = 10) -> List[SearchResult]:

        model        = self._load_or_get_cached_model()

        query_vector = model.encode([user_query])[0]

        try:
            raw_results = self.vector_db.query(
                query_embeddings = [query_vector.tolist()],
                n_results        = top_k,
                include          = ['metadatas', 'distances', 'embeddings'],
            )

            if not raw_results or not raw_results.get('ids') or not raw_results['ids'][0]:
                logger.warning("No results returned from ChromaDB query")
                return []

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        search_results = self._extract_results(raw_results)

        return search_results
    """================= End method search ================="""

    """================= Startup method search_batch ================="""
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[SearchResult]]:
        model = self._load_or_get_cached_model()
        query_vectors = model.encode(queries)

        results = []
        for query_vector in query_vectors:
            raw_results = self.vector_db.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                include=['metadatas', 'distances', 'embeddings'],
            )
            search_results = self._extract_results(raw_results)
            results.append(search_results)
        return results
    """================= End method search_batch ================="""

    """================= Startup method _extract_results ================="""
    def _extract_results(self, raw_results) -> List[SearchResult]:
        search_results = []
        for i in range(len(raw_results['ids'][0])):
            rowid          = raw_results['metadatas'][0][i].get('rowid')
            distance       = raw_results['distances'][0][i]
            source         = raw_results['metadatas'][0][i].get('source', '')
            table_or_sheet = raw_results['metadatas'][0][i].get('table_or_sheet', '')

            search_results.append(SearchResult(
                rowid          = rowid,
                distance       = distance,
                source         = source,
                table_or_sheet = table_or_sheet,
            ))
        return search_results
    """================= End method _extract_results ================="""

    """================= Startup method _load_or_get_cached_model ================="""
    def _load_or_get_cached_model(self):
        global _cached_model, _cached_model_name
        if _cached_model is not None and _cached_model_name == self.model_name:
            return _cached_model

        from sentence_transformers import SentenceTransformer
        import os

        logger.info(f"Loading query model: {self.model_name} on {self.device}")

        try:
            _cached_model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=True,
                trust_remote_code=False
            )
            _cached_model_name = self.model_name
            logger.info(f"✅ Successfully loaded model from local cache: {self.model_name}")
            return _cached_model

        except Exception as e:
            logger.warning(f"Failed to load model locally: {e}")

            try:
                logger.info("Attempting to download model from HuggingFace...")
                _cached_model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=False
                )
                _cached_model_name = self.model_name
                logger.info(f"✅ Successfully downloaded and loaded model: {self.model_name}")
                return _cached_model

            except Exception as e2:
                logger.error(f"Failed to download model {self.model_name}: {e2}")

                if self.model_name != "all-MiniLM-L6-v2":
                    logger.info("Trying fallback model: all-MiniLM-L6-v2")
                    try:
                        _cached_model = SentenceTransformer(
                            "all-MiniLM-L6-v2",
                            device=self.device,
                            local_files_only=True,
                            trust_remote_code=False
                        )
                        _cached_model_name = "all-MiniLM-L6-v2"
                        logger.info("✅ Successfully loaded fallback model from local cache")
                        return _cached_model
                    except Exception as e3:
                        logger.error(f"Fallback model also failed: {e3}")

                raise Exception(f"Could not load any embedding model. Original error: {e2}")
    """================= End method _load_or_get_cached_model ================="""

"""================= End class QueryEngine ================="""


if __name__ == "__main__":
    print("QueryEngine: import OK")
    sr = SearchResult(rowid=42, distance=0.12, source="mydb", table_or_sheet="inventory")
    print(f"SearchResult : {sr}")
    print(f"Similarity   : {1 - sr.distance:.2f}")