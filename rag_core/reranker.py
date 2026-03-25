"""
File Summary:
Cross-encoder reranking for the HAUP RAG engine. The single biggest precision
improvement after basic vector search. A cross-encoder sees both the query and
document together, producing a much more accurate relevance score than bi-encoder
retrieval alone. Gracefully degrades to original retriever ordering if the model
is unavailable.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

Reranker()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Store config, call _load_model()
│       │
│       └── _load_model()  [Function] -------------------> Load CrossEncoder model
│               │
│               ├── [Exception Block] ImportError --------> Log warning, reranking disabled
│               │
│               └── [Exception Block] load failure ------> Log warning, fall back to vector order
│
├── rerank()  [Function] ---------------------------------> Rerank retrieved rows with cross-encoder
│       │
│       ├── [Early Exit Branch] disabled / no model -----> Return rows[:top_n] (graceful degradation)
│       │
│       ├── _row_to_text()  [Function] × N rows ---------> Convert each row to scoreable text
│       │
│       ├── model.predict(pairs) ------------------------> Score all (query, document) pairs
│       │       │
│       │       └── [Exception Block] predict fails -----> Log warning, return original order
│       │
│       ├── Sort by score descending ---------------------> Best candidates first
│       │
│       └── _sigmoid() -----------------------------------> Normalise raw score to 0-1 range
│
├── is_available()  [Function] ---------------------------> Return True if model loaded
│
├── _load_model()  [Function] ----------------------------> Load CrossEncoder from sentence-transformers
│
└── _row_to_text()  [Function] ---------------------------> Build document text from full_row or document

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from rag_core import logger as log
from rag_core.retriever import RetrievedRow


_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


"""================= Startup class Reranker ================="""
class Reranker:
    """
    Cross-encoder reranker.  Call rerank() after retrieval.

    Usage:
        reranker = Reranker()
        rows = reranker.rerank(query="active users India", rows=rows, top_n=5)
    """

    """================= Startup method __init__ ================="""
    def __init__(
        self,
        model_name: str  = _DEFAULT_MODEL,
        top_n:      int  = 5,
        enabled:    bool = True,
    ):
        self._top_n    = top_n
        self._enabled  = enabled
        self._model    = None
        self._log      = log.get("reranker")

        if enabled:
            self._load_model(model_name)
    """================= End method __init__ ================="""

    """================= Startup method rerank ================="""
    def rerank(
        self,
        query: str,
        rows:  List[RetrievedRow],
        top_n: Optional[int] = None,
    ) -> List[RetrievedRow]:
        """
        Rerank retrieved rows using the cross-encoder.

        Args:
            query:  The original user query (not expanded variants).
            rows:   Candidates from the retriever, ordered by vector similarity.
            top_n:  Override the default top_n if needed.

        Returns:
            Reranked list, best first.  Length ≤ top_n.
        """
        n = top_n or self._top_n

        if not self._enabled or self._model is None or not rows:
            return rows[:n]

        t0 = time.perf_counter()

        pairs: List[Tuple[str, str]] = []
        for row in rows:
            doc = self._row_to_text(row)
            pairs.append((query, doc))

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            self._log.warning("Reranker predict failed, using original order: %s", exc)
            return rows[:n]

        scored   = sorted(zip(scores, rows), key=lambda x: float(x[0]), reverse=True)
        reranked = [row for _, row in scored[:n]]

        for i, (score, row) in enumerate(
            sorted(zip(scores, rows), key=lambda x: float(x[0]), reverse=True)[:n]
        ):
            reranked[i].similarity = float(_sigmoid(float(score)))

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log.debug(
            "Reranked %d → %d candidates in %.0fms",
            len(rows), len(reranked), latency_ms,
        )
        return reranked
    """================= End method rerank ================="""

    """================= Startup method is_available ================="""
    def is_available(self) -> bool:
        return self._model is not None
    """================= End method is_available ================="""

    """================= Startup method _load_model ================="""
    def _load_model(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._log.info("Loading cross-encoder: %s", model_name)
            self._model = CrossEncoder(model_name, max_length=512)
            self._log.info("Cross-encoder loaded")
        except ImportError:
            self._log.warning(
                "sentence-transformers CrossEncoder not available. "
                "Install sentence-transformers>=2.6.0 to enable reranking."
            )
        except Exception as exc:
            self._log.warning(
                "Failed to load cross-encoder '%s': %s. "
                "Reranking disabled — falling back to vector similarity order.",
                model_name, exc,
            )
    """================= End method _load_model ================="""

    """================= Startup method _row_to_text ================="""
    def _row_to_text(self, row: RetrievedRow) -> str:
        """Convert a row to the text the cross-encoder will score."""
        if row.full_row:
            skip  = {"password", "password_hash", "secret", "token", "salt"}
            parts = [
                f"{k}: {v}"
                for k, v in row.full_row.items()
                if not any(s in k.lower() for s in skip)
                and v is not None
                and str(v).strip()
            ]
            return " | ".join(parts)
        return row.document or ""
    """================= End method _row_to_text ================="""

"""================= End class Reranker ================="""


"""================= Startup function _sigmoid ================="""
def _sigmoid(x: float) -> float:
    """Normalise cross-encoder raw score to 0-1 range."""
    import math
    return 1.0 / (1.0 + math.exp(-x))
"""================= End function _sigmoid ================="""


"""================= Startup class PassthroughReranker ================="""
class PassthroughReranker(Reranker):
    """
    No-op reranker that just truncates results.
    Use in tests or when you want to skip reranking without
    changing the rest of the pipeline.
    """

    """================= Startup method __init__ ================="""
    def __init__(self, top_n: int = 5):
        # Do NOT call super().__init__() to avoid loading any model
        self._top_n   = top_n
        self._enabled = False
        self._model   = None
        self._log     = log.get("reranker.passthrough")
    """================= End method __init__ ================="""

    """================= Startup method rerank ================="""
    def rerank(
        self,
        query: str,
        rows:  List[RetrievedRow],
        top_n: Optional[int] = None,
    ) -> List[RetrievedRow]:
        return rows[: top_n or self._top_n]
    """================= End method rerank ================="""

"""================= End class PassthroughReranker ================="""