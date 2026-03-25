"""
HAUP v3.0 — reverse_core/text_filter/__init__.py

Thin router that delegates every document to the heuristic parser.
"""

from __future__ import annotations

from typing import Optional

from .heuristic_parser import parse as heuristic_parse


def route(
    doc:      Optional[str],
    meta:     dict,
    strategy,                 # SchemaStrategy
    chromadb_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Parse one vector-DB document into a {col: value} dict.

    Handles both JSON documents (v3.0 forward pipeline) and
    pipe-separated heuristic template strings.

    Returns None if parsing fails or doc is empty.
    """
    return heuristic_parse(doc, strategy, meta, chromadb_id)


__all__ = ["route"]