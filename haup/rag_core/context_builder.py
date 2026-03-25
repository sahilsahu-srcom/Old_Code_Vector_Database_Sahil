"""
File Summary:
Converts retrieved rows into a formatted, token-budgeted context string
with multiple output formats and citation tracking.

====================================================================
             STARTUP
====================================================================

ContextBuilder  [Class → Object]
||
├── __init__(cfg)  ------------------------------> Store configuration
│
├── build(rows, schema_summary)  [CORE]
│       │
│       ├── _format_row()
│       │       └── _row_data()
│       │               └── _parse_document() (fallback)
│       │
│       ├── Budget Control ----------------------> Token limit enforcement
│       │
│       ├── Output Builder
│       │       ├── _build_markdown_table()
│       │       ├── _build_json_block()
│       │       └── _build_key_value_block()
│       │
│       └── Citation Builder --------------------> Build metadata list
│
├── _format_row()
├── _row_data()
├── _build_markdown_table()
├── _build_json_block()
└── _build_key_value_block()

--------------------------------------------------------------------

Global Functions
||
├── build_schema_summary() ----------------------> Read DB schema summary
├── _trunc() ------------------------------------> String truncation
└── _parse_document() ---------------------------> Parse embedded string

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from rag_core.config import ContextConfig
from rag_core.retriever import RetrievedRow


_CHARS_PER_TOKEN = 4


"""================= Startup class ContextBuilder ================="""
class ContextBuilder:

    """================= Startup function __init__ ================="""
    def __init__(self, cfg: ContextConfig):
        self._cfg = cfg
    """================= End function __init__ ================="""

    # ── Public API ─────────────────────────────────────────────────────────

    """================= Startup function build ================="""
    def build(
        self,
        rows: List[RetrievedRow],
        schema_summary: Optional[str] = None,

    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Flow:
            Iterate rows → Apply token budget → Select rows
            → Build formatted context → Generate citations
        """
        cfg = self._cfg
        budget_chars = cfg.max_context_tokens * _CHARS_PER_TOKEN

        used_chars = 0
        selected: List[Tuple[int, RetrievedRow]] = []

        for idx, row in enumerate(rows, start=1):
            row_str = self._format_row(row, idx, cfg.row_format)
            cost = len(row_str)
            if used_chars + cost > budget_chars:
                break
            selected.append((idx, row))
            used_chars += cost

        if not selected:
            return "(No relevant records found in the database.)", []

        parts: List[str] = []

        if cfg.include_schema_summary and schema_summary:
            parts.append("### Database Schema Summary\n" + schema_summary)

        parts.append(
            f"### Retrieved Records ({len(selected)} rows, ranked by relevance)"
        )

        if cfg.row_format == "markdown_table":
            parts.append(self._build_markdown_table(selected))
        elif cfg.row_format == "json":
            parts.append(self._build_json_block(selected))
        else:
            parts.append(self._build_key_value_block(selected))

        parts.append(
            f"\n*Retrieved {len(selected)} of {len(rows)} candidate rows "
            f"(similarity threshold applied).*"
        )

        context_str = "\n\n".join(parts)

        citations = [
            {
                "index": idx,
                "rowid": row.rowid,
                "similarity": round(row.similarity, 3),
                "source": row.metadata.get("source", "unknown"),
            }
            for idx, row in selected
        ]

        return context_str, citations
    """================= End function build ================="""

    # ── Formatters ─────────────────────────────────────────────────────────

    """================= Startup function _format_row ================="""
    def _format_row(self, row: RetrievedRow, idx: int, fmt: str) -> str:
        data = self._row_data(row)
        if fmt == "markdown_table":
            values = " | ".join(
                _trunc(str(v), self._cfg.truncate_long_values)
                for v in data.values()
            )
            return f"| [{idx}] | " + values + " |"
        elif fmt == "json":
            return json.dumps({"_ref": idx, **data}, default=str, ensure_ascii=False)
        else:
            kv = "\n".join(
                f"  {k}: {_trunc(str(v), self._cfg.truncate_long_values)}"
                for k, v in data.items()
            )
            return f"[{idx}] (similarity={row.similarity:.2f})\n{kv}"
    """================= End function _format_row ================="""

    """================= Startup function _row_data ================="""
    def _row_data(self, row: RetrievedRow) -> Dict[str, Any]:
        if row.full_row:
            skip = {"password", "password_hash", "secret", "token", "salt"}
            return {
                k: v for k, v in row.full_row.items()
                if not any(s in k.lower() for s in skip)
            }
        return _parse_document(row.document)
    """================= End function _row_data ================="""

    """================= Startup function _build_markdown_table ================="""
    def _build_markdown_table(
        self, selected: List[Tuple[int, RetrievedRow]]

    ) -> str:
        if not selected:
            return ""

        all_cols: List[str] = []
        rows_data = []

        for idx, row in selected:
            data = self._row_data(row)
            rows_data.append((idx, data))
            for k in data:
                if k not in all_cols:
                    all_cols.append(k)

        header = "| # | " + " | ".join(all_cols) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(all_cols)) + " |"

        lines = [header, sep]

        for idx, data in rows_data:
            cells = [
                _trunc(str(data.get(c, "")), self._cfg.truncate_long_values)
                for c in all_cols
            ]
            lines.append(f"| [{idx}] | " + " | ".join(cells) + " |")

        return "\n".join(lines)
    """================= End function _build_markdown_table ================="""

    """================= Startup function _build_json_block ================="""
    def _build_json_block(
        self, selected: List[Tuple[int, RetrievedRow]]

    ) -> str:
        records = []
        for idx, row in selected:
            data = self._row_data(row)
            records.append({"_ref": idx, "similarity": round(row.similarity, 3), **data})
        return "```json\n" + json.dumps(records, indent=2, default=str, ensure_ascii=False) + "\n```"
    """================= End function _build_json_block ================="""

    """================= Startup function _build_key_value_block ================="""
    def _build_key_value_block(
        self, selected: List[Tuple[int, RetrievedRow]]

    ) -> str:
        blocks = []
        for idx, row in selected:
            data = self._row_data(row)
            kv = "\n".join(
                f"  {k}: {_trunc(str(v), self._cfg.truncate_long_values)}"
                for k, v in data.items()
            )
            blocks.append(f"[{idx}] Similarity={row.similarity:.2f}\n{kv}")
        return "\n\n".join(blocks)
    """================= End function _build_key_value_block ================="""

"""================= End class ContextBuilder ================="""


"""================= Startup function build_schema_summary ================="""
def build_schema_summary(checkpoint_db_path: str) -> str:
    try:
        import sqlite3, json as _json
        conn = sqlite3.connect(checkpoint_db_path)
        row = conn.execute(
            "SELECT strategy_json FROM schema_strategy ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            return ""
        s = _json.loads(row[0])
        lines = [
            f"Table columns: {', '.join(s.get('all_cols', []))}",
            f"Primary key: {s.get('rowid_col', 'id')}",
            f"Semantic columns (embedded for search): {', '.join(s.get('semantic_cols', []))}",
            f"Numeric columns: {', '.join(s.get('numeric_cols', []))}",
            f"Skipped columns (not embedded): {', '.join(s.get('skip_cols', []))}",
        ]
        return "\n".join(lines)
    except Exception:
        return ""
"""================= End function build_schema_summary ================="""


"""================= Startup function _trunc ================="""
def _trunc(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"
"""================= End function _trunc ================="""


"""================= Startup function _parse_document ================="""
def _parse_document(doc: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for part in doc.split("|"):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            result[k.strip()] = v.strip()
    return result
"""================= End function _parse_document ================="""