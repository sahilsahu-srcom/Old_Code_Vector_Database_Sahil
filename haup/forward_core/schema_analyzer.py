"""
File Summary:
Schema analyzer for HAUP v2.0. Classifies columns and builds row-serialization templates for embeddings.

====================================================================
Startup
====================================================================

SchemaAnalyzer()
||
├── __init__()  [Function] -------------------------------> Set checkpoint_db_path, call _init_checkpoint_db()
│
├── _init_checkpoint_db()  [Function] -------------------> Create schema_strategy table if not exists
│
├── analyze()  [Function] --------------------------------> Main analysis entry point
│       │
│       ├── Detect rowid_col -----------------------------> Scan columns for primary key pattern
│       │       │
│       │       └── [Conditional Branch] no match --------> Fallback to columns[0]
│       │
│       ├── Initialize buckets ---------------------------> Empty lists for each column category
│       │
│       ├── Classify columns -----------------------------> Iterate columns, apply pattern matching
│       │       │
│       │       ├── _ID_PATTERNS match -------------------> Append to id_cols
│       │       │
│       │       ├── _DATE_PATTERNS match -----------------> Append to date_cols
│       │       │
│       │       ├── _SKIP_PATTERNS match -----------------> Append to skip_cols
│       │       │
│       │       ├── _infer_dtype() [Function] ------------> Detect value type from sample rows
│       │       │       │
│       │       │       ├── TEXT type --------------------> _avg_cardinality() check
│       │       │       │       │
│       │       │       │       ├── High cardinality -----> Append to semantic_cols
│       │       │       │       └── Low cardinality ------> Append to id_cols
│       │       │       │
│       │       │       └── NUMERIC type -----------------> Append to numeric_cols
│       │       │
│       │       └── _avg_cardinality()  [Function] -------> Unique value ratio calculation
│       │
│       ├── _build_template()  [Function] ----------------> Build row serialization template
│       │       │
│       │       └── Heuristic template -------------------> "col: {col} | ..." pipe-join
│       │
│       └── _save_strategy()  [Function] -----------------> Persist SchemaStrategy to SQLite
│               │
│               └── Returns SchemaStrategy  [Class → Object] -> Dataclass with all column buckets
│
└── load_saved_strategy()  [Function] -------------------> Load latest strategy from checkpoint DB
        │
        └── [Conditional Branch] no row found -----------> Return None

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import re
import sqlite3
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


_ID_PATTERNS   = re.compile(r'\b(id|_id|pk|rowid|uuid|serial|key|code|number|num|no)\b', re.IGNORECASE)
_DATE_PATTERNS = re.compile(r'\b(date|time|created|modified|timestamp|updated)\b', re.IGNORECASE)
_SKIP_PATTERNS = re.compile(r'(password|hash|token|blob|binary|secret)', re.IGNORECASE)
_NUMERIC_TYPES = {'int', 'float', 'decimal', 'numeric'}
_TEXT_TYPES    = {'varchar', 'text', 'nvarchar', 'string'}


"""================= Startup class SchemaStrategy ================="""
@dataclass
class SchemaStrategy:
    rowid_col:     str
    template:      str
    semantic_cols: List[str]
    id_cols:       List[str]
    date_cols:     List[str]
    skip_cols:     List[str]
    numeric_cols:  List[str]
"""================= End class SchemaStrategy ================="""


"""================= Startup class SchemaAnalyzer ================="""
class SchemaAnalyzer:

    CARDINALITY_THRESHOLD = 0.3
    # FIX: Removed unused CHECKPOINT_DB class variable — the instance always
    # uses self.checkpoint_db_path; this constant was never read anywhere.

    """================= Startup method __init__ ================="""
    def __init__(self, checkpoint_db_path: str = "haup_checkpoint.db"):
        self.checkpoint_db_path = checkpoint_db_path
        self._init_checkpoint_db()
    """================= End method __init__ ================="""

    """================= Startup method _init_checkpoint_db ================="""
    def _init_checkpoint_db(self) -> None:
        conn = sqlite3.connect(self.checkpoint_db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_strategy "
            "(id INTEGER PRIMARY KEY, strategy_json TEXT, "
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()
        conn.close()
    """================= End method _init_checkpoint_db ================="""

    """================= Startup method analyze ================="""
    def analyze(self,
                first_chunk: List[Dict[str, Any]],
                columns: List[str]) -> SchemaStrategy:

        rowid_col = None
        for col in columns:
            if _ID_PATTERNS.search(col):
                rowid_col = col
                break
        if rowid_col is None:
            rowid_col = columns[0]

        id_cols       = []
        date_cols     = []
        skip_cols     = []
        semantic_cols = []
        numeric_cols  = []

        for col in columns:
            if _ID_PATTERNS.search(col):
                id_cols.append(col)
            elif _DATE_PATTERNS.search(col):
                date_cols.append(col)
            elif _SKIP_PATTERNS.search(col):
                skip_cols.append(col)
            else:
                dtype = self._infer_dtype(col, first_chunk)
                if dtype in _TEXT_TYPES:
                    if self._avg_cardinality(col, first_chunk) > self.CARDINALITY_THRESHOLD:
                        semantic_cols.append(col)
                    else:
                        id_cols.append(col)
                elif dtype in _NUMERIC_TYPES:
                    numeric_cols.append(col)

        template = self._build_template(semantic_cols, numeric_cols, id_cols, date_cols)

        strategy = SchemaStrategy(
            rowid_col     = rowid_col,
            template      = template,
            semantic_cols = semantic_cols,
            id_cols       = id_cols,
            date_cols     = date_cols,
            skip_cols     = skip_cols,
            numeric_cols  = numeric_cols,
        )
        self._save_strategy(strategy)
        return strategy
    """================= End method analyze ================="""

    """================= Startup method _build_template ================="""
    def _build_template(self, semantic_cols, numeric_cols, id_cols, date_cols) -> str:
        """
        Build template including ALL columns that should be embedded or stored.
        Now includes id_cols (except rowid) to ensure data completeness for reverse extraction.
        """
        # Include semantic and numeric columns (high-value content)
        embed_cols = semantic_cols + numeric_cols
        
        # Also include id_cols (like country_code) for data completeness
        # These are important for reverse extraction even if low cardinality
        embed_cols += id_cols
        
        # Optionally include date columns as they may be useful
        embed_cols += date_cols
        
        parts = [f"{col}: {{{col}}}" for col in embed_cols]
        return " | ".join(parts)
    """================= End method _build_template ================="""

    """================= Startup method _avg_cardinality ================="""
    @staticmethod
    def _avg_cardinality(col: str, chunk) -> float:
        values = [row.get(col) for row in chunk if row.get(col) is not None]
        if not values:
            return 0.0
        return len(set(values)) / len(values)
    """================= End method _avg_cardinality ================="""

    """================= Startup method _infer_dtype ================="""
    @staticmethod
    def _infer_dtype(col: str, chunk) -> str:
        for row in chunk:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, bool):  return 'bool'
            if isinstance(val, int):   return 'int'
            if isinstance(val, float): return 'float'
            if isinstance(val, str):   return 'varchar'
            return 'unknown'
        return 'varchar'
    """================= End method _infer_dtype ================="""

    """================= Startup method _save_strategy ================="""
    def _save_strategy(self, strategy: SchemaStrategy) -> None:
        payload = json.dumps({
            "rowid_col":     strategy.rowid_col,
            "template":      strategy.template,
            "semantic_cols": strategy.semantic_cols,
            "id_cols":       strategy.id_cols,
            "date_cols":     strategy.date_cols,
            "skip_cols":     strategy.skip_cols,
            "numeric_cols":  strategy.numeric_cols,
        })
        conn = sqlite3.connect(self.checkpoint_db_path)
        conn.execute("INSERT INTO schema_strategy (strategy_json) VALUES (?)", (payload,))
        conn.commit()
        conn.close()
    """================= End method _save_strategy ================="""

    """================= Startup method load_saved_strategy ================="""
    def load_saved_strategy(self) -> Optional[SchemaStrategy]:
        conn = sqlite3.connect(self.checkpoint_db_path)
        row  = conn.execute(
            "SELECT strategy_json FROM schema_strategy ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if not row:
            return None
        return SchemaStrategy(**json.loads(row[0]))
    """================= End method load_saved_strategy ================="""

"""================= End class SchemaAnalyzer ================="""


if __name__ == "__main__":
    cols  = ["id", "product_name", "description", "price", "created_at", "password_hash"]
    chunk = [
        {"id": 1, "product_name": "Widget A", "description": "A great widget",
         "price": 9.99, "created_at": "2024-01-01", "password_hash": "abc"},
        {"id": 2, "product_name": "Gadget B", "description": "Another gadget",
         "price": 19.99, "created_at": "2024-01-02", "password_hash": "def"},
    ]
    s = SchemaAnalyzer().analyze(chunk, cols)
    print(f"RowID      : {s.rowid_col}")
    print(f"Template   : {s.template}")
    print(f"Semantic   : {s.semantic_cols}")
    print(f"Skip       : {s.skip_cols}")