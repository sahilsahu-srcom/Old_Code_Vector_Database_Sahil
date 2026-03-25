"""
File Summary:
Reads haup_checkpoint.db written by the forward pipeline.
Recovers SchemaStrategy and infers SQL column types from sample docs.

====================================================================
Startup
====================================================================

schema_loader
||
├── load()  [Function] -----------------------------------> Load SchemaStrategy from checkpoint DB
│       │
│       ├── Path existence check -------------------------> Raise MissingCheckpointError if missing
│       │
│       ├── Open haup_checkpoint.db ----------------------> SQLite connection
│       │
│       ├── Query schema_strategy table ------------------> Get latest saved strategy JSON
│       │
│       ├── [Early Exit Branch] row is None --------------> Raise MissingCheckpointError
│       │
│       ├── json.loads() ---------------------------------> Parse strategy JSON
│       │
│       ├── _parse_json_list()  [Function] ---------------> Parse each column list field
│       │
│       ├── _dedupe_ordered()  [Function] ----------------> Build all_cols without duplicates
│       │
│       └── Returns SchemaStrategy  [Class → Object] ----> Populated strategy dataclass
│
├── infer_sql_types()  [Function] -----------------------> Infer SQL types from sample documents
│       │
│       ├── heuristic_parse() ---------------------------> Parse each sample doc to row dict
│       │
│       ├── Collect values per column --------------------> col_samples dict
│       │
│       ├── [Conditional Branch] numeric_cols -----------> _infer_numeric_type()
│       │
│       ├── [Conditional Branch] other cols -------------> _infer_text_type()
│       │
│       └── Returns TypeMap  [Class → Object] ----------> mapping, col_types, final_cols
│
├── _infer_numeric_type()  [Function] -------------------> Detect INTEGER or REAL from samples
│
├── _infer_text_type()  [Function] ----------------------> Detect TEXT or DATE from samples
│
└── _dedupe_ordered()  [Function] -----------------------> Remove duplicates preserving order

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


"""================= Startup class SchemaStrategy ================="""
@dataclass
class SchemaStrategy:
    rowid_col:     str
    template:      str
    template_mode: str
    semantic_cols: list
    numeric_cols:  list
    id_cols:       list
    skip_cols:     list
    all_cols:      list
    date_cols:     list = field(default_factory=list)
"""================= End class SchemaStrategy ================="""


"""================= Startup class TypeMap ================="""
@dataclass
class TypeMap:
    """Maps column name -> SQL type string."""
    mapping:    dict = field(default_factory=dict)
    col_types:  list = field(default_factory=list)
    final_cols: list = field(default_factory=list)

    def get(self, col: str, default: str = "TEXT") -> str:
        return self.mapping.get(col, default)
"""================= End class TypeMap ================="""


"""================= Startup class MissingCheckpointError ================="""
class MissingCheckpointError(RuntimeError):
    pass
"""================= End class MissingCheckpointError ================="""


"""================= Startup function load ================="""
def load(checkpoint_db_path: str | Path) -> SchemaStrategy:
    """
    Load SchemaStrategy from haup_checkpoint.db produced by the forward pipeline.
    Raises MissingCheckpointError if the file does not exist or has no saved strategy.
    """
    path = Path(checkpoint_db_path)
    if not path.exists():
        raise MissingCheckpointError(
            f"Checkpoint not found: {path}\n"
            "Run the forward pipeline first to generate haup_checkpoint.db."
        )

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT strategy_json
            FROM   schema_strategy
            ORDER  BY created_at DESC
            LIMIT  1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        conn.close()

    if row is None or not row["strategy_json"]:
        raise MissingCheckpointError(
            f"schema_strategy table empty or missing in {path}.\n"
            "Run the forward pipeline first."
        )

    try:
        data = json.loads(row["strategy_json"])
    except (json.JSONDecodeError, TypeError) as e:
        raise MissingCheckpointError(
            f"Invalid strategy_json in {path}: {e}\n"
            "The checkpoint database may be corrupted."
        )

    def _parse_json_list(val: Optional[str | list]) -> list[str]:
        if not val:
            return []
        if isinstance(val, list):
            return val
        try:
            result = json.loads(val)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, TypeError):
            return [c.strip() for c in str(val).split(",") if c.strip()]

    semantic_cols = _parse_json_list(data.get("semantic_cols"))
    numeric_cols  = _parse_json_list(data.get("numeric_cols"))
    id_cols       = _parse_json_list(data.get("id_cols"))
    skip_cols     = _parse_json_list(data.get("skip_cols"))

    all_cols = _dedupe_ordered(semantic_cols + numeric_cols + id_cols)

    strategy = SchemaStrategy(
        rowid_col     = data.get("rowid_col", "id"),
        template      = data.get("template", ""),
        template_mode = data.get("template_mode", "heuristic"),
        semantic_cols = semantic_cols,
        numeric_cols  = numeric_cols,
        id_cols       = id_cols,
        skip_cols     = skip_cols,
        all_cols      = all_cols,
    )

    logger.info(
        "[SchemaLoader] rowid_col=%s  mode=%s  cols=%d",
        strategy.rowid_col, strategy.template_mode, len(all_cols),
    )
    print(
        f"[SchemaLoader]\n"
        f"  rowid_col    : {strategy.rowid_col}\n"
        f"  template_mode: {strategy.template_mode}\n"
        f"  columns      : {len(all_cols)}\n"
    )
    return strategy
"""================= End function load ================="""


"""================= Startup function infer_sql_types ================="""
def infer_sql_types(
    strategy:     SchemaStrategy,
    sample_docs:  list[Optional[str]],
    max_samples:  int = 200,
    source_conn   = None,   # PostgreSQL/Neon connection for direct type inference
    source_table: str = "",
) -> TypeMap:
    """
    Inspect up to max_samples document strings to guess SQL column types.
    Type priority: INTEGER > REAL > DATE > TEXT.
    Always falls back to TEXT (safe).

    If source_conn is provided, fetch actual column types from the Neon/PostgreSQL
    source database instead of inferring from document strings.
    """
    from reverse_core.text_filter.heuristic_parser import parse as heuristic_parse

    if source_conn and source_table:
        logger.info("[SchemaLoader] Fetching column types from Neon/PostgreSQL source")
        try:
            mapping    = _get_source_column_types(source_conn, source_table, strategy.all_cols)
            col_types  = [(col, mapping.get(col, "TEXT")) for col in strategy.all_cols]
            final_cols = list(strategy.all_cols)
            type_map   = TypeMap(mapping=mapping, col_types=col_types, final_cols=final_cols)
            logger.info("[SchemaLoader] inferred types from source: %s", mapping)
            return type_map
        except Exception as e:
            logger.warning(
                "[SchemaLoader] Failed to get source types: %s — falling back to document inference", e
            )

    col_samples: dict[str, list[str]] = {col: [] for col in strategy.all_cols}

    for doc in sample_docs[:max_samples]:
        if not doc:
            continue
        try:
            if doc.strip().startswith("{"):
                row = json.loads(doc)
            else:
                row = heuristic_parse(doc, strategy) or {}
        except Exception:
            row = {}

        for col in strategy.all_cols:
            val = row.get(col)
            if val is not None and str(val).strip():
                col_samples[col].append(str(val).strip())

    mapping: dict[str, str] = {}
    for col in strategy.all_cols:
        samples = col_samples[col]
        if col in strategy.numeric_cols:
            mapping[col] = _infer_numeric_type(samples)
        else:
            mapping[col] = _infer_text_type(samples)

    col_types  = [(col, mapping.get(col, "TEXT")) for col in strategy.all_cols]
    final_cols = list(strategy.all_cols)
    type_map   = TypeMap(mapping=mapping, col_types=col_types, final_cols=final_cols)
    logger.info("[SchemaLoader] inferred types: %s", mapping)
    return type_map
"""================= End function infer_sql_types ================="""


"""================= Startup function _get_source_column_types ================="""
def _get_source_column_types(conn, table: str, columns: list[str]) -> dict[str, str]:
    """
    Fetch actual column types from the Neon/PostgreSQL source database via
    information_schema.columns.  Returns mapping of column_name -> SQL type
    (TEXT, INTEGER, REAL, etc.).
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT column_name, data_type
        FROM   information_schema.columns
        WHERE  table_name = %s
        ORDER  BY ordinal_position
        """,
        (table,),
    )
    rows = cursor.fetchall()
    cursor.close()

    pg_to_sql: dict[str, str] = {
        "integer":                     "INTEGER",
        "bigint":                      "INTEGER",
        "smallint":                    "INTEGER",
        "double precision":            "REAL",
        "real":                        "REAL",
        "numeric":                     "REAL",
        "decimal":                     "REAL",
        "boolean":                     "INTEGER",
        "text":                        "TEXT",
        "character varying":           "TEXT",
        "character":                   "TEXT",
        "uuid":                        "TEXT",
        "date":                        "TEXT",
        "timestamp without time zone": "TEXT",
        "timestamp with time zone":    "TEXT",
    }

    mapping: dict[str, str] = {}
    for col_name, data_type in rows:
        if col_name in columns:
            mapping[col_name] = pg_to_sql.get(data_type.lower(), "TEXT")

    # Default TEXT for any column not found in information_schema
    for col in columns:
        if col not in mapping:
            mapping[col] = "TEXT"

    return mapping
"""================= End function _get_source_column_types ================="""


_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"
)


"""================= Startup function _infer_numeric_type ================="""
def _infer_numeric_type(samples: list[str]) -> str:
    if not samples:
        return "REAL"
    has_float = any("." in s for s in samples)
    try:
        for s in samples:
            float(s)
        return "REAL" if has_float else "INTEGER"
    except ValueError:
        return "TEXT"
"""================= End function _infer_numeric_type ================="""


"""================= Startup function _infer_text_type ================="""
def _infer_text_type(samples: list[str]) -> str:
    if not samples:
        return "TEXT"
    date_hits = sum(1 for s in samples if _DATE_RE.match(s))
    if date_hits / len(samples) > 0.9:
        return "TEXT"
    return "TEXT"
"""================= End function _infer_text_type ================="""


"""================= Startup function _dedupe_ordered ================="""
def _dedupe_ordered(lst: list[str]) -> list[str]:
    seen   = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
"""================= End function _dedupe_ordered ================="""