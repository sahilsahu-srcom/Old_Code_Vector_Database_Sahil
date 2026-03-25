"""
File Summary:
Schema reconciler for HAUP v3.0 Reverse Pipeline. Builds the final agreed Schema
for the output SQL table or Excel workbook. Merges checkpoint column order with
sample row discovery, strips internal tags, and assigns SQL types.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

reconcile()  [Function]
||
├── Step 1: base column list from strategy.all_cols (checkpoint-defined order)
│
├── Step 2: discover extra columns from sample_rows (schema drift handling)
│       │
│       └── [Conditional Branch] sample_rows provided --> Extend base_cols with new cols
│
├── Step 3: strip internal tags (__rowid__, __orig_rowid__)
│
├── Step 4: assign SQL types from type_map (default TEXT)
│
└── Returns Schema  [Class → Object] ---------------------> final_cols, col_types, output_cols

Schema()  [Class → Object]
||
├── sql_col_defs()  [Function] ---------------------------> Generate column definitions for CREATE TABLE
│
├── placeholders()  [Function] ---------------------------> Return "%s, %s, ..." for parameterised INSERT
│
└── insert_cols()  [Function] ----------------------------> Return quoted column name string for INSERT

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)


"""================= Startup class Schema ================="""
@dataclass
class Schema:
    """Final agreed schema for the output PostgreSQL table or xlsx workbook."""
    final_cols:  list[str]
    col_types:   list[tuple[str, str]]
    output_cols: list[str]         = field(default_factory=list)
    constraints: Optional[Any]     = None

    """================= Startup method sql_col_defs ================="""
    def sql_col_defs(self) -> str:
        """
        Returns '"col1" TEXT, "col2" INTEGER, ...' for CREATE TABLE.

        Only UNIQUE and DEFAULT are copied from source constraints.
        NOT NULL is intentionally omitted: extracted rows may have null
        values for columns that could not be recovered from document text.
        
        UNIQUE constraints are also skipped for columns that may have nulls,
        as this can cause insertion failures when data is incomplete.
        """
        defs = []
        for col, typ in self.col_types:
            parts = [f'"{col}" {typ}']

            if self.constraints:
                constraint = self.constraints.get(col)
                if constraint:
                    # Skip UNIQUE constraint to avoid conflicts with null values
                    # if constraint.is_unique:
                    #     parts.append("UNIQUE")
                    if constraint.default_value is not None:
                        parts.append(f"DEFAULT {constraint.default_value}")

            defs.append(" ".join(parts))
        return ", ".join(defs)
    """================= End method sql_col_defs ================="""

    """================= Startup method placeholders ================="""
    def placeholders(self) -> str:
        """Returns '%s, %s, ...' for parameterised PostgreSQL INSERT."""
        return ", ".join(["%s"] * len(self.final_cols))
    """================= End method placeholders ================="""

    """================= Startup method insert_cols ================="""
    def insert_cols(self) -> str:
        """Returns double-quoted column names for INSERT."""
        return ", ".join(f'"{c}"' for c in self.final_cols)
    """================= End method insert_cols ================="""

"""================= End class Schema ================="""


"""================= Startup function reconcile ================="""
def reconcile(
    strategy,
    type_map,
    sample_rows: Optional[list[dict]] = None,
) -> Schema:
    """
    Build the final Schema for the output table / workbook.

    1. Start with the checkpoint-defined column order (most authoritative).
    2. Append any extra columns seen in sample_rows (schema drift handling).
    3. Strip internal tags (__rowid__).
    4. Assign SQL types from type_map.
    """
    # ── Step 1: base column list from checkpoint ──────────────────
    base_cols = list(strategy.all_cols)

    # ── Step 2: discover extra cols from sample rows ──────────────
    if sample_rows:
        seen: set[str] = set()
        for row in sample_rows:
            seen.update(row.keys())

        internal  = {"__rowid__", "__orig_rowid__"}
        seen     -= internal
        seen     -= set(base_cols)

        if seen:
            logger.info(
                "[SchemaReconciler] %d extra cols found in sample rows: %s",
                len(seen), sorted(seen),
            )
            base_cols.extend(sorted(seen))

    # ── Step 3: strip internal tags ──────────────────────────────
    final_cols = [c for c in base_cols if not c.startswith("__")]

    # ── Step 4: assign SQL types ──────────────────────────────────
    col_types = [
        (col, type_map.get(col, "TEXT"))
        for col in final_cols
    ]

    # output_cols adds __orig_rowid__ at the end (audit column)
    output_cols = final_cols + ["__orig_rowid__"]

    schema = Schema(
        final_cols  = final_cols,
        col_types   = col_types,
        output_cols = output_cols,
    )

    logger.info(
        "[SchemaReconciler] final_cols=%d  types=%s",
        len(final_cols), dict(col_types),
    )
    print(
        f"[SchemaReconciler]\n"
        f"  columns    : {len(final_cols)}\n"
        f"  output     : {', '.join(final_cols[:5])}{', ...' if len(final_cols) > 5 else ''}\n"
    )
    return schema
"""================= End function reconcile ================="""


"""================= Startup function normalise_row ================="""
def normalise_row(row, schema) -> Optional[list]:
    """Convert row dict to ordered list aligned to schema.final_cols. Missing → None."""
    if row is None:
        return None
    return [row.get(col) for col in schema.final_cols]
"""================= End function normalise_row ================="""