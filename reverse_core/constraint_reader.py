"""
File Summary:
Constraint reader for HAUP v3.0 Reverse Pipeline. Reads original table schema from
PostgreSQL/Neon to preserve column constraints (UNIQUE, NOT NULL, DEFAULT, etc.)
when creating the extracted output table.

====================================================================
Startup
====================================================================

constraint_reader
||
├── ColumnConstraint  [Class] ----------------------------> Single column constraint metadata container
│
├── TableConstraints  [Class] ----------------------------> Table-level collection of ColumnConstraints
│       │
│       ├── get()  [Function] ----------------------------> Lookup ColumnConstraint by column name
│       │
│       └── has_unique()  [Function] ---------------------> Check if a column has UNIQUE constraint
│
├── read_postgresql_constraints()  [Function] ------------> Read constraints from PostgreSQL/Neon
│       │
│       ├── Query information_schema.columns -------------> Fetch type, nullable, default info
│       │
│       ├── Query table_constraints + key_column_usage --> Determine is_unique flag
│       │
│       └── Returns TableConstraints  [Class → Object] --> Populated constraint collection
│
└── apply_constraints_to_col_defs()  [Function] ---------> Enhance SQL column definitions with constraints
        │
        ├── Split col_defs string into parts -------------> Parse each column definition
        │
        ├── Extract column name (double-quoted) ---------> Identify column in constraint map
        │
        ├── [Conditional Branch] constraint found --------> Append UNIQUE / NOT NULL / DEFAULT
        │       │
        │       ├── is_unique → append "UNIQUE" ----------> Add uniqueness constraint
        │       ├── not is_nullable → append "NOT NULL" --> Add nullability constraint
        │       └── default_value → append "DEFAULT x" --> Add default value
        │
        └── Return enhanced SQL string -------------------> Rejoined column definitions

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)


"""================= Startup class ColumnConstraint ================="""
@dataclass
class ColumnConstraint:
    """Represents constraints for a single column."""
    name:          str
    data_type:     str
    is_nullable:   bool = True
    is_unique:     bool = False
    default_value: Optional[str] = None
    extra:         str = ""
"""================= End class ColumnConstraint ================="""


"""================= Startup class TableConstraints ================="""
@dataclass
class TableConstraints:
    """Collection of column constraints for a table."""
    columns: dict[str, ColumnConstraint] = field(default_factory=dict)

    """================= Startup method get ================="""
    def get(self, col_name: str) -> Optional[ColumnConstraint]:
        return self.columns.get(col_name)
    """================= End method get ================="""

    """================= Startup method has_unique ================="""
    def has_unique(self, col_name: str) -> bool:
        col = self.columns.get(col_name)
        return col.is_unique if col else False
    """================= End method has_unique ================="""

"""================= End class TableConstraints ================="""


"""================= Startup function read_postgresql_constraints ================="""
def read_postgresql_constraints(
    conn: Any,
    table_name: str,
    schema_name: str = "public",
) -> TableConstraints:
    """
    Read column constraints from PostgreSQL/Neon information_schema.

    Returns TableConstraints with:
    - data_type: PostgreSQL type (VARCHAR, INTEGER, etc.)
    - is_nullable: True if column allows NULLs
    - is_unique: True if column has a UNIQUE or PRIMARY KEY constraint
    - default_value: DEFAULT expression if any
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT column_name,
               data_type,
               is_nullable,
               column_default
        FROM   information_schema.columns
        WHERE  table_schema = %s
          AND  table_name   = %s
        ORDER  BY ordinal_position
        """,
        (schema_name, table_name),
    )

    columns: dict[str, ColumnConstraint] = {}
    for col_name, data_type, is_nullable, column_default in cursor.fetchall():
        columns[col_name] = ColumnConstraint(
            name          = col_name,
            data_type     = data_type,
            is_nullable   = (is_nullable == "YES"),
            is_unique     = False,   # updated below
            default_value = column_default,
        )

    # Identify columns covered by UNIQUE or PRIMARY KEY constraints
    cursor.execute(
        """
        SELECT kcu.column_name
        FROM   information_schema.table_constraints  tc
        JOIN   information_schema.key_column_usage   kcu
               ON  tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
        WHERE  tc.table_schema   = %s
          AND  tc.table_name     = %s
          AND  tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
        """,
        (schema_name, table_name),
    )

    for (col_name,) in cursor.fetchall():
        if col_name in columns:
            columns[col_name].is_unique = True

    cursor.close()

    logger.info(
        "[ConstraintReader] Read %d columns from PostgreSQL table %s.%s",
        len(columns), schema_name, table_name,
    )

    return TableConstraints(columns=columns)
"""================= End function read_postgresql_constraints ================="""


"""================= Startup function apply_constraints_to_col_defs ================="""
def apply_constraints_to_col_defs(
    col_defs:    str,
    constraints: TableConstraints,
) -> str:
    """
    Enhance column definitions with constraints from the original PostgreSQL table.

    Args:
        col_defs:    Base column definitions e.g. '"name" TEXT, "email" TEXT'
        constraints: TableConstraints read from the source table

    Returns:
        Enhanced column definitions with UNIQUE, NOT NULL, DEFAULT appended.
    """
    parts    = [part.strip() for part in col_defs.split(",")]
    enhanced = []

    for part in parts:
        if not part:
            continue

        # Extract column name from double-quoted identifier
        if '"' in part:
            col_name = part.split('"')[1]
        else:
            col_name = part.split()[0]

        constraint = constraints.get(col_name)
        if not constraint:
            enhanced.append(part)
            continue

        tokens = [part]
        if constraint.is_unique:
            tokens.append("UNIQUE")
        if constraint.default_value is not None:
            tokens.append(f"DEFAULT {constraint.default_value}")

        enhanced.append(" ".join(tokens))

    return ",\n                    ".join(enhanced)
"""================= End function apply_constraints_to_col_defs ================="""