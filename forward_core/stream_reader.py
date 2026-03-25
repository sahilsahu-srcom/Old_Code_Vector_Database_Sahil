"""
File Summary:
Streaming data reader for HAUP v2.0. Memory-efficient PostgreSQL data streaming
without loading all rows into RAM.

====================================================================
Startup
====================================================================

StreamReader System
||
├── FileStats  [Class] -----------------------------------> row count and columns
│
├── Chunk  [Class] --------------------------------------->  chunk_id and rows
│
└── SQLStreamReader  [Class] -----------------------------> PostgreSQL table streaming reader
        │
        ├── __init__()  [Function] ----------------------> Store connection, table name, primary key
        │
        ├── get_file_stats()  [Function] -----------------> Fetch total row count and column names in ordinal order
        │       │
        │       ├── COUNT(*) query
        │       │
        │       └── INFORMATION_SCHEMA query
        │
        └── stream_chunks()  [Function] -----------------> One batch at a time, break when no rows returned
                │
                ├── YIELD Chunk
                │
                └── [Early Exit Branch]

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, List, Any, Dict


"""================= Startup class FileStats ================="""
@dataclass
class FileStats:
    total_rows: int
    columns:    List[str]
"""================= End class FileStats ================="""


"""================= Startup class Chunk ================="""
@dataclass
class Chunk:
    chunk_id: int
    data:     List[Dict[str, Any]]
"""================= End class Chunk ================="""


"""================= Startup class SQLStreamReader ================="""
class SQLStreamReader:

    """================= Startup method __init__ ================="""
    def __init__(self, sql_connection, table_name: str, primary_key: str = "id"):
        self.conn        = sql_connection
        self.table       = table_name
        self.primary_key = primary_key
    """================= End method __init__ ================="""

    """================= Startup method get_file_stats ================="""
    def get_file_stats(self) -> FileStats:
        cursor = self.conn.cursor()

        cursor.execute(f'SELECT COUNT(*) FROM "{self.table}"')
        total_rows = cursor.fetchone()[0]

        cursor.execute(
            "SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE table_name = %s AND table_schema = CURRENT_SCHEMA() "
            "ORDER BY ordinal_position",
            (self.table,)
        )

        columns = [row[0] for row in cursor.fetchall()]

        cursor.close()
        return FileStats(total_rows=total_rows, columns=columns)
    """================= End method get_file_stats ================="""

    """================= Startup method stream_chunks ================="""
    def stream_chunks(self, chunk_size: int) -> Generator[Chunk, None, None]:
        from psycopg2.extras import RealDictCursor
        cursor   = self.conn.cursor(cursor_factory=RealDictCursor)
        offset   = 0
        chunk_id = 0

        while True:
            cursor.execute(
                f'SELECT * FROM "{self.table}" '
                f'ORDER BY "{self.primary_key}" '
                f"LIMIT {chunk_size} OFFSET {offset}"
            )
            rows = cursor.fetchall()

            if not rows:
                break

            yield Chunk(chunk_id=chunk_id, data=rows)
            offset   += chunk_size
            chunk_id += 1

        cursor.close()
    """================= End method stream_chunks ================="""

"""================= End class SQLStreamReader ================="""