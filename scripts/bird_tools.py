"""Read schema and execute bounded read-only queries in one BIRD database."""

import sqlite3
from contextlib import closing
from time import monotonic

from bird_ex import DATABASE_ROOT, DEFAULT_TIMEOUT_SECONDS, database_path, execute_for_ex


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List available tables and views in the current SQLite database.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get CREATE statements including columns and declared keys for selected tables or views.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["table_names"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Submit and execute the final read-only SQLite query. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


class BirdTools:
    def __init__(self, db_id):
        self.path = database_path(db_id, DATABASE_ROOT)
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as connection:
            self.schema = dict(connection.execute(
                "SELECT name, sql FROM main.sqlite_schema WHERE type IN ('table', 'view') "
                "AND name NOT GLOB 'sqlite_*' AND sql IS NOT NULL ORDER BY type, name"
            ))

    def list_tables(self):
        return {"tables": list(self.schema)}

    def get_schema(self, table_names):
        if not isinstance(table_names, list) or not table_names or any(
            not isinstance(name, str) or name not in self.schema for name in table_names
        ):
            raise ValueError("table_names 必须是当前数据库中有效表名的非空数组")
        return {
            "tables": {
                name: self.schema[name].rstrip().removesuffix(";") + ";"
                for name in dict.fromkeys(table_names)
            }
        }

    def execute_sql(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空 SQL 字符串")
        result = execute_for_ex(self.path, query, monotonic() + DEFAULT_TIMEOUT_SECONDS)
        # EX retains complete result sets. This tool exposes a bounded sample of
        # distinct rows, explicitly labelled so it cannot be mistaken for all rows.
        preview = []
        if result.rows is not None:
            for row in sorted(result.rows, key=repr)[:5]:
                preview.append([
                    {"blob_hex_preview": value[:128].hex(), "bytes": len(value)}
                    if isinstance(value, bytes) else value[:256] if isinstance(value, str) else value
                    for value in row
                ])
        return {
            **result.summary(),
            "distinct_row_count": len(result.rows) if result.rows is not None else None,
            "distinct_rows_preview": preview,
            "preview_row_limit": 5,
            "preview_text_limit": 256,
        }
