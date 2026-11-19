"""Read table metadata from the local SQLite demo database."""

import sqlite3
from contextlib import closing

from build_demo_db import DEMO_DB


CATEGORY_VALUE_QUERIES = {
    "orders": ("region", "SELECT DISTINCT region FROM orders ORDER BY region LIMIT 10"),
    "products": ("category", "SELECT DISTINCT category FROM products ORDER BY category LIMIT 10"),
}


GET_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_schema",
        "description": "查询指定 SQLite 表的字段、类型、主键和外键。",
        "parameters": {
            "type": "object",
            "properties": {
                "table_names": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            },
            "required": ["table_names"],
            "additionalProperties": False,
        },
    },
}


def get_schema(table_names):
    if not isinstance(table_names, list) or not table_names or any(
        not isinstance(name, str) or not name.strip() for name in table_names
    ):
        raise ValueError("table_names 必须是非空的表名字符串数组")

    uri = DEMO_DB.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        available_tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT GLOB 'sqlite_*'"
            )
        }
        unknown_tables = sorted(set(table_names) - available_tables)
        if unknown_tables:
            raise ValueError(f"演示库中没有这些表：{unknown_tables}")

        tables = {}
        for table_name in dict.fromkeys(table_names):
            columns = connection.execute(
                'SELECT name, type, "notnull" AS not_null, pk AS primary_key_position '
                "FROM pragma_table_info(?) ORDER BY cid",
                (table_name,),
            )
            foreign_keys = connection.execute(
                'SELECT id, seq, "from" AS from_column, "table" AS to_table, "to" AS to_column '
                "FROM pragma_foreign_key_list(?) ORDER BY id, seq",
                (table_name,),
            )
            tables[table_name] = {
                "columns": [dict(column) for column in columns],
                "foreign_keys": [dict(key) for key in foreign_keys],
            }
            if table_name in CATEGORY_VALUE_QUERIES:
                column_name, query = CATEGORY_VALUE_QUERIES[table_name]
                tables[table_name]["sample_values"] = {
                    column_name: [value for (value,) in connection.execute(query)]
                }
    return {"tables": tables}
