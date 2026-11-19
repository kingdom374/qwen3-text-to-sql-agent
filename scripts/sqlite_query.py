"""Execute bounded, read-only queries against the local demo database."""

import sqlite3
from contextlib import closing
from time import monotonic

from build_demo_db import DEMO_DB


DEMO_TABLES = {"orders", "products"}


def execute_sql(query, *, max_rows=20, timeout_seconds=2.0):
    if not isinstance(query, str) or not query.strip():
        return {"status": "invalid_query", "error": "query 必须是非空 SQL 字符串"}
    if not 1 <= max_rows <= 20 or timeout_seconds <= 0:
        raise ValueError("max_rows 须在 1 到 20 之间，timeout_seconds 须大于 0")

    rejected = False

    def authorize(action, argument1, argument2, database, source):
        nonlocal rejected
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if database == "main":
                return sqlite3.SQLITE_OK
            # SQLite can omit the database name for table-level reads.
            if database is None and argument2 == "" and argument1 in DEMO_TABLES:
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (argument2 or "").lower() not in (
            "load_extension", "readfile", "writefile"
        ):
            return sqlite3.SQLITE_OK
        rejected = True
        return sqlite3.SQLITE_DENY

    try:
        with closing(sqlite3.connect(DEMO_DB.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.set_authorizer(authorize)
            deadline = monotonic() + timeout_seconds
            connection.set_progress_handler(lambda: int(monotonic() >= deadline), 1000)

            cursor = connection.execute(query)
            if cursor.description is None:
                return {"status": "invalid_query", "error": "SQL 没有返回结果列"}
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as error:
        if rejected:
            return {"status": "rejected", "error": "只允许对演示库执行单条只读查询"}
        if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
            return {"status": "timeout", "error": "SQLite 查询超过执行时限"}
        return {"status": "sql_error", "error": str(error)}

    return {
        "status": "ok",
        "columns": columns,
        "rows": [
            [{"blob_hex": value.hex()} if isinstance(value, bytes) else value for value in row]
            for row in rows[:max_rows]
        ],
        "truncated": len(rows) > max_rows,
    }
