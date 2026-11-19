"""Evaluate SQLite predictions with BIRD Mini-Dev's result-set EX rule."""

import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from time import monotonic


BIRD_DIR = Path(__file__).resolve().parents[1] / "data" / "bird_mini_dev"
DATABASE_ROOT = BIRD_DIR / "dev_databases"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class QueryResult:
    status: str
    rows: set[tuple] | None = None
    row_count: int | None = None
    error: str | None = None

    def summary(self):
        return {"status": self.status, "row_count": self.row_count, "error": self.error}


def database_path(db_id, database_root):
    if re.fullmatch(r"[A-Za-z0-9_]+", db_id) is None:
        raise ValueError(f"无效的数据库 ID：{db_id!r}")
    root = Path(database_root).resolve()
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"SQLite 数据库不存在：{path}")
    return path


def execute_for_ex(path, sql, deadline):
    if not isinstance(sql, str) or not sql.strip():
        return QueryResult("sql_error", error="SQL 必须是非空字符串")
    remaining = deadline - monotonic()
    if remaining <= 0:
        return QueryResult("timeout", error="达到 EX 执行时限")

    denied = False

    def authorize(action, argument1, argument2, database, source):
        nonlocal denied
        if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            # SQLite uses an empty column and no database name for some table-level
            # reads, including reads from query-local CTEs.
            if database == "main" or (database is None and argument2 == "" and argument1):
                return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (argument2 or "").lower() not in {
            "load_extension", "readfile", "writefile"
        }:
            return sqlite3.SQLITE_OK
        denied = True
        return sqlite3.SQLITE_DENY

    try:
        uri = path.as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=remaining)) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.set_authorizer(authorize)
            connection.set_progress_handler(lambda: int(monotonic() >= deadline), 1000)
            cursor = connection.execute(sql)
            if cursor.description is None:
                return QueryResult("sql_error", error="SQL 没有返回结果列")

            rows = set()
            row_count = 0
            for row in cursor:
                if monotonic() >= deadline:
                    return QueryResult("timeout", error="达到 EX 执行时限")
                rows.add(row)
                row_count += 1
            if monotonic() >= deadline:
                return QueryResult("timeout", error="达到 EX 执行时限")
            return QueryResult("ok", rows, row_count)
    except (sqlite3.Error, ValueError) as error:
        if denied:
            return QueryResult("rejected", error="只允许执行只读查询")
        if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
            return QueryResult("timeout", error="达到 EX 执行时限")
        return QueryResult("sql_error", error=str(error))


def evaluate_sql(db_id, predicted_sql, gold_sql, *, database_root=DATABASE_ROOT, timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    """Return EX and the two query statuses for a single prediction."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须是有限正数")
    path = database_path(db_id, database_root)
    started = monotonic()
    deadline = started + timeout_seconds
    prediction = execute_for_ex(path, predicted_sql, deadline)
    reference = QueryResult("not_run")
    if prediction.status == "ok":
        reference = execute_for_ex(path, gold_sql, deadline)
    correct = prediction.status == reference.status == "ok" and prediction.rows == reference.rows
    return {
        "db_id": db_id,
        "ex": int(correct),
        "prediction": prediction.summary(),
        "reference": reference.summary(),
        "elapsed_seconds": monotonic() - started,
    }
