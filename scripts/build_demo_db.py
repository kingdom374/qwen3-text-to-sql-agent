"""Create a small SQLite fixture with predictable query results."""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path


DEMO_DB = Path(__file__).resolve().parents[1] / "data" / "sample_db" / "sales.sqlite"


def build_demo_db():
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    DEMO_DB.touch(exist_ok=False)
    try:
        with closing(sqlite3.connect(DEMO_DB)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL
                );
                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    product_id INTEGER NOT NULL REFERENCES products(product_id),
                    order_date TEXT NOT NULL,
                    region TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price_cents INTEGER NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO products VALUES (?, ?, ?)",
                [(1, "咖啡", "饮品"), (2, "饼干", "零食"), (3, "茶", "饮品")],
            )
            connection.executemany(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, 1, "2025-04-12", "华东", 5, 2000),
                    (2, 2, "2025-05-02", "华北", 8, 1500),
                    (3, 3, "2025-06-10", "华东", 4, 1200),
                    (4, 1, "2026-04-12", "华东", 3, 2000),
                    (5, 2, "2026-05-02", "华北", 6, 1500),
                    (6, 3, "2026-06-10", "华东", 2, 1200),
                ],
            )
    except (Exception, KeyboardInterrupt):
        DEMO_DB.unlink()
        raise


if __name__ == "__main__":
    try:
        build_demo_db()
    except FileExistsError:
        sys.exit(f"数据库已存在，保留原文件：{DEMO_DB}")
    print(f"演示库已创建：{DEMO_DB}")
    print("预期表名：orders, products")
