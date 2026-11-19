"""Check the SQLite query tool with fixed SQL and expected results."""

import json
import unittest

from build_demo_db import DEMO_DB
from sqlite_query import execute_sql


class SqliteQuerySmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEMO_DB.is_file():
            raise FileNotFoundError(f"请先运行 build_demo_db.py 创建演示库：{DEMO_DB}")

    def test_sales_total_in_cents(self):
        result = execute_sql(
            """
            SELECT SUM(quantity * unit_price_cents) AS sales_cents
            FROM orders
            WHERE region = '华东'
              AND order_date >= '2026-04-01'
              AND order_date < '2026-07-01'
            """
        )
        print("\n实际销售额查询结果", json.dumps(result, ensure_ascii=False), flush=True)
        self.assertEqual(
            result,
            {"status": "ok", "columns": ["sales_cents"], "rows": [[8400]], "truncated": False},
        )

    def test_write_is_blocked(self):
        result = execute_sql("UPDATE orders SET quantity = 0")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(execute_sql("SELECT quantity FROM orders WHERE order_id = 1")["rows"], [[5]])

    def test_multiple_statements_are_rejected(self):
        result = execute_sql("SELECT order_id FROM orders; SELECT product_id FROM products")
        self.assertEqual(result["status"], "sql_error")

    def test_missing_column_is_reported(self):
        result = execute_sql("SELECT missing_column FROM orders")
        self.assertEqual(result["status"], "sql_error")
        self.assertIn("missing_column", result["error"])

    def test_empty_result_preserves_columns(self):
        result = execute_sql("SELECT order_id FROM orders WHERE order_id = 0")
        self.assertEqual(result, {"status": "ok", "columns": ["order_id"], "rows": [], "truncated": False})

    def test_preview_marks_truncated_results(self):
        result = execute_sql("SELECT order_id FROM orders ORDER BY order_id", max_rows=2)
        self.assertEqual(result, {"status": "ok", "columns": ["order_id"], "rows": [[1], [2]], "truncated": True})

    def test_slow_query_is_interrupted(self):
        result = execute_sql(
            """
            WITH RECURSIVE counter(n) AS (
                SELECT 1 UNION ALL SELECT n + 1 FROM counter WHERE n < 10000000
            )
            SELECT SUM(n) FROM counter
            """,
            timeout_seconds=0.02,
        )
        self.assertEqual(result["status"], "timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
