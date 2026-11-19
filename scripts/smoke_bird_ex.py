"""Check BIRD EX scoring on a real Mini-Dev SQLite database."""

import json
import unittest

from bird_ex import BIRD_DIR, evaluate_sql


class BirdExecutionAccuracySmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gold_file = BIRD_DIR / "mini_dev_sqlite_gold.sql"
        with gold_file.open(encoding="utf-8-sig") as gold:
            cls.gold_sql, cls.db_id = next(line.strip().rsplit("\t", 1) for line in gold if line.strip())

    def test_gold_matches_itself(self):
        result = evaluate_sql(self.db_id, self.gold_sql, self.gold_sql)
        print("\n标准 SQL 自对照", json.dumps(result, ensure_ascii=False), flush=True)
        self.assertEqual(result["prediction"]["status"], "ok")
        self.assertEqual(result["reference"]["status"], "ok")
        self.assertEqual(result["ex"], 1)

    def test_executable_but_wrong_result(self):
        result = evaluate_sql(self.db_id, "SELECT '__bird_ex_smoke_wrong_result__'", self.gold_sql)
        self.assertEqual(result["prediction"]["status"], "ok")
        self.assertEqual(result["reference"]["status"], "ok")
        self.assertEqual(result["ex"], 0)

    def test_invalid_sql(self):
        result = evaluate_sql(self.db_id, "SELECT FROM", self.gold_sql)
        self.assertEqual(result["prediction"]["status"], "sql_error")
        self.assertEqual(result["ex"], 0)

    def test_row_order_and_duplicates(self):
        result = evaluate_sql(self.db_id, "SELECT NULL UNION ALL SELECT 1 UNION ALL SELECT 1", "SELECT 1 UNION ALL SELECT NULL")
        self.assertEqual(result["prediction"]["row_count"], 3)
        self.assertEqual(result["reference"]["row_count"], 2)
        self.assertEqual(result["ex"], 1)

    def test_column_order(self):
        result = evaluate_sql(self.db_id, "SELECT 1, 2", "SELECT 2, 1")
        self.assertEqual(result["prediction"]["status"], "ok")
        self.assertEqual(result["reference"]["status"], "ok")
        self.assertEqual(result["ex"], 0)

    def test_table_level_read_of_cte(self):
        query = "WITH cte_result AS (SELECT 7 AS value) SELECT (SELECT value FROM cte_result) FROM cte_result"
        result = evaluate_sql(self.db_id, query, "SELECT 7")
        self.assertEqual(result["prediction"]["status"], "ok")
        self.assertEqual(result["ex"], 1)

    def test_read_only_restrictions(self):
        for query, expected_status in (
            ("CREATE TABLE ex_smoke_blocked_write (id INTEGER)", "rejected"),
            ("ATTACH DATABASE ':memory:' AS aux", "rejected"),
            ("SELECT 1; SELECT 7", "sql_error"),
        ):
            with self.subTest(query=query):
                result = evaluate_sql(self.db_id, query, "SELECT 7")
                self.assertEqual(result["prediction"]["status"], expected_status)
                self.assertEqual(result["ex"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
