"""Offline contract checks using a temporary SQLite DB and scripted model replies.

No network or Qwen3 inference is used. Passing these checks does not measure EX
on BIRD; it checks the experiment's execution, isolation and accounting rules.
"""

import argparse
import copy
import csv
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack, closing, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import bird_v2_tools as value_tools
import run_bird_semantic_feedback as semantic_feedback
import run_bird_v2 as runner
from bird_ex import evaluate_sql
from bird_v2_tools import DatabaseTools


SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, county TEXT);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id),
    amount_cents INTEGER NOT NULL, order_date TEXT NOT NULL
);
CREATE TABLE "strange table" ("quoted\"\"column" TEXT);
"""
QUESTION = {"id": 0, "db_id": "v2_fixture", "question": "What is the total sales amount in cents?", "evidence": ""}
CORRECT_SQL = "SELECT SUM(amount_cents) FROM orders"
# This evaluator-only alias makes accidental gold leakage visible in requests.
GOLD_SQL = "SELECT 8400 AS evaluator_only_marker"
SALES_PLAN = {
    "outputs": [{"what": "total sales in cents", "columns": ["ORDERS.Amount_Cents"]}],
    "filters": [],
    "joins": [],
    "grain": "one grand total",
}


def write_column_csv(directory, table, rows, *, encoding="utf-8"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{table}.csv"
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "original_column_name", "column_name", "column_description", "data_format", "value_description", "",
        ])
        writer.writerows(rows)
    return path


def text_choice(text, finish="stop"):
    return {"finish_reason": finish, "message": {"role": "assistant", "content": text}}


def native_choice(name, arguments, call_id="call_1"):
    return {"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }}


def lookup_choice(call_id="lookup_1"):
    return native_choice("lookup_values", {"table": "customers", "column": "county", "search_text": "Orange County"}, call_id)


class TemporaryDatabaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v2-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / QUESTION["db_id"] / (QUESTION["db_id"] + ".sqlite")
        self.path.parent.mkdir()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript(SCHEMA)
            connection.executemany("INSERT INTO customers VALUES (?, ?, ?)", [
                (1, "Alice", "Orange"), (2, "Bob", "Los Angeles"),
                (3, "100%_literal", "COUNTY_3"), (4, "O'Reilly", "COUNTY_4"),
                (5, "VALUE_5", "COUNTY_5"), (6, "VALUE_6", "COUNTY_6"),
                (7, "VALUE_7", "COUNTY_7"), (8, "X" * 300, "COUNTY_8"),
            ])
            connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [
                (1, 1, 6000, "2026-04-01"), (2, 2, 2400, "2026-05-01"),
            ])
            connection.execute('INSERT INTO "strange table" VALUES (?)', ("quoted value",))
            connection.commit()
        self.database = DatabaseTools(self.path)


class ValueToolTest(TemporaryDatabaseTest):
    def values(self, table, column, text):
        return self.database.lookup_values(table, column, text)

    def test_grounding_uses_actual_values_and_both_substring_directions(self):
        for fragment in ("Orange County", "orange", "rang"):
            with self.subTest(fragment=fragment):
                result = self.values("CUSTOMERS", "COUNTY", fragment)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["values"], [{"value": "Orange", "text_truncated": False}])
        self.assertEqual(self.values("customers", "county", "does not exist")["values"], [])

    def test_sample_and_long_values_are_explicitly_bounded(self):
        sample = self.values("customers", "name", "")
        self.assertEqual(len(sample["values"]), 5)
        self.assertTrue(sample["more_matches"])
        long_value = self.values("customers", "name", "XXX")["values"][0]
        self.assertEqual(len(long_value["value"]), 256)
        self.assertTrue(long_value["text_truncated"])

    def test_search_is_parameterized_and_identifiers_are_validated(self):
        for fragment, expected in (("%_", "100%_literal"), ("O'Reilly", "O'Reilly")):
            with self.subTest(fragment=fragment):
                self.assertEqual(self.values("customers", "name", fragment)["values"][0]["value"], expected)
        self.assertEqual(self.values("customers", "name", "' OR 1=1 --")["values"], [])
        for table, column in (("customers; DROP TABLE orders;", "name"), ("customers", "name; DROP TABLE orders;")):
            with self.subTest(table=table, column=column), self.assertRaises(ValueError):
                self.values(table, column, "")
        self.assertEqual(self.values("strange table", 'quoted"column', "quoted")["values"][0]["value"], "quoted value")

    def test_connection_rejects_writes_and_long_search_input(self):
        with closing(self.database.connect()) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("UPDATE orders SET amount_cents = 0")
            self.assertEqual(connection.execute("SELECT SUM(amount_cents) FROM orders").fetchone(), (8400,))
        with self.assertRaises(ValueError):
            self.values("customers", "name", "x" * 129)

    def test_lookup_timeout_is_returned_as_a_tool_observation(self):
        with patch.object(value_tools, "LOOKUP_TIMEOUT_SECONDS", 0):
            self.assertEqual(self.values("customers", "name", "")["status"], "timeout")

    def test_diagnosis_reports_metadata_and_generic_dialect_without_gold(self):
        result = self.database.diagnose("no such column: c.amount_cents")
        self.assertEqual(result["error_type"], "missing_column")
        owner = result["exact_column_owners"][0]
        self.assertEqual((owner["table"], owner["column"]), ("orders", "amount_cents"))
        self.assertEqual(owner["foreign_keys"][0]["to_table"], "customers")
        typo = self.database.diagnose("no such column: amount_cent")
        self.assertIn("amount_cents", typo["similar_spellings"])
        year = self.database.diagnose("no such function: YEAR")
        self.assertIn("strftime", year["sqlite_equivalent"])
        self.assertNotIn("evaluator_only_marker", json.dumps([result, typo, year]))


REPLAY_LOG = None


def batch_choice(*choices):
    return {"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [copy.deepcopy(call) for choice in choices for call in choice["message"]["tool_calls"]],
    }}


class ColumnDescriptionTest(TemporaryDatabaseTest):
    def test_csv_preserves_encoding_and_unnamed_value_cell(self):
        source = self.root / "csv"
        path = write_column_csv(source, "customers", [
            ["county", "", "Customer’s county", "text", "", '"OC" stands for Orange County'],
        ], encoding="cp1252")
        descriptions, hashes = value_tools.load_column_descriptions(source)
        self.assertEqual(set(hashes), {path.name})
        self.assertEqual(descriptions[0][:3], ("customers", "county", "Customer’s county"))
        notes = value_tools.select_column_descriptions(
            descriptions, self.database.tables, 'List customers in "OC".', "",
        )
        self.assertEqual(notes, ['customers.county: Customer’s county | "OC" stands for Orange County'])

    def test_selector_keeps_literal_mappings_and_rejects_ungrounded_text(self):
        tables = {
            "schools": {"columns": {"StatusType": "TEXT", "category": "TEXT"}},
            "cards": {"columns": {"borderColor": "TEXT"}},
        }
        descriptions = [
            ("SCHOOLS", "statustype", "District status.",
             "Active: District is in operation.\nMerged: District has combined with another district."),
            ("cards", "borderColor", "Color of the border.", '"black", "borderless", "white"'),
            ("schools", "category", "", "'A' stands for category A"),
            ("schools", "missingColumn", "", "Merged: Not an actual column."),
            ("other", "StatusType", "", "Merged: Not an actual table."),
            ("schools", "category", "", "Historical narrative mentions merging and being borderless."),
        ]
        notes = value_tools.select_column_descriptions(
            descriptions, tables, "Find a merged district and a borderless card.", "",
        )
        self.assertCountEqual(notes, [
            'cards.borderColor: Color of the border. | "black", "borderless", "white"',
            "schools.StatusType: District status. | Merged: District has combined with another district.",
        ])

    def test_selector_honors_limits_without_truncating_notes(self):
        columns = [f"Measure{i}" for i in range(10)]
        tables = {"sales": {"columns": dict.fromkeys(columns, "TEXT")}}
        descriptions = [("sales", name, "A complete description of the recorded measure.", "") for name in columns]
        question = "Select " + ", ".join(columns)
        notes = value_tools.select_column_descriptions(descriptions, tables, question, "")
        self.assertEqual(len(notes), value_tools.COLUMN_DESCRIPTION_LIMIT)
        with patch.object(value_tools, "COLUMN_DESCRIPTION_CHAR_LIMIT", 160):
            compact = value_tools.select_column_descriptions(descriptions, tables, question, "")
        self.assertTrue(compact)
        self.assertLessEqual(len("\n".join(compact)), 160)
        self.assertEqual(compact, notes[:len(compact)])
        self.assertTrue(all(note.endswith("recorded measure.") for note in compact))


class AgentContractTest(TemporaryDatabaseTest):
    def run_scripted(
        self, mode, replies, tokens=None, *, check_explicit_filters=False, column_descriptions=None,
        use_structured_plan=False,
    ):
        """Use real DB/tools/evaluator; replace only HTTP and the schema source."""
        replies = iter(zip(replies, [20] * len(replies) if tokens is None else tokens))
        trace, requests, score_requests = io.StringIO(), [], []
        pending = self

        class ScriptedHTTP:
            def open(self, request, timeout):
                payload = json.loads(request.data)
                requests.append(payload)
                pending.assertNotIn("evaluator_only_marker", json.dumps(payload))
                choice, count = next(replies)
                return io.BytesIO(json.dumps({"choices": [choice], "usage": {
                    "prompt_tokens": 100, "completion_tokens": count,
                }}).encode("utf-8"))

        def score(question, sql):
            score_requests.append(sql)
            return evaluate_sql(question["db_id"], sql, GOLD_SQL, database_root=self.root)

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "DATABASE_ROOT", self.root))
            stack.enter_context(patch.object(runner, "read_schema", return_value=SCHEMA))
            stack.enter_context(patch.object(runner, "build_opener", return_value=ScriptedHTTP()))
            stack.enter_context(patch.object(runner, "score_prediction", side_effect=score))
            result = runner.run_question(
                QUESTION, mode, trace,
                check_explicit_filters=check_explicit_filters, column_descriptions=column_descriptions,
                use_structured_plan=use_structured_plan,
            )
        return result, requests, [json.loads(line) for line in trace.getvalue().splitlines()], score_requests

    def test_same_tool_free_generator_is_used_in_all_modes(self):
        generation_requests = []
        retrieval_requests = []
        for mode in runner.MODES:
            with self.subTest(mode=mode):
                replies = [text_choice(CORRECT_SQL)]
                if mode != "A2":
                    replies.insert(0, text_choice("NO_LOOKUP"))
                result, requests, _, scoring = self.run_scripted(mode, replies)
                self.assertEqual((result["ex"], result["agent_status"]), (1, "submitted"))
                self.assertEqual(scoring, [CORRECT_SQL])
                generation_requests.append(requests[-1])
                self.assertNotIn("tools", requests[-1])
                self.assertNotIn("tool_choice", requests[-1])
                self.assertEqual(result["sql_submission_formats"], {"plain_sql": 1})
                if mode == "A2":
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(requests[0]["max_tokens"], 1024)
                else:
                    retrieval_requests.append(requests[0])
                    self.assertEqual(requests[0]["tools"], [runner.LOOKUP_TOOL])
                    self.assertEqual([p["max_tokens"] for p in requests], [256, 1004])
        self.assertTrue(all(p["messages"] == generation_requests[0]["messages"] for p in generation_requests))
        self.assertEqual(retrieval_requests[0], retrieval_requests[1])

    def test_a2_filter_check_only_changes_the_system_message(self):
        baseline, baseline_requests, _, baseline_scoring = self.run_scripted("A2", [text_choice(CORRECT_SQL)])
        variant, variant_requests, _, variant_scoring = self.run_scripted(
            "A2", [text_choice(CORRECT_SQL)], check_explicit_filters=True,
        )
        expected = copy.deepcopy(baseline_requests)
        expected[0]["messages"][0]["content"] = runner.EXPLICIT_FILTERS_SYSTEM_PROMPT
        self.assertEqual(variant_requests, expected)
        self.assertNotEqual(variant_requests, baseline_requests)
        self.assertEqual((variant["ex"], variant["model_calls"], variant["sql_submissions"]), (1, 1, 1))
        self.assertEqual(variant_scoring, baseline_scoring)
        self.assertNotIn("check_explicit_filters", baseline)
        self.assertTrue(variant["check_explicit_filters"])

    def test_column_descriptions_only_append_to_the_a2_request(self):
        _, baseline_requests, _, baseline_scoring = self.run_scripted("A2", [text_choice(CORRECT_SQL)])
        descriptions = [("orders", "amount_cents", "Sales amount in cents.", "")]
        result, requests, events, scoring = self.run_scripted(
            "A2", [text_choice(CORRECT_SQL)], column_descriptions=descriptions,
        )
        note = "orders.amount_cents: Sales amount in cents."
        expected = copy.deepcopy(baseline_requests)
        expected[0]["messages"][1]["content"] += f"\n\n{runner.COLUMN_DESCRIPTION_HEADER}\n{note}"
        self.assertEqual(requests, expected)
        self.assertEqual(scoring, baseline_scoring)
        self.assertEqual((result["ex"], result["model_calls"]), (1, 1))
        self.assertTrue(result["use_column_descriptions"])
        self.assertEqual((result["column_description_count"], result["column_description_chars"]), (1, len(note)))
        event = next(event for event in events if event["event"] == "column_descriptions")
        self.assertEqual(event["descriptions"], [note])

        empty, requests, _, _ = self.run_scripted("A2", [text_choice(CORRECT_SQL)], column_descriptions=[])
        self.assertEqual(requests, baseline_requests)
        self.assertEqual(empty["column_description_count"], 0)

    def test_a2_variants_are_restricted_to_their_comparison_splits(self):
        cases = [
            ([flag, *arguments], flag)
            for flag in ("--check-explicit-filters", "--use-column-descriptions")
            for arguments in (["--mode", "B2"], ["--mode", "C2"], ["--mode", "A2", "--split", "held-out"])
        ]
        cases.append((["--mode", "A2", "--check-explicit-filters", "--use-column-descriptions"],
                      "--use-column-descriptions"))
        cases.extend([
            (["--mode", "A2", "--use-structured-plan"], "--use-structured-plan"),
            (["--mode", "A2", "--split", "held-out", "--use-structured-plan"], "--use-structured-plan"),
            (["--mode", "B2", "--split", "diagnostic-55"], "diagnostic-55"),
            (["--mode", "A2", "--split", "diagnostic-55", "--limit", "1"], "--limit"),
            (["--mode", "A2", "--split", "diagnostic-55", "--check-explicit-filters"], "--check-explicit-filters"),
            (["--mode", "A2", "--split", "diagnostic-55", "--fallback-from", "A2"], "--fallback-from"),
        ])
        for arguments, expected_flag in cases:
            with self.subTest(arguments=arguments), ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["run_bird_v2.py", *arguments]))
                load_questions = stack.enter_context(patch.object(runner, "load_questions"))
                stderr = stack.enter_context(redirect_stderr(io.StringIO()))
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(expected_flag, stderr.getvalue())
                load_questions.assert_not_called()

    def test_plan_uses_an_independent_request_and_shares_completion_budget(self):
        _, baseline_requests, _, baseline_scoring = self.run_scripted("A2", [text_choice(CORRECT_SQL)])
        fence = chr(96) * 3
        plan_reply = text_choice(f"{fence}json\n{json.dumps(SALES_PLAN)}\n{fence}")
        result, requests, events, scoring = self.run_scripted(
            "A2", [plan_reply, text_choice(CORRECT_SQL)], tokens=[130, 894], use_structured_plan=True,
        )
        self.assertEqual((result["model_calls"], result["sql_submissions"], result["completion_tokens"]), (2, 1, 1024))
        self.assertEqual((result["query_plan_status"], result["ex"]), ("accepted", 1))
        self.assertEqual(scoring, baseline_scoring)
        self.assertEqual([request["max_tokens"] for request in requests], [320, 894])
        self.assertEqual(requests[0]["messages"][0]["content"], runner.QUERY_PLAN_PROMPT)
        self.assertEqual(requests[0]["messages"][1], baseline_requests[0]["messages"][1])
        self.assertNotIn("tool_choice", requests[0])
        self.assertNotIn("tools", requests[0])
        plan = next(event["plan"] for event in events if event["event"] == "query_plan_accepted")
        self.assertEqual(plan["outputs"][0]["columns"], ["orders.amount_cents"])
        expected = copy.deepcopy(baseline_requests[0])
        expected["messages"][1]["content"] += f"\n\n{runner.QUERY_PLAN_HEADER}\n{json.dumps(plan, ensure_ascii=False)}"
        expected["max_tokens"] = 894
        self.assertEqual(requests[1], expected)
        self.assertEqual(result["stage_usage"][runner.PLAN_STAGE]["completion_tokens"], 130)

    def test_plan_validates_filter_and_join_columns(self):
        plan = copy.deepcopy(SALES_PLAN)
        plan["filters"] = [{"column": "CUSTOMERS.County", "condition": "Orange only"}]
        plan["joins"] = [["orders.customer_id", "CUSTOMERS.ID"]]
        validated = value_tools.validate_query_plan(json.dumps(plan), self.database.tables)
        self.assertEqual(validated["filters"], [{"column": "customers.county", "condition": "Orange only"}])
        self.assertEqual(validated["joins"], [["orders.customer_id", "customers.id"]])
        plan["joins"][0][1] = "customers.absent"
        with self.assertRaisesRegex(ValueError, "customers.absent"):
            value_tools.validate_query_plan(json.dumps(plan), self.database.tables)

    def test_invalid_plan_has_no_sql_request_or_evaluation(self):
        for reply, tokens, expected in (
            (text_choice("SELECT 1"), 20, "invalid"),
            (text_choice(json.dumps({**SALES_PLAN, "filters": [
                {"column": "unlisted.alias", "condition": "exists"},
            ]})), 50, "invalid"),
            (lookup_choice(), 20, "invalid"),
            (text_choice("{", finish="length"), 320, "truncated"),
            (text_choice(json.dumps(SALES_PLAN)), 321, "request_error"),
        ):
            with self.subTest(expected=expected, reply=reply):
                result, requests, _, scoring = self.run_scripted(
                    "A2", [reply], tokens=[tokens], use_structured_plan=True,
                )
                self.assertEqual((result["query_plan_status"], result["agent_status"]), (expected, "agent_error"))
                self.assertEqual((len(requests), result["sql_submissions"], result["ex"]), (1, 0, 0))
                self.assertEqual(scoring, [])

    def test_fallback_keeps_executable_baseline_regardless_of_ex(self):
        for sql, expected_ex in ((CORRECT_SQL, 1), ("SELECT 1", 0)):
            with self.subTest(sql=sql):
                baseline, _, _, _ = self.run_scripted("A2", [text_choice(sql)])
                with patch.object(runner, "run_question") as run_question:
                    result = runner.run_fallback_question(QUESTION, baseline, io.StringIO())
                run_question.assert_not_called()
                self.assertEqual((result["ex"], result["predicted_sql"]), (expected_ex, sql))
                self.assertEqual((result["selected_source"], result["query_plan_status"]), ("baseline", "not_attempted"))
                self.assertEqual((result["baseline_usage"]["model_calls"], result["model_calls"]), (1, 0))
                self.assertIsNone(result["fallback_attempt"])

    def test_fallback_selects_an_executable_candidate_even_if_its_ex_is_zero(self):
        baseline, _, _, _ = self.run_scripted("A2", [text_choice("SELECT absent FROM orders")])
        plan_reply = text_choice(json.dumps(SALES_PLAN))
        candidate, _, _, _ = self.run_scripted(
            "A2", [plan_reply, text_choice("SELECT 1")], use_structured_plan=True,
        )
        trace = io.StringIO()
        with patch.object(runner, "run_question", return_value=candidate) as run_question:
            result = runner.run_fallback_question(QUESTION, baseline, trace)
        run_question.assert_called_once_with(QUESTION, "A2", trace, use_structured_plan=True)
        self.assertEqual((result["selected_source"], result["predicted_sql"], result["ex"]), ("structured_plan", "SELECT 1", 0))
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["baseline_prediction_status"], "sql_error")
        decision = json.loads(trace.getvalue())
        self.assertEqual(decision["candidate_prediction_status"], "ok")
        self.assertNotIn("ex", decision)

    def test_failed_fallback_keeps_original_sql_and_attempt_details(self):
        original_sql = "SELECT absent FROM orders"
        baseline, _, _, _ = self.run_scripted("A2", [text_choice(original_sql)])
        candidate, _, _, _ = self.run_scripted(
            "A2", [text_choice("{}")], use_structured_plan=True,
        )
        with patch.object(runner, "run_question", return_value=candidate):
            result = runner.run_fallback_question(QUESTION, baseline, io.StringIO())
        self.assertEqual((result["selected_source"], result["predicted_sql"]), ("baseline", original_sql))
        self.assertEqual(result["evaluation"], baseline["evaluation"])
        self.assertEqual(result["query_plan_status"], "invalid")
        self.assertEqual(result["fallback_attempt"]["error"], candidate["error"])
        self.assertEqual(result["model_calls"], 1)

    def test_batch_lookups_reach_generator_as_data_and_share_budget(self):
        batch = batch_choice(lookup_choice("a"), native_choice(
            "lookup_values", {"table": "customers", "column": "name", "search_text": "Alice"}, "b"))
        result, requests, events, scoring = self.run_scripted(
            "B2", [batch, text_choice(CORRECT_SQL)], tokens=[100, 80])
        self.assertEqual((result["ex"], result["lookup_requests"], result["lookup_calls"], result["sql_submissions"]), (1, 2, 2, 1))
        self.assertEqual([request["max_tokens"] for request in requests], [256, 924])
        self.assertEqual((result["prompt_tokens"], result["completion_tokens"]), (200, 180))
        self.assertEqual(result["stage_usage"]["value_retrieval"]["completion_tokens"], 100)
        self.assertEqual(result["stage_usage"]["sql_generation"]["completion_tokens"], 80)
        self.assertNotIn("tools", requests[1])
        self.assertEqual([message["role"] for message in requests[1]["messages"]], ["system", "user"])
        content = requests[1]["messages"][1]["content"]
        self.assertIn('"value": "Orange"', content)
        self.assertIn('"value": "Alice"', content)
        self.assertNotIn("tool_calls", content)
        self.assertEqual(scoring, [CORRECT_SQL])
        self.assertEqual(sum(event["event"] == "lookup_result" for event in events), 2)

    def test_over_budget_and_sql_calls_are_recorded_without_executing_them(self):
        batch = batch_choice(lookup_choice("a"), lookup_choice("b"), lookup_choice("c"),
                             native_choice("execute_sql", {"query": "DELETE FROM orders"}, "d"))
        result, _, events, _ = self.run_scripted("B2", [batch, text_choice(CORRECT_SQL)])
        self.assertEqual((result["lookup_requests"], result["lookup_calls"], result["lookup_skipped_budget"]), (3, 2, 1))
        self.assertEqual(result["retrieval_calls_rejected"], 1)
        self.assertEqual((result["ex"], result["sql_submissions"]), (1, 1))
        statuses = [event["result"]["status"] for event in events if event["event"] == "retrieval_call_result"]
        self.assertEqual(statuses, ["ok", "ok", "budget_exhausted", "not_available"])
        self.assertEqual([event["query"] for event in events if event["event"] == "sql_execution"], [CORRECT_SQL])

    def test_retrieval_execute_only_call_is_never_a_candidate(self):
        result, requests, events, scoring = self.run_scripted("B2", [
            native_choice("execute_sql", {"query": "SELECT 123"}, "bad_stage"), text_choice(CORRECT_SQL)])
        self.assertEqual((result["lookup_calls"], result["sql_submissions"], result["retrieval_calls_rejected"]), (0, 1, 1))
        self.assertEqual(scoring, [CORRECT_SQL])
        self.assertNotIn("SELECT 123", json.dumps(requests[1]))
        self.assertEqual(result["ex"], 1)

    def test_malformed_retrieval_and_duplicate_ids_fall_back_to_generator(self):
        malformed = lookup_choice()
        malformed["message"]["tool_calls"][0]["function"]["arguments"] = "broken JSON"
        for choice in (malformed, batch_choice(lookup_choice("same"), lookup_choice("same")),
                       text_choice("SELECT 999"), text_choice("partial", finish="length")):
            with self.subTest(choice=choice):
                result, requests, events, _ = self.run_scripted("B2", [choice, text_choice(CORRECT_SQL)])
                self.assertEqual((result["ex"], result["lookup_calls"]), (1, 0))
                self.assertEqual((result["protocol_errors"], result["retrieval_status"]), (1, "protocol_error_fallback"))
                self.assertEqual(result["agent_status"], "submitted")
                self.assertNotIn("Database value lookup observations", requests[1]["messages"][1]["content"])
                self.assertTrue(any(e["event"] == "retrieval_fallback" for e in events))

    def test_bad_second_envelope_is_validated_before_any_lookup(self):
        malformed = lookup_choice("bad")
        malformed["message"]["tool_calls"][0]["function"]["arguments"] = "broken JSON"
        result, _, _, _ = self.run_scripted("B2", [
            batch_choice(lookup_choice("good"), malformed), text_choice(CORRECT_SQL)])
        self.assertEqual((result["lookup_calls"], result["ex"]), (0, 1))

    def test_invalid_lookup_arguments_are_recorded_then_generation_continues(self):
        invalid = native_choice("lookup_values", {"table": "customers"}, "invalid")
        unknown_table = native_choice("lookup_values", {"table": "not_a_table", "column": "x", "search_text": ""}, "unknown")
        result, _, events, _ = self.run_scripted("B2", [
            batch_choice(invalid, unknown_table), text_choice(CORRECT_SQL)])
        self.assertEqual((result["lookup_requests"], result["lookup_calls"], result["retrieval_calls_rejected"]), (2, 1, 1))
        self.assertEqual(result["ex"], 1)
        statuses = [e["result"]["status"] for e in events if e["event"] == "retrieval_call_result"]
        self.assertEqual(statuses, ["invalid_argument", "invalid_argument"])

    def test_retrieval_token_limit_and_unverifiable_usage_stop_further_requests(self):
        for count in (257, None):
            with self.subTest(count=count):
                result, requests, _, scoring = self.run_scripted("B2", [text_choice("NO_LOOKUP")], tokens=[count])
                self.assertEqual((result["agent_status"], len(requests)), ("agent_error", 1))
                self.assertEqual((result["sql_submissions"], scoring), (0, []))
        result, _, _, _ = self.run_scripted("A2", [text_choice(CORRECT_SQL)], tokens=[1025])
        self.assertEqual((result["agent_status"], result["sql_submissions"]), ("agent_error", 0))

    def test_full_retrieval_budget_still_leaves_768_tokens_for_sql(self):
        result, requests, _, _ = self.run_scripted("B2", [
            text_choice("truncated", finish="length"), text_choice(CORRECT_SQL)], tokens=[256, 50])
        self.assertEqual([p["max_tokens"] for p in requests], [256, 768])
        self.assertEqual((result["ex"], result["completion_tokens"]), (1, 306))

    def test_sql_generation_cannot_trigger_retrieval(self):
        for mode in runner.MODES:
            with self.subTest(mode=mode):
                replies = [lookup_choice()]
                if mode != "A2":
                    replies.insert(0, text_choice("NO_LOOKUP"))
                result, _, _, scoring = self.run_scripted(mode, replies)
                self.assertEqual((result["lookup_calls"], result["sql_submissions"]), (0, 0))
                self.assertEqual((result["agent_status"], result["protocol_errors"]), ("agent_error", 1))
                self.assertEqual(scoring, [])

    def test_b_stops_after_execution_error_without_feedback(self):
        result, requests, events, _ = self.run_scripted("B2", [
            text_choice("NO_LOOKUP"), text_choice("SELECT missing FROM orders")])
        self.assertEqual((result["sql_submissions"], len(requests)), (1, 2))
        self.assertEqual(result["evaluation"]["prediction"]["status"], "sql_error")
        self.assertFalse(result["repair_triggered"])
        self.assertFalse(any(event["event"] == "repair_feedback" for event in events))

    def test_c_receives_metadata_diagnosis_and_rebuilds_sql_once(self):
        bad = "SELECT SUM(c.amount_cents) FROM customers c"
        result, requests, events, scoring = self.run_scripted("C2", [
            text_choice("NO_LOOKUP"), text_choice(bad), text_choice(CORRECT_SQL)])
        self.assertEqual((result["ex"], result["sql_submissions"], len(requests)), (1, 2, 3))
        self.assertTrue(result["repair_triggered"])
        self.assertNotIn("tools", requests[1])
        self.assertNotIn("tools", requests[2])
        self.assertEqual(requests[2]["messages"][-2], {"role": "assistant", "content": bad})
        feedback = requests[2]["messages"][-1]
        self.assertEqual(feedback["role"], "user")
        self.assertIn('"failed_query": "' + bad + '"', feedback["content"])
        self.assertIn('"table": "orders"', feedback["content"])
        self.assertEqual(scoring, [CORRECT_SQL])
        self.assertEqual(sum(event["event"] == "repair_feedback" for event in events), 1)
        self.assertEqual([p["max_tokens"] for p in requests], [256, 1004, 984])

    def test_c_never_retries_successful_but_wrong_or_empty_queries(self):
        for query in ("SELECT 1", "SELECT id FROM orders WHERE 0"):
            with self.subTest(query=query):
                result, requests, _, _ = self.run_scripted("C2", [text_choice("NO_LOOKUP"), text_choice(query)])
                self.assertEqual(result["ex"], 0)
                self.assertEqual(result["first_execution"]["status"], "ok")
                self.assertEqual(len(requests), 2)
                self.assertFalse(result["repair_triggered"])

    def test_c_stops_after_one_failed_revision(self):
        result, requests, _, _ = self.run_scripted("C2", [
            text_choice("NO_LOOKUP"), text_choice("SELECT missing FROM orders"),
            text_choice("SELECT missing2 FROM orders")])
        self.assertEqual((result["sql_submissions"], len(requests)), (2, 3))
        self.assertEqual(result["evaluation"]["prediction"]["status"], "sql_error")

    def test_unchanged_revision_is_not_executed_again(self):
        query = "SELECT missing FROM orders"
        result, _, events, _ = self.run_scripted("C2", [
            text_choice("NO_LOOKUP"), text_choice(query), text_choice(query)])
        self.assertEqual((result["agent_status"], result["sql_submissions"]), ("agent_error", 1))
        self.assertTrue(any(event["event"] == "repair_unchanged" for event in events))

    def test_c_cannot_lookup_during_repair(self):
        result, requests, _, _ = self.run_scripted("C2", [
            text_choice("NO_LOOKUP"), text_choice("SELECT missing FROM orders"), lookup_choice()])
        self.assertEqual((result["lookup_calls"], result["sql_submissions"]), (0, 1))
        self.assertEqual(result["agent_status"], "agent_error")
        self.assertNotIn("tools", requests[-1])

    def test_no_repair_after_total_budget_is_exhausted(self):
        result, requests, _, _ = self.run_scripted("C2", [
            text_choice("NO_LOOKUP"), text_choice("SELECT missing FROM orders")], tokens=[20, 1004])
        self.assertEqual((result["repair_skipped"], len(requests)), ("budget_exhausted", 2))
        self.assertEqual(result["completion_tokens"], 1024)

    def test_text_function_and_fenced_sql_keep_original_output_in_trace(self):
        fence = chr(96) * 3
        for text, source in ((f"execute_sql({json.dumps(CORRECT_SQL)})", "text_function_call"),
                             (f"{fence}sql\n{CORRECT_SQL}\n{fence}", "plain_sql")):
            with self.subTest(source=source):
                result, _, events, _ = self.run_scripted("A2", [text_choice(text)])
                self.assertEqual(result["ex"], 1)
                normalized = next(event for event in events if event["event"] == "sql_format_normalization")
                self.assertEqual((normalized["source"], normalized["original_content"]), (source, text))

    def test_malformed_sql_or_executable_python_is_not_run(self):
        malformed = native_choice("execute_sql", {"query": CORRECT_SQL})
        malformed["message"]["tool_calls"][0]["function"]["arguments"] = "broken JSON"
        for choice in (malformed, text_choice('execute_sql(__import__("os").getcwd())'),
                       text_choice("SELECT 1; SELECT 2;"), text_choice("SELECT 1", finish="length")):
            with self.subTest(choice=choice):
                result, _, _, scoring = self.run_scripted("A2", [choice])
                self.assertEqual((result["protocol_errors"], result["sql_submissions"]), (1, 0))
                self.assertEqual(scoring, [])

    def test_write_sql_is_rejected_by_real_executor(self):
        result, _, _, _ = self.run_scripted("A2", [
            native_choice("execute_sql", {"query": "DELETE FROM orders"})])
        self.assertEqual(result["evaluation"]["prediction"]["status"], "rejected")
        with closing(self.database.connect()) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM orders").fetchone(), (2,))

    def test_completed_results_resume_without_new_model_calls(self):
        result, _, _, _ = self.run_scripted("A2", [text_choice(CORRECT_SQL)])
        directory = self.root / "run"
        directory.mkdir()
        with (directory / "results-previous.jsonl").open("x", encoding="utf-8") as output:
            runner.record(output, "question_result", **result)
        manifest = {"mode": "A2", "split": "smoke", "workflow_version": runner.WORKFLOW_VERSION}
        with patch.object(runner, "run_question", side_effect=AssertionError("completed question was rerun")):
            with redirect_stdout(io.StringIO()):
                status = runner.run_batch(directory, [QUESTION], manifest)
        self.assertEqual(status, 0)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["total"], summary["ex_correct"], summary["workflow_version"]),
                         (1, 1, runner.WORKFLOW_VERSION))

    def test_filter_check_run_records_prompt_and_resumes_in_its_own_directory(self):
        result, _, _, _ = self.run_scripted("A2", [text_choice(CORRECT_SQL)], check_explicit_filters=True)
        gold_file = self.root / "gold.sql"
        gold_file.write_text(GOLD_SQL, encoding="utf-8")
        log_root = self.root / "logs"
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "LOG_ROOT", log_root))
            stack.enter_context(patch.object(runner, "GOLD_PATH", gold_file))
            stack.enter_context(patch.object(runner, "load_questions", return_value=([QUESTION], "fixture")))
            stack.enter_context(patch.object(runner, "split_questions", return_value={"smoke": [QUESTION]}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_question = stack.enter_context(patch.object(runner, "run_question", return_value=result))
            with patch.object(sys, "argv", ["run_bird_v2.py", "--mode", "A2", "--check-explicit-filters"]):
                self.assertEqual(runner.main(), 0)
            (directory,) = log_root.iterdir()
            self.assertIn("explicit-filters-v1", directory.name)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["system_prompt"], runner.EXPLICIT_FILTERS_SYSTEM_PROMPT)
            self.assertTrue(manifest["check_explicit_filters"])
            with patch.object(sys, "argv", ["run_bird_v2.py", "--resume", str(directory)]):
                self.assertEqual(runner.main(), 0)
            run_question.assert_called_once()
            self.assertTrue(run_question.call_args.kwargs["check_explicit_filters"])

        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["check_explicit_filters"])
        with self.assertRaisesRegex(ValueError, "提示词配置"):
            runner.run_batch(directory, [QUESTION], {**manifest, "check_explicit_filters": False})

    def test_column_description_run_pins_csv_bytes_for_resume(self):
        descriptions = [("orders", "amount_cents", "Sales amount in cents.", "")]
        result, _, _, _ = self.run_scripted("A2", [text_choice(CORRECT_SQL)], column_descriptions=descriptions)
        source = self.path.parent / "database_description"
        path = write_column_csv(source, "orders", [
            ["amount_cents", "", "Sales amount in cents.", "integer", "", ""],
        ])
        gold_file = self.root / "gold.sql"
        gold_file.write_text(GOLD_SQL, encoding="utf-8")
        log_root = self.root / "l"
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "DATABASE_ROOT", self.root))
            stack.enter_context(patch.object(runner, "LOG_ROOT", log_root))
            stack.enter_context(patch.object(runner, "GOLD_PATH", gold_file))
            stack.enter_context(patch.object(runner, "load_questions", return_value=([QUESTION], "fixture")))
            stack.enter_context(patch.object(runner, "split_questions", return_value={"smoke": [QUESTION]}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_question = stack.enter_context(patch.object(runner, "run_question", return_value=result))
            with patch.object(sys, "argv", ["run_bird_v2.py", "--mode", "A2", "--use-column-descriptions"]):
                self.assertEqual(runner.main(), 0)
            (directory,) = log_root.iterdir()
            self.assertIn("column-descriptions-v1", directory.name)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["use_column_descriptions"])
            self.assertEqual(manifest["system_prompt"], runner.SYSTEM_PROMPT)
            self.assertEqual(set(manifest["column_description_sha256"][QUESTION["db_id"]]), {"orders.csv"})
            self.assertEqual(run_question.call_args.kwargs["column_descriptions"][0][:3], descriptions[0][:3])
            with patch.object(sys, "argv", ["run_bird_v2.py", "--resume", str(directory)]):
                self.assertEqual(runner.main(), 0)
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "代码、数据或配置改变"):
                    runner.main()
            run_question.assert_called_once()
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["questions_with_column_descriptions"], 1)

    def test_fallback_cli_pins_the_baseline_and_separates_request_cost(self):
        baseline, _, _, _ = self.run_scripted("A2", [text_choice("SELECT 1; unexpected text")])
        self.assertIsNone(baseline["predicted_sql"])
        self.assertEqual(baseline["protocol_errors"], 1)
        protected_question = {**QUESTION, "id": QUESTION["id"] + 1}
        protected, _, _, _ = self.run_scripted("A2", [text_choice(CORRECT_SQL)])
        protected["question_id"] = protected_question["id"]
        selected = [QUESTION, protected_question]
        candidate, _, _, _ = self.run_scripted(
            "A2", [text_choice(json.dumps(SALES_PLAN)), text_choice(CORRECT_SQL)], use_structured_plan=True,
        )
        source = self.root / "base"
        source.mkdir()
        gold_file = self.root / "gold.sql"
        gold_file.write_text(GOLD_SQL, encoding="utf-8")
        log_root = self.root / "l"
        splits = {"smoke": [], "held-out": selected}
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "LOG_ROOT", log_root))
            stack.enter_context(patch.object(runner, "GOLD_PATH", gold_file))
            stack.enter_context(patch.object(runner, "load_questions", return_value=(selected, "fixture")))
            stack.enter_context(patch.object(runner, "split_questions", return_value=splits))
            stack.enter_context(patch.object(runner, "select_questions", return_value=selected))
            stack.enter_context(redirect_stdout(io.StringIO()))
            run_question = stack.enter_context(patch.object(runner, "run_question", return_value=candidate))
            manifest = runner.make_manifest("A2", runner.DIAGNOSTIC_SPLIT, selected, splits, "fixture")
            manifest["code_sha256"]["run_bird_v2.py"] = "original-runner"
            runner.write_json(source / "manifest.json", manifest)
            source_results = source / "results-original.jsonl"
            with source_results.open("x", encoding="utf-8") as output:
                runner.record(output, "question_result", **baseline)
                runner.record(output, "question_result", **protected)
            args = [
                "run_bird_v2.py", "--mode", "A2", "--split", "diagnostic-55", "--use-structured-plan",
                "--fallback-from", str(source),
            ]
            with patch.object(sys, "argv", args):
                self.assertEqual(runner.main(), 0)
            (directory,) = log_root.iterdir()
            self.assertIn("plan-fallback-v1", directory.name)
            saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["fallback_baseline"]["eligible_ids"], [QUESTION["id"]])
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["baseline_ex_correct"], summary["net_ex_gain"]), (1, 1))
            self.assertEqual((summary["fallback_attempted"], summary["fallback_selected"]), (1, 1))
            self.assertEqual(summary["baseline_usage"]["model_calls"], 2)
            self.assertEqual(summary["additional_usage"]["model_calls"], 2)
            self.assertEqual(summary["combined_usage"]["model_calls"], 4)
            self.assertEqual(summary["mean_agent_seconds"], candidate["agent_seconds"] / 2)
            self.assertEqual(summary["mean_total_seconds"], candidate["total_seconds"] / 2)

            with patch.object(sys, "argv", ["run_bird_v2.py", "--resume", str(directory)]):
                self.assertEqual(runner.main(), 0)
                with source_results.open("w", encoding="utf-8") as output:
                    runner.record(output, "question_result", **{**baseline, "total_seconds": baseline["total_seconds"] + 1})
                    runner.record(output, "question_result", **protected)
                with self.assertRaisesRegex(ValueError, "代码、数据或配置改变"):
                    runner.main()
            run_question.assert_called_once()

    def test_old_result_version_is_not_mixed_into_new_run(self):
        directory = self.root / "old_run"
        directory.mkdir()
        old = {"question_id": QUESTION["id"], "db_id": QUESTION["db_id"], "mode": "A2"}
        with (directory / "results-old.jsonl").open("x", encoding="utf-8") as output:
            runner.record(output, "question_result", **old)
        before = set(directory.iterdir())
        with self.assertRaises(ValueError):
            runner.run_batch(directory, [QUESTION], {
                "mode": "A2", "split": "smoke", "workflow_version": runner.WORKFLOW_VERSION})
        self.assertEqual(set(directory.iterdir()), before)


class SemanticFeedbackTest(TemporaryDatabaseTest):
    def test_derivation_recovers_final_sql_but_rejects_multiple_statements(self):
        content = "Output: the total amount.\n" + CORRECT_SQL + ";"
        self.assertEqual(semantic_feedback.parse_candidate(text_choice(content), "derive"), CORRECT_SQL + ";")
        with self.assertRaises(ValueError):
            semantic_feedback.parse_candidate(text_choice(content + "\nSELECT 1;"), "derive")

    def test_case_suffix_repair_preserves_literals_and_comments(self):
        query = "SELECT COUNT(CASE WHEN region = 'END.0)' THEN 1 END.0) FROM sales -- END.0)"
        expected = "SELECT COUNT(CASE WHEN region = 'END.0)' THEN 1 END) FROM sales -- END.0)"
        self.assertEqual(semantic_feedback.case_suffix_repair(query), expected)
        self.assertIsNone(semantic_feedback.case_suffix_repair("SELECT 'END.0)'"))

    def test_precision_repair_only_wraps_a_numeric_scalar(self):
        question = {**QUESTION, "question": "Provide the average divided by 37 with three decimal places."}
        baseline = {"predicted_sql": "SELECT AVG(amount_cents) / 37 FROM orders",
                    "evaluation": {"prediction": {"status": "ok", "row_count": 1}}}
        with patch.object(runner, "DATABASE_ROOT", self.root):
            query = semantic_feedback.precision_repair(question, baseline)
            self.assertIn("ROUND(value, 3)", query)
            self.assertEqual(evaluate_sql(
                QUESTION["db_id"], query, "SELECT ROUND(4200.0 / 37, 3)", database_root=self.root
            )["ex"], 1)
            self.assertIsNone(semantic_feedback.precision_repair(
                question, {**baseline, "predicted_sql": "SELECT '113.5136'"},
            ))

    def test_group_top_rejects_candidate_without_the_requested_partition(self):
        question = {**QUESTION, "question": "List the top 5 in each county.",
                    "evidence": "Rank by amount using PARTITION BY County"}
        original = "SELECT county FROM orders ORDER BY amount_cents DESC LIMIT 5"
        baseline = {"predicted_sql": original, "ex": 0,
                    "evaluation": {"prediction": {"status": "ok", "row_count": 5}}}
        candidate = semantic_feedback.make_repair_candidate(
            "group_top", "SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount) FROM orders",
        )
        with patch.object(semantic_feedback, "generate", return_value=candidate):
            chosen, attempts = semantic_feedback.select_repair(question, "fixture", baseline, io.StringIO())
        self.assertIs(chosen, baseline)
        self.assertEqual(attempts[0]["status"], "guard_rejected")


class DiagnosticSplitTest(unittest.TestCase):
    def diagnostic_fixture(self):
        candidates = [
            {**QUESTION, "id": 100 * db + offset, "db_id": f"db{db:02d}"}
            for db in range(11) for offset in range(1, 8)
        ]
        return {"smoke": [QUESTION], "held-out": candidates}

    def test_diagnostic_sample_is_balanced_stable_and_held_out(self):
        splits = self.diagnostic_fixture()
        sample = runner.select_questions(splits, runner.DIAGNOSTIC_SPLIT)
        reversed_splits = {**splits, "held-out": splits["held-out"][::-1]}
        self.assertEqual(sample, runner.select_questions(reversed_splits, runner.DIAGNOSTIC_SPLIT))
        counts = Counter(question["db_id"] for question in sample)
        self.assertEqual(counts, {f"db{db:02d}": 5 for db in range(11)})
        sample_ids = {question["id"] for question in sample}
        self.assertEqual(len(sample_ids), 55)
        self.assertNotIn(QUESTION["id"], sample_ids)
        self.assertTrue(sample_ids <= {question["id"] for question in splits["held-out"]})

    def test_diagnostic_cli_archives_two_arms_with_identical_ids(self):
        splits = self.diagnostic_fixture()
        with tempfile.TemporaryDirectory(prefix="v2-") as directory, ExitStack() as stack:
            root = Path(directory)
            gold_file = root / "gold.sql"
            gold_file.write_text(GOLD_SQL, encoding="utf-8")
            log_root = root / "l"
            stack.enter_context(patch.object(runner, "LOG_ROOT", log_root))
            stack.enter_context(patch.object(runner, "GOLD_PATH", gold_file))
            stack.enter_context(patch.object(runner, "load_questions", return_value=([], "fixture")))
            stack.enter_context(patch.object(runner, "split_questions", return_value=splits))
            run_batch = stack.enter_context(patch.object(runner, "run_batch", return_value=0))
            common = ["run_bird_v2.py", "--mode", "A2", "--split", "diagnostic-55"]
            for options in ([], ["--use-structured-plan"]):
                with patch.object(sys, "argv", common + options):
                    self.assertEqual(runner.main(), 0)
            baseline, structured = [call.args for call in run_batch.call_args_list]
            self.assertEqual(baseline[1], structured[1])
            self.assertEqual(baseline[2]["selected_ids"], structured[2]["selected_ids"])
            self.assertEqual(baseline[2]["system_prompt"], structured[2]["system_prompt"])
            self.assertEqual(baseline[2]["diagnostic_selection"], structured[2]["diagnostic_selection"])
            self.assertNotIn("use_structured_plan", baseline[2])
            self.assertTrue(structured[2]["use_structured_plan"])
            self.assertNotEqual(baseline[0], structured[0])
            self.assertIn("query-plan-v1", structured[0].name)
            with patch.object(sys, "argv", ["run_bird_v2.py", "--resume", str(structured[0])]):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(run_batch.call_args.args, structured)

    def test_diagnostic_rejects_rows_from_the_other_arm(self):
        with tempfile.TemporaryDirectory(prefix="v2-") as directory:
            root = Path(directory)
            with (root / "results-old.jsonl").open("x", encoding="utf-8") as output:
                runner.record(output, "question_result", **runner.new_result(QUESTION, "A2"))
            manifest = {
                "mode": "A2", "split": runner.DIAGNOSTIC_SPLIT,
                "workflow_version": runner.WORKFLOW_VERSION, "use_structured_plan": True,
            }
            self.assertRaisesRegex(ValueError, "提示词配置", runner.run_batch, root, [QUESTION], manifest)
            self.assertEqual([path.name for path in root.iterdir()], ["results-old.jsonl"])


class HistoricalDispatchReplayTest(unittest.TestCase):
    def test_saved_multi_call_responses_obey_dispatch_policy(self):
        if REPLAY_LOG is None:
            self.skipTest("Specify --replay-log for historical dispatch-only replay")
        paths = sorted(REPLAY_LOG.glob("trace-*.jsonl")) if REPLAY_LOG.is_dir() else [REPLAY_LOG]
        self.assertTrue(paths)
        report = []
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event["event"] != "model_response":
                    continue
                choice = event["response"]["choices"][0]
                calls = choice["message"].get("tool_calls") or []
                if len(calls) < 2:
                    continue
                original = copy.deepcopy(choice)
                observed = []

                class RecordingLookup:
                    def lookup_values(self, **arguments):
                        observed.append(arguments)
                        return {"status": "ok", "values": [], "source": "offline_dispatch_stub"}

                question = {"id": event["question_id"], "db_id": "offline_replay"}
                result = runner.new_result(question, "B2")
                trace = io.StringIO()
                with patch.object(runner, "execute_for_ex", side_effect=AssertionError("replay executed SQL")) as sql:
                    observations = runner.dispatch_lookup_batch(choice, RecordingLookup(), trace, result)
                    sql.assert_not_called()
                requested = sum(call["function"]["name"] == "lookup_values" for call in calls)
                rejected = len(calls) - requested
                self.assertEqual(result["lookup_requests"], requested)
                self.assertEqual((len(observed), result["lookup_calls"]), (min(requested, 2), min(requested, 2)))
                self.assertEqual(result["lookup_skipped_budget"], max(requested - 2, 0))
                self.assertEqual(result["retrieval_calls_rejected"], rejected)
                self.assertEqual(len(observations), len(observed))
                self.assertEqual(choice, original)
                report.append({"id": question["id"], "raw_calls": len(calls),
                               "lookup_requested": requested, "lookup_dispatched": len(observed),
                               "over_budget": result["lookup_skipped_budget"], "other_tools_rejected": rejected})
        self.assertTrue(report, "No historical multi-call responses found")
        print("\nHistorical dispatch replay only; no Qwen3 inference, real BIRD query, or EX re-scoring:")
        print(json.dumps(report, ensure_ascii=False))
        if REPLAY_LOG.is_dir():
            manifest_path = REPLAY_LOG / "manifest.json"
            if manifest_path.exists():
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(runner.SYSTEM_PROMPT, old_manifest["system_prompt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--replay-log", type=Path)
    options, unittest_args = parser.parse_known_args()
    REPLAY_LOG = options.replay_log
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
