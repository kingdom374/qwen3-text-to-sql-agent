"""Offline checks for protocol recovery and its B/C integration."""

import copy
import io
import json
import unittest
from unittest.mock import patch

import run_bird_tool_assisted as runner
from tool_protocol import normalize_tool_choice, parse_text_sql


def text_choice(text, finish="stop"):
    return {"finish_reason": finish, "message": {"role": "assistant", "content": text}}


def native_choice(name, arguments):
    return {
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "type": "function", "id": "native_1",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        },
    }


class ToolProtocolSmokeTest(unittest.TestCase):
    def test_native_call_is_preserved(self):
        choice = native_choice("get_schema", {"table_names": ["orders"]})
        original = copy.deepcopy(choice)
        message, source = normalize_tool_choice(choice, mode="recover-sql", recovered_id="adapter_1")
        self.assertEqual(source, "native")
        self.assertEqual(message, original["message"])
        self.assertEqual(choice, original)

    def test_strict_mode_rejects_text(self):
        with self.assertRaises(ValueError):
            normalize_tool_choice(text_choice("SELECT 1"), recovered_id="adapter_1")

    def test_text_function_handles_quotes_and_newlines(self):
        sql = "SELECT phone FROM member\nWHERE first_name = 'Carlo';"
        text = "execute_sql(" + json.dumps(sql) + ")"
        self.assertEqual(parse_text_sql(text), (sql, "text_function_call"))

    def test_keyword_argument(self):
        self.assertEqual(parse_text_sql('execute_sql(query="SELECT 1")'), ("SELECT 1", "text_function_call"))

    def test_plain_and_fenced_sql(self):
        for text in ("SELECT 1", "```sql\nSELECT 1\n```", "WITH x AS (SELECT 1) SELECT * FROM x"):
            with self.subTest(text=text):
                self.assertEqual(parse_text_sql(text)[1], "plain_sql")

    def test_semicolons_in_literals_and_comments(self):
        for text in ("SELECT ';';", "SELECT 'it''s;ok';", "SELECT 1 /* ; */;", "SELECT 1 -- ;\n;"):
            with self.subTest(text=text):
                self.assertEqual(parse_text_sql(text)[0], text)

    def test_rejects_executable_python_and_extra_text(self):
        for text in (
            'execute_sql(__import__("os").getcwd())',
            'execute_sql("SELECT " + "1")',
            'execute_sql(f"SELECT {1}")',
            'other.execute_sql("SELECT 1")',
            'execute_sql("SELECT 1", "SELECT 2")',
            'execute_sql(**{"query": "SELECT 1"})',
            'execute_sql("SELECT 1"); print("extra")',
            'Here is the SQL: SELECT 1',
            'SELECT 1; Explanation follows.',
            'SELECT 1; SELECT 2;',
            'DELETE FROM orders;',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_text_sql(text)

    def test_rejects_unclosed_sql(self):
        for text in ("SELECT 'unfinished", "SELECT (1", "SELECT 1 /* comment", "SELECT 1)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_text_sql(text)

    def test_length_reply_cannot_be_recovered(self):
        with self.assertRaises(ValueError):
            normalize_tool_choice(text_choice("SELECT 1", finish="length"), mode="recover-sql", recovered_id="adapter_1")

    def test_malformed_native_call_has_no_text_fallback(self):
        choice = native_choice("execute_sql", {"query": "SELECT 1"})
        choice["message"]["content"] = "SELECT 1"
        choice["message"]["tool_calls"][0]["function"]["arguments"] = "invalid JSON"
        with self.assertRaises(ValueError):
            normalize_tool_choice(choice, mode="recover-sql", recovered_id="adapter_1")

    def test_recovery_keeps_original_and_marks_source(self):
        choice = text_choice('execute_sql("SELECT 1")')
        original = copy.deepcopy(choice)
        message, source = normalize_tool_choice(choice, mode="recover-sql", recovered_id="adapter_1")
        call = message["tool_calls"][0]
        self.assertEqual(source, "text_function_call")
        self.assertEqual(call["id"], "adapter_1")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "SELECT 1"})
        self.assertEqual(choice, original)


class RunnerProtocolIntegrationTest(unittest.TestCase):
    def run_scripted_question(self, replies, observations, *, mode, repair=False):
        replies = iter(replies)
        trace = io.StringIO()
        requests = []

        def respond(messages, remaining, log, question_id, stats):
            requests.append(copy.deepcopy(messages))
            self.assertGreater(remaining, 0)
            stats["model_calls"] += 1
            stats["completion_tokens"] += 20
            stats["prompt_tokens"] += 100
            return next(replies)

        evaluation = {"ex": 1, "prediction": {"status": "ok"}, "reference": {"status": "ok"}}
        with patch.object(runner, "request_model", side_effect=respond):
            with patch.object(runner, "BirdTools") as tool_class:
                tool_class.return_value.execute_sql.side_effect = observations
                with patch.object(runner, "score_prediction", return_value=evaluation) as score:
                    result = runner.run_question(
                        {"id": 1, "db_id": "offline_fixture", "question": "test", "evidence": ""},
                        trace, repair_on_sql_error=repair, protocol_mode=mode,
                    )
                executions = tool_class.return_value.execute_sql.call_count
                score_calls = score.call_count
        return result, [json.loads(line) for line in trace.getvalue().splitlines()], requests, executions, score_calls

    def test_b_recovered_sql_uses_existing_execution_and_scoring(self):
        result, events, _, executions, score_calls = self.run_scripted_question(
            [text_choice('execute_sql("SELECT 1")')], [{"status": "ok"}], mode="recover-sql"
        )
        self.assertEqual(result["ex"], 1)
        self.assertEqual((executions, score_calls), (1, 1))
        self.assertEqual(result["recovered_sql_calls"], 1)
        self.assertEqual(result["native_tool_calls"], 0)
        self.assertTrue(any(event["event"] == "protocol_recovery" for event in events))

    def test_b_strict_does_not_execute_text(self):
        result, _, _, executions, score_calls = self.run_scripted_question(
            [text_choice("SELECT 1")], [], mode="strict"
        )
        self.assertEqual(result["status"], "agent_error")
        self.assertEqual((executions, score_calls), (0, 0))

    def test_c_can_return_feedback_after_recovered_submission(self):
        result, _, requests, executions, score_calls = self.run_scripted_question(
            [text_choice('execute_sql("SELECT missing FROM orders")'), text_choice("SELECT 1")],
            [{"status": "sql_error", "error": "no such column: missing"}, {"status": "ok"}],
            mode="recover-sql", repair=True,
        )
        self.assertEqual((executions, score_calls), (2, 1))
        self.assertEqual(result["recovered_sql_calls"], 2)
        self.assertTrue(result["repair_triggered"])
        assistant = requests[1][-3]
        tool_result = requests[1][-2]
        self.assertEqual(assistant["tool_calls"][0]["id"], tool_result["tool_call_id"])
        self.assertIn("no such column: missing", tool_result["content"])

    def test_native_path_is_same_in_both_modes(self):
        for mode in ("strict", "recover-sql"):
            with self.subTest(mode=mode):
                result, _, _, executions, _ = self.run_scripted_question(
                    [native_choice("execute_sql", {"query": "SELECT 1"})], [{"status": "ok"}], mode=mode
                )
                self.assertEqual(executions, 1)
                self.assertEqual(result["native_tool_calls"], 1)
                self.assertEqual(result["recovered_sql_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
