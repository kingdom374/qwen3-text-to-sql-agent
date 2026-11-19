"""Run the fixed P0 cases and save independent case traces and a summary."""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from urllib.error import HTTPError, URLError

from build_demo_db import DEMO_DB
from p0_cases import CASES
from smoke_sqlite_roundtrip import list_tables
from smoke_text_to_sql import MAX_TOOL_CALLS, complete, record
from sqlite_query import execute_sql
from sqlite_schema import get_schema


SYSTEM_PROMPT = (
    "你是 SQLite 数据分析助手。理解当前问题与对话历史，必要时先查 Schema。"
    "每次只调用一个工具，每个用户问题都要实际执行 SQL。"
    "筛选连续日期范围时，使用 日期 >= 起始日 AND 日期 < 结束边界，结束边界为所选时段结束后的第一天。"
    "销售额为订单数量乘以单价，SQL 中保留分，最终答复以元为单位。"
    '最终只返回 JSON 对象 {"rows": [...]}。rows 是二维数组，每个查询结果行对应一个数组。'
    '例如返回一个数字时使用 {"rows": [[数值]]}，查询结果为零行时使用 {"rows": []}。'
)


def answer_rows(content):
    answer = content.strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.partition("\n")[2].removesuffix("```").strip()
    result = json.loads(answer)
    if not isinstance(result, dict) or set(result) != {"rows"} or not isinstance(result["rows"], list):
        raise ValueError("最终答复必须是只包含 rows 数组的 JSON 对象")
    rows = result["rows"]
    if any(not isinstance(row, list) or any(isinstance(value, bool) for value in row) for row in rows):
        raise ValueError("rows 必须是二维数组，布尔值不能替代数值")
    return rows


def run_turn(case, turn, messages, trace):
    messages.append({"role": "user", "content": turn.question})
    last_result = None
    sql_calls = 0
    sql_errors = 0
    failed_query = None
    error_feedback = None
    refreshed_schema = False

    for step in range(MAX_TOOL_CALLS + 1):
        choice = complete(messages, trace)
        message = choice["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            if choice.get("finish_reason") != "stop" or last_result is None or last_result["status"] != "ok":
                raise ValueError("模型未在本轮成功执行 SQL 后正常回答")
            messages.append(message)
            return last_result, message.get("content") or "", sql_errors

        if step == MAX_TOOL_CALLS:
            raise ValueError(f"达到本轮工具调用上限（{MAX_TOOL_CALLS} 次）")
        if len(calls) != 1 or choice.get("finish_reason") != "tool_calls":
            raise ValueError("每次只接受一个完整的工具调用")
        (call,) = calls
        call_id = call.get("id")
        if call.get("type") != "function" or not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("工具调用类型或调用 ID 无效")
        function = call["function"]
        name = function["name"]
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        record(trace, "tool_call", id=call_id, name=name, arguments=arguments)

        if sql_errors and last_result["status"] == "ok":
            raise ValueError("修订 SQL 已执行，模型应根据结果直接回答")
        if sql_errors and name != "execute_sql":
            if name != "get_schema" or refreshed_schema or step + 1 >= MAX_TOOL_CALLS:
                raise ValueError("执行反馈后只允许查一次 Schema，再执行一次修订 SQL")
            refreshed_schema = True

        if name == "list_tables" and not arguments:
            result = {"tables": list_tables()}
        elif name == "get_schema" and set(arguments) == {"table_names"}:
            result = get_schema(arguments["table_names"])
        elif name == "execute_sql" and set(arguments) == {"query"} and isinstance(arguments["query"], str):
            query = arguments["query"]
            injected = sql_calls < len(case.injected_sql)
            if injected:
                query = case.injected_sql[sql_calls]
            elif sql_errors and query.strip() == failed_query.strip():
                record(trace, "repair_unchanged", id=call_id, query=query)
                raise ValueError("模型原样提交了失败 SQL")
            record(trace, "sql_execution", id=call_id, query=query, injected=injected, retry=bool(sql_errors))
            sql_calls += 1
            result = execute_sql(query)
            last_result = result
        else:
            raise ValueError(f"未知工具或参数不匹配：{name}")

        sql_failed = name == "execute_sql" and result["status"] != "ok"
        retry_allowed = (
            sql_failed and result["status"] == "sql_error" and case.repair_on_error
            and sql_errors == 0 and step + 1 < MAX_TOOL_CALLS
        )
        feedback = result
        if retry_allowed:
            failed_query = query
            error_feedback = {
                "failed_query": query,
                "sqlite_error": result["error"],
                "next_action": "根据报错修订 SQL。保持用户的条件与原有单位，最多提交一次修订。",
            }
            feedback = {**result, "repair_context": error_feedback}
        elif sql_errors and name == "get_schema":
            feedback = {**result, "repair_context": error_feedback}

        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(feedback, ensure_ascii=False),
        }
        record(trace, "tool_result", name=name, result=result, message=tool_message)
        if sql_failed:
            sql_errors += 1
            if not retry_allowed:
                if case.injected_sql and sql_errors == 2 and result["status"] == "sql_error":
                    record(trace, "termination", reason="injected_retry_exhausted")
                    return result, None, sql_errors
                raise ValueError(f"SQL 执行失败（{result['status']}）：{result['error']}")
        messages.extend([message, tool_message])


def run_case(name, case, directory):
    path = directory / f"{name}.jsonl"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    with path.open("x", encoding="utf-8") as trace:
        record(trace, "run_config", case=name, repair_on_error=case.repair_on_error, injected=bool(case.injected_sql))
        try:
            for number, turn in enumerate(case.turns, 1):
                record(trace, "turn_start", turn=number, question=turn.question)
                expected_rows = None
                if turn.reference_sql is not None:
                    reference = execute_sql(turn.reference_sql)
                    if reference["status"] != "ok" or reference["truncated"]:
                        raise ValueError(f"标准查询执行失败或被截断：{reference}")
                    expected_rows = reference["rows"]

                result, answer, sql_errors = run_turn(case, turn, messages, trace)
                if case.injected_sql:
                    if sql_errors != 2 or answer is not None:
                        raise ValueError("预期注入两次执行错误后停止")
                    record(trace, "turn_validation", turn=number, passed=True, observed_expected_failure=True)
                    continue

                expected_answer = [[expected_rows[0][0] / 100]] if turn.answer_in_yuan else expected_rows
                sql_matches = not result["truncated"] and result["rows"] == expected_rows
                answer_error = None
                try:
                    answer_matches = answer_rows(answer) == expected_answer
                except ValueError as error:
                    answer_matches = False
                    answer_error = str(error)
                record(
                    trace, "turn_validation", turn=number, passed=sql_matches and answer_matches,
                    sql_matches=sql_matches, answer_matches=answer_matches, expected_sql_rows=expected_rows,
                    actual_sql_rows=result["rows"], expected_answer_rows=expected_answer, answer=answer,
                    answer_error=answer_error, execution_error_repaired=bool(sql_errors),
                )
                if not sql_matches or not answer_matches:
                    raise ValueError(f"第 {number} 轮未通过：SQL 结果匹配={sql_matches}，答复匹配={answer_matches}")
        except HTTPError as error:
            failure = f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        except (URLError, TimeoutError, sqlite3.Error, ValueError, KeyError, IndexError, TypeError) as error:
            failure = str(error)
        else:
            record(trace, "validation", passed=True)
            return {"case": name, "passed": True, "trace": path.name}
        record(trace, "validation", passed=False, error=failure)
        return {"case": name, "passed": False, "trace": path.name, "error": failure}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, help="仅运行一个用例，默认运行全部八个")
    args = parser.parse_args()
    if not DEMO_DB.is_file():
        sys.exit(f"请先运行 build_demo_db.py 创建演示库：{DEMO_DB}")

    selected = [args.case] if args.case else CASES
    directory = DEMO_DB.parents[2] / ".local" / "logs" / f"p0-suite-{datetime.now():%Y%m%d-%H%M%S-%f}"
    directory.mkdir(parents=True)
    print(f"日志目录：{directory}")
    results = []
    for name in selected:
        result = run_case(name, CASES[name], directory)
        results.append(result)
        status = "PASS" if result["passed"] else f"FAIL - {result['error']}"
        print(f"{name}: {status}", flush=True)
    passed = sum(result["passed"] for result in results)
    summary = {"passed": passed, "total": len(results), "cases": results}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"P0 用例通过数：{passed}/{len(results)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
