"""Test one complete Text-to-SQL turn using the local Qwen3 and demo database."""

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from build_demo_db import DEMO_DB
from smoke_sqlite_roundtrip import list_tables
from smoke_tool_call import ENDPOINT, TOOL as LIST_TABLES_TOOL
from sqlite_query import execute_sql
from sqlite_schema import GET_SCHEMA_TOOL, get_schema


EXECUTE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "在演示库上执行单条只读 SQL，并返回列名和结果行。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
TOOLS = [LIST_TABLES_TOOL, GET_SCHEMA_TOOL, EXECUTE_SQL_TOOL]
MAX_TOOL_CALLS = 6
QUESTION = "2026 年第二季度华东地区的销售额是多少？请以人民币元回答。"
EXPECTED_CENTS = 8400
EXPECTED_YUAN = Decimal("84")
YUAN_AMOUNT = re.compile(r"(?<![\d.,])([+\-−]?\d+(?:\.\d+)?)\s*(?:`|\*\*)?\s*元")
SYSTEM_PROMPT = (
    "你是 SQLite 数据分析助手。每次只请求一个工具。先查相关表的 Schema，再写并执行 SQL。"
    "销售额是订单数量乘以单价的总和。SQL 汇总时保留整数分，最终根据实际查询结果换算成元。"
    "取得足够的查询结果后直接回答，金额和单位用普通文本输出。"
)


def record(trace, event, **data):
    line = json.dumps({"event": event, **data}, ensure_ascii=False)
    trace.write(line + "\n")
    trace.flush()


def read_trace(path):
    events = defaultdict(list)
    with path.open(encoding="utf-8-sig") as trace:
        for line_number, line in enumerate(trace, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"日志第 {line_number} 行不是有效 JSON") from error
            if not isinstance(entry, dict) or not isinstance(entry.get("event"), str):
                raise ValueError(f"日志第 {line_number} 行缺少 event 字段")
            events[entry["event"]].append(entry)
    return events


def score_trace(events):
    requests = events["model_request"]
    if not requests or not any(
        message.get("role") == "user" and message.get("content") == QUESTION
        for message in requests[0]["payload"]["messages"]
    ):
        raise ValueError("这个评分只适用于 2026 年第二季度华东销售额的固定演示题")

    sql_results = [event for event in events["tool_result"] if event["name"] == "execute_sql"]
    last_sql_result = sql_results[-1]["result"] if sql_results else None
    repair_attempts = events["repair_attempt"]
    repair_result = None
    if repair_attempts:
        repair_id = repair_attempts[-1]["id"]
        repair_result = next(
            (event["result"] for event in sql_results if event["message"]["tool_call_id"] == repair_id),
            None,
        )

    if repair_result is not None:
        repair_status = "pass" if repair_result["status"] == "ok" else "fail"
    elif events["repair_feedback"]:
        repair_status = "no_completed_execution"
    else:
        repair_status = "not_triggered"

    sql_contract_status = "not_observed"
    if last_sql_result is not None and last_sql_result["status"] == "ok":
        matches = last_sql_result["rows"] == [[EXPECTED_CENTS]] and not last_sql_result["truncated"]
        sql_contract_status = "pass" if matches else "fail"

    answer = None
    if events["model_response"]:
        choice = events["model_response"][-1]["response"]["choices"][0]
        if choice.get("finish_reason") == "stop" and not choice["message"].get("tool_calls"):
            answer = choice["message"].get("content") or None
    amounts = {Decimal(amount.replace("−", "-")) for amount in YUAN_AMOUNT.findall(answer or "")}
    answer_status = "not_observed"
    if answer is not None:
        answer_status = "pass" if amounts == {EXPECTED_YUAN} else "fail"

    original_validation = events["validation"]
    return {
        "question": QUESTION,
        "execution_repair": {
            "status": repair_status,
            "unchanged_retry": bool(events["repair_unchanged"]),
            "retry_sql_status": repair_result["status"] if repair_result is not None else None,
        },
        "sql_result_cents_contract": {
            "status": sql_contract_status,
            "expected_rows": [[EXPECTED_CENTS]],
            "observed_rows": last_sql_result.get("rows") if last_sql_result is not None else None,
        },
        "final_answer_amount": {
            "status": answer_status,
            "expected_yuan": str(EXPECTED_YUAN),
            "observed_yuan": [str(amount) for amount in sorted(amounts)],
            "answer": answer,
        },
        "original_validation": original_validation[-1] if original_validation else None,
    }


def print_trace_score(path):
    try:
        result = score_trace(read_trace(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"无法评分：{error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def complete(messages, trace):
    payload = {
        "model": "qwen3-local",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    record(trace, "model_request", payload=payload)
    started = monotonic()
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        data = json.load(response)
    record(trace, "model_response", elapsed_seconds=monotonic() - started, response=data)
    return data["choices"][0]


def run_turn(trace, *, repair_on_sql_error=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    queried_tables = set()
    last_sql_result = None
    awaiting_repair = False
    repair_schema_used = False
    repair_attempted = False
    previous_sql_failure = None
    print(f"问题：{QUESTION}")

    for step in range(MAX_TOOL_CALLS + 1):
        choice = complete(messages, trace)
        message = choice["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            if awaiting_repair:
                raise ValueError("模型收到 SQLite 报错后没有提交修订 SQL")
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"模型未正常结束：{choice.get('finish_reason')}")
            answer = message.get("content") or ""
            print(f"最终答复：{answer}")
            if last_sql_result is None:
                raise ValueError("模型在实际查询前就给出了最终答复")
            if last_sql_result["truncated"] or last_sql_result["rows"] != [[EXPECTED_CENTS]]:
                raise ValueError(f"最后一次 SQL 的结果不等于预期的 {EXPECTED_CENTS} 分")
            if re.search(r"(?<![\d.\-−])84(?:\.0+)?\s*元", answer) is None:
                raise ValueError("最终答复缺少预期的 84 元")
            return

        if step == MAX_TOOL_CALLS:
            raise ValueError(f"达到工具调用上限（{MAX_TOOL_CALLS} 次）")
        if repair_attempted:
            raise ValueError("已执行一次修订 SQL，接下来应直接给出最终答复")
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
        print(f"工具请求：{name} {json.dumps(arguments, ensure_ascii=False)}")
        if awaiting_repair:
            if name == "get_schema" and set(arguments) == {"table_names"}:
                if repair_schema_used or step + 1 >= MAX_TOOL_CALLS:
                    raise ValueError("已达到修复阶段的 Schema 查询上限，请提交修订 SQL")
                repair_schema_used = True
            elif name != "execute_sql" or set(arguments) != {"query"}:
                raise ValueError("修复阶段只接受一次 get_schema 和一次修订 SQL")

        if name == "list_tables" and not arguments:
            result = {"tables": list_tables()}
        elif name == "get_schema" and set(arguments) == {"table_names"}:
            result = get_schema(arguments["table_names"])
            queried_tables.update(result["tables"])
            if awaiting_repair:
                record(trace, "repair_schema_refresh", id=call_id, tables=list(result["tables"]))
        elif name == "execute_sql" and set(arguments) == {"query"}:
            if "orders" not in queried_tables:
                raise ValueError("需要先查询 orders 的 Schema，再执行本题 SQL")
            query = arguments["query"]
            if awaiting_repair:
                if isinstance(query, str) and query.strip() == previous_sql_failure["failed_query"].strip():
                    record(trace, "repair_unchanged", id=call_id, query=query)
                    raise ValueError("修订 SQL 与失败 SQL 相同，未再次执行")
                awaiting_repair = False
                repair_attempted = True
                record(trace, "repair_attempt", id=call_id, query=query)
            result = execute_sql(query)
            last_sql_result = result
        else:
            raise ValueError(f"未知工具或参数不匹配：{name}")

        sql_failed = name == "execute_sql" and result["status"] != "ok"
        will_retry = (
            repair_on_sql_error
            and sql_failed
            and result["status"] == "sql_error"
            and not repair_attempted
            and step + 1 < MAX_TOOL_CALLS
        )
        feedback = result
        if will_retry:
            previous_sql_failure = {"failed_query": query, "sqlite_error": result["error"]}
            next_action = "根据 SQLite 报错和真实 Schema 修正 SQL。"
            if step + 2 < MAX_TOOL_CALLS:
                next_action += "需要时可先调用一次 get_schema 核对字段。"
            feedback = {
                **result,
                "failed_query": query,
                "retry_remaining": 1,
                "next_action": next_action + "调用一次 execute_sql 提交修订 SQL。"
                "保持原问题的条件，SQL 汇总结果仍使用分。",
            }
        elif awaiting_repair and name == "get_schema":
            feedback = {
                **result,
                "repair_context": {
                    **previous_sql_failure,
                    "next_action": "对照上面的表结构和失败 SQL 的 FROM/JOIN，核对报错字段属于哪张表。"
                    "保持原问题条件与分单位，调用一次 execute_sql 提交修改后的 SQL。",
                },
            }
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(feedback, ensure_ascii=False),
        }
        record(trace, "tool_result", name=name, result=result, message=tool_message)
        print(f"工具结果：{tool_message['content']}")
        if sql_failed and not will_retry:
            raise ValueError(f"SQL 执行失败（{result['status']}）：{result['error']}")
        if will_retry:
            awaiting_repair = True
            record(trace, "repair_feedback", id=call_id, error=result["error"])
            print("已回传 SQLite 报错，允许一次修订。")
        messages.extend([message, tool_message])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--repair-on-sql-error",
        action="store_true",
        help="回传 SQLite 报错，并允许模型修订一次 SQL",
    )
    mode.add_argument("--score-trace", type=Path, help="离线复核已有的 text-to-sql JSONL 日志")
    args = parser.parse_args()
    if args.score_trace is not None:
        return print_trace_score(args.score_trace)
    if not DEMO_DB.is_file():
        sys.exit(f"FAIL: 请先运行 build_demo_db.py 创建演示库：{DEMO_DB}")
    log_dir = DEMO_DB.parents[2] / ".local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = log_dir / f"text-to-sql-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    print(f"运行日志：{trace_path}")
    failure = None
    with trace_path.open("x", encoding="utf-8") as trace:
        record(trace, "run_config", repair_on_sql_error=args.repair_on_sql_error)
        try:
            run_turn(trace, repair_on_sql_error=args.repair_on_sql_error)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            failure = f"HTTP {error.code}: {details}"
        except (URLError, TimeoutError, sqlite3.Error, ValueError, KeyError, IndexError, TypeError) as error:
            failure = str(error)
        if failure is None:
            record(trace, "validation", passed=True, expected_sales_cents=EXPECTED_CENTS)
        else:
            record(trace, "validation", passed=False, error=failure)

    print("\n分项评分")
    print_trace_score(trace_path)
    if failure is not None:
        print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"PASS: SQLite 返回 {EXPECTED_CENTS} 分，最终答复包含 {EXPECTED_YUAN} 元。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
