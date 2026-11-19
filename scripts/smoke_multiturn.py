"""Verify that follow-ups preserve or replace order-count filters."""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from urllib.error import HTTPError, URLError

from build_demo_db import DEMO_DB
from smoke_sqlite_roundtrip import list_tables
from smoke_text_to_sql import MAX_TOOL_CALLS, complete, record
from sqlite_query import execute_sql
from sqlite_schema import get_schema


CASES = {
    "add-region": [
        ("2026 年第二季度共有多少笔订单？", 3),
        ("只看华东呢？", 2),
    ],
    "replace-region": [
        ("2026 年第二季度，华东地区共有多少笔订单？", 2),
        ("改看华北呢？", 1),
    ],
}
SYSTEM_PROMPT = (
    "你是 SQLite 数据分析助手。根据当前问题和之前的对话理解指标与筛选条件。"
    "需要时查 Schema，每次只调用一个工具。每个用户问题都应执行查询，并根据实际结果回答。"
    "订单笔数使用阿拉伯数字并标注单位‘笔’。"
)
ORDER_COUNT = re.compile(r"(?<![\d.+\-−])([0-9]+)\s*(?:`|\*\*)?\s*笔")


def run_turn(messages, question, expected_count, trace):
    messages.append({"role": "user", "content": question})
    print(f"问题：{question}")
    latest_sql_result = None

    for step in range(MAX_TOOL_CALLS + 1):
        choice = complete(messages, trace)
        message = choice["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"模型未正常结束：{choice.get('finish_reason')}")
            answer = message.get("content") or ""
            print(f"答复：{answer}")
            if latest_sql_result is None:
                raise ValueError("本轮没有执行 SQL")
            if latest_sql_result["truncated"] or latest_sql_result["rows"] != [[expected_count]]:
                raise ValueError(f"SQL 的实际结果为 {latest_sql_result['rows']}，预期 {expected_count} 笔")
            if {int(value) for value in ORDER_COUNT.findall(answer)} != {expected_count}:
                raise ValueError(f"答复中的笔数未匹配预期的 {expected_count} 笔")
            messages.append(message)
            return

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
        print(f"工具请求：{name} {json.dumps(arguments, ensure_ascii=False)}")

        if name == "list_tables" and not arguments:
            result = {"tables": list_tables()}
        elif name == "get_schema" and set(arguments) == {"table_names"}:
            result = get_schema(arguments["table_names"])
        elif name == "execute_sql" and set(arguments) == {"query"}:
            result = execute_sql(arguments["query"])
            latest_sql_result = result
        else:
            raise ValueError(f"未知工具或参数不匹配：{name}")

        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }
        record(trace, "tool_result", name=name, result=result, message=tool_message)
        print(f"工具结果：{tool_message['content']}")
        if name == "execute_sql" and result["status"] != "ok":
            raise ValueError(f"SQL 执行失败（{result['status']}）：{result['error']}")
        messages.extend([message, tool_message])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="add-region", help="选择要执行的两轮用例")
    args = parser.parse_args()
    if not DEMO_DB.is_file():
        sys.exit(f"FAIL: 请先运行 build_demo_db.py 创建演示库：{DEMO_DB}")
    log_dir = DEMO_DB.parents[2] / ".local" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"multiturn-{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
    print(f"运行日志：{path}")
    print(f"用例：{args.case}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    with path.open("x", encoding="utf-8") as trace:
        record(trace, "run_config", case=args.case)
        for turn, (question, expected_count) in enumerate(CASES[args.case], 1):
            record(trace, "turn_start", turn=turn, question=question)
            print(f"\n第 {turn} 轮")
            try:
                run_turn(messages, question, expected_count, trace)
            except HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                failure = f"HTTP {error.code}: {details}"
            except (URLError, TimeoutError, sqlite3.Error, ValueError, KeyError, IndexError, TypeError) as error:
                failure = str(error)
            else:
                record(trace, "turn_validation", turn=turn, passed=True, expected_count=expected_count)
                print(f"本轮通过：查询与回答均为 {expected_count} 笔。")
                continue
            record(trace, "turn_validation", turn=turn, passed=False, error=failure)
            record(trace, "validation", passed=False)
            print(f"FAIL: 第 {turn} 轮：{failure}", file=sys.stderr)
            return 1
        record(trace, "validation", passed=True)
    print("PASS: 两轮查询结果与回答均符合预期。请核对第二轮 SQL 的时间和地区条件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
