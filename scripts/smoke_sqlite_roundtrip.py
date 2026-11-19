"""Exercise one model-tool-model round trip against the SQLite fixture."""

import json
import sqlite3
import sys
from contextlib import closing
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from build_demo_db import DEMO_DB
from smoke_tool_call import ENDPOINT, TOOL


def complete(messages):
    payload = {
        "model": "qwen3-local",
        "messages": messages,
        "tools": [TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        return json.load(response)["choices"][0]


def list_tables():
    uri = DEMO_DB.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = ? "
            "AND name NOT GLOB ? ORDER BY name",
            ("table", "sqlite_*"),
        ).fetchall()
    return [name for (name,) in rows]


def show(title, payload):
    print(f"\n{title}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    if not DEMO_DB.is_file():
        raise ValueError(f"演示库不存在，请先运行 build_demo_db.py：{DEMO_DB}")

    messages = [
        {
            "role": "system",
            "content": "需要查表名时调用 list_tables。收到工具结果后，用实际表名直接回答。",
        },
        {"role": "user", "content": "当前 SQLite 数据库有哪些表？请列出表名。"},
    ]
    show("1. 用户问题", messages[-1])
    first_choice = complete(messages)
    show("2. 模型发出的调用请求", first_choice)
    assistant_message = first_choice["message"]
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"预期收到一个工具调用，实际收到 {len(calls)} 个")
    call = calls[0]
    if call["type"] != "function" or call["function"]["name"] != "list_tables":
        raise ValueError("本次演示只允许调用 list_tables")
    if json.loads(call["function"]["arguments"]) != {}:
        raise ValueError("list_tables 的参数须为空 JSON 对象")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("工具调用缺少有效的 ID")

    tables = list_tables()
    tool_message = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps({"tables": tables}, ensure_ascii=False),
    }
    show("3. SQLite 实查结果与工具消息", tool_message)
    if tables != ["orders", "products"]:
        raise ValueError(f"演示库的表名与预期不一致：{tables}")

    messages.extend([assistant_message, tool_message])
    final_choice = complete(messages)
    show("4. 模型收到工具结果后的回复", final_choice)
    final_message = final_choice["message"]
    answer = final_message.get("content") or ""
    if final_message.get("tool_calls") or final_choice.get("finish_reason") != "stop":
        raise ValueError("模型没有在收到结果后正常结束")
    if any(table not in answer for table in tables):
        raise ValueError("模型答复未包含 SQLite 返回的全部表名")

    print("\nPASS: SQLite 返回预期表名，模型答复也包含两个表名。")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        sys.exit(f"FAIL: HTTP {error.code}: {detail}")
    except (URLError, TimeoutError, sqlite3.Error, ValueError, KeyError, IndexError, TypeError) as error:
        sys.exit(f"FAIL: {error}")
