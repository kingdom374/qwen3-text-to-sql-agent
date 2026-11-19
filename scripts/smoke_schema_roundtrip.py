"""Exercise get_schema with one model-tool-model round trip."""

import json
import sqlite3
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from build_demo_db import DEMO_DB
from smoke_tool_call import ENDPOINT
from sqlite_schema import GET_SCHEMA_TOOL, get_schema


EXPECTED_COLUMNS = [
    "order_id", "product_id", "order_date", "region", "quantity", "unit_price_cents"
]


def complete(messages):
    payload = {
        "model": "qwen3-local",
        "messages": messages,
        "tools": [GET_SCHEMA_TOOL],
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
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        return json.load(response)["choices"][0]


def show(title, payload):
    print(f"\n{title}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    if not DEMO_DB.is_file():
        raise ValueError(f"演示库不存在，请先运行 build_demo_db.py：{DEMO_DB}")

    messages = [
        {
            "role": "system",
            "content": "先用 get_schema 查询用户所问的表。收到结果后直接回答。描述外键时用完整的 表名.字段名。",
        },
        {
            "role": "user",
            "content": "orders 表有哪些字段？请逐一列出，并说明它通过哪个字段关联 products 表。",
        },
    ]
    show("1. 用户问题", messages[-1])
    first_choice = complete(messages)
    show("2. 模型发出的 get_schema 请求", first_choice)
    assistant_message = first_choice["message"]
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1 or first_choice.get("finish_reason") != "tool_calls":
        raise ValueError("预期模型发出一个工具调用")
    (call,) = calls
    function = call["function"]
    if call.get("type") != "function" or function["name"] != "get_schema":
        raise ValueError("本次演示只允许调用 get_schema")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("工具调用缺少有效的 ID")
    arguments = json.loads(function["arguments"])
    if not isinstance(arguments, dict) or set(arguments) != {"table_names"}:
        raise ValueError("get_schema 需要且只接受 table_names 参数")

    schema = get_schema(arguments["table_names"])
    show("3. SQLite 实查 Schema", schema)
    orders = schema["tables"].get("orders")
    if orders is None or [column["name"] for column in orders["columns"]] != EXPECTED_COLUMNS:
        raise ValueError("查到的 orders 字段与演示库预期不一致")
    if not any(
        (key["from_column"], key["to_table"], key["to_column"])
        == ("product_id", "products", "product_id")
        for key in orders["foreign_keys"]
    ):
        raise ValueError("未查到预期的 orders.product_id → products.product_id 外键")

    tool_message = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(schema, ensure_ascii=False),
    }
    show("4. 回传的工具消息", tool_message)
    messages.extend([assistant_message, tool_message])
    final_choice = complete(messages)
    show("5. 模型根据 Schema 给出的回答", final_choice)
    final_message = final_choice["message"]
    if final_message.get("tool_calls") or final_choice.get("finish_reason") != "stop":
        raise ValueError("模型没有在收到 Schema 后正常结束")
    answer = final_message.get("content") or ""
    for name in [*EXPECTED_COLUMNS, "orders.product_id", "products.product_id"]:
        if name not in answer:
            raise ValueError(f"答复缺少预期的字段标识：{name}")
    print("\nPASS: 实查字段和外键符合预期，模型答复包含所需字段与两个关联字段标识。")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        sys.exit(f"FAIL: HTTP {error.code}: {detail}")
    except (URLError, TimeoutError, sqlite3.Error, ValueError, KeyError, IndexError, TypeError) as error:
        sys.exit(f"FAIL: {error}")
