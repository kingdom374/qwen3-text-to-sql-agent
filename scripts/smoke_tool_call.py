"""Check whether the local Qwen3 server returns structured tool calls."""

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
TOOL = {
    "type": "function",
    "function": {
        "name": "list_tables",
        "description": "列出当前 SQLite 数据库中可查询的表名。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


def send_probe(question, tool_choice):
    payload = {
        "model": "qwen3-local",
        "messages": [
            {"role": "user", "content": question}
        ],
        "tools": [TOOL],
        "tool_choice": tool_choice,
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
    # The request stays on loopback even if a proxy is configured for the shell.
    with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
        return json.load(response)


def check_probe(label, question, tool_choice, *, expect_tool):
    print(f"\n{label}", flush=True)
    try:
        response = send_probe(question, tool_choice)
        choice = response["choices"][0]
        message = choice["message"]
        print(json.dumps(choice, ensure_ascii=False, indent=2))

        calls = message.get("tool_calls") or []
        if not expect_tool:
            if calls or not (message.get("content") or "").strip():
                raise ValueError("expected a text reply without tool calls")
            print("PASS: text reply without tool calls")
            return True

        if len(calls) != 1:
            raise ValueError(f"expected one tool call, got {len(calls)}")
        call = calls[0]
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("the tool call has no valid id")
        if call.get("type") != "function":
            raise ValueError("unexpected tool call type")
        function = call["function"]
        if function["name"] != "list_tables":
            raise ValueError(f"unexpected function: {function['name']}")
        if json.loads(function["arguments"]) != {}:
            raise ValueError("list_tables requires an empty JSON object")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {error.code}: {body}", file=sys.stderr)
        return False
    except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return False

    print(f"PASS: list_tables() id={call_id}")
    return True


def main():
    lookup_question = "请查询当前 SQLite 数据库有哪些表。"
    control_question = "这是一条连通性测试，无需查询数据库。请只回复 OK。"
    auto_lookup_ok = check_probe("查表问题 | auto", lookup_question, "auto", expect_tool=True)
    auto_control_ok = check_probe("无需查库 | auto", control_question, "auto", expect_tool=False)
    required_control_ok = check_probe(
        "无需查库 | required", control_question, "required", expect_tool=True
    )
    if auto_control_ok and required_control_ok:
        print("RESULT: 同一问题在 auto 下返回文本，在 required 下请求调用工具。")
    else:
        print("RESULT: 此次对照未能验证 required 的效果。")
    return 0 if auto_lookup_ok and auto_control_ok and required_control_ok else 1


if __name__ == "__main__":
    sys.exit(main())
