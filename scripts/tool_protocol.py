"""Normalize native tool calls or explicitly enabled textual SQL submissions."""

import ast
import json
import re


SQL_FENCE = re.compile(r"```(?:sql|sqlite)?[ \t]*\r?\n(.*?)\r?\n?```", re.IGNORECASE | re.DOTALL)


def check_sql_submission(sql):
    """Check submission boundaries. SQLite still validates syntax and read access."""
    sql = sql.strip()
    if re.match(r"(?:SELECT|WITH)\b", sql, re.IGNORECASE) is None:
        raise ValueError("文本恢复只接受以 SELECT 或 WITH 开头的 SQL")
    quote = None
    comment = None
    depth = 0
    index = 0
    while index < len(sql):
        char = sql[index]
        pair = sql[index:index + 2]
        if comment == "line":
            if char in "\r\n":
                comment = None
        elif comment == "block":
            if pair == "*/":
                comment = None
                index += 1
        elif quote is not None:
            if char == quote:
                if quote != "]" and index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif pair in ("--", "/*"):
            comment = "line" if pair == "--" else "block"
            index += 1
        elif char in "'\"`[":
            quote = "]" if char == "[" else char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("SQL 括号不完整")
        elif char == ";" and sql[index + 1:].strip():
            raise ValueError("SQL 语句结束后有额外内容")
        index += 1
    if quote is not None or comment == "block" or depth:
        raise ValueError("SQL 含有未闭合的引号、注释或括号")
    return sql


def parse_text_sql(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型没有提交有效 SQL")
    text = content.strip()
    fenced = SQL_FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    if re.match(r"(?:SELECT|WITH)\b", text, re.IGNORECASE):
        return check_sql_submission(text), "plain_sql"
    try:
        expression = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError) as error:
        raise ValueError("回复不是完整 SQL 或 execute_sql 字符串调用") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != "execute_sql":
        raise ValueError("文本恢复只允许 execute_sql 调用")
    if len(expression.args) == 1 and not expression.keywords:
        argument = expression.args[0]
    elif not expression.args and len(expression.keywords) == 1 and expression.keywords[0].arg == "query":
        argument = expression.keywords[0].value
    else:
        raise ValueError("execute_sql 必须且只能传一个 query 字符串")
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        raise ValueError("execute_sql 参数必须是字符串常量")
    return check_sql_submission(argument.value), "text_function_call"


def normalize_tool_choice(choice, *, mode="strict", recovered_id):
    if mode not in ("strict", "recover-sql"):
        raise ValueError(f"未知协议模式：{mode}")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("无效的 assistant 消息")
    calls = message.get("tool_calls") or []
    if choice.get("finish_reason") == "tool_calls" and isinstance(calls, list) and len(calls) == 1:
        call = calls[0]
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("无效的工具调用类型")
        if not isinstance(call.get("id"), str) or not call["id"].strip():
            raise ValueError("工具调用缺少 ID")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
            raise ValueError("工具调用缺少 JSON 参数字符串")
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        name = function.get("name")
        valid = name == "list_tables" and not arguments
        if name == "get_schema" and set(arguments) == {"table_names"}:
            tables = arguments["table_names"]
            valid = isinstance(tables, list) and bool(tables) and all(isinstance(table, str) and table.strip() for table in tables)
        if name == "execute_sql" and set(arguments) == {"query"}:
            query = arguments["query"]
            valid = isinstance(query, str) and bool(query.strip())
        if not valid:
            raise ValueError("未知工具或参数结构不符合工具定义")
        return message, "native"

    if mode != "recover-sql" or choice.get("finish_reason") != "stop" or calls:
        raise ValueError(f"需要一个完整工具调用，实际结束原因为 {choice.get('finish_reason')}")
    sql, source = parse_text_sql(message.get("content"))
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "type": "function", "id": recovered_id,
            "function": {"name": "execute_sql", "arguments": json.dumps({"query": sql}, ensure_ascii=False)},
        }],
    }, source
