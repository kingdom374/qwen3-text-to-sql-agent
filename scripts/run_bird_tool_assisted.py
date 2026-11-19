"""Run BIRD strategy B with schema tools and a single SQL submission."""

import argparse
import hashlib
import json
import msvcrt
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import bird_ex
from bird_ex import DEFAULT_TIMEOUT_SECONDS
from bird_tools import TOOLS, BirdTools
from run_bird_direct import (
    ENDPOINT, GENERATION_SEED, GOLD_PATH, LOG_ROOT, MAX_TOKENS, MODEL,
    REQUEST_TIMEOUT_SECONDS, SPLIT_SEED, load_questions, read_schema, score_prediction, split_questions, write_json,
)
from smoke_text_to_sql import record
from tool_protocol import normalize_tool_choice


MAX_TOOL_CALLS = 6
SYSTEM_PROMPT = (
    "You answer questions over a SQLite database using tools. "
    "Every assistant response must contain exactly one tool call. "
    "Do not return SQL as plain text, explanations, or a final natural-language answer. "
    "Use list_tables and get_schema as needed to inspect the schema. "
    "If the inspected schema does not contain a field needed by the question, "
    "inspect other relevant tables before deciding how to query it. "
    "Use table names and declared foreign keys to find those tables. "
    "Use the evidence and schema to write exactly one read-only SQLite query. "
    "Submit the query only through execute_sql, never as plain text. "
    "That submission ends the task, including on error."
)
REPAIR_PROMPT = (
    "Runtime update: the first SQL execution failed. For this recovery phase, "
    "the earlier instruction to end on SQL submission is overridden. "
    "Use the actual SQLite error and schema to correct the query. "
    "You may call get_schema at most once, then must call execute_sql exactly once "
    "with a revised query. You may also submit the revised query immediately. "
    "Preserve the user's requirements. Return a structured tool call, not text. "
    "The revised SQL execution ends the task."
)


def request_model(messages, remaining_tokens, trace, question_id, stats):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "seed": GENERATION_SEED,
        "max_tokens": remaining_tokens,
        "stream": False,
    }
    record(trace, "model_request", question_id=question_id, payload=payload)
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    stats["model_calls"] += 1
    started = perf_counter()
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.load(response)
    finally:
        stats["model_seconds"] += perf_counter() - started
    record(trace, "model_response", question_id=question_id, response=data)
    usage = data.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens"):
        count = usage.get(key)
        if type(count) is not int or count < 0:
            raise ValueError(f"服务未返回有效的 {key}，无法核对预算")
        stats[key] += count
    if stats["completion_tokens"] > MAX_TOKENS:
        raise ValueError("服务返回的累计生成 token 超出预算")
    return data["choices"][0]


def run_agent(question, trace, result, *, repair_on_sql_error=False, protocol_mode="strict", schema_mode="on-demand"):
    if schema_mode not in ("on-demand", "full"):
        raise ValueError(f"未知 Schema 模式：{schema_mode}")
    tools = BirdTools(question["db_id"])
    user_content = f"Database: {question['db_id']}\n"
    if schema_mode == "full":
        user_content += f"Schema:\n{read_schema(question['db_id'])}\n\n"
    user_content += f"Evidence:\n{question['evidence']}\n\nQuestion:\n{question['question']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    call_ids = set()
    awaiting_repair = False
    schema_refreshed = False
    repair_context = None
    for step in range(MAX_TOOL_CALLS):
        remaining = MAX_TOKENS - result["completion_tokens"]
        if remaining <= 0:
            raise ValueError("累计生成 token 预算耗尽")
        choice = request_model(messages, remaining, trace, question["id"], result)
        result["protocol_responses"] += 1
        try:
            message, submission_source = normalize_tool_choice(
                choice, mode=protocol_mode, recovered_id=f"adapter_{question['id']}_{step + 1}"
            )
        except ValueError:
            result["native_protocol_failures"] += 1
            raise
        if submission_source == "native":
            result["native_tool_calls"] += 1
        else:
            result["native_protocol_failures"] += 1
            result["recovered_sql_calls"] += 1
            record(trace, "protocol_recovery", question_id=question["id"],
                   source=submission_source, original_content=choice["message"].get("content"),
                   normalized_message=message)
        calls = message["tool_calls"]
        (call,) = calls
        call_id = call.get("id")
        if call.get("type") != "function" or not isinstance(call_id, str) or not call_id.strip() or call_id in call_ids:
            raise ValueError("工具调用类型或 ID 无效")
        call_ids.add(call_id)
        function = call["function"]
        name = function["name"]
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        result["tool_calls"] += 1
        record(trace, "tool_call", question_id=question["id"], step=step + 1,
               id=call_id, name=name, arguments=arguments, submission_source=submission_source)

        if awaiting_repair:
            if name == "get_schema" and set(arguments) == {"table_names"}:
                if schema_refreshed or step + 1 >= MAX_TOOL_CALLS:
                    raise ValueError("修复阶段最多查一次 Schema，并须为修订 SQL 留出调用次数")
                schema_refreshed = True
            elif name != "execute_sql" or set(arguments) != {"query"}:
                raise ValueError("修复阶段只接受 get_schema 或 execute_sql")

        started = perf_counter()
        try:
            if name == "list_tables" and not arguments:
                observation = tools.list_tables()
            elif name == "get_schema" and set(arguments) == {"table_names"}:
                observation = tools.get_schema(arguments["table_names"])
            elif name == "execute_sql" and set(arguments) == {"query"}:
                if not isinstance(arguments["query"], str) or not arguments["query"].strip():
                    raise ValueError("query 必须是非空 SQL 字符串")
                if awaiting_repair and arguments["query"].strip() == result["first_predicted_sql"].strip():
                    record(trace, "repair_unchanged", question_id=question["id"], query=arguments["query"])
                    raise ValueError("模型原样提交了失败 SQL，未重复执行")
                result["sql_submissions"] += 1
                result["predicted_sql"] = arguments["query"]
                record(trace, "sql_execution", question_id=question["id"], query=result["predicted_sql"],
                       submission=result["sql_submissions"])
                observation = tools.execute_sql(result["predicted_sql"])
                result["tool_execution"] = observation
                if result["sql_submissions"] == 1:
                    result["first_predicted_sql"] = result["predicted_sql"]
                    result["first_tool_execution"] = observation
            else:
                raise ValueError(f"未知工具或参数不匹配：{name}")
        finally:
            result["tool_seconds"] += perf_counter() - started

        will_retry = (
            repair_on_sql_error and name == "execute_sql"
            and result["sql_submissions"] == 1 and observation["status"] == "sql_error"
            and step + 1 < MAX_TOOL_CALLS and result["completion_tokens"] < MAX_TOKENS
        )
        feedback = observation
        if will_retry:
            repair_context = {
                "failed_query": result["predicted_sql"], "sqlite_error": observation["error"],
                "sql_retries_remaining": 1,
            }
            feedback = {**observation, "repair_context": repair_context}
        elif awaiting_repair and name == "get_schema":
            feedback = {**observation, "repair_context": repair_context}
        tool_message = {
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps(feedback, ensure_ascii=False),
        }
        record(trace, "tool_result", question_id=question["id"], name=name,
               result=observation, message=tool_message)
        if will_retry:
            awaiting_repair = True
            result["repair_triggered"] = True
            record(trace, "repair_feedback", question_id=question["id"], **repair_context)
            messages.extend([message, tool_message, {"role": "system", "content": REPAIR_PROMPT}])
            continue
        if name == "execute_sql":
            reason = "first_sql_executed" if result["sql_submissions"] == 1 else "revised_sql_executed"
            record(trace, "termination", question_id=question["id"], reason=reason)
            return
        messages.extend([message, tool_message])
    raise ValueError("达到工具调用上限，尚未提交 SQL")


def run_question(question, trace, *, repair_on_sql_error=False, protocol_mode="strict", schema_mode="on-demand"):
    result = {
        "question_id": question["id"], "db_id": question["db_id"],
        "predicted_sql": None, "ex": 0, "model_calls": 0, "tool_calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "model_seconds": 0.0, "tool_seconds": 0.0,
        "sql_submissions": 0, "repair_triggered": False,
        "protocol_responses": 0, "native_tool_calls": 0,
        "native_protocol_failures": 0, "recovered_sql_calls": 0,
    }
    started = perf_counter()
    try:
        run_agent(question, trace, result, repair_on_sql_error=repair_on_sql_error,
                  protocol_mode=protocol_mode, schema_mode=schema_mode)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        result.update(status="http_error", error=f"HTTP {error.code}: {detail}")
    except (URLError, TimeoutError, sqlite3.Error, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        result.update(status="agent_error", error=str(error))
    else:
        result["status"] = "submitted"
    result["agent_seconds"] = perf_counter() - started

    if result["status"] == "submitted":
        try:
            # Scoring re-executes the saved prediction independently, just as in
            # A. Gold SQL and scoring results are never appended to messages.
            evaluation = score_prediction(question, result["predicted_sql"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            result.update(status="evaluation_error", error=str(error))
        else:
            result.update(status="evaluated", ex=evaluation["ex"], evaluation=evaluation)
    result["total_seconds"] = perf_counter() - started
    return result


def make_manifest(question_hash, splits, selected, split, *, repair_on_sql_error=False,
                  protocol_mode="strict", schema_mode="on-demand"):
    return {
        "strategy": "execution_feedback" if repair_on_sql_error else "tool_assisted",
        "split": split, "question_sha256": question_hash,
        "split_seed": SPLIT_SEED,
        "splits": {name: [question["id"] for question in items] for name, items in splits.items()},
        "selected_ids": [question["id"] for question in selected],
        "model_alias": MODEL, "endpoint": ENDPOINT, "temperature": 0,
        "generation_seed": GENERATION_SEED, "max_completion_tokens_per_question": MAX_TOKENS,
        "max_tool_calls": MAX_TOOL_CALLS, "max_sql_submissions": 2 if repair_on_sql_error else 1,
        "repair_on_sql_error": repair_on_sql_error,
        "protocol_mode": protocol_mode,
        "schema_mode": schema_mode,
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("tool_protocol.py").read_bytes()).hexdigest(),
        "repair_prompt": REPAIR_PROMPT if repair_on_sql_error else None,
        "system_prompt": SYSTEM_PROMPT, "tools": TOOLS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "ex_timeout_seconds": DEFAULT_TIMEOUT_SECONDS, "sqlite_version": sqlite3.sqlite_version,
        "evaluator_sha256": hashlib.sha256(Path(bird_ex.__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tools_sha256": hashlib.sha256(Path(__file__).with_name("bird_tools.py").read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest(),
    }


@contextmanager
def run_lock(run_dir):
    # Windows releases this byte-range lock even if the Python process is killed.
    with (run_dir / "run.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise ValueError("这个运行目录正被另一个进程使用") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def read_completed_results(run_dir, selected):
    expected = {question["id"]: question["db_id"] for question in selected}
    completed = {}
    for path in sorted(run_dir.glob("results-*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                # An interrupted final write is retained as evidence. Only a
                # complete, newline-terminated result commits a question.
                if not line.endswith("\n"):
                    print(f"未提交的末行保留在 {path.name}:{number}，对应题目可重跑。", flush=True)
                    continue
                result = json.loads(line)
                question_id = result["question_id"]
                if result.get("event") != "question_result" or expected.get(question_id) != result["db_id"]:
                    raise ValueError(f"已有结果不符合当前题单：{path.name}:{number}")
                if question_id in completed:
                    raise ValueError(f"已有结果包含重复题号：{question_id}")
                completed[question_id] = result
    return completed


def summarize(results, split, *, repair_on_sql_error=False, protocol_mode="strict", schema_mode="on-demand"):
    total = len(results)
    correct = sum(result["ex"] for result in results)
    executable = sum(result.get("evaluation", {}).get("prediction", {}).get("status") == "ok" for result in results)
    responses = sum(result["protocol_responses"] for result in results)
    native_calls = sum(result["native_tool_calls"] for result in results)
    return {
        "strategy": "execution_feedback" if repair_on_sql_error else "tool_assisted",
        "split": split, "total": total,
        "protocol_mode": protocol_mode,
        "schema_mode": schema_mode,
        "protocol_responses": responses, "native_tool_calls": native_calls,
        "native_tool_call_rate": native_calls / responses if responses else None,
        "native_protocol_failures": sum(result["native_protocol_failures"] for result in results),
        "recovered_sql_calls": sum(result["recovered_sql_calls"] for result in results),
        "questions_with_protocol_recovery": sum(result["recovered_sql_calls"] > 0 for result in results),
        "recovered_questions_ex_correct": sum(result["recovered_sql_calls"] > 0 and result["ex"] == 1 for result in results),
        "ex_correct": correct, "ex": correct / total,
        "sql_executable": executable, "sql_execution_rate": executable / total,
        "reference_timeouts": sum(result.get("evaluation", {}).get("reference", {}).get("status") == "timeout" for result in results),
        "mean_tool_calls": sum(result["tool_calls"] for result in results) / total,
        "mean_prompt_tokens": sum(result["prompt_tokens"] for result in results) / total,
        "mean_completion_tokens": sum(result["completion_tokens"] for result in results) / total,
        "mean_agent_seconds": sum(result["agent_seconds"] for result in results) / total,
        "mean_total_seconds": sum(result["total_seconds"] for result in results) / total,
        "repair_triggered": sum(result["repair_triggered"] for result in results),
        "repair_attempted": sum(result["sql_submissions"] == 2 for result in results),
        "repair_execution_success": sum(
            result["sql_submissions"] == 2 and result.get("tool_execution", {}).get("status") == "ok"
            for result in results
        ),
        "repair_ex_correct": sum(result["sql_submissions"] == 2 and result["ex"] == 1 for result in results),
    }


def run_batch(run_dir, selected, manifest):
    completed = read_completed_results(run_dir, selected)
    session = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    repair = manifest["repair_on_sql_error"]
    protocol_mode = manifest["protocol_mode"]
    schema_mode = manifest["schema_mode"]
    group = "C" if repair else "B"
    print(f"{group} 组日志（{manifest['split']}）：{run_dir}", flush=True)
    print(f"协议模式：{protocol_mode}", flush=True)
    print(f"Schema 模式：{schema_mode}", flush=True)
    print(f"已有结果 {len(completed)}/{len(selected)}，本次补跑 {len(selected) - len(completed)} 题。", flush=True)
    with (run_dir / f"trace-{session}.jsonl").open("x", encoding="utf-8") as trace:
        with (run_dir / f"results-{session}.jsonl").open("x", encoding="utf-8") as output:
            for question in selected:
                if question["id"] in completed:
                    continue
                record(trace, "question_start", question_id=question["id"], session=session)
                result = run_question(question, trace, repair_on_sql_error=repair,
                                      protocol_mode=protocol_mode, schema_mode=schema_mode)
                record(output, "question_result", **result)
                completed[question["id"]] = result
                evaluation = result.get("evaluation")
                status = evaluation["prediction"]["status"] if evaluation else result["status"]
                print(f"题目 {question['id']}: EX={result['ex']}，执行状态={status}，"
                      f"工具调用={result['tool_calls']}，生成 token={result['completion_tokens']}，"
                      f"SQL 提交={result['sql_submissions']}", flush=True)
                if result.get("error"):
                    print(f"原因：{result['error']}", flush=True)

    results = [completed[question["id"]] for question in selected]
    summary = summarize(results, manifest["split"], repair_on_sql_error=repair,
                        protocol_mode=protocol_mode, schema_mode=schema_mode)
    summary_path = run_dir / f"summary-{session}.json"
    write_json(summary_path, summary)
    temporary = run_dir / f"summary-{session}.tmp"
    temporary.write_bytes(summary_path.read_bytes())
    temporary.replace(run_dir / "summary.json")
    print(f"{group} 组 {manifest['split']} EX：{summary['ex_correct']}/{summary['total']}，"
          f"SQL 可执行：{summary['sql_executable']}/{summary['total']}")
    print(f"原生合法工具回复：{summary['native_tool_calls']}/{summary['protocol_responses']}，"
          f"适配恢复：{summary['recovered_sql_calls']} 次，涉及 {summary['questions_with_protocol_recovery']} 题，"
          f"这些题中 EX 正确：{summary['recovered_questions_ex_correct']}")
    if repair:
        print(f"触发修复：{summary['repair_triggered']}，实际提交修订：{summary['repair_attempted']}，"
              f"修订可执行：{summary['repair_execution_success']}，修订 EX 正确：{summary['repair_ex_correct']}")
    return 1 if any(result["status"] != "evaluated" for result in results) else 0


def main(*, repair_on_sql_error=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("smoke", "held-out"), help="默认 smoke，正式集选 held-out")
    parser.add_argument("--limit", type=int, help="仅用于冒烟清单，默认运行 1 题，最多 22")
    parser.add_argument("--resume", type=Path, help="续跑本版本创建的运行目录，保留已记录的失败结果")
    parser.add_argument("--protocol-mode", choices=("strict", "recover-sql"), help="默认 strict，可选文本 SQL 适配恢复")
    parser.add_argument("--schema-mode", choices=("on-demand", "full"), help="默认按需查 Schema，full 使用与 A 相同的完整 Schema")
    args = parser.parse_args()
    if args.resume is not None and any(
        option is not None for option in (args.split, args.limit, args.protocol_mode, args.schema_mode)
    ):
        parser.error("--resume 使用原目录配置，不能同时指定划分、题数、协议或 Schema 模式")
    if args.split == "held-out" and args.limit is not None:
        parser.error("--limit 仅适用于 smoke 划分")
    if args.limit is not None and not 1 <= args.limit <= 22:
        parser.error("--limit 必须在 1 到 22 之间")

    questions, question_hash = load_questions()
    splits = split_questions(questions)
    if args.resume is not None:
        run_dir = args.resume.resolve()
        saved = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        split = saved["split"]
        selected_ids = saved["selected_ids"]
        available_ids = [question["id"] for question in splits[split]]
        if not selected_ids or selected_ids != available_ids[:len(selected_ids)]:
            raise ValueError("续跑题单与固定划分不一致")
        if split == "held-out" and len(selected_ids) != len(available_ids):
            raise ValueError("正式运行须包含完整 478 题")
        selected = splits[split][:len(selected_ids)]
        protocol_mode = saved.get("protocol_mode", "strict")
        schema_mode = saved.get("schema_mode", "on-demand")
        manifest = make_manifest(question_hash, splits, selected, split,
                                 repair_on_sql_error=repair_on_sql_error, protocol_mode=protocol_mode,
                                 schema_mode=schema_mode)
        if saved != manifest:
            raise ValueError("代码、配置或数据版本与原运行不同，请新建运行目录；旧版日志不支持本续跑入口")
    else:
        split = args.split or "smoke"
        limit = args.limit if args.limit is not None else 1
        selected = splits[split][:limit] if split == "smoke" else splits[split]
        protocol_mode = args.protocol_mode or "strict"
        schema_mode = args.schema_mode or "on-demand"
        manifest = make_manifest(question_hash, splits, selected, split,
                                 repair_on_sql_error=repair_on_sql_error, protocol_mode=protocol_mode,
                                 schema_mode=schema_mode)
        strategy = "execution-feedback" if repair_on_sql_error else "tool-assisted"
        run_dir = LOG_ROOT / f"bird-{strategy}-{split}-{protocol_mode}-{schema_mode}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "manifest.json", manifest)

    with run_lock(run_dir):
        return run_batch(run_dir, selected, manifest)


if __name__ == "__main__":
    sys.exit(main())
