"""Generate one SQLite query per BIRD Mini-Dev question and score it with EX."""

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from bird_ex import BIRD_DIR, DATABASE_ROOT, DEFAULT_TIMEOUT_SECONDS, database_path, evaluate_sql
from smoke_text_to_sql import record
from smoke_tool_call import ENDPOINT


QUESTION_PATH = BIRD_DIR / "mini_dev_sqlite.json"
GOLD_PATH = BIRD_DIR / "mini_dev_sqlite_gold.sql"
LOG_ROOT = BIRD_DIR.parents[1] / ".local" / "logs"
MODEL = "qwen3-local"
SPLIT_SEED = 42
GENERATION_SEED = 42
MAX_TOKENS = 512
REQUEST_TIMEOUT_SECONDS = 120
SYSTEM_PROMPT = (
    "You translate questions into SQLite SQL. Use the given schema and evidence. "
    "Return exactly one read-only SELECT query, optionally starting with WITH. "
    "Return only the SQL, with no explanation or Markdown."
)
SQL_FENCE = re.compile(r"```(?:sql|sqlite)?[ \t]*\r?\n(.*?)\r?\n?```", re.IGNORECASE | re.DOTALL)


def load_questions():
    content = QUESTION_PATH.read_bytes()
    source = json.loads(content.decode("utf-8-sig"))
    if not isinstance(source, list) or len(source) != 500:
        raise ValueError("预期 500 道原版 SQLite Mini-Dev 题目")
    questions = []
    for index, item in enumerate(source):
        public_fields = {key: item[key] for key in ("db_id", "question", "evidence")}
        if any(not isinstance(value, str) for value in public_fields.values()):
            raise ValueError(f"题目 {index} 的输入字段类型不合法")
        questions.append({"id": index, **public_fields})
    return questions, hashlib.sha256(content).hexdigest()


def split_questions(questions):
    by_database = defaultdict(list)
    for question in questions:
        by_database[question["db_id"]].append(question["id"])
    if len(by_database) != 11 or any(len(ids) < 2 for ids in by_database.values()):
        raise ValueError("预期 11 个数据库，每个库至少有两道题")

    rng = random.Random(SPLIT_SEED)
    smoke_ids = {
        question_id
        for db_id in sorted(by_database)
        for question_id in rng.sample(by_database[db_id], 2)
    }
    return {
        "smoke": [question for question in questions if question["id"] in smoke_ids],
        "held-out": [question for question in questions if question["id"] not in smoke_ids],
    }


@lru_cache(maxsize=None)
def read_schema(db_id):
    path = database_path(db_id, DATABASE_ROOT)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        statements = [
            sql for (sql,) in connection.execute(
                "SELECT sql FROM main.sqlite_schema WHERE type IN ('table', 'view') "
                "AND name NOT GLOB 'sqlite_*' AND sql IS NOT NULL ORDER BY type, name"
            )
        ]
    if not statements:
        raise ValueError(f"数据库 {db_id} 无可用 Schema")
    return ";\n".join(sql.rstrip().removesuffix(";") for sql in statements) + ";"


def generate_sql(question, trace):
    schema = read_schema(question["db_id"])
    user_message = (
        f"Database: {question['db_id']}\n"
        f"Schema:\n{schema}\n\n"
        f"Evidence:\n{question['evidence']}\n\n"
        f"Question:\n{question['question']}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "seed": GENERATION_SEED,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    record(trace, "model_request", question_id=question["id"], payload=payload)
    with build_opener(ProxyHandler({})).open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.load(response)
    record(trace, "model_response", question_id=question["id"], response=data)

    choice = data["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"模型未正常结束：{choice.get('finish_reason')}")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型返回了空 SQL")
    sql = content.strip()
    fenced = SQL_FENCE.fullmatch(sql)
    if fenced:
        sql = fenced.group(1).strip()
    if not sql:
        raise ValueError("模型返回了空 SQL")
    return sql, data.get("usage")


@lru_cache(maxsize=1)
def read_gold():
    with GOLD_PATH.open(encoding="utf-8-sig") as source:
        rows = [line.strip().rsplit("\t", 1) for line in source if line.strip()]
    if len(rows) != 500 or any(len(pair) != 2 for pair in rows):
        raise ValueError("标准 SQL 文件格式与原版 Mini-Dev 不匹配")
    return rows


def score_prediction(question, sql):
    reference_sql, reference_db = read_gold()[question["id"]]
    if reference_db != question["db_id"]:
        raise ValueError(f"题目 {question['id']} 的数据库与标准 SQL 不匹配")
    return evaluate_sql(reference_db, sql, reference_sql)


def run_question(question, trace):
    result = {"question_id": question["id"], "db_id": question["db_id"], "predicted_sql": None, "ex": 0}
    start = perf_counter()
    try:
        sql, usage = generate_sql(question, trace)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        result.update(status="http_error", error=f"HTTP {error.code}: {detail}")
    except (URLError, TimeoutError, sqlite3.Error, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        result.update(status="generation_error", error=str(error))
    else:
        result.update(predicted_sql=sql, usage=usage)

    result["generation_seconds"] = perf_counter() - start
    if result["predicted_sql"] is not None:
        try:
            score = score_prediction(question, result["predicted_sql"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            result.update(status="evaluation_error", error=str(error))
        else:
            result.update(status="evaluated", ex=score["ex"], evaluation=score)
    return result


def write_json(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("smoke", "held-out"), default="smoke",
        help="运行 22 道冒烟题或正式保留的 478 道题",
    )
    parser.add_argument("--limit", type=int, help="仅运行冒烟清单的前 N 题")
    args = parser.parse_args()
    if args.limit is not None:
        if args.split != "smoke":
            parser.error("--limit 仅适用于 smoke 划分")
        if not 1 <= args.limit <= 22:
            parser.error("--limit 必须在 1 到 22 之间")

    questions, question_sha256 = load_questions()
    splits = split_questions(questions)
    selected = splits[args.split][:args.limit]
    run_dir = LOG_ROOT / f"bird-direct-{args.split}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "manifest.json", {
        "question_sha256": question_sha256,
        "id_definition": "zero-based row index in the original SQLite question file",
        "split_seed": SPLIT_SEED,
        "selected_split": args.split,
        "splits": {name: [question["id"] for question in cases] for name, cases in splits.items()},
        "selected_ids": [question["id"] for question in selected],
        "model_alias": MODEL,
        "endpoint": ENDPOINT,
        "temperature": 0,
        "generation_seed": GENERATION_SEED,
        "max_tokens": MAX_TOKENS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "ex_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "sqlite_version": sqlite3.sqlite_version,
    })
    print(f"A 组日志（{args.split}，{len(selected)} 题）：{run_dir}", flush=True)

    results = []
    with (run_dir / "trace.jsonl").open("x", encoding="utf-8") as trace:
        with (run_dir / "results.jsonl").open("x", encoding="utf-8") as output:
            for question in selected:
                result = run_question(question, trace)
                record(output, "question_result", **result)
                results.append(result)
                evaluation = result.get("evaluation")
                status = evaluation["prediction"]["status"] if evaluation else result["status"]
                print(f"题目 {question['id']} ({question['db_id']}): EX={result['ex']}，执行状态={status}", flush=True)

    correct = sum(result["ex"] for result in results)
    executable = sum(
        result.get("evaluation", {}).get("prediction", {}).get("status") == "ok" for result in results
    )
    reference_timeouts = sum(
        result.get("evaluation", {}).get("reference", {}).get("status") == "timeout" for result in results
    )
    summary = {
        "strategy": "direct_sql", "split": args.split, "total": len(results),
        "ex_correct": correct, "ex": correct / len(results),
        "sql_executable": executable, "sql_execution_rate": executable / len(results),
        "reference_timeouts": reference_timeouts,
        "mean_generation_seconds": sum(result["generation_seconds"] for result in results) / len(results),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"{args.split} EX：{correct}/{len(results)}，SQL 可执行：{executable}/{len(results)}，"
        f"标准 SQL 超时：{reference_timeouts}"
    )
    return 1 if any(result["status"] != "evaluated" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
