"""Bounded value grounding, column descriptions and diagnostics for BIRD v2."""

import csv
import difflib
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from time import monotonic


LOOKUP_LIMIT = 5
LOOKUP_TIMEOUT_SECONDS = 2.0
VALUE_TEXT_LIMIT = 256
COLUMN_DESCRIPTION_LIMIT = 6
COLUMN_DESCRIPTION_CHAR_LIMIT = 1200
COLUMN_DESCRIPTION_STOPWORDS = set("""
a an and are as at be been between but by can could did do does for from gives given has have
how in into is it its list mean name number of on only or records refers s shows that the their
these they this those to use used was were what when where which who with within year
""".split())
_QUOTED_VALUE = r"""["']([^"']+)["']"""
_ENUMERATION = re.compile(r"""["'][^"']+["'](?:\s*,\s*["'][^"']+["'])+""")
_VALUE_MAPPING = re.compile(
    r"""^("[^"]+"|'[^']+'|[A-Za-z0-9][\w -]{0,30})\s*(?::| – | - | stands for\b)""",
    re.IGNORECASE,
)
LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_values",
        "description": (
            "Find actual values in a specified database column using a short search fragment. "
            "Matches are case-insensitive substrings in either direction. An empty search_text "
            "returns a small sample. Returned candidates require interpretation; a match alone "
            "does not establish that it satisfies the user's requirement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "column": {"type": "string"},
                "search_text": {"type": "string", "maxLength": 128},
            },
            "required": ["table", "column", "search_text"],
            "additionalProperties": False,
        },
    },
}
EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Submit one read-only SQLite candidate query for actual execution.",
        "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}},
            "required": ["query"], "additionalProperties": False,
        },
    },
}

# Generic dialect equivalents. No question IDs, reference SQL or gold results.
SQLITE_EQUIVALENTS = {
    "year": "YEAR(x): CAST(strftime('%Y', x) AS INTEGER)",
    "month": "MONTH(x): CAST(strftime('%m', x) AS INTEGER)",
    "curdate": "CURDATE(): date('now')",
    "now": "NOW(): datetime('now')",
    "subtract": "SUBTRACT(x, y): (x - y)",
    "divide": "DIVIDE(x, y): (CAST(x AS REAL) / y)",
    "multiply": "MULTIPLY(x, y): (x * y)",
}


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def resolve_identifier(name, available):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("表名和字段名必须是非空字符串")
    matches = [actual for actual in available if actual.casefold() == name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Schema 中不存在唯一匹配的标识符：{name}")
    return matches[0]


def load_column_descriptions(directory):
    """Read one database's CSV documentation and hash the bytes used for it."""
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"找不到字段说明 CSV：{directory}")
    descriptions = []
    hashes = {}
    for path in paths:
        data = path.read_bytes()
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252")
        reader = csv.reader(io.StringIO(text, newline=""))
        header = ["original_column_name", "column_name", "column_description", "data_format", "value_description"]
        if next(reader, [])[:len(header)] != header:
            raise ValueError(f"无法识别字段说明 CSV 表头：{path}")
        for row in reader:
            if len(row) < len(header) or not row[0].strip():
                continue
            column, label, definition = (" ".join(value.split()) for value in row[:3])
            definition = definition or label
            if _description_phrase(definition).replace(" ", "") == _description_phrase(column).replace(" ", ""):
                definition = ""
            # A source CSV can carry the value notes in a trailing unnamed cell.
            descriptions.append((path.stem, column, definition, "\n".join(row[4:])))
    return descriptions, hashes


def _description_phrase(text):
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()))


def _description_words(text):
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = _description_phrase(separated).split()
    return {
        word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")) else word
        for word in words if len(word) > 1 and word not in COLUMN_DESCRIPTION_STOPWORDS
    }


def _mentioned_description_value(text, task_phrase):
    normalized = _description_phrase(text)
    return bool(normalized and f" {normalized} " in task_phrase)


def _matching_value_notes(values, task_phrase, quoted_task_values):
    for source_line in values.splitlines():
        line = " ".join(source_line.split()).strip(" \u2022\u00b7\u0095")
        if not 0 < len(line) <= 155:
            continue
        mapping = _VALUE_MAPPING.match(line)
        if _ENUMERATION.fullmatch(line):
            keys = re.findall(_QUOTED_VALUE, line)
        elif mapping:
            keys = [mapping[1]]
        else:
            continue
        for key in keys:
            normalized = _description_phrase(key)
            if len(normalized) == 1 and normalized not in quoted_task_values:
                continue
            if _mentioned_description_value(key, task_phrase):
                yield line
                break


def select_column_descriptions(descriptions, tables, question, evidence):
    """Select compact, complete snippets using the task text and actual schema."""
    task = question + "\n" + evidence
    task_phrase = f" {_description_phrase(task)} "
    task_words = _description_words(task)
    quoted_task_values = {_description_phrase(value) for value in re.findall(_QUOTED_VALUE, task)}
    known_columns = {
        (table.casefold(), column.casefold()): (table, column)
        for table, metadata in tables.items() for column in metadata["columns"]
    }
    ranked = []
    for table, column, definition, values in descriptions:
        identifiers = known_columns.get((table.casefold(), column.casefold()))
        if identifiers is None:
            continue
        table, column = identifiers
        column_words = _description_words(column)
        explicit_column = bool(column_words and _mentioned_description_value(column, task_phrase))
        table_match = bool(_description_words(table) & task_words)
        column_match = column_words & task_words
        definition_match = _description_words(definition) & task_words
        notes = _matching_value_notes(values, task_phrase, quoted_task_values)
        notes = list(dict.fromkeys(notes))[:2]
        if not (notes or explicit_column or len(column_match) >= 2
                or (table_match and column_match and len(definition_match) >= 2)
                or len(definition_match) >= 3):
            continue
        parts = [part for part in (definition, *notes) if 0 < len(part) <= 160]
        line = f"{table}.{column}: " + " | ".join(parts)
        if not parts or len(line) > 240:
            continue
        score = (8 * bool(notes) + 6 * explicit_column + 2 * table_match
                 + 2 * len(column_match) + len(definition_match - column_match))
        ranked.append((-score, table.casefold(), column.casefold(), line))

    selected = []
    seen = set()
    chars = 0
    for _, table, column, line in sorted(ranked):
        length = len(line) + bool(selected)
        if (table, column) in seen or chars + length > COLUMN_DESCRIPTION_CHAR_LIMIT:
            continue
        selected.append(line)
        seen.add((table, column))
        chars += length
        if len(selected) == COLUMN_DESCRIPTION_LIMIT:
            break
    return selected


def validate_query_plan(content, tables):
    """Validate the plan format and resolve every declared column to its schema name."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("查询计划必须是非空 JSON 对象")
    content = content.strip()
    fenced = re.fullmatch(r"\x60{3}(?:json)?[ \t]*\r?\n(.*?)\r?\n?\x60{3}", content, re.IGNORECASE | re.DOTALL)
    plan = json.loads(fenced[1] if fenced else content)
    if not isinstance(plan, dict) or plan.keys() != {"outputs", "filters", "joins", "grain"}:
        raise ValueError("查询计划必须包含 outputs、filters、joins、grain")
    available = {}
    for table, metadata in tables.items():
        for column in metadata["columns"]:
            name = f"{table}.{column}"
            available.setdefault(name.casefold(), []).append(name)

    def known_column(ref):
        matches = available.get(ref.strip().casefold(), ()) if isinstance(ref, str) else ()
        if len(matches) != 1:
            raise ValueError(f"计划中找不到唯一的表.列：{ref!r}")
        return matches[0]

    def require_text(value, context):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} 必须是非空文本")
        return value.strip()

    if any(not isinstance(plan[key], list) for key in ("outputs", "filters", "joins")) or not plan["outputs"]:
        raise ValueError("outputs、filters、joins 必须是列表，且 outputs 不能为空")

    outputs = []
    for output in plan["outputs"]:
        if (not isinstance(output, dict) or output.keys() != {"what", "columns"}
                or not isinstance(output["columns"], list)):
            raise ValueError("outputs 的每项必须包含 what 和 columns 列表")
        outputs.append({
            "what": require_text(output["what"], "输出说明"),
            "columns": [known_column(ref) for ref in output["columns"]],
        })

    filters = []
    for condition in plan["filters"]:
        if not isinstance(condition, dict) or condition.keys() != {"column", "condition"}:
            raise ValueError("filters 的每项必须包含 column 和 condition")
        filters.append({
            "column": known_column(condition["column"]),
            "condition": require_text(condition["condition"], "筛选条件"),
        })

    joins = []
    for join in plan["joins"]:
        if not isinstance(join, list) or len(join) != 2:
            raise ValueError("joins 的每项必须是一对表.列")
        joins.append([known_column(ref) for ref in join])
    return {
        "outputs": outputs, "filters": filters, "joins": joins,
        "grain": require_text(plan["grain"], "结果粒度"),
    }


class DatabaseTools:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.tables = {}
        with closing(self.connect()) as connection:
            objects = connection.execute(
                "SELECT name FROM main.sqlite_schema WHERE type IN ('table', 'view') "
                "AND name NOT GLOB 'sqlite_*' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
            for (table,) in objects:
                columns = connection.execute("SELECT name, type FROM pragma_table_info(?)", (table,)).fetchall()
                keys = connection.execute(
                    'SELECT "from", "table", "to" FROM pragma_foreign_key_list(?)', (table,)
                ).fetchall()
                self.tables[table] = {
                    "columns": dict(columns),
                    "foreign_keys": [
                        {"from_column": source, "to_table": target, "to_column": column}
                        for source, target, column in keys
                    ],
                }

    def connect(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=LOOKUP_TIMEOUT_SECONDS)
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def lookup_values(self, table, column, search_text):
        if not isinstance(search_text, str) or len(search_text) > 128:
            raise ValueError("search_text 必须是长度不超过 128 的字符串")
        table = resolve_identifier(table, self.tables)
        column = resolve_identifier(column, self.tables[table]["columns"])
        # Identifiers come from inspected metadata. Search text and the limit are
        # bound parameters, including quotes, percent signs and underscores.
        expression = f"CAST({quote_identifier(column)} AS TEXT)"
        query = (
            f"SELECT DISTINCT {expression} AS value FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column)} IS NOT NULL AND trim({expression}) <> '' "
            f"AND (instr(lower({expression}), lower(?)) > 0 "
            f"OR instr(lower(?), lower({expression})) > 0) ORDER BY value LIMIT ?"
        )
        deadline = monotonic() + LOOKUP_TIMEOUT_SECONDS
        try:
            with closing(self.connect()) as connection:
                connection.set_progress_handler(lambda: int(monotonic() >= deadline), 1000)
                rows = connection.execute(query, (search_text, search_text, LOOKUP_LIMIT + 1)).fetchall()
                if monotonic() >= deadline:
                    return {"status": "timeout", "error": "实际取值检索超时"}
        except (sqlite3.Error, UnicodeError) as error:
            status = "timeout" if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT else "read_error"
            return {"status": status, "error": str(error)}
        return {
            "status": "ok", "table": table, "column": column, "search_text": search_text,
            "matching_rule": "case-insensitive substring in either direction",
            "values": [
                {"value": value[:VALUE_TEXT_LIMIT], "text_truncated": len(value) > VALUE_TEXT_LIMIT}
                for (value,) in rows[:LOOKUP_LIMIT]
            ],
            "more_matches": len(rows) > LOOKUP_LIMIT, "limit": LOOKUP_LIMIT,
        }

    def diagnose(self, error):
        diagnosis = {"sqlite_error": error, "source": "schema_metadata_and_generic_sqlite_rules"}
        missing = re.fullmatch(r"no such column: (.+)", error)
        function = re.fullmatch(r"no such function: (.+)", error)
        if missing:
            name = missing.group(1).rsplit(".", 1)[-1].strip('`"[] ')
            owners = [
                {"table": table, "column": column, "type": data_type, "foreign_keys": metadata["foreign_keys"]}
                for table, metadata in self.tables.items()
                for column, data_type in metadata["columns"].items()
                if column.casefold() == name.casefold()
            ]
            diagnosis.update(error_type="missing_column", column=name, exact_column_owners=owners)
            if not owners:
                columns = sorted({column for metadata in self.tables.values() for column in metadata["columns"]})
                diagnosis["similar_spellings"] = difflib.get_close_matches(name, columns, n=3, cutoff=0.6)
        elif function:
            diagnosis.update(error_type="unsupported_function", function=function.group(1))
            equivalent = SQLITE_EQUIVALENTS.get(function.group(1).casefold())
            if equivalent:
                diagnosis["sqlite_equivalent"] = equivalent
        else:
            diagnosis["error_type"] = "other_sql_error"
        return diagnosis
