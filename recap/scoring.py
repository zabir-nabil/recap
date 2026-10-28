"""Selected deterministic final-answer scoring policy."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


NUMBER = re.compile(
    r"(?<![\w.])[-+]?[$€£]?\s*\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"\s*(?:%|percent|percentage\s+points?|USD|dollars?|euros?|pounds?|"
    r"thousand|million|billion|trillion|mn|bn)?(?!\w)", re.I)
PREFIX = re.compile(r"^\s*(?:(?:the\s+)?final\s+answer|answer|result)\s*(?:is|:)\s*", re.I)
DATE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
YEAR = re.compile(r"\b(?:FY\s*)?(?:19|20)\d{2}\b", re.I)
EMPTY_SQL = {"[]", "no rows", "no results", "empty result", "empty set"}
ORDERING_RE = re.compile(
    r"\b(order(?:ed)?\s+by|sort(?:ed)?|ascending|descending|"
    r"chronological(?:ly)?|alphabetical(?:ly)?|rank(?:ed)?\s+by|"
    r"in\s+order\s+of|arranged\s+(?:by|alphabetically|in)|"
    r"top\s+\d+|bottom\s+\d+|first\s+\d+|last\s+\d+|"
    r"highest\s+to\s+lowest|lowest\s+to\s+highest|"
    r"earliest\s+to\s+latest|latest\s+to\s+earliest|"
    r"in\s+(?:increasing|decreasing)\s+order|"
    r"from\s+(?:highest|lowest|largest|smallest|earliest|latest))\b",
    re.I,
)


def question_requires_order(question: str) -> bool:
    return bool(ORDERING_RE.search(question or ""))


def clean(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = PREFIX.sub("", text).strip().strip("`").strip()
    return re.sub(r"\s+", " ", text).strip(" .\n\t")


def norm(value: object) -> str:
    return clean(value).casefold()


def numeric_token(value: object) -> tuple[Decimal, str] | None:
    text = clean(value)
    match = re.fullmatch(
        r"(?i)\s*([$€£])?\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
        r"(%|percent|percentage\s+points?|usd|dollars?|euros?|pounds?|"
        r"thousand|million|billion|trillion|mn|bn)?\s*",
        text,
    )
    if not match:
        return None
    try:
        value = Decimal(match.group(2).replace(",", ""))
    except InvalidOperation:
        return None
    unit = (match.group(3) or match.group(1) or "").casefold()
    return value, unit


def equivalent_number(predicted: object, expected: object, *, discrete: bool = False,
                      question: str = "") -> bool:
    p = numeric_token(predicted)
    e = numeric_token(expected)
    if p is None or e is None:
        return False
    pv, pu = p
    ev, eu = e
    scale = {"thousand": Decimal(1000), "million": Decimal(10) ** 6,
             "mn": Decimal(10) ** 6, "billion": Decimal(10) ** 9,
             "bn": Decimal(10) ** 9, "trillion": Decimal(10) ** 12}
    if pu in scale and eu in scale:
        pv = pv * scale[pu] / scale[eu]
    elif pu in scale and eu in {"$", "€", "£", "usd", "dollar", "dollars"}:
        if pu not in question.casefold():
            return False
    elif eu in scale and pu in {"$", "€", "£", "usd", "dollar", "dollars"}:
        if eu not in question.casefold():
            return False
    elif pu and eu and pu != eu:
        aliases = ({"%", "percent"}, {"$", "usd", "dollar", "dollars"},
                   {"€", "euro", "euros"}, {"£", "pound", "pounds"})
        if not any({pu, eu} <= group for group in aliases):
            return False
    if discrete or ev == ev.to_integral() and "." not in str(expected):
        return pv == ev
    last_unit = Decimal(1).scaleb(ev.as_tuple().exponent)
    return pv.quantize(last_unit, rounding=ROUND_HALF_UP) == ev


def one_answer_number(text: str, question: str) -> str | None:
    """Find one number after excluding dates and years supplied by the question."""
    text = DATE.sub(" ", text)
    question_years = {m.group().lstrip("FYfy ") for m in YEAR.finditer(question)}
    found = []
    for match in NUMBER.finditer(text):
        token = match.group().strip()
        if (match.start() >= 2 and text[match.start() - 1] == "-"
                and text[match.start() - 2].isalpha()):
            continue
        bare = token.replace(",", "").strip()
        if (bare in question_years and not re.search(r"[$€£%]|\b(?:mn|bn|million|billion)\b", token, re.I)
                and len(text.strip()) > len(token) + 5):
            continue
        found.append(token)
    return found[0] if len(found) == 1 else None


def parse_rows(value: object) -> list[dict | list | tuple | object] | None:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n|\n```\s*$", "", text).strip()
    candidates = [text]
    for left, right in (("[", "]"), ("{", "}")):
        start, end = text.find(left), text.rfind(right)
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])
    for candidate in candidates:
        if not candidate.lstrip().startswith(("[", "{", "(")):
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                rows = parser(candidate)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(rows, dict):
                return [rows]
            if isinstance(rows, (list, tuple)):
                return list(rows)
    return None


def cell_equal(predicted: object, expected: object) -> bool:
    if predicted is None or expected is None:
        return predicted is None and expected is None
    if norm(predicted) == norm(expected):
        return True
    return equivalent_number(predicted, expected)


def row_equal(predicted: object, expected: object) -> bool:
    if isinstance(expected, dict):
        if isinstance(predicted, dict):
            exp_keys = {str(k).casefold(): k for k in expected}
            pred_keys = {str(k).casefold(): k for k in predicted}
            if exp_keys.keys() <= pred_keys.keys():
                return all(cell_equal(predicted[pred_keys[k]], expected[exp_keys[k]])
                           for k in exp_keys)
            if len(predicted) != len(expected):
                return False
            remaining = list(predicted.values())
            for gold_value in expected.values():
                index = next((i for i, value in enumerate(remaining)
                              if cell_equal(value, gold_value)), None)
                if index is None:
                    return False
                remaining.pop(index)
            return not remaining
        if isinstance(predicted, (list, tuple)):
            return len(predicted) == len(expected) and all(
                cell_equal(p, e) for p, e in zip(predicted, expected.values()))
        return len(expected) == 1 and cell_equal(predicted, next(iter(expected.values())))
    if isinstance(expected, (list, tuple)):
        return isinstance(predicted, (list, tuple)) and len(predicted) == len(expected) and all(
            cell_equal(p, e) for p, e in zip(predicted, expected))
    return cell_equal(predicted, expected)


def rows_equal(predicted: list, expected: list, ordered: bool) -> bool:
    if len(predicted) != len(expected):
        return False
    if ordered:
        return all(row_equal(p, e) for p, e in zip(predicted, expected))
    remaining = list(predicted)
    for gold_row in expected:
        index = next((i for i, row in enumerate(remaining) if row_equal(row, gold_row)), None)
        if index is None:
            return False
        remaining.pop(index)
    return not remaining


def score_sql(predicted: str, expected: str, question: str) -> tuple[bool, str]:
    ordered = question_requires_order(question)

    gold_rows = parse_rows(expected)
    if gold_rows is None:
        return False, "sql_gold_unparseable"
    if not clean(predicted):
        return False, "empty"
    pred_rows = parse_rows(predicted)
    if pred_rows is not None:
        if (len(gold_rows) == 1 and isinstance(gold_rows[0], dict)
                and len(gold_rows[0]) > 1 and len(pred_rows) == len(gold_rows[0])
                and all(not isinstance(row, (dict, list, tuple)) for row in pred_rows)):
            pred_rows = [pred_rows]
        return rows_equal(pred_rows, gold_rows, ordered), "sql_rows"
    if not gold_rows:
        return norm(predicted) in EMPTY_SQL, "sql_empty"
    if len(gold_rows) == 1:
        row = gold_rows[0]
        values = list(row.values()) if isinstance(row, dict) else (
            list(row) if isinstance(row, (list, tuple)) else [row])
        if len(values) == 1:
            value = values[0]
            number = one_answer_number(predicted, question)
            return (cell_equal(predicted, value) or
                    (number is not None and equivalent_number(number, value))), "sql_one_cell"
        rendered = " ".join(str(v) for v in values)
        # Permit punctuation as a separator, but do not accept extra prose/items.
        delimited = lambda s: re.sub(r"\s+", " ", re.sub(r"[,;|]+", " ", norm(s))).strip()
        return delimited(predicted) == delimited(rendered), "sql_one_row_text"
    if all(isinstance(r, dict) and len(r) == 1 for r in gold_rows):
        parts = [part.strip(" -•\t") for part in re.split(r"[,;\n]", predicted) if part.strip()]
        candidates = [[{next(iter(row)): part} for part, row in zip(parts, gold_rows)]] if len(parts) == len(gold_rows) else []
        return any(rows_equal(rows, gold_rows, ordered) for rows in candidates), "sql_one_column_text"
    return False, "sql_unstructured_multirow"


def score_simple(dataset: str, entry: dict, record: dict | None = None) -> tuple[bool, str]:
    predicted = str(entry.get("agent_answer") or "")
    expected = str(entry.get("expected_answer") or "")
    question = str(entry.get("question") or "")
    if not clean(predicted):
        return False, "empty"
    if dataset in {"sqlapibench", "sql_api_bench"}:
        return score_sql(predicted, expected, question)
    if norm(predicted) == norm(expected):
        return True, "normalized_exact"
    if dataset == "financebench" and re.match(r"^\s*(yes|no)\b", expected, re.I):
        gold = re.match(r"^\s*(yes|no)\b", expected, re.I).group(1).casefold()
        answer = re.match(r"^\s*(yes|no)\b", clean(predicted), re.I)
        return bool(answer and answer.group(1).casefold() == gold), "yes_no"
    gold_number = one_answer_number(expected, question)
    if dataset == "financebench":
        numeric_question = bool(re.search(
            r"\b(how many|how much|what (?:was|is|were|are)\b.*\b"
            r"(?:amount|value|total|ratio|price|ebitda|margin|income|revenue|"
            r"sales|cost|turnover|liabilit\w*|asset\w*|balance|percentage|percent|rate))\b",
            question, re.I))
        numeric_gold = (numeric_token(expected) is not None or
                        (record or {}).get("answer_type") == "numeric" or
                        (numeric_question and gold_number is not None))
    else:
        numeric_gold = numeric_token(expected) is not None
    if numeric_gold:
        candidate = one_answer_number(predicted, question)
        if candidate is None:
            return False, "numeric_ambiguous"
        discrete = bool(re.search(r"\b(how many|what year|which year|number of|count of)\b", question, re.I))
        return equivalent_number(candidate, gold_number or expected,
                                 discrete=discrete, question=question), "numeric_single"
    return False, "text_not_exact"
