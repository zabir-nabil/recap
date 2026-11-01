#!/usr/bin/env python3
"""Add templated question text and placeholder mappings to sampled questions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    kind: str  # "table" | "col" | "entity"
    value: str
    source_name: str | None = None


@dataclass(frozen=True)
class SchemaTerm:
    kind: str
    pattern: re.Pattern[str]
    source_name: str | None = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def normalize_identifier(text: str) -> str:
    # Split camelCase / PascalCase before applying regular normalization.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return normalize_text(text)


def org_variants(text: str) -> Set[str]:
    variants = {text}
    variants.add(text.replace("organisation", "organization"))
    variants.add(text.replace("organisations", "organizations"))
    variants.add(text.replace("organization", "organisation"))
    variants.add(text.replace("organizations", "organisations"))
    return {v for v in variants if v}


def singular_plural_variants(text: str) -> Set[str]:
    variants = {text}
    tokens = text.split()
    if not tokens:
        return variants

    last = tokens[-1]
    if len(last) > 3 and last.endswith("ies"):
        variants.add(" ".join(tokens[:-1] + [last[:-3] + "y"]))
    if len(last) > 3 and last.endswith("s"):
        variants.add(" ".join(tokens[:-1] + [last[:-1]]))
    elif len(last) > 2:
        variants.add(" ".join(tokens[:-1] + [last + "s"]))
    return {v for v in variants if v}


def generate_variants(raw: str) -> Set[str]:
    base = normalize_text(raw)
    if not base:
        return set()

    variants = set()
    for v in org_variants(base):
        variants.update(singular_plural_variants(v))
    return {v for v in variants if v}


LINKING_WORDS = {"of", "for", "from", "to", "in", "by", "with"}


def generate_column_variants(raw: str) -> Set[str]:
    # Columns are often referenced with reordered phrasing in NL questions,
    # e.g. "transaction amount" vs schema "amount_of_transaction".
    variants = set(generate_variants(raw))
    derived: Set[str] = set()

    for phrase in list(variants):
        tokens = phrase.split()
        content = [t for t in tokens if t not in LINKING_WORDS]
        if len(content) == 2:
            a, b = content
            derived.add(f"{a} {b}")
            derived.add(f"{b} {a}")

    for phrase in list(derived):
        variants.update(singular_plural_variants(phrase))

    return {v for v in variants if v}


def compile_phrase_pattern(normalized_phrase: str) -> re.Pattern[str]:
    words = normalized_phrase.split()
    body = r"[\s_]*".join(re.escape(word) for word in words)
    pattern = rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def tokens_with_aliases(text: str) -> List[str]:
    return [KEY_TOKEN_MAP.get(tok, tok) for tok in normalize_identifier(text).split()]


def score_token_overlap(a: str, b: str) -> float:
    a_tokens = set(tokens_with_aliases(a))
    b_tokens = set(tokens_with_aliases(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def choose_best_column_name(key: str, candidate_names: Set[str]) -> str | None:
    if not candidate_names:
        return None
    ranked = sorted(
        candidate_names,
        key=lambda name: (
            -score_token_overlap(key, name),
            abs(len(tokens_with_aliases(name)) - len(tokens_with_aliases(key))),
            len(name),
            name,
        ),
    )
    best = ranked[0]
    if score_token_overlap(key, best) <= 0:
        return None
    return best


def build_schema_terms(
    schema: Dict[str, Any]
) -> Tuple[List[SchemaTerm], List[SchemaTerm], Set[str], Dict[str, Set[str]]]:
    table_terms: List[SchemaTerm] = []
    col_terms: List[SchemaTerm] = []
    schema_vocab: Set[str] = set()
    col_variant_to_names: Dict[str, Set[str]] = {}

    table_names = schema["table_names"]
    table_names_original = schema["table_names_original"]
    for tid, table_name in enumerate(table_names):
        table_original = table_name
        if tid < len(table_names_original):
            table_original = table_names_original[tid]
        originals = {table_name, table_original}
        for raw in originals:
            for variant in generate_variants(raw):
                schema_vocab.add(variant)
                table_terms.append(
                    SchemaTerm(
                        kind="table",
                        pattern=compile_phrase_pattern(variant),
                        source_name=table_original,
                    )
                )

    column_names = schema["column_names"]
    column_names_original = schema["column_names_original"]
    for cid, (table_id, column_name) in enumerate(column_names):
        if table_id == -1:
            continue
        table_original = table_names[table_id]
        if table_id < len(table_names_original):
            table_original = table_names_original[table_id]
        column_original = column_name
        if cid < len(column_names_original):
            column_original = column_names_original[cid][1]
        qualified_column_original = f"{table_original}.{column_original}"
        originals = {column_name, column_original}
        for raw in originals:
            for variant in generate_column_variants(raw):
                schema_vocab.add(variant)
                col_terms.append(
                    SchemaTerm(
                        kind="col",
                        pattern=compile_phrase_pattern(variant),
                        source_name=qualified_column_original,
                    )
                )
                col_variant_to_names.setdefault(variant, set()).add(qualified_column_original)

    table_terms.sort(key=lambda t: (-len(t.pattern.pattern), t.source_name or ""))
    col_terms.sort(key=lambda t: (-len(t.pattern.pattern), t.source_name or ""))
    return table_terms, col_terms, schema_vocab, col_variant_to_names


def is_overlap(start: int, end: int, matches: Sequence[Match]) -> bool:
    for m in matches:
        if start < m.end and m.start < end:
            return True
    return False


def select_non_overlapping(matches: Sequence[Match]) -> List[Match]:
    priority = {"col": 0, "table": 1, "entity": 2}
    ordered = sorted(
        matches,
        key=lambda m: (-(m.end - m.start), priority.get(m.kind, 10), m.start, m.end),
    )
    chosen: List[Match] = []
    for m in ordered:
        if not is_overlap(m.start, m.end, chosen):
            chosen.append(m)
    return sorted(chosen, key=lambda m: m.start)


def phrase_match_pattern(raw_phrase: str) -> re.Pattern[str]:
    if not raw_phrase:
        return re.compile(r"$^")
    escaped = re.escape(raw_phrase)
    prefix = r"(?<![A-Za-z0-9])" if raw_phrase[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9])" if raw_phrase[-1].isalnum() else ""
    return re.compile(prefix + escaped + suffix, flags=re.IGNORECASE)


def iter_scalar_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from iter_scalar_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_scalar_strings(v)
    elif isinstance(value, str):
        yield value


def iter_dict_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                yield k
            yield from iter_dict_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_dict_keys(v)


KEY_TOKEN_MAP = {
    "apt": "apartment",
    "num": "number",
    "no": "number",
}

CONTEXT_PREFIXES = {
    "apartment",
    "booking",
    "customer",
    "document",
    "guest",
    "invoice",
    "lot",
    "order",
    "organisation",
    "organization",
    "product",
    "project",
    "research",
    "shipment",
    "staff",
    "transaction",
}


def key_to_column_phrases(key: str) -> Set[str]:
    normalized = normalize_identifier(key)
    if not normalized:
        return set()
    if re.fullmatch(r"col \d+", normalized):
        return set()

    tokens = [KEY_TOKEN_MAP.get(tok, tok) for tok in normalized.split()]
    if not tokens:
        return set()

    phrases: Set[str] = {" ".join(tokens)}
    if len(tokens) >= 3 and tokens[0] in CONTEXT_PREFIXES:
        phrases.add(" ".join(tokens[1:]))
    if len(tokens) >= 4 and tokens[0] in CONTEXT_PREFIXES and tokens[1] in CONTEXT_PREFIXES:
        phrases.add(" ".join(tokens[2:]))
    if len(tokens) >= 2 and tokens[-1] == "date" and tokens[-2] in {"start", "end"}:
        phrases.add(" ".join(tokens[-2:]))

    variants: Set[str] = set()
    for phrase in phrases:
        if phrase:
            variants.update(generate_variants(phrase))
    return {v for v in variants if len(v) > 2}


def collect_schema_matches(question: str, table_terms: Sequence[SchemaTerm], col_terms: Sequence[SchemaTerm]) -> List[Match]:
    candidates: List[Match] = []
    for term in col_terms:
        for found in term.pattern.finditer(question):
            candidates.append(
                Match(
                    start=found.start(),
                    end=found.end(),
                    kind="col",
                    value=question[found.start() : found.end()],
                    source_name=term.source_name,
                )
            )
    for term in table_terms:
        for found in term.pattern.finditer(question):
            candidates.append(
                Match(
                    start=found.start(),
                    end=found.end(),
                    kind="table",
                    value=question[found.start() : found.end()],
                    source_name=term.source_name,
                )
            )
    return select_non_overlapping(candidates)


def collect_ground_truth_key_column_matches(
    question: str,
    ground_truth_raw: str,
    blocked: Sequence[Match],
    col_variant_to_names: Dict[str, Set[str]],
) -> List[Match]:
    if not ground_truth_raw:
        return []
    try:
        parsed = json.loads(ground_truth_raw)
    except json.JSONDecodeError:
        return []

    key_to_phrases: Dict[str, Set[str]] = {}
    for key in iter_dict_keys(parsed):
        key_to_phrases[key] = key_to_column_phrases(key)

    matches: List[Match] = []
    for key, phrases in key_to_phrases.items():
        candidate_names: Set[str] = set()
        for phrase in phrases:
            candidate_names.update(col_variant_to_names.get(phrase, set()))
        column_name = choose_best_column_name(key, candidate_names)
        if not column_name:
            continue
        for phrase in sorted(phrases, key=len, reverse=True):
            pattern = compile_phrase_pattern(phrase)
            for found in pattern.finditer(question):
                if is_overlap(found.start(), found.end(), blocked + matches):
                    continue
                matches.append(
                    Match(
                        start=found.start(),
                        end=found.end(),
                        kind="col",
                        value=question[found.start() : found.end()],
                        source_name=column_name,
                    )
                )
    return matches


def collect_quoted_entity_matches(question: str, blocked: Sequence[Match], schema_vocab: Set[str]) -> List[Match]:
    quote_pattern = re.compile(r"'([^']+)'|\"([^\"]+)\"")
    entities: List[Match] = []
    for found in quote_pattern.finditer(question):
        if is_overlap(found.start(), found.end(), blocked):
            continue
        inner = found.group(1) if found.group(1) is not None else found.group(2)
        if not inner:
            continue
        normalized = normalize_text(inner)
        if not normalized or normalized in schema_vocab:
            continue
        entities.append(
            Match(
                start=found.start(),
                end=found.end(),
                kind="entity",
                value=inner.strip(),
            )
        )
    return entities


def collect_ground_truth_entity_matches(
    question: str,
    ground_truth_raw: str,
    blocked: Sequence[Match],
    schema_vocab: Set[str],
) -> List[Match]:
    if not ground_truth_raw:
        return []
    try:
        parsed = json.loads(ground_truth_raw)
    except json.JSONDecodeError:
        return []

    values: Set[str] = set()
    for scalar in iter_scalar_strings(parsed):
        candidate = scalar.strip()
        if len(candidate) < 2:
            continue
        normalized = normalize_text(candidate)
        if not normalized or normalized in schema_vocab:
            continue
        values.add(candidate)

    entities: List[Match] = []
    for candidate in sorted(values, key=len, reverse=True):
        pattern = phrase_match_pattern(candidate)
        for found in pattern.finditer(question):
            if is_overlap(found.start(), found.end(), blocked + entities):
                continue
            entities.append(
                Match(
                    start=found.start(),
                    end=found.end(),
                    kind="entity",
                    value=question[found.start() : found.end()],
                )
            )
    return entities


def apply_replacements(text: str, replacements: Sequence[Tuple[int, int, str]]) -> str:
    pieces: List[str] = []
    cursor = 0
    for start, end, repl in sorted(replacements, key=lambda x: x[0]):
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        pieces.append(repl)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def template_question(
    question: str,
    ground_truth_raw: str,
    table_terms: Sequence[SchemaTerm],
    col_terms: Sequence[SchemaTerm],
    schema_vocab: Set[str],
    col_variant_to_names: Dict[str, Set[str]],
) -> Tuple[str, List[Dict[str, str]]]:
    schema_matches = collect_schema_matches(question, table_terms, col_terms)
    gt_key_cols = collect_ground_truth_key_column_matches(question, ground_truth_raw, schema_matches, col_variant_to_names)
    quote_entities = collect_quoted_entity_matches(question, schema_matches + gt_key_cols, schema_vocab)
    gt_entities = collect_ground_truth_entity_matches(question, ground_truth_raw, schema_matches + gt_key_cols + quote_entities, schema_vocab)
    all_matches = select_non_overlapping(schema_matches + gt_key_cols + quote_entities + gt_entities)

    placeholder_prefix = {"table": "TABLE", "col": "COL", "entity": "ENT"}
    counters = {"table": 0, "col": 0, "entity": 0}
    key_to_placeholder: Dict[Tuple[str, str, str], str] = {}
    placeholders: List[Dict[str, str]] = []
    replacements: List[Tuple[int, int, str]] = []

    for match in all_matches:
        if match.kind == "col" and not (match.source_name or "").strip():
            continue
        norm_value = normalize_text(match.value)
        source_name = match.source_name or ""
        key = (match.kind, norm_value, source_name)
        if key not in key_to_placeholder:
            counters[match.kind] += 1
            placeholder = f"{{{placeholder_prefix[match.kind]}_{counters[match.kind]}}}"
            key_to_placeholder[key] = placeholder
            placeholder_item: Dict[str, str] = {"name": placeholder, "value": match.value}
            if match.kind == "col":
                placeholder_item["column_name"] = source_name
            if match.kind == "table":
                placeholder_item["table_name"] = source_name
            placeholders.append(placeholder_item)
        replacements.append((match.start, match.end, key_to_placeholder[key]))

    template = apply_replacements(question, replacements)
    return template, placeholders


def enrich_question_item(
    item: Dict[str, Any],
    schema_index: Dict[str, Tuple[List[SchemaTerm], List[SchemaTerm], Set[str], Dict[str, Set[str]]]],
) -> Tuple[str, List[Dict[str, str]]]:
    question = item.get("Question", "")
    db_name = item.get("DBName", "")
    ground_truth_raw = item.get("GroundTruthRowsString", "")
    if db_name not in schema_index:
        return question, []

    table_terms, col_terms, schema_vocab, col_variant_to_names = schema_index[db_name]
    return template_question(question, ground_truth_raw, table_terms, col_terms, schema_vocab, col_variant_to_names)


def iter_all_question_items(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in data.get("questions", []):
        if isinstance(item, dict):
            yield item

    for group in data.get("questions_by_group", []):
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, dict):
                yield item

    by_db = data.get("questions_by_database", {})
    if isinstance(by_db, dict):
        for items in by_db.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Add question templates and placeholders.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data/sampled_questions_templated.json")
    parser.add_argument("--in-place", action="store_true", help="Write changes back to --questions")
    args = parser.parse_args()
    for option in ('questions', 'tables'):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")

    questions_data = json.loads(args.questions.read_text(encoding="utf-8"))
    tables_data = json.loads(args.tables.read_text(encoding="utf-8"))

    schema_by_db = {schema["db_id"]: schema for schema in tables_data}
    schema_index = {db: build_schema_terms(schema) for db, schema in schema_by_db.items()}

    templates_by_qid: Dict[str, Tuple[str, List[Dict[str, str]]]] = {}
    updated_count = 0
    for item in iter_all_question_items(questions_data):
        qid = item.get("QuestionID")
        if isinstance(qid, str) and qid in templates_by_qid:
            template, placeholders = templates_by_qid[qid]
        else:
            template, placeholders = enrich_question_item(item, schema_index)
            if isinstance(qid, str):
                templates_by_qid[qid] = (template, placeholders)
        item["template"] = template
        item["placeholders"] = placeholders
        updated_count += 1

    output_path = args.questions if args.in_place else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(questions_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {updated_count} question records.")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
