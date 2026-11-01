#!/usr/bin/env python3
"""Generate typed question variants by swapping column placeholders."""

from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"\{[A-Z]+_\d+\}")


def humanize_column_name(qualified_column: str) -> str:
    # "Table.column_name" -> "column name"
    if "." not in qualified_column:
        return qualified_column
    _, column = qualified_column.split(".", 1)
    return column.replace("_", " ")


def parse_registry_key(key: str) -> Tuple[str, str, str]:
    # "db.Table.column" -> ("db", "Table", "column")
    db_name, table_name, column_name = key.split(".", 2)
    return db_name, table_name, column_name


def build_registry_indexes(
    registry: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str, str], List[str]]]:
    lower_key_index: Dict[str, Dict[str, Any]] = {}
    by_db_table_type: Dict[Tuple[str, str, str], List[str]] = {}

    for key, meta in registry.items():
        lower_key_index[key.lower()] = meta
        db_name, table_name, column_name = parse_registry_key(key)
        inferred_type = meta.get("inferred_type")
        if not inferred_type:
            continue
        idx_key = (db_name, table_name, inferred_type)
        by_db_table_type.setdefault(idx_key, []).append(f"{table_name}.{column_name}")

    for cols in by_db_table_type.values():
        cols.sort()
    return lower_key_index, by_db_table_type


def get_registry_entry(
    registry: Dict[str, Any], lower_key_index: Dict[str, Dict[str, Any]], db_name: str, table_column: str
) -> Dict[str, Any] | None:
    full_key = f"{db_name}.{table_column}"
    if full_key in registry:
        return registry[full_key]
    return lower_key_index.get(full_key.lower())


def iter_questions(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in payload.get("questions", []):
        if isinstance(item, dict):
            yield item


def generate_combos(candidates: Sequence[List[str]], originals: Sequence[str], limit: int = 3) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    seen = set()
    original_tuple = tuple(originals)
    original_available = False

    for combo in product(*candidates):
        # Enforce diversity across placeholders inside one variant.
        if len(set(combo)) != len(combo):
            continue
        if combo in seen:
            continue
        seen.add(combo)
        if combo == original_tuple:
            original_available = True
            continue
        out.append(combo)
        if len(out) >= limit:
            return out

    if original_available and len(out) < limit:
        out.append(original_tuple)
    return out


def substitute_tokens(template: str, token_values: Dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return token_values.get(token, token)

    return TOKEN_PATTERN.sub(repl, template)


def build_question_variants(
    item: Dict[str, Any],
    registry: Dict[str, Any],
    lower_key_index: Dict[str, Dict[str, Any]],
    by_db_table_type: Dict[Tuple[str, str, str], List[str]],
    per_template_limit: int = 3,
) -> List[Dict[str, Any]]:
    db_name = item.get("DBName", "")
    template = item.get("template", "")
    placeholders = item.get("placeholders", [])
    question_id = item.get("QuestionID", "")
    if not isinstance(db_name, str) or not isinstance(template, str) or not isinstance(question_id, str):
        return []
    if not isinstance(placeholders, list):
        return []

    col_placeholders = [p for p in placeholders if isinstance(p, dict) and str(p.get("name", "")).startswith("{COL_")]
    if not col_placeholders:
        return []

    originals: List[str] = []
    candidate_lists: List[List[str]] = []

    for ph in col_placeholders:
        col_ref = ph.get("column_name")
        token = ph.get("name")
        if not isinstance(col_ref, str) or "." not in col_ref or not isinstance(token, str):
            return []
        entry = get_registry_entry(registry, lower_key_index, db_name, col_ref)
        if not entry:
            return []
        inferred_type = entry.get("inferred_type")
        if not isinstance(inferred_type, str) or not inferred_type:
            return []
        table_name, _ = col_ref.split(".", 1)
        candidates = by_db_table_type.get((db_name, table_name, inferred_type), [])
        if not candidates:
            return []

        alternatives = [c for c in candidates if c != col_ref]
        ordered = alternatives + ([col_ref] if col_ref in candidates else [])
        if not ordered:
            return []
        originals.append(col_ref)
        candidate_lists.append(ordered)

    combos = generate_combos(candidate_lists, originals, limit=per_template_limit)
    if not combos:
        return []

    variants: List[Dict[str, Any]] = []
    seen_variants = set()
    for combo in combos:
        token_to_column = {ph["name"]: chosen for ph, chosen in zip(col_placeholders, combo)}
        updated_placeholders: List[Dict[str, Any]] = []
        token_to_value: Dict[str, str] = {}

        for ph in placeholders:
            if not isinstance(ph, dict):
                continue
            ph_new = dict(ph)
            name = ph_new.get("name")
            if not isinstance(name, str):
                continue
            if name in token_to_column:
                chosen_col = token_to_column[name]
                ph_new["column_name"] = chosen_col
                ph_new["value"] = humanize_column_name(chosen_col)
                token_to_value[name] = ph_new["value"]
            else:
                value = ph_new.get("value")
                if isinstance(value, str):
                    token_to_value[name] = value
            updated_placeholders.append(ph_new)

        question_text = substitute_tokens(template, token_to_value)
        col_signature = tuple(
            sorted(
                (
                    str(p.get("name", "")),
                    str(p.get("column_name", "")),
                )
                for p in updated_placeholders
                if str(p.get("name", "")).startswith("{COL_")
            )
        )
        variant_signature = (db_name, template, question_text, col_signature)
        if variant_signature in seen_variants:
            continue
        seen_variants.add(variant_signature)
        idx = len(variants) + 1
        variants.append(
            {
                "QuestionID": f"{question_id}.type{idx}",
                "DBName": db_name,
                "Question": question_text,
                "GroundTruthRowsString": [],
                "template": template,
                "placeholders": updated_placeholders,
            }
        )
    return variants


def build_original_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    question_id = item.get("QuestionID")
    db_name = item.get("DBName")
    question_text = item.get("Question")
    template = item.get("template")
    placeholders = item.get("placeholders")
    if not isinstance(question_id, str):
        return None
    if not isinstance(db_name, str):
        return None
    if not isinstance(question_text, str):
        return None
    if not isinstance(template, str):
        return None
    if not isinstance(placeholders, list):
        return None

    return {
        "QuestionID": question_id,
        "DBName": db_name,
        "Question": question_text,
        "GroundTruthRowsString": [],
        "template": template,
        "placeholders": placeholders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate typed column-substituted questions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data/sampled_questions_typed_variants.json")
    parser.add_argument("--max-per-template", type=int, default=3)
    args = parser.parse_args()
    for option in ('input', 'registry'):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))

    lower_key_index, by_db_table_type = build_registry_indexes(registry)

    generated_questions: List[Dict[str, Any]] = []
    original_count = 0
    for item in iter_questions(payload):
        original_item = build_original_item(item)
        if original_item is not None:
            generated_questions.append(original_item)
            original_count += 1
        generated_questions.extend(
            build_question_variants(
                item,
                registry,
                lower_key_index,
                by_db_table_type,
                per_template_limit=max(args.max_per_template, 0),
            )
        )

    out_payload = {
        "metadata": {
            "source_questions": len(payload.get("questions", [])),
            "original_questions_included": original_count,
            "generated_questions": len(generated_questions),
            "max_per_template": max(args.max_per_template, 0),
        },
        "questions": generated_questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(generated_questions)} generated questions to {args.output}")


if __name__ == "__main__":
    main()
