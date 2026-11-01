#!/usr/bin/env python3
"""Group Spider columns by matching names and conservative suffix stems."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")

# Deliberately narrow for conservative stem grouping.
ALLOWED_SUFFIX_STEMS = {
    "address",
    "email",
    "phone",
    "city",
    "country",
    "state",
    "province",
    "postal",
    "postcode",
    "zip",
    "zipcode",
    "website",
    "url",
    "fax",
}

# Extra guardrail: avoid ambiguous quantity/political phrases for stem matching.
DISALLOWED_PREFIX_TOKENS = {
    "number",
    "numbers",
    "num",
    "count",
    "counts",
    "total",
    "totals",
    "head",
    "of",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find same-name and conservative same-stem column groups from Spider tables.json"
    )
    parser.add_argument("--tables", type=Path, required=True, help="Path to Spider tables.json")
    parser.add_argument(
        "--output",
        type=Path, default=Path(__file__).resolve().parent.parent / "data/column_name_families_conservative.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    for option in ('tables',):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")
    return args


def tokenize(name: str) -> list[str]:
    if not name:
        return []
    name = CAMEL_1.sub(r"\1 \2", name)
    name = CAMEL_2.sub(r"\1 \2", name)
    name = (
        name.replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(".", " ")
        .strip()
        .lower()
    )
    return [tok for tok in re.split(r"[^a-z0-9]+", name) if tok]


def light_stem(token: str) -> str:
    """Tiny stemmer for plural normalization without external deps."""
    token = token.lower().strip()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            r["db"],
            r["table"],
            r["column_original"],
            r["column_nl"],
        ),
    )


def load_columns(tables_path: Path) -> list[dict[str, Any]]:
    with tables_path.open("r", encoding="utf-8") as f:
        schemas = json.load(f)

    columns: list[dict[str, Any]] = []

    for schema in schemas:
        db = schema["db_id"]
        table_names_original = schema["table_names_original"]
        column_names_original = schema["column_names_original"]
        column_names_nl = schema["column_names"]

        for idx, (table_idx, col_original) in enumerate(column_names_original):
            if table_idx == -1:
                continue

            col_nl = column_names_nl[idx][1]
            table = table_names_original[table_idx]

            # Prefer the Spider NL label for matching; fallback to original.
            tokens = tokenize(col_nl) or tokenize(col_original)
            normalized = " ".join(tokens)

            columns.append(
                {
                    "db": db,
                    "table": table,
                    "column_original": col_original,
                    "column_nl": col_nl,
                    "normalized_name": normalized,
                    "tokens": tokens,
                }
            )

    return columns


def build_exact_name_groups(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for col in columns:
        if col["normalized_name"]:
            by_name[col["normalized_name"]].append(col)

    groups: list[dict[str, Any]] = []
    for name, cols in by_name.items():
        unique_locations = {(c["db"], c["table"], c["column_original"]) for c in cols}
        if len(unique_locations) < 2:
            continue

        db_count = len({c["db"] for c in cols})
        table_count = len({(c["db"], c["table"]) for c in cols})

        groups.append(
            {
                "name": name,
                "match_type": "exact_name",
                "count": len(cols),
                "db_count": db_count,
                "table_count": table_count,
                "columns": sort_records(
                    [
                        {
                            "db": c["db"],
                            "table": c["table"],
                            "column_original": c["column_original"],
                            "column_nl": c["column_nl"],
                        }
                        for c in cols
                    ]
                ),
            }
        )

    groups.sort(key=lambda g: (-g["count"], g["name"]))
    return groups


def build_conservative_stem_groups(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for col in columns:
        tokens = col["tokens"]
        if len(tokens) < 2:
            continue

        raw_suffix = tokens[-1]
        stem = light_stem(raw_suffix)
        if stem not in ALLOWED_SUFFIX_STEMS:
            continue

        prefix_tokens = tokens[:-1]
        if any(tok in DISALLOWED_PREFIX_TOKENS for tok in prefix_tokens):
            continue

        entry = {
            "db": col["db"],
            "table": col["table"],
            "column_original": col["column_original"],
            "column_nl": col["column_nl"],
            "normalized_name": col["normalized_name"],
            "prefix": " ".join(prefix_tokens),
        }
        by_stem[stem].append(entry)

    groups: list[dict[str, Any]] = []

    for stem, cols in by_stem.items():
        variants = sorted({c["normalized_name"] for c in cols})
        prefixes = {c["prefix"] for c in cols if c["prefix"]}
        db_count = len({c["db"] for c in cols})
        table_count = len({(c["db"], c["table"]) for c in cols})

        # Conservative thresholds.
        if len(cols) < 3:
            continue
        if len(variants) < 2:
            continue
        if len(prefixes) < 2:
            continue
        if db_count < 2:
            continue

        groups.append(
            {
                "stem": stem,
                "match_type": "suffix_stem_conservative",
                "count": len(cols),
                "db_count": db_count,
                "table_count": table_count,
                "variant_count": len(variants),
                "variants": variants,
                "columns": sort_records(
                    [
                        {
                            "db": c["db"],
                            "table": c["table"],
                            "column_original": c["column_original"],
                            "column_nl": c["column_nl"],
                        }
                        for c in cols
                    ]
                ),
            }
        )

    groups.sort(key=lambda g: (-g["count"], g["stem"]))
    return groups


def main() -> None:
    args = parse_args()
    tables_path = Path(args.tables)
    output_path = Path(args.output)

    columns = load_columns(tables_path)
    exact_groups = build_exact_name_groups(columns)
    stem_groups = build_conservative_stem_groups(columns)

    payload = {
        "metadata": {
            "tables_path": str(tables_path),
            "total_columns": len(columns),
            "exact_group_count": len(exact_groups),
            "conservative_stem_group_count": len(stem_groups),
            "allowed_suffix_stems": sorted(ALLOWED_SUFFIX_STEMS),
            "notes": [
                "Exact groups match on normalized column labels.",
                "Stem groups are conservative and only use a narrow suffix stem whitelist.",
            ],
        },
        "exact_name_groups": exact_groups,
        "conservative_stem_groups": stem_groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        "Wrote "
        f"{len(exact_groups)} exact groups and {len(stem_groups)} conservative stem groups "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
