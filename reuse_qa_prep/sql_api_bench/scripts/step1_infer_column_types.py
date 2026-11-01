#!/usr/bin/env python3
"""Infer column types from sampled Spider rows and schema metadata."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

INT_RE = re.compile(r"^[+-]?\d+$")

# Common date/time formats present in Spider sample dumps.
DATE_TIME_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%H:%M",
    "%H:%M:%S",
    "%d-%b-%Y",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%Y%m%d",
    "%Y/%m/%d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer true column types from Spider table samples and write "
            "column_type_registry.json"
        )
    )
    parser.add_argument(
        "--samples",
        type=Path, required=True,
        help="Path to spider_table_samples_5.json",
    )
    parser.add_argument(
        "--tables",
        type=Path, required=True,
        help="Path to Spider tables.json",
    )
    parser.add_argument(
        "--output",
        type=Path, default=Path(__file__).resolve().parent.parent / "data/column_type_registry.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    for option in ('samples', 'tables'):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")
    return args


def is_integer_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False

    if isinstance(value, int):
        return True

    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer()

    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and bool(INT_RE.fullmatch(stripped))

    return False


def is_real_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        return math.isfinite(float(value))

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            parsed = float(stripped)
        except ValueError:
            return False
        return math.isfinite(parsed)

    return False


def is_datetime_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    stripped = value.strip()
    if not stripped:
        return False

    for fmt in DATE_TIME_FORMATS:
        try:
            datetime.strptime(stripped, fmt)
            return True
        except ValueError:
            continue

    return False


def spider_type_fallback(spider_type: str) -> str:
    normalized = (spider_type or "").strip().lower()

    if normalized in {"number", "numeric", "integer", "int"}:
        return "integer"
    if normalized in {"real", "float", "double", "decimal"}:
        return "real"
    if normalized in {"time", "date", "datetime", "timestamp", "year"}:
        return "date_time"

    return "text"


def infer_type(sample_values: list[Any], column_name: str, spider_type: str) -> str:
    non_null_values = [value for value in sample_values if value is not None]

    if not non_null_values:
        inferred = spider_type_fallback(spider_type)
    elif all(is_integer_value(value) for value in non_null_values):
        inferred = "integer"
    elif all(is_real_value(value) for value in non_null_values):
        inferred = "real"
    elif all(is_datetime_value(value) for value in non_null_values):
        inferred = "date_time"
    else:
        inferred = "text"

    if inferred == "integer" and column_name.lower().endswith("_id"):
        return "id"

    return inferred


def build_registry(samples_path: Path, tables_path: Path) -> dict[str, dict[str, Any]]:
    with samples_path.open("r", encoding="utf-8") as f:
        samples_payload = json.load(f)

    with tables_path.open("r", encoding="utf-8") as f:
        tables = json.load(f)

    samples_by_db = samples_payload.get("samples", {})

    registry: dict[str, dict[str, Any]] = {}

    for schema in tables:
        db_id = schema["db_id"]
        table_names_original = schema["table_names_original"]
        column_names_original = schema["column_names_original"]
        column_types = schema["column_types"]

        db_samples = samples_by_db.get(db_id, {})

        for col_idx, (table_idx, column_name) in enumerate(column_names_original):
            if table_idx == -1:
                # Skip Spider's synthetic "*" column.
                continue

            table_name = table_names_original[table_idx]
            spider_type = column_types[col_idx]

            table_rows = db_samples.get(table_name, [])
            sample_values = [row.get(column_name) for row in table_rows]

            registry_key = f"{db_id}.{table_name}.{column_name}"
            registry[registry_key] = {
                "spider_type": spider_type,
                "inferred_type": infer_type(sample_values, column_name, spider_type),
                "sample_values": sample_values,
            }

    return dict(sorted(registry.items()))


def main() -> None:
    args = parse_args()

    samples_path = Path(args.samples)
    tables_path = Path(args.tables)
    output_path = Path(args.output)

    registry = build_registry(samples_path=samples_path, tables_path=tables_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"Wrote {len(registry)} entries to {output_path} "
        f"(samples={samples_path}, tables={tables_path})"
    )


if __name__ == "__main__":
    main()
