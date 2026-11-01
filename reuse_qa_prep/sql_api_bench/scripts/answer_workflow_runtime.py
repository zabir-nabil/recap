#!/usr/bin/env python3
"""Apply the supported filters and sorting operations in an answer workflow."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            try:
                f = float(stripped)
            except ValueError:
                return None
            if math.isfinite(f) and f.is_integer():
                return int(f)
            return None
        return parsed
    return None


def is_prime(value: Any) -> bool:
    n = _to_int(value)
    if n is None or n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n))
    d = 3
    while d <= limit:
        if n % d == 0:
            return False
        d += 2
    return True


def is_perfect_square(value: Any) -> bool:
    n = _to_int(value)
    if n is None or n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def is_fibonacci(value: Any) -> bool:
    # n is Fibonacci iff 5n^2+4 or 5n^2-4 is a perfect square.
    n = _to_int(value)
    if n is None or n < 0:
        return False
    x1 = 5 * n * n + 4
    x2 = 5 * n * n - 4
    return is_perfect_square(x1) or is_perfect_square(x2)


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        sig = tuple(sorted((k, str(v)) for k, v in row.items()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def apply_workflow(rows: List[Dict[str, Any]], workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply a postprocess plan to SQL result rows.

    The input rows should be dicts keyed by the SQL aliases emitted in
    `workflow["sql_template"]` (e.g., "COL_1", "COL_2", ...).
    """

    plan = workflow.get("postprocess_plan", [])
    if not isinstance(plan, list):
        return rows

    out = list(rows)

    for step in plan:
        if not isinstance(step, dict):
            continue
        op = step.get("op")
        if op == "filter_prime":
            key = step.get("column")
            if isinstance(key, str):
                out = [r for r in out if is_prime(r.get(key))]
        elif op == "filter_perfect_square":
            key = step.get("column")
            if isinstance(key, str):
                out = [r for r in out if is_perfect_square(r.get(key))]
        elif op == "filter_fibonacci":
            key = step.get("column")
            if isinstance(key, str):
                out = [r for r in out if is_fibonacci(r.get(key))]
        elif op == "filter_divisible":
            key = step.get("column")
            n = step.get("n")
            if isinstance(key, str) and isinstance(n, int) and n != 0:
                out = [r for r in out if (_to_int(r.get(key)) is not None and _to_int(r.get(key)) % n == 0)]
        elif op == "order_by":
            key = step.get("column")
            direction = (step.get("direction") or "ASC").upper()
            if isinstance(key, str):
                reverse = direction == "DESC"

                def sort_key(row: Dict[str, Any]) -> Any:
                    v = row.get(key)
                    as_int = _to_int(v)
                    if as_int is not None:
                        return (0, as_int)
                    return (1, str(v))

                out.sort(key=sort_key, reverse=reverse)
        elif op == "distinct":
            out = _dedupe_rows(out)

    return out

