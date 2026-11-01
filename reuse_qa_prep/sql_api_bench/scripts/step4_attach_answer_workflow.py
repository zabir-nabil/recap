#!/usr/bin/env python3
"""Attach SQL join skeletons and limited postprocessing plans to questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


COL_TOKEN_RE = re.compile(r"\{COL_(\d+)\}")


@dataclass(frozen=True)
class JoinEdge:
    left_table: str
    left_column: str
    right_table: str
    right_column: str


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_table_column(table: str, column: str) -> str:
    return f"{quote_ident(table)}.{quote_ident(column)}"


def parse_table_column(qualified: str) -> Optional[Tuple[str, str]]:
    if not isinstance(qualified, str) or "." not in qualified:
        return None
    table, col = qualified.split(".", 1)
    table = table.strip()
    col = col.strip()
    if not table or not col:
        return None
    return table, col


def iter_all_question_items(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in payload.get("questions", []):
        if isinstance(item, dict):
            yield item

    for group in payload.get("questions_by_group", []):
        if isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    yield item

    by_db = payload.get("questions_by_database", {})
    if isinstance(by_db, dict):
        for items in by_db.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item


def schema_by_db(tables_payload: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for schema in tables_payload:
        db_id = schema.get("db_id")
        if isinstance(db_id, str) and db_id:
            out[db_id] = schema
    return out


def build_join_graph(schema: Dict[str, Any]) -> Dict[str, List[Tuple[str, JoinEdge]]]:
    table_names = schema["table_names_original"]
    col_names = schema["column_names_original"]

    # column index -> (table_name, col_name)
    col_index_to_ref: Dict[int, Tuple[str, str]] = {}
    for idx, (table_idx, col_name) in enumerate(col_names):
        if table_idx == -1:
            continue
        col_index_to_ref[idx] = (table_names[table_idx], col_name)

    graph: Dict[str, List[Tuple[str, JoinEdge]]] = {}
    for a_idx, b_idx in schema.get("foreign_keys", []):
        if a_idx not in col_index_to_ref or b_idx not in col_index_to_ref:
            continue
        a_table, a_col = col_index_to_ref[a_idx]
        b_table, b_col = col_index_to_ref[b_idx]
        edge = JoinEdge(left_table=a_table, left_column=a_col, right_table=b_table, right_column=b_col)
        graph.setdefault(a_table, []).append((b_table, edge))
        graph.setdefault(b_table, []).append((a_table, edge))
    return graph


def bfs_tree(
    graph: Dict[str, List[Tuple[str, JoinEdge]]], root: str
) -> Dict[str, Tuple[str, JoinEdge]]:
    """Return parent pointers: table -> (parent_table, edge_used)."""
    parent: Dict[str, Tuple[str, JoinEdge]] = {}
    q = deque([root])
    seen = {root}
    while q:
        cur = q.popleft()
        for nxt, edge in graph.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = (cur, edge)
            q.append(nxt)
    return parent


def tables_for_question(placeholders: List[Dict[str, Any]]) -> Set[str]:
    tables: Set[str] = set()
    for ph in placeholders:
        if not isinstance(ph, dict):
            continue
        name = str(ph.get("name", ""))
        if name.startswith("{TABLE_"):
            t = ph.get("table_name")
            if isinstance(t, str) and t:
                tables.add(t)
        if name.startswith("{COL_"):
            col_ref = ph.get("column_name")
            parsed = parse_table_column(col_ref)
            if parsed:
                tables.add(parsed[0])
    return tables


def col_placeholders(placeholders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cols = [ph for ph in placeholders if isinstance(ph, dict) and str(ph.get("name", "")).startswith("{COL_")]

    def key(ph: Dict[str, Any]) -> int:
        m = re.search(r"\{COL_(\d+)\}", str(ph.get("name", "")))
        return int(m.group(1)) if m else 10**9

    cols.sort(key=key)
    return cols


def first_table_placeholder(placeholders: List[Dict[str, Any]]) -> Optional[str]:
    for ph in placeholders:
        if isinstance(ph, dict) and str(ph.get("name", "")).startswith("{TABLE_"):
            t = ph.get("table_name")
            if isinstance(t, str) and t:
                return t
    return None


def join_plan_for_tables(
    root: str,
    required: Set[str],
    graph: Dict[str, List[Tuple[str, JoinEdge]]],
) -> Tuple[List[str], List[JoinEdge], List[str]]:
    """Return (join_tables_in_order, join_edges_in_order, warnings)."""
    warnings: List[str] = []
    if root not in graph and len(required) > 1:
        warnings.append(f"Root table '{root}' has no foreign-key edges in schema.")

    parent = bfs_tree(graph, root)

    join_tables: Set[str] = {root}
    for t in sorted(required):
        if t == root:
            continue
        if t not in parent:
            warnings.append(f"Could not connect table '{t}' to root '{root}' via foreign keys.")
            continue
        cur = t
        while cur != root:
            join_tables.add(cur)
            cur = parent[cur][0]
        join_tables.add(root)

    # Build a join order using the BFS tree restricted to join_tables.
    children: Dict[str, List[str]] = {}
    for child, (par, _edge) in parent.items():
        if child in join_tables and par in join_tables:
            children.setdefault(par, []).append(child)
    for ch in children.values():
        ch.sort()

    ordered_tables: List[str] = [root]
    ordered_edges: List[JoinEdge] = []
    q = deque([root])
    while q:
        cur = q.popleft()
        for child in children.get(cur, []):
            _par, edge = parent[child]
            ordered_tables.append(child)
            ordered_edges.append(edge)
            q.append(child)

    # Keep only tables that are actually needed.
    ordered_tables = [t for t in ordered_tables if t in join_tables]

    return ordered_tables, ordered_edges, warnings


def detect_postprocess_plan(template: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    t = template.lower()
    plan: List[Dict[str, Any]] = []
    helpers: Set[str] = set()

    if "distinct" in t:
        plan.append({"op": "distinct"})

    def add_filter(op: str, col_token: str, extra: Dict[str, Any] | None = None) -> None:
        alias = col_token.strip("{}")
        step: Dict[str, Any] = {"op": op, "column": alias}
        if extra:
            step.update(extra)
        plan.append(step)

    # prime/fibonacci/perfect-square detection when a {COL_n} is near the keyword.
    for m in re.finditer(r"(\{COL_\d+\})[^{}]{0,80}\bprime numbers?\b", template, flags=re.IGNORECASE):
        add_filter("filter_prime", m.group(1))
        helpers.add("is_prime")

    for m in re.finditer(r"(\{COL_\d+\})[^{}]{0,80}\bfibonacci number\b", template, flags=re.IGNORECASE):
        add_filter("filter_fibonacci", m.group(1))
        helpers.add("is_fibonacci")

    for m in re.finditer(r"(\{COL_\d+\})[^{}]{0,80}\bperfect square\b", template, flags=re.IGNORECASE):
        add_filter("filter_perfect_square", m.group(1))
        helpers.add("is_perfect_square")

    for m in re.finditer(r"(\{COL_\d+\})[^{}]{0,80}\bdivisible by (\d+)\b", template, flags=re.IGNORECASE):
        add_filter("filter_divisible", m.group(1), {"n": int(m.group(2))})

    order_patterns = [
        (re.compile(r"(?:ascending order of|ordered ascending by)\s+(\{COL_\d+\})", re.IGNORECASE), "ASC"),
        (re.compile(r"(?:descending order of|ordered descending by)\s+(\{COL_\d+\})", re.IGNORECASE), "DESC"),
    ]
    for pat, direction in order_patterns:
        m = pat.search(template)
        if not m:
            continue
        plan.append({"op": "order_by", "column": m.group(1).strip("{}"), "direction": direction})
        break

    return plan, sorted(helpers)


def build_sql_template(
    db_name: str,
    template: str,
    placeholders: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    cols = col_placeholders(placeholders)
    if not cols:
        return {"status": "no_columns"}, ["No {COL_n} placeholders; cannot build SQL SELECT list."]

    required_tables = tables_for_question(placeholders)

    root = first_table_placeholder(placeholders)
    if root is None:
        parsed = parse_table_column(cols[0].get("column_name"))
        root = parsed[0] if parsed else None
    if root is None:
        return {"status": "no_root"}, ["Could not infer a root table from placeholders."]

    graph = build_join_graph(schema)
    join_tables_order, join_edges, join_warnings = join_plan_for_tables(root, required_tables, graph)
    warnings.extend(join_warnings)

    select_items: List[str] = []
    token_to_alias: Dict[str, str] = {}
    for ph in cols:
        token = str(ph.get("name", ""))
        col_ref = ph.get("column_name")
        parsed = parse_table_column(col_ref)
        if not parsed:
            warnings.append(f"Bad column_name for {token}: {col_ref!r}")
            continue
        table, col = parsed
        alias = token.strip("{}")
        token_to_alias[token] = alias
        select_items.append(f"{quote_table_column(table, col)} AS {quote_ident(alias)}")

    distinct = "distinct" in template.lower()
    select_kw = "SELECT DISTINCT" if distinct else "SELECT"

    from_clause = f"FROM {quote_ident(root)}"
    join_clauses: List[str] = []
    joined: Set[str] = {root}

    # Join edges follow the child-table order in the BFS tree.
    edge_iter = iter(join_edges)
    for table in join_tables_order[1:]:
        try:
            edge = next(edge_iter)
        except StopIteration:
            break
        if table in joined:
            continue
        # edge stores one FK equality between two tables; use it directly.
        join_clauses.append(
            f"JOIN {quote_ident(table)} ON "
            f"{quote_table_column(edge.left_table, edge.left_column)} = {quote_table_column(edge.right_table, edge.right_column)}"
        )
        joined.add(table)

    sql_lines = [select_kw, "  " + ",\n  ".join(select_items), from_clause]
    sql_lines.extend(join_clauses)
    sql_template = "\n".join(sql_lines)

    plan, helpers = detect_postprocess_plan(template)

    workflow = {
        "version": 1,
        "status": "ok" if not warnings else "partial",
        "dialect": "sqlite",
        "db_name": db_name,
        "db_path_hint": f"data/spider_sqlite_dbs/{db_name}.sqlite",
        "sql_template": sql_template,
        "token_to_alias": token_to_alias,
        "postprocess_plan": plan,
        "python_runtime": {
            "module": "scripts.answer_workflow_runtime",
            "function": "apply_workflow",
            "helpers": helpers,
        },
    }
    return workflow, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach answer workflows to templated questions JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data/sampled_questions_templated_with_workflow.json")
    args = parser.parse_args()
    for option in ('input', 'tables'):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    tables_payload = json.loads(args.tables.read_text(encoding="utf-8"))
    schemas = schema_by_db(tables_payload)

    warnings_by_qid: Dict[str, List[str]] = {}
    workflows_by_qid: Dict[str, Dict[str, Any]] = {}

    for item in iter_all_question_items(payload):
        qid = item.get("QuestionID")
        db = item.get("DBName")
        tmpl = item.get("template")
        placeholders = item.get("placeholders")
        if not isinstance(qid, str) or not isinstance(db, str) or not isinstance(tmpl, str) or not isinstance(placeholders, list):
            continue
        if qid in workflows_by_qid:
            item["answer_workflow"] = workflows_by_qid[qid]
            if qid in warnings_by_qid:
                item["answer_workflow_warnings"] = warnings_by_qid[qid]
            continue
        schema = schemas.get(db)
        if not schema:
            wf = {
                "version": 1,
                "status": "no_schema",
                "dialect": "sqlite",
                "db_name": db,
                "sql_template": None,
                "postprocess_plan": [],
                "python_runtime": {"module": "scripts.answer_workflow_runtime", "function": "apply_workflow", "helpers": []},
            }
            workflows_by_qid[qid] = wf
            item["answer_workflow"] = wf
            item["answer_workflow_warnings"] = [f"No schema found for DBName={db!r} in {args.tables}"]
            continue

        wf, warns = build_sql_template(db, tmpl, placeholders, schema)
        workflows_by_qid[qid] = wf
        if warns:
            warnings_by_qid[qid] = warns
            item["answer_workflow_warnings"] = warns
        item["answer_workflow"] = wf

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

