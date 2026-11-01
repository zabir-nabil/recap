#!/usr/bin/env python3
"""Extract Spider schemas, sample rows, and optional SQLite databases."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import shutil


DEFAULT_SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Spider tables.json and sample rows.")
    parser.add_argument(
        "--spider-zip",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data/spider_data.zip",
        help="Path to Spider spider_data.zip (downloaded archive).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If set and --spider-zip does not exist, download from Google Drive via gdown.",
    )
    parser.add_argument(
        "--spider-drive-id",
        default=DEFAULT_SPIDER_DRIVE_ID,
        help="Google Drive file id for spider_data.zip (used with --download).",
    )
    parser.add_argument(
        "--tables-out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data/tables.json",
        help="Where to write extracted tables.json.",
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data/spider_table_samples_5.json",
        help="Where to write sampled rows JSON.",
    )
    parser.add_argument(
        "--limit-per-table",
        type=int,
        default=5,
        help="How many rows to sample per table.",
    )
    parser.add_argument(
        "--db-ids",
        nargs="*",
        default=None,
        help="Optional list of db_id values to restrict sampling to.",
    )
    parser.add_argument(
        "--db-ids-from-sampled-questions",
        type=Path,
        default=None,
        help="If set, restrict to DBName values in this sampled_questions.json.",
    )
    parser.add_argument(
        "--extract-sqlite-dir",
        type=Path,
        default=None,
        help="Optional directory to extract sqlite DB files to.",
    )
    args = parser.parse_args()
    if args.db_ids_from_sampled_questions is not None and not args.db_ids_from_sampled_questions.is_file():
        parser.error(f"Question file not found: {args.db_ids_from_sampled_questions}")
    return args


def maybe_download_spider_zip(spider_zip: Path, drive_id: str) -> None:
    if spider_zip.exists():
        return
    try:
        import gdown  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Spider zip is missing and gdown is not available.\n"
            f"Tried to import gdown and got: {e}\n"
            "Either:\n"
            "- Download spider_data.zip yourself and pass --spider-zip\n"
            "- Or install gdown (pip install gdown) and rerun with --download"
        ) from e

    spider_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={drive_id}"
    print(f"Downloading Spider zip to {spider_zip} ...")
    gdown.download(url, str(spider_zip), quiet=False)


def _safe_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _find_tables_json_in_zip(z: zipfile.ZipFile) -> str:
    names = z.namelist()
    candidates = [
        n
        for n in names
        if n.lower().endswith("tables.json")
        and "__macosx" not in n.lower()
        and not Path(n).name.startswith("._")
    ]
    if not candidates:
        near = [n for n in names if "table" in n.lower() and "__macosx" not in n.lower()][:50]
        raise FileNotFoundError(f"tables.json not found in zip. Related entries: {near}")
    candidates.sort(key=lambda p: ("/" in p, len(p)))
    return candidates[0]


def _find_sqlite_in_zip(z: zipfile.ZipFile, db_id: str) -> str:
    names = z.namelist()
    suffix = f"/{db_id}/{db_id}.sqlite"
    candidates = [
        n
        for n in names
        if n.endswith(suffix)
        and "__macosx" not in n.lower()
        and not Path(n).name.startswith("._")
    ]
    if candidates:
        return candidates[0]

    candidates = [
        n
        for n in names
        if n.endswith(f"/{db_id}.sqlite")
        and "__macosx" not in n.lower()
        and not Path(n).name.startswith("._")
    ]
    if candidates:
        candidates.sort(key=lambda p: ("/database/" not in p.lower(), len(p)))
        return candidates[0]

    raise FileNotFoundError(f"Could not find sqlite for db_id='{db_id}' inside the zip.")


def load_tables_from_zip(z: zipfile.ZipFile) -> List[Dict[str, Any]]:
    tables_name = _find_tables_json_in_zip(z)
    return json.loads(z.read(tables_name).decode("utf-8"))


def write_tables_json(tables: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tables, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def db_ids_from_sampled_questions(sampled_questions_path: Path) -> Set[str]:
    payload = json.loads(sampled_questions_path.read_text(encoding="utf-8"))
    dbs = set()
    for item in payload.get("questions", []):
        if isinstance(item, dict) and isinstance(item.get("DBName"), str):
            dbs.add(item["DBName"])
    return dbs


def iter_target_dbs(tables: List[Dict[str, Any]], allowed: Set[str] | None) -> Iterable[Dict[str, Any]]:
    for schema in tables:
        db_id = schema.get("db_id")
        if not isinstance(db_id, str) or not db_id:
            continue
        if allowed is not None and db_id not in allowed:
            continue
        yield schema


def sample_rows_from_zip(
    z: zipfile.ZipFile,
    tables: List[Dict[str, Any]],
    allowed_db_ids: Set[str] | None,
    limit_per_table: int,
) -> Dict[str, Any]:
    tmp_root = Path(tempfile.mkdtemp(prefix="spider_tmp_"))
    try:
        out: Dict[str, Any] = {}
        for schema in iter_target_dbs(tables, allowed_db_ids):
            db_id = schema["db_id"]
            table_names = schema.get("table_names_original", [])
            if not isinstance(table_names, list):
                out[db_id] = {"__error__": "Invalid table_names_original"}
                continue

            try:
                sqlite_name = _find_sqlite_in_zip(z, db_id)
            except FileNotFoundError as e:
                out[db_id] = {"__error__": str(e)}
                continue

            db_tmp_dir = tmp_root / db_id
            db_tmp_dir.mkdir(parents=True, exist_ok=True)
            sqlite_tmp = db_tmp_dir / f"{db_id}.sqlite"
            sqlite_tmp.write_bytes(z.read(sqlite_name))

            db_samples: Dict[str, Any] = {}
            conn = sqlite3.connect(str(sqlite_tmp))
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                existing_tables = {r[0] for r in cur.fetchall()}

                for table_name in table_names:
                    if not isinstance(table_name, str):
                        continue
                    if table_name not in existing_tables:
                        db_samples[table_name] = []
                        continue
                    try:
                        cur.execute(f"PRAGMA table_info({_safe_ident(table_name)});")
                        cols = [r[1] for r in cur.fetchall()]
                        cur.execute(
                            f"SELECT * FROM {_safe_ident(table_name)} LIMIT {int(limit_per_table)};"
                        )
                        rows = cur.fetchall()
                        db_samples[table_name] = [dict(zip(cols, row)) for row in rows]
                    except Exception as e:  # pragma: no cover
                        db_samples[table_name] = {"__error__": str(e)}
            finally:
                conn.close()

            out[db_id] = db_samples
            try:
                sqlite_tmp.unlink()
            except Exception:
                pass

        return out
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def extract_sqlite_dbs(
    z: zipfile.ZipFile,
    tables: List[Dict[str, Any]],
    allowed_db_ids: Set[str] | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for schema in iter_target_dbs(tables, allowed_db_ids):
        db_id = schema["db_id"]
        try:
            sqlite_name = _find_sqlite_in_zip(z, db_id)
        except FileNotFoundError:
            continue
        out_path = out_dir / f"{db_id}.sqlite"
        if out_path.exists():
            continue
        out_path.write_bytes(z.read(sqlite_name))


def main() -> None:
    args = parse_args()
    spider_zip = args.spider_zip

    if not spider_zip.exists():
        if args.download:
            maybe_download_spider_zip(spider_zip, args.spider_drive_id)
        else:
            raise SystemExit(
                f"Missing spider zip: {spider_zip}\n"
                "Provide --spider-zip or pass --download to fetch it (requires gdown)."
            )

    allowed: Set[str] | None = None
    if args.db_ids_from_sampled_questions is not None:
        allowed = db_ids_from_sampled_questions(args.db_ids_from_sampled_questions)
    if args.db_ids:
        allowed = set(args.db_ids) if allowed is None else (allowed & set(args.db_ids))

    with zipfile.ZipFile(spider_zip, "r") as z:
        tables = load_tables_from_zip(z)
        write_tables_json(tables, args.tables_out)

        samples = sample_rows_from_zip(
            z=z,
            tables=tables,
            allowed_db_ids=allowed,
            limit_per_table=max(int(args.limit_per_table), 0),
        )

        payload = {
            "metadata": {
                "spider_zip": str(spider_zip),
                "limit_per_table": int(args.limit_per_table),
                "db_filter": sorted(allowed) if allowed is not None else None,
            },
            "samples": samples,
        }
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        args.samples_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if args.extract_sqlite_dir is not None:
            extract_sqlite_dbs(
                z=z,
                tables=tables,
                allowed_db_ids=allowed,
                out_dir=args.extract_sqlite_dir,
            )

    print(f"Wrote: {args.tables_out}")
    print(f"Wrote: {args.samples_out}")
    if args.extract_sqlite_dir is not None:
        print(f"Extracted sqlite DBs to: {args.extract_sqlite_dir}")


if __name__ == "__main__":
    main()
