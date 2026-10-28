"""Prepare the Spider databases used by the published REUSE-QA SQL questions.

Required database names come from the published RECAP-data/reuse-qa SQL split, 
or from an existing local JSONL question file supplied with --questions. 
Question data are not regenerated.

Use an existing Spider distribution without downloading anything:
    python scripts/prepare_sql_api_bench.py --spider-dir /path/to/spider

Or download and extract Spider into a separate working directory:
    python scripts/prepare_sql_api_bench.py --work-dir ./sql_api_bench_data

Only the download path needs the optional gdown package. No package is installed
automatically. The complete archive, including upstream notices, is retained.
This script prepares databases, not the scalar tools or an agent environment.
Loading the published questions needs the optional datasets package; --questions
uses only the standard library for reading an existing question file.
"""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sqlite3
import stat
import sys
import zipfile


SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
QUESTION_REPOSITORY = "RECAP-data/reuse-qa"


def load_required_databases(question_file=None):
    """Read names from published or local SQL questions without changing them."""
    if question_file is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Reading the published REUSE-QA questions requires datasets. "
                "Install it explicitly, or supply --questions /path/to/sql_api_bench.jsonl."
            ) from exc
        records = load_dataset(
            QUESTION_REPOSITORY, "default", split="sql_api_bench"
        )
        source = QUESTION_REPOSITORY
    else:
        question_file = Path(question_file)
        records = []
        with question_file.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"Blank JSONL line {line_number} in {question_file}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL line {line_number} in {question_file}: {exc}"
                    ) from exc
                records.append(row)
        source = str(question_file)

    required = set()
    question_count = 0
    for record_number, row in enumerate(records, 1):
        if not isinstance(row, dict) or row.get("dataset") != "sql_api_bench":
            raise ValueError(f"Unexpected source on record {record_number} in {source}")
        context = row.get("context")
        db_name = context.get("db_name") if isinstance(context, dict) else None
        if not isinstance(db_name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", db_name):
            raise ValueError(f"Invalid context.db_name on record {record_number} in {source}")
        required.add(db_name)
        question_count += 1
    if not required:
        raise ValueError(f"No SQL questions found in {source}")
    return question_count, required


def database_path(database_dir, db_name):
    """Support Spider's name/name.sqlite layout and flat database directories."""
    for path in (
        database_dir / db_name / f"{db_name}.sqlite",
        database_dir / f"{db_name}.sqlite",
    ):
        if path.is_file():
            return path
    return None


def find_database_dir(spider_dir, required):
    """Find the directory containing the requested Spider databases."""
    candidates = [
        spider_dir,
        spider_dir / "spider_data" / "database",
        spider_dir / "database",
        spider_dir / "spider" / "database",
    ]
    if spider_dir.is_dir():
        for root, dirs, _ in os.walk(spider_dir, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != "__MACOSX")
            if Path(root).name == "database":
                candidates.append(Path(root))

    best = None
    best_count = 0
    seen = set()
    for candidate in candidates:
        key = candidate.resolve()
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        count = sum(database_path(candidate, name) is not None for name in required)
        if count == len(required):
            return candidate
        if count > best_count:
            best, best_count = candidate, count
    return best


def safe_extract(archive, destination):
    """Extract every archive member after rejecting unsafe paths and links."""
    destination = destination.resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(
            f"Extraction directory is not empty: {destination}. "
            "Choose a fresh --work-dir or use --spider-dir for an existing copy."
        )

    with zipfile.ZipFile(archive) as zf:
        members = []
        targets = {}
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            windows_path = PureWindowsPath(name)
            if (
                not name
                or relative.is_absolute()
                or windows_path.drive
                or ".." in relative.parts
                or any(
                    part.endswith((" ", "."))
                    or any(ch in '<>:"|?*' or ord(ch) < 32 for ch in part)
                    for part in relative.parts
                )
            ):
                raise ValueError(f"Unsafe archive member path: {info.filename!r}")
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported archive link or special file: {info.filename!r}")
            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination) or target == destination:
                raise ValueError(f"Unsafe archive member path: {info.filename!r}")
            key = target.as_posix().casefold()
            is_dir = info.is_dir() or file_type == stat.S_IFDIR
            if key in targets:
                if is_dir and targets[key]:
                    continue
                raise ValueError(f"Conflicting archive member path: {info.filename!r}")
            targets[key] = is_dir
            members.append((info, target, is_dir))

        # A regular file cannot also be a parent directory of another member.
        for _, target, _ in members:
            for parent in target.parents:
                if parent == destination:
                    break
                if targets.get(parent.as_posix().casefold()) is False:
                    raise ValueError(f"Archive file/directory conflict: {target}")

        destination.mkdir(parents=True, exist_ok=True)
        for info, target, is_dir in members:
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def download_spider(work_dir, required):
    """Download the original Spider archive and keep its complete contents."""
    work_dir = work_dir.resolve()
    spider_dir = work_dir / "spider"
    if spider_dir.exists():
        database_dir = find_database_dir(spider_dir, required)
        if database_dir and all(database_path(database_dir, n) for n in required):
            print(f"Using existing extracted Spider copy: {spider_dir}")
            return database_dir
        raise ValueError(
            f"Existing extraction is incomplete: {spider_dir}. "
            "Choose a fresh --work-dir or provide a complete --spider-dir."
        )

    archive = work_dir / "spider_data.zip"
    if not archive.is_file():
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "Downloading Spider requires gdown. Install it explicitly with:\n"
                f'  "{sys.executable}" -m pip install gdown\n'
                "Then rerun, or provide --spider-dir for an existing distribution."
            ) from exc
        work_dir.mkdir(parents=True, exist_ok=True)
        partial = work_dir / "spider_data.zip.part"
        print("Downloading the Spider archive from Google Drive...")
        try:
            result = gdown.download(
                f"https://drive.google.com/uc?id={SPIDER_DRIVE_ID}",
                str(partial),
                quiet=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Spider download failed: {exc}") from exc
        if not result or not partial.is_file() or not zipfile.is_zipfile(partial):
            raise RuntimeError(
                "Spider download did not produce a ZIP archive. "
                "Check the download source or use --spider-dir."
            )
        partial.replace(archive)

    print(f"Extracting the full Spider archive to: {spider_dir}")
    safe_extract(archive, spider_dir)
    print(f"Archive retained, including upstream notices: {archive}")
    database_dir = find_database_dir(spider_dir, required)
    if database_dir is None:
        raise FileNotFoundError(f"No requested Spider databases found in {spider_dir}")
    return database_dir


def verify_databases(database_dir, required):
    """Check required files and their SQLite integrity without writing to them."""
    missing = [name for name in sorted(required) if database_path(database_dir, name) is None]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required databases in {database_dir}: "
            + ", ".join(missing)
        )
    details = []
    for name in sorted(required):
        path = database_path(database_dir, name).resolve()
        uri = path.as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            check = [row[0] for row in connection.execute("PRAGMA integrity_check")]
            if check != ["ok"]:
                raise ValueError(f"SQLite integrity check failed for {path}: {check}")
            table_count = connection.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
        details.append((name, path, table_count))
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spider-dir", type=Path,
        help="Existing Spider root or database directory; read in place without downloading.",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("sql_api_bench_data"),
        help="Download/extraction directory when --spider-dir is omitted (default: %(default)s).",
    )
    parser.add_argument(
        "--questions", type=Path,
        help="Existing REUSE-QA SQL JSONL file; otherwise read the published SQL split.",
    )
    args = parser.parse_args(argv)
    try:
        question_count, required = load_required_databases(args.questions)
        print(f"SQL questions: {question_count}; required databases: {len(required)}")
        if args.spider_dir is not None:
            spider_dir = args.spider_dir.resolve()
            if not spider_dir.is_dir():
                raise FileNotFoundError(f"Existing Spider directory not found: {spider_dir}")
            database_dir = find_database_dir(spider_dir, required)
            if database_dir is None:
                raise FileNotFoundError(f"No requested Spider databases found in {spider_dir}")
        else:
            database_dir = download_spider(args.work_dir, required)
        details = verify_databases(database_dir, required)
        print(f"Ready: {len(details)} databases for {question_count} SQL questions.")
        print(f"Database directory: {database_dir.resolve()}")
        for name, path, tables in details:
            print(f"  {name}: {path} ({tables} tables)")
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error, zipfile.BadZipFile) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
