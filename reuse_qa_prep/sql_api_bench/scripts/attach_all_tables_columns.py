"""Attach database table and column names to Benchmark II questions."""

import argparse
import json
from pathlib import Path


def attach_schema(data, db_tables):
    if isinstance(db_tables, list):
        db_tables = {schema["db_id"]: schema for schema in db_tables}
    questions = []
    for item in data["questions"]:
        database = db_tables[item["DBName"]]
        question = item.copy()
        question["db_information"] = {
            "table_names": database["table_names_original"],
            "table_name_aliases": database["table_names"],
            "column_names": [
                column[1] for column in database["column_names_original"]
                if column[1] != "*"
            ],
            "column_name_aliases": [
                column[1] for column in database["column_names"] if column[1] != "*"
            ],
        }
        questions.append(question)
    return {**data, "questions": questions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True
    )
    parser.add_argument("--schemas", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "data/sql_api_bench_questions_full_db_info.json",
    )
    args = parser.parse_args()
    for option in ('input', 'schemas'):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    db_tables = json.loads(args.schemas.read_text(encoding="utf-8"))
    payload = attach_schema(data, db_tables)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    print(f"Wrote {len(payload['questions'])} questions to {args.output}")


if __name__ == "__main__":
    main()
