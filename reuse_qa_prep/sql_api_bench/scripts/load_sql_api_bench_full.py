"""Download SQL-API-Bench Benchmark II questions."""

import argparse
import json
from pathlib import Path


def load_sql_api_bench():
    from datasets import load_dataset

    dataset = load_dataset("ibm-research/SQL-API-Bench", "benchmarkII")
    return {item["QuestionID"]: item for item in dataset["train"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent.parent / "data/sql_api_bench_questions_full.json"
    )
    args = parser.parse_args()

    questions = list(load_sql_api_bench().values())
    payload = {
        "metadata": {
            "total_questions": len(questions),
            "source": "SQL-API-Bench benchmarkII",
        },
        "questions": questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=6), encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
