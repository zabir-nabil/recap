"""Remove schema underscores and backticks from generated paraphrases."""

import argparse
import json
from pathlib import Path


def clean_paraphrases(data):
    cleaned = []
    for item in data:
        paraphrases = item["paraphrases"]
        if not isinstance(paraphrases, dict):
            raise ValueError(f"Invalid paraphrases for {item.get('QuestionID')}")
        question = item.copy()
        question["paraphrases"] = {
            "paraphrases": [
                text.replace("_", " ").replace("`", "")
                for text in paraphrases["paraphrases"]
            ]
        }
        cleaned.append(question)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "data/paraphrased_questions_full_cor_2338.json",
    )
    args = parser.parse_args()
    for option in ('input',):
        path = Path(getattr(args, option))
        if not path.is_file():
            parser.error(f"--{option.replace('_', '-')} file not found: {path}")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    cleaned = clean_paraphrases(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cleaned, indent=4), encoding="utf-8")
    print(f"Wrote {len(cleaned)} questions to {args.output}")


if __name__ == "__main__":
    main()
