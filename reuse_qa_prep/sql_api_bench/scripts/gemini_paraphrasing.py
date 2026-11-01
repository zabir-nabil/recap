"""Generate SQL-API-Bench paraphrases with Gemini."""

import argparse
import json
import time
from pathlib import Path



PROMPT = """
Your task is to generate 5 paraphrases of the input question.

INPUT QUESTION: {QUESTION}

The question refers to one or multiple database tables and columns. The table and column names, their aliases and ground truth column names are provided to you. These table and column names or any variant of these MUST remain the same verbatim **as in the original question** while paraphrasing. Any mathematical or comparison expression conidtion/constraint in the original question must preserve the exact wording from the original question.
DATABSE INFORMATION: {DB}
GROUNDTRUTH COLUMNS: {GROUND_TRUTH}
Provide your output in the specified structured JSON format.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "paraphrases": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
            }
        }
    },
    "required": [
        "paraphrases"
    ]
}


def generate_with_retry(client, model_name, contents, config, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
        except Exception as error:
            if "503" not in str(error) and "high demand" not in str(error):
                raise
            wait_time = (2 ** attempt) + 1
            print(f"Server busy. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError("Max retries exceeded")


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--log-dir", type=Path, default=root / "logs")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"--input file not found: {args.input}")

    from google import genai
    from google.genai import types
    from tqdm import tqdm

    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)
    client = genai.Client()
    data_dir = args.output_dir
    log_dir = args.log_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    json_errors = 0
    for question in tqdm(data["questions"]):

        columns = set()
        for row in json.loads(question["GroundTruthRowsString"]):
            columns.update(row)
        contents = PROMPT.format(
            QUESTION=question["Question"],
            DB=str(question["db_information"]),
            GROUND_TRUTH=str(columns),
        )

        response = generate_with_retry(
            client,
            model_name="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                max_output_tokens=4096,
            ),
        )
        record = question.copy()
        try:
            generated = json.loads(response.text)
        except json.JSONDecodeError:
            generated = response.text
            json_errors += 1

            with (log_dir / "json_error_paraphrasing_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({question["Question"]: response.text}) + "\n")

        record["paraphrases"] = generated
        results.append(record)
        with (log_dir / "paraphrasing_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    output = {key: value for key, value in data.items() if key != "questions"}
    output["questions"] = results

    with (data_dir / "paraphrased_full.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=4)
    with (data_dir / f"paraphrased_questions_full_{len(results)}.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(f"Generated {len(results)} records; JSON errors: {json_errors}")


if __name__ == "__main__":
    main()
