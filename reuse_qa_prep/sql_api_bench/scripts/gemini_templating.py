"""Generate SQL-API-Bench schema-aware templates with Gemini."""

import argparse
import json
import time
from pathlib import Path



PROMPT = """
Your task is to generate a masked natural language template for the input question by masking only explicit database entity mentions in the question.

A span may refer to a database table or column even if it does NOT appear verbatim in the schema. Match mentions using:
1. table names
2. table aliases
3. column names
4. column aliases
5. natural-language noun-phrase variants of schema names

Important distinction:
- Mask only explicit contiguous spans in the question that denote a table or column.
- Do NOT mask words or phrases that only imply a condition or operation.
- Do NOT infer and mask implicit columns from verbs or predicates.

Examples:
- Good column mentions to mask: "transaction amount", "document ID", "response received dates", "opening year", "organization IDs"
- Do NOT mask: "described", "opened", "granted", "sorted in ascending order", "divisible by 5", "perfect square", "Fibonacci number"

Instructions:
- Replace each table mention with [TABLE_n]
- Replace each column mention with [COL_n]
- n must start from 1
- Number columns separately from tables
- Preserve all other words in the question exactly
- Prefer the longest valid explicit mention span
- Use the exact span from the question as `original_value`
- Do not split one mention into multiple placeholders
- Reuse the same placeholder if the exact same schema item is mentioned multiple times
- The final template must remain grammatical and preserve the original meaning

INPUT QUESTION: {QUESTION}
DATABASE INFORMATION: {DB}

Return JSON with:
- template
- values_mapping
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "template": {"type": "STRING"},
        "values_mapping": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "placeholder": {
                        "type": "STRING"
                    },
                    "kind": {
                        "type": "STRING",
                        "enum": ["column", "table"]
                    },
                    "original_value": {"type": "STRING"},
                    "schema_item": {
                        "type": "STRING",
                        "description": "If column, format as table_name.column_name. If table, format as table_name."
                    }
                },
                "required": ["placeholder", "kind", "original_value", "schema_item"]
            }
        }
    },
    "required": ["template", "values_mapping"]
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

        contents = PROMPT.format(
            QUESTION=question["Question"], DB=str(question["db_information"])
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

        record["template"] = generated
        results.append(record)
        with (log_dir / "templating_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    output = {key: value for key, value in data.items() if key != "questions"}
    output["questions"] = results

    with (data_dir / "templates_full_flash.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=4)
    with (data_dir / f"templated_questions_full_flash{len(results)}.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(f"Generated {len(results)} records; JSON errors: {json_errors}")


if __name__ == "__main__":
    main()
