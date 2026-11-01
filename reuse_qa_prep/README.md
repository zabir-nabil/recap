# Dataset preparation

Scripts accept external input files and write outputs to the paths supplied when run.

| Path | Purpose |
| --- | --- |
| `financebench/scripts/paraphrase_questions.py` | Generate template-based FinanceBench paraphrase candidates. |
| `financebench/scripts/augment_constraints.py` | Generate numeric-property and multi-year condition candidates. |
| `financebench_extended/scripts/replace_entities.py` | Rebuild synthetic company/year variants from original records using validated replacements. |
| `financebench_extended/scripts/paraphrase_questions.py` | Generate paraphrases for company/year variants. |
| `financebench_extended/scripts/augment_constraints.py` | Generate conditions for company/year variants. |
| `sql_api_bench/scripts/load_sql_api_bench_full.py` | Download SQL-API-Bench Benchmark II questions. |
| `sql_api_bench/scripts/download_sql_api_bench_schema.py` | Prepare Spider schemas, table samples, and optional SQLite files. |
| `sql_api_bench/scripts/attach_all_tables_columns.py` | Attach schema names and aliases to questions. |
| `sql_api_bench/scripts/gemini_paraphrasing.py` | Generate SQL paraphrases; the prompt is in `PROMPT`. |
| `sql_api_bench/scripts/gemini_templating.py` | Mask explicit table/column mentions; the prompt is in `PROMPT`. |
| `sql_api_bench/scripts/post_process.py` | Remove underscores and backticks from generated paraphrases. |
| `sql_api_bench/scripts/step0_find_column_name_families.py` | Group related column names. |
| `sql_api_bench/scripts/step1_infer_column_types.py` | Infer column types from schemas and sampled rows. |
| `sql_api_bench/scripts/step2_template_questions.py` | Build rule-based question templates with brace placeholders. |
| `sql_api_bench/scripts/step3_generate_typed_questions.py` | Substitute same-table, type-compatible columns in step 2 templates. |
| `sql_api_bench/scripts/step4_attach_answer_workflow.py` | Attach partial SQL and postprocessing workflows to step 2 templates. |
| `sql_api_bench/scripts/answer_workflow_runtime.py` | Apply supported postprocessing operations to SQL result rows. |
| `prompts/preserve_entities.txt` | Generic named-entity-preserving paraphrase prompt from the notebook. |
| `prompts/preserve_words.txt` | Generic paraphrase prompt with an explicit list of protected words. |

Dependencies are listed in `requirements.txt`. Script arguments are available through `--help`; `answer_workflow_runtime.py` is an importable helper.
