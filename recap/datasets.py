"""Load REUSE-QA questions and the corresponding execution tools."""

from __future__ import annotations

from copy import deepcopy


DATASET_REPOSITORIES = {
    "financebench": "RECAP-data/reuse-qa",
    "sql_api_bench": "RECAP-data/reuse-qa",
    "gaia_l1": "RECAP-data/reuse-qa-gaia",
    "gaia_l2": "RECAP-data/reuse-qa-gaia",
}

PREAMBLE_FINANCEBENCH = ""

PREAMBLE_SQL_API_BENCH = """Important workflow for this dataset:
1. ALWAYS call get_schema(db_name="...") FIRST to see the table structure and column names.
2. Write SQL queries using sql_execute(db_name="...", query="SELECT ...").
3. For questions involving numeric properties (prime, fibonacci, perfect square, even, divisible), first get the data via SQL, then use the appropriate API tool on each value.
4. For questions involving syllable counts, get the text values via SQL, then use count_syllables.
5. For questions involving geography/distance, get place names via SQL, then use geocode and geo_distance.
6. SQL queries must be read-only SELECT statements."""

PREAMBLE_GAIA = """You are answering questions from the GAIA benchmark.
GAIA questions require precise, concise answers. Your final answer should be:
- A number (no commas, no units unless asked), OR
- A few words (no articles, no abbreviations for cities), OR
- A comma-separated list of numbers and/or strings.

CRITICAL RULES:
- You MUST call at least one search or reasoning tool BEFORE calling finish().
- Never call finish() as your first action. Always search or compute first.
- Do not guess from memory. Use web_search, wikipedia_search, or other tools to find and verify facts.
- If a web_fetch fails, try a different URL or use web_search with a refined query.
- Use search tools ONLY to gather the underlying FACTS and evidence, then REASON or COMPUTE the answer yourself (python_exec / calculator for any math). Do NOT search for benchmark answer keys, leaked question/answer datasets, or files such as GAIA answer lists — derive the answer from primary, authoritative sources only.
- As soon as your Observations contain enough to determine the answer, STOP searching and call finish(). You have a limited step budget; a well-supported answer now beats more searching, and repeated searches will not help.
- Be precise. Do not explain. Just give the answer value in finish()."""


def normalize_questions(records):
    """Copy records and map the released ``answer`` field to ``expected``.

    Custom records may already use ``expected`` or omit reference answers.
    Neither reference field is passed to the agent as task context.
    """
    questions = []
    for record in records:
        item = deepcopy(dict(record))
        if not item.get("id") or not isinstance(item.get("question"), str):
            raise ValueError("Each question needs a nonempty id and question text")
        item["expected"] = item.get("expected", item.get("answer", ""))
        item.pop("answer", None)
        item["context"] = item.get("context") or {}
        item.setdefault("variant_type", "original")
        item.setdefault("parent_id", None)
        questions.append(item)
    return questions


def order_questions(questions, *, groups=None, order="file"):
    """Select complete families in file order or original-then-variants order.

    ``order='family'`` retains the existing experiment ordering: each original
    is immediately followed by its variants, preserving their relative order.
    ``order='file'`` preserves the source file's order. ``groups=None`` or 0
    selects every family. This function does not mutate its input records.
    """
    if order not in {"file", "family"}:
        raise ValueError("order must be 'file' or 'family'")
    if groups is not None and groups < 0:
        raise ValueError("groups must be nonnegative")
    questions = list(questions)
    originals = [item for item in questions if item.get("variant_type") == "original"]
    selected = originals[:groups] if groups else originals
    if order == "file":
        if not groups:
            return questions
        selected_ids = {item["id"] for item in selected}
        return [item for item in questions
                if (item.get("parent_id") or item["id"]) in selected_ids]
    variants_by_parent = {}
    for item in questions:
        parent = item.get("parent_id")
        if parent:
            variants_by_parent.setdefault(parent, []).append(item)
    ordered = []
    for original in selected:
        ordered.append(original)
        ordered.extend(variants_by_parent.get(original["id"], []))
    return ordered


def _financebench_stores(questions):
    """Build the document-tool stores from each record's own supplied evidence."""
    evidence_store = {}
    metadata_store = {}
    for item in questions:
        context = item["context"]
        passages = []
        for evidence in context.get("evidence") or []:
            if isinstance(evidence, str):
                passages.append({"evidence_text": evidence})
            elif isinstance(evidence, dict):
                passages.append(deepcopy(evidence))
        if passages:
            evidence_store[item["id"]] = passages
        doc_name = context.get("doc_name", "")
        if doc_name and doc_name not in metadata_store:
            metadata_store[doc_name] = {
                "company": context.get("company", ""),
                "doc_name": doc_name,
                "fiscal_year": "",
                "doc_type": "",
            }
    return evidence_store, metadata_store


def load_benchmark(dataset, *, groups=None, order="file", sql_data_dir=None,
                   revision=None, token=None):
    """Return ``(questions, tools, preamble)`` for a published REUSE-QA split.

    Library calls preserve file order by default; the experiment CLI explicitly
    selects family order. FinanceBench evidence is installed in the document
    tool, then removed only from the copied runtime context. Its full record ID
    is supplied as ``question_id``. GAIA tools support text and web execution;
    attachment-dependent questions require additional tools. Attachments are not
    downloaded implicitly.
    """
    if dataset not in DATASET_REPOSITORIES:
        raise ValueError(f"Unknown benchmark split: {dataset}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install benchmark dependencies with pip install '.[datasets]'") from exc
    from recap import config
    from recap.tools.manager import (
        get_financebench_tools, get_gaia_tools, get_sql_api_tools,
    )

    token = token if token is not None else (config.HF_TOKEN or None)
    records = load_dataset(
        DATASET_REPOSITORIES[dataset], "default", split=dataset,
        revision=revision, token=token,
    )
    questions = order_questions(
        normalize_questions(records), groups=groups, order=order,
    )
    if dataset == "financebench":
        evidence_store, metadata_store = _financebench_stores(questions)
        tools = get_financebench_tools(evidence_store, metadata_store)
        for item in questions:
            item["context"]["question_id"] = item["id"]
            item["context"].pop("evidence", None)
        return questions, tools, PREAMBLE_FINANCEBENCH
    if dataset == "sql_api_bench":
        return questions, get_sql_api_tools(data_dir=sql_data_dir), PREAMBLE_SQL_API_BENCH
    from recap.tools.gaia import set_benchmark_questions
    set_benchmark_questions(questions)
    return questions, get_gaia_tools(), PREAMBLE_GAIA
