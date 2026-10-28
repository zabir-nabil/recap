from recap.tools import BaseTool
from recap import config
from recap.llm import call_llm


class DocSearch(BaseTool):
    name = "doc_search"
    description = (
        "search a financial document for passages relevant to a query. "
        "takes a question_id and query, returns the most relevant text excerpts."
    )

    def __init__(self, evidence_store, observation_max_chars=None):
        """
        evidence_store: dict mapping question_id -> list of evidence dicts.
        keyed by question_id (not doc_name) because multiple questions about
        the same document need different evidence passages.
        """
        self.evidence_store = evidence_store
        self.observation_max_chars = observation_max_chars

    def execute(self, question_id, query):
        if question_id not in self.evidence_store:
            return {"status": "error", "output": f"no evidence for question: {question_id}", "output_type": "text"}

        evidence_list = self.evidence_store[question_id]
        passages = []
        for ev in evidence_list:
            text = ev.get("evidence_text_full_page") or ev.get("evidence_text", "")
            if text:
                passages.append(text)

        if not passages:
            return {"status": "error", "output": f"evidence list empty for: {question_id}", "output_type": "text"}
        result = {"status": "success", "output": passages, "output_type": "text"}
        if self.observation_max_chars is not None:
            result["observation_max_chars"] = self.observation_max_chars
        return result


class LLMAnalyze(BaseTool):
    name = "llm_analyze"
    description = (
        "send a financial document passage and a specific question to a small llm. "
        "the llm reads the passage and extracts the answer. "
        "use for extracting numbers, line items, or interpreting financial data."
    )
    is_deterministic = False
    llm_needed_output = True

    def execute(self, passage, question):
        messages = [
            {
                "role": "system",
                "content": (
                    "you are a financial analyst assistant. "
                    "given a passage from a financial document, answer the question precisely. "
                    "provide only the answer with numbers and units. do not explain."
                ),
            },
            {
                "role": "user",
                "content": f"passage:\n{passage}\n\nquestion: {question}",
            },
        ]
        # pre-initialize token dict so it's returned even if call_llm raises
        tokens_dict = {"input": 0, "output": 0, "reasoning": 0,
                       "model": config.SMALL_LLM_MODEL, "provider": config.SMALL_LLM_PROVIDER}
        try:
            response = call_llm(
                messages,
                model=config.SMALL_LLM_MODEL,
                provider=config.SMALL_LLM_PROVIDER,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            tokens_dict["input"] = response.input_tokens
            tokens_dict["output"] = response.output_tokens
            tokens_dict["reasoning"] = getattr(response, "reasoning_tokens", 0) or 0
            return {
                "status": "success",
                "output": response.text,
                "output_type": "text",
                "tokens": tokens_dict,
            }
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text",
                    "tokens": tokens_dict}


class GetFilingMetadata(BaseTool):
    name = "get_filing_metadata"
    description = "get metadata about a filing: company, fiscal year, doc type, sector."

    def __init__(self, metadata_store):
        """metadata_store: dict mapping doc_name -> metadata dict"""
        self.metadata_store = metadata_store

    def execute(self, doc_name):
        if doc_name not in self.metadata_store:
            return {"status": "error", "output": f"metadata not found: {doc_name}", "output_type": "text"}
        return {"status": "success", "output": self.metadata_store[doc_name], "output_type": "structured"}



def _normalize_evidence(raw_evidence):
    """
    normalize evidence from huggingface datasets.

    huggingface datasets can return list-of-dicts fields
    in two different formats:
      - list of dicts: [{"evidence_text": "...", "doc_name": "..."}, ...]
      - dict of lists: {"evidence_text": ["...", ...], "doc_name": ["...", ...]}

    this function normalizes both to list of dicts.
    """
    if not raw_evidence:
        return []

    # already a list of dicts
    if isinstance(raw_evidence, list):
        if len(raw_evidence) > 0 and isinstance(raw_evidence[0], dict):
            return raw_evidence
        return []

    # dict - could be dict-of-lists (arrow columnar) or a single evidence dict
    if isinstance(raw_evidence, dict):
        keys = list(raw_evidence.keys())
        if not keys:
            return []

        # check if values are lists (dict-of-lists format)
        first_val = raw_evidence[keys[0]]
        if isinstance(first_val, list):
            n = len(first_val)
            try:
                return [{k: raw_evidence[k][i] for k in keys} for i in range(n)]
            except (IndexError, TypeError):
                return []
        else:
            # single evidence dict - wrap it in a list
            return [raw_evidence]

    return []


def _get_first(row, *field_names, default=""):
    """try multiple field names, return the first non-empty value.
    handles the merged dataset where field names may vary
    (e.g. 'company' vs 'company_x', 'comany_sector_gics' vs 'gics_sector')."""
    for name in field_names:
        val = row.get(name)
        if val is not None and val != "":
            return val
    return default
