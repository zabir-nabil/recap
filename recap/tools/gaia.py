"""
GAIA benchmark tools.

tools for the GAIA (General AI Assistants) benchmark, covering web search,
url text extraction, wikipedia lookup, LLM reasoning, arxiv search, and
pubmed search. these are the text-only tools used for GAIA-Text evaluation.

tool implementations are ported from octotools (https://github.com/octotools/octotools)
and adapted to the BaseTool interface used by this project.
Adapted portions: Copyright (c) 2025 Pan Lu, MIT License.
The full upstream notice is retained in licenses/OctoTools-MIT.txt.
"""

import os
import re
from functools import lru_cache
import xml.etree.ElementTree as ET
from recap.tools import BaseTool
from recap import config
from recap.llm import call_llm


# ---------------------------------------------------------------------------
# GAIA answer-key guard
# ---------------------------------------------------------------------------
# GAIA's own question/answer files are publicly mirrored, so an agent with open
# web access can "solve" a question by finding the benchmark's leaked answers
# instead of reasoning. Known sources and explicit answer-key searches are
# blocked. We also compare queries and returned content with question text
# (never gold answers), which catches unknown mirrors and papers quoting items.
_ANSWER_KEY_DENYLIST = (
    "gaia_web.jsonl",
    "gaia-benchmark",
    "albertvillanova/answers",
    "huggingface.co/datasets/gaia",
)

_ANSWER_KEY_MARKERS = (
    "answer key",
    "benchmark answer",
    "expected_answer",
    "gold answer",
    "golden answer",
    "ground truth answer",
    "reference answer",
    "true_answer",
)

_QUESTION_NGRAM_SIZE = 10
_WORD_RE = re.compile(r"[a-z0-9]+")
_BENCHMARK_QUESTION_TEXTS = ()


def _normalized_words(value):
    return tuple(_WORD_RE.findall(str(value or "").lower()))


def set_benchmark_questions(questions):
    """Register only question text for the in-memory benchmark lookup guard.

    Dataset adapters call this before constructing GAIA tools. Direct tool users
    may pass question strings or dictionaries with a ``question`` field. No
    reference answers or other record fields are retained or written to disk.
    """
    global _BENCHMARK_QUESTION_TEXTS
    _BENCHMARK_QUESTION_TEXTS = tuple(
        str((question.get("question", "") if isinstance(question, dict) else question) or "")
        for question in questions
    )
    _benchmark_question_ngrams.cache_clear()


@lru_cache(maxsize=4)
def _benchmark_question_ngrams(size=_QUESTION_NGRAM_SIZE):
    """Return long token sequences from registered question text, without gold."""
    ngrams = set()
    for question in _BENCHMARK_QUESTION_TEXTS:
        words = _normalized_words(question)
        for index in range(len(words) - size + 1):
            ngrams.add(words[index:index + size])
    return frozenset(ngrams)


def _quotes_benchmark_question(value, size=_QUESTION_NGRAM_SIZE):
    """True when text reproduces a long contiguous span from a GAIA item."""
    words = _normalized_words(value)
    if len(words) < size:
        return False
    known = _benchmark_question_ngrams(size)
    return any(
        words[index:index + size] in known
        for index in range(len(words) - size + 1)
    )


def _has_answer_key_marker(value):
    text = str(value or "").lower()
    return any(marker in text for marker in _ANSWER_KEY_MARKERS)


def _is_answer_key_query(query):
    """True for explicit answer-key or verbatim benchmark-item searches."""
    text = str(query or "")
    lowered = text.lower()
    if _has_answer_key_marker(text) or _quotes_benchmark_question(text):
        return True
    if "gaia" in lowered:
        if _quotes_benchmark_question(text, size=4):
            return True
        if any(
            marker in lowered
            for marker in ("answer", "dataset", "gold", "question", "solution")
        ):
            return True
    if "site:huggingface.co/datasets" in lowered:
        return True
    return False


def _is_answer_key_url(url):
    """True if a URL points at a known GAIA answer-key source."""
    u = (url or "").lower()
    return any(p in u for p in _ANSWER_KEY_DENYLIST)


def _is_answer_key_result(r):
    """True if a result references or reproduces GAIA answer-key material."""
    if not isinstance(r, dict):
        return False
    hay = " ".join(
        str(r.get(k, "")) for k in ("link", "url", "title", "snippet")
    )
    lowered = hay.lower()
    return (
        any(pattern in lowered for pattern in _ANSWER_KEY_DENYLIST)
        or _has_answer_key_marker(hay)
        or _quotes_benchmark_question(hay)
    )


def _blocked_answer_key_response():
    return {
        "status": "error",
        "output": (
            "blocked: benchmark answer-key lookup or quoted benchmark content "
            "is not allowed; search underlying facts in primary sources"
        ),
        "output_type": "text",
    }


# ---------------------------------------------------------------------------
# HTTP with retry — web tools were failing (and wasting an agent step) on
# transient network errors (connection reset after sleep/wake, timeouts, brief
# drops). Retry ONLY connection-level exceptions with backoff; do NOT retry HTTP
# 4xx/5xx (those come back as normal responses and are handled by the caller —
# e.g. a 400 from a malformed query must not be retried).
# ---------------------------------------------------------------------------

_HTTP_RETRIES = 3
_HTTP_BACKOFF = 1.0   # seconds; grows 1s, 2s, 4s


def _http_request(method, url, **kwargs):
    import time as _t
    import requests
    transient = (requests.exceptions.ConnectionError,
                 requests.exceptions.Timeout,
                 requests.exceptions.ChunkedEncodingError)
    last = None
    for attempt in range(_HTTP_RETRIES):
        try:
            return requests.request(method, url, **kwargs)
        except transient as e:
            last = e
            if attempt < _HTTP_RETRIES - 1:
                _t.sleep(_HTTP_BACKOFF * (2 ** attempt))
    raise last


def _http_get(url, **kwargs):
    return _http_request("GET", url, **kwargs)


def _http_post(url, **kwargs):
    return _http_request("POST", url, **kwargs)


# ---- web search ----

class WebSearch(BaseTool):
    name = "web_search"
    description = (
        "search the web for information. returns a list of results with "
        "title, snippet, and url. supports serper, serpapi, and google backends."
    )
    is_deterministic = False
    has_side_effect = False

    def execute(self, query, num_results=10):
        if _is_answer_key_query(query):
            return _blocked_answer_key_response()
        try:
            num_results = int(num_results)
        except (ValueError, TypeError):
            num_results = 10
        provider = config.SEARCH_PROVIDER.lower()
        try:
            if provider == "serper":
                result = self._search_serper(query, num_results)
            elif provider == "serpapi":
                result = self._search_serpapi(query, num_results)
            elif provider == "google":
                result = self._search_google(query, num_results)
            else:
                return {"status": "error", "output": f"unknown search provider: {provider}", "output_type": "text"}
        except Exception as e:
            return {"status": "error", "output": f"search error: {e}", "output_type": "text"}

        # drop benchmark answer-key sources so the agent can't look up GAIA's answers
        if result.get("status") == "success" and isinstance(result.get("output"), list):
            result["output"] = [r for r in result["output"] if not _is_answer_key_result(r)]
        return result

    def _search_serper(self, query, num_results):
        import requests
        api_key = config.SERPER_API_KEY
        if not api_key:
            return {"status": "error", "output": "SERPER_API_KEY not set", "output_type": "text"}
        resp = _http_post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })
        return {"status": "success", "output": results, "output_type": "structured"}

    def _search_serpapi(self, query, num_results):
        import requests
        api_key = config.SERPAPI_API_KEY
        if not api_key:
            return {"status": "error", "output": "SERPAPI_API_KEY not set", "output_type": "text"}
        resp = _http_get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": api_key, "num": num_results, "engine": "google"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })
        return {"status": "success", "output": results, "output_type": "structured"}

    def _search_google(self, query, num_results):
        import requests
        api_key = config.GOOGLE_API_KEY
        cx = config.GOOGLE_CX
        if not api_key or not cx:
            return {"status": "error", "output": "GOOGLE_API_KEY and GOOGLE_CX must be set", "output_type": "text"}
        resp = _http_get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx, "q": query, "num": min(num_results, 10)},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })
        return {"status": "success", "output": results, "output_type": "structured"}


# ---- url text extraction ----

class WebFetch(BaseTool):
    name = "web_fetch"
    description = (
        "fetch a web page and extract its text content. "
        "takes a url, returns the visible text from the page."
    )
    is_deterministic = False
    has_side_effect = False

    # max chars to return to avoid flooding context
    MAX_TEXT_LENGTH = 20000

    def execute(self, url, max_length=None):
        if _is_answer_key_url(url):
            return _blocked_answer_key_response()
        try:
            max_length = int(max_length) if max_length is not None else self.MAX_TEXT_LENGTH
        except (ValueError, TypeError):
            max_length = self.MAX_TEXT_LENGTH
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = _http_get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            # handle plain text / json directly
            if "text/plain" in content_type or "application/json" in content_type:
                text = resp.text[:max_length]
                if _has_answer_key_marker(text) or _quotes_benchmark_question(text):
                    return _blocked_answer_key_response()
                return {"status": "success", "output": text, "output_type": "text"}

            soup = BeautifulSoup(resp.text, "html.parser")
            # remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # collapse multiple blank lines
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text[:max_length]
            if _has_answer_key_marker(text) or _quotes_benchmark_question(text):
                return _blocked_answer_key_response()
            return {"status": "success", "output": text, "output_type": "text"}
        except Exception as e:
            return {"status": "error", "output": f"fetch error for '{url}': {e}", "output_type": "text"}


# ---- wikipedia search ----

class WikipediaSearch(BaseTool):
    name = "wikipedia_search"
    description = (
        "search wikipedia for information. takes a query, returns article "
        "summaries with titles and urls. useful for factual knowledge lookup."
    )
    is_deterministic = False
    has_side_effect = False

    # wikipedia requires a descriptive user-agent or it returns 403
    _HEADERS = {"User-Agent": "AgentPlanReuse/1.0 (tool-benchmark; contact: research)"}

    def execute(self, query, num_results=3):
        import requests
        try:
            # step 1: search for matching articles
            search_resp = _http_get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": num_results,
                    "format": "json",
                },
                headers=self._HEADERS,
                timeout=10,
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            hits = search_data.get("query", {}).get("search", [])

            if not hits:
                return {"status": "success", "output": [], "output_type": "structured"}

            # step 2: get summaries for top results
            results = []
            for hit in hits:
                title = hit.get("title", "")
                # get page summary via REST API
                summary_resp = _http_get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                    headers=self._HEADERS,
                    timeout=10,
                )
                if summary_resp.status_code == 200:
                    summary_data = summary_resp.json()
                    results.append({
                        "title": summary_data.get("title", title),
                        "summary": summary_data.get("extract", ""),
                        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    })
                else:
                    # fallback: use the search snippet (has html tags)
                    snippet = re.sub(r'<[^>]+>', '', hit.get("snippet", ""))
                    results.append({
                        "title": title,
                        "summary": snippet,
                        "url": f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}",
                    })
            return {"status": "success", "output": results, "output_type": "structured"}
        except Exception as e:
            return {"status": "error", "output": f"wikipedia search error: {e}", "output_type": "text"}


# ---- llm reasoning (generalist solution generator) ----

class LLMReasoner(BaseTool):
    name = "llm_reason"
    description = (
        "send a question or prompt to an LLM for general reasoning, analysis, "
        "or synthesis. use when no specialized tool is needed - just thinking."
    )
    is_deterministic = False
    llm_needed_output = True

    def execute(self, prompt, system_prompt=None):
        system_prompt = system_prompt or (
            "you are a helpful reasoning assistant. think step by step "
            "and provide a clear, precise answer."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
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


# ---- arxiv search ----

class ArxivSearch(BaseTool):
    name = "arxiv_search"
    description = (
        "search arxiv for academic papers. takes a query, returns paper "
        "titles, authors, abstracts, and links."
    )
    is_deterministic = False

    def execute(self, query, max_results=5):
        import requests
        try:
            resp = _http_get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                },
                timeout=15,
            )
            resp.raise_for_status()

            # parse atom xml
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)

            results = []
            for entry in entries:
                title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
                # truncate long abstracts
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
                link = ""
                for link_el in entry.findall("atom:link", ns):
                    if link_el.get("type") == "text/html":
                        link = link_el.get("href", "")
                        break
                if not link:
                    link = entry.findtext("atom:id", "", ns)
                results.append({
                    "title": title,
                    "authors": authors[:5],  # cap at 5 authors
                    "abstract": summary,
                    "link": link,
                })
            return {"status": "success", "output": results, "output_type": "structured"}
        except Exception as e:
            return {"status": "error", "output": f"arxiv search error: {e}", "output_type": "text"}


# ---- pubmed search ----

class PubMedSearch(BaseTool):
    name = "pubmed_search"
    description = (
        "search pubmed for biomedical and life science articles. "
        "takes a query, returns article titles, authors, and abstracts."
    )
    is_deterministic = False

    def execute(self, query, max_results=5):
        import requests
        try:
            # step 1: search for PMIDs
            search_resp = _http_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
                timeout=15,
            )
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return {"status": "success", "output": [], "output_type": "structured"}

            # step 2: fetch article details
            fetch_resp = _http_get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
                timeout=15,
            )
            fetch_resp.raise_for_status()

            root = ET.fromstring(fetch_resp.text)
            results = []
            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                if medline is None:
                    continue
                art = medline.find("Article")
                if art is None:
                    continue
                title = art.findtext("ArticleTitle", "")
                # structured abstracts have multiple labeled <AbstractText>
                # sections (Background/Methods/Results/Conclusions); join them
                # all rather than keeping only the first.
                sections = []
                for el in art.findall(".//AbstractText"):
                    text = "".join(el.itertext()).strip()
                    if not text:
                        continue
                    label = el.get("Label") or el.get("NlmCategory")
                    sections.append(f"{label}: {text}" if label else text)
                abstract = " ".join(sections)
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                # authors
                authors = []
                for author in art.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    first = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{first} {last}".strip())
                pmid = medline.findtext("PMID", "")
                results.append({
                    "title": title,
                    "authors": authors[:5],
                    "abstract": abstract,
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                })
            return {"status": "success", "output": results, "output_type": "structured"}
        except Exception as e:
            return {"status": "error", "output": f"pubmed search error: {e}", "output_type": "text"}
