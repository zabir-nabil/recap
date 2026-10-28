import os
import random


# ---- .env loader (zero-dependency) ----
# Load the working directory's .env file, or RECAP_ENV_FILE when specified.
# No environment file is bundled with the package.
# real shell environment variables always take precedence over .env values.
def _load_dotenv(path=None):
    path = path or os.getenv("RECAP_ENV_FILE", ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            # do not override real environment variables
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

# ---- api keys (loaded from .env or the shell; no secrets in source) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---- model configuration ----
# Two tiers, named by capability/cost:
#   LLM_BIG   — the expensive, capable model (A_big): novel solves, fallbacks.
#   LLM_SMALL — the cheap model (A_small): tool-internal calls, and RECAP's
#               reuse path (judge, guided agent, format+verify).
# default: gemini-3.5-flash ($1.50/$9 per M) + gemini-3.1-flash-lite ($0.25/$1.50 per M).
LLM_BIG = os.getenv("LLM_BIG", os.getenv("PLANNER_MODEL", "gemini-3.5-flash"))
LLM_BIG_PROVIDER = os.getenv("LLM_BIG_PROVIDER", os.getenv("PLANNER_PROVIDER", "gemini"))
LLM_SMALL = os.getenv("LLM_SMALL", os.getenv("SMALL_LLM_MODEL", "gemini-3.1-flash-lite"))
LLM_SMALL_PROVIDER = os.getenv("LLM_SMALL_PROVIDER", os.getenv("SMALL_LLM_PROVIDER", "gemini"))

# Back-compat aliases (agents' planner/executor roles + tool-internal small LLM).
# Planner and executor are the SAME big tier by default; keeping the names means
# AgentConfig.resolve() and Plan-and-Act retain these role defaults.
PLANNER_MODEL = LLM_BIG
PLANNER_PROVIDER = LLM_BIG_PROVIDER
EXECUTOR_MODEL = LLM_BIG
EXECUTOR_PROVIDER = LLM_BIG_PROVIDER
# cheap llm used as a tool inside financebench (llm_analyze) and gaia (llm_reason)
SMALL_LLM_MODEL = LLM_SMALL
SMALL_LLM_PROVIDER = LLM_SMALL_PROVIDER

# ---- embedding configuration ----
# provider: "local" (sentence-transformers + FAISS) or "openai" (API)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
# local models: any sentence-transformers compatible model from HuggingFace
#   fast/small:  "all-MiniLM-L6-v2" (22M, 384d, ~10ms/query CPU)
#   balanced:    "all-MiniLM-L12-v2" (33M, 384d, ~15ms/query CPU)
#   ibm granite: "ibm-granite/granite-embedding-30m-english" (30M, 384d)
#                "ibm-granite/granite-embedding-125m-english" (125M, 768d)
#   bge family:  "BAAI/bge-small-en-v1.5" (33M, 384d)
# openai models: "text-embedding-3-small" (API, ~200ms/query)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---- RECAP retrieval threshold (tau) ----
# tau is RECAP's retrieval pre-filter: a new question retrieves a cached plan only
# if the cosine similarity of their question embeddings >= tau (else RECAP skips
# the judge and runs A_big). It is deliberately a LOOSE, high-recall filter, not
# a precision gate: its only job is to reject clearly-unrelated candidates (and a
# cold/sparse index, where it returns nothing). The real screening happens
# downstream — the A_small judge decides reusability per candidate, and
# format+verify guards the answer — so tau is set low to keep essentially all
# genuine reuse opportunities and let the cheap-to-reject negatives fall to the
# judge. DEFAULT = 0.5: a generic low threshold that keeps ~all true reuses
# across datasets while still filtering the obviously-irrelevant. (A per-dataset
# sensitivity sweep can refine this later.)
#
RECAP_TAU_DEFAULT = float(os.getenv("RECAP_TAU_DEFAULT", "0.5"))

# Optional per-dataset overrides (env: RECAP_TAU_SQL_API_BENCH=<value> etc.), kept
# for the sensitivity sweep; unset -> RECAP_TAU_DEFAULT (the shipped behavior).
RECAP_TAU_BY_DATASET = {
    ds: (float(os.environ[env]) if env in os.environ else None)
    for ds, env in (
        ("sql_api_bench", "RECAP_TAU_SQL_API_BENCH"),
        ("financebench", "RECAP_TAU_FINANCEBENCH"),
        ("gaia_l1", "RECAP_TAU_GAIA_L1"),
        ("gaia_l2", "RECAP_TAU_GAIA_L2"),
        ("gaia", "RECAP_TAU_GAIA"),
        ("toolqa", "RECAP_TAU_TOOLQA"),
    )
}


def recap_tau(dataset=None):
    """resolve RECAP's retrieval tau for a dataset: a per-dataset override if set,
    else RECAP_TAU_DEFAULT (0.5)."""
    if dataset and RECAP_TAU_BY_DATASET.get(dataset) is not None:
        return RECAP_TAU_BY_DATASET[dataset]
    return RECAP_TAU_DEFAULT

# ---- per-token pricing (USD per million tokens) ----
# Configured provider prices. Output rate applies to reasoning
# tokens too. "cached" = cached-input rate where the provider offers one.
MODEL_PRICING = {
    # gemini 3.x  (A_big = gemini-3.5-flash, A_small = gemini-3.1-flash-lite)
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached": 0.15},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    # gemini 2.5  (kept for fallback / alternate runs)
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached": 0.01},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # openai gpt-5  (A_big = gpt-5, A_small = gpt-5-mini)
    "gpt-5": {"input": 1.25, "output": 10.00, "cached": 0.125},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached": 0.025},
    # legacy
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

# warn once per unknown model per process (avoids log spam)
_PRICING_WARNED = set()

def get_token_cost(model, input_tokens, output_tokens):
    """compute USD cost for a given model and token counts.

    output_tokens is the BILLABLE output (raw output + reasoning); reasoning
    tokens are billed at the output rate by Gemini and OpenAI.
    """
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        if model not in _PRICING_WARNED:
            import logging
            logging.getLogger(__name__).warning(
                "no pricing entry for model %r; using fallback $1.0/$4.0 per million. "
                "add it to config.MODEL_PRICING for accurate USD reporting.",
                model,
            )
            _PRICING_WARNED.add(model)
        pricing = {"input": 1.0, "output": 4.0}
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

# ---- llm settings ----
# TEMPERATURE is shared by agents, RECAP decision calls, and tool-internal LLMs.
# None leaves the temperature unset so each provider uses its own default.
# Set RECAP_TEMPERATURE to request a fixed value for an experiment.
_raw_temp = os.getenv("RECAP_TEMPERATURE")
TEMPERATURE = (
    None
    if _raw_temp is None
    or _raw_temp.strip().lower() in {"", "default", "none", "unset"}
    else float(_raw_temp)
)
MAX_TOKENS = 4096

# A process-wide seed is shared by provider calls, including tool-internal calls.
# Runners call set_experiment_seed() before constructing any agents.
_raw_seed = os.getenv("RECAP_SEED")
EXPERIMENT_SEED = int(_raw_seed) if _raw_seed not in (None, "") else None


def set_experiment_seed(seed):
    """Resolve the seed used by all provider calls in this process."""
    global EXPERIMENT_SEED
    EXPERIMENT_SEED = None if seed is None else int(seed)
    if EXPERIMENT_SEED is not None:
        random.seed(EXPERIMENT_SEED)
        try:
            import numpy as np
            np.random.seed(EXPERIMENT_SEED)
        except ImportError:
            pass
    return EXPERIMENT_SEED

# RECAP decision calls request a 1024-token thinking budget by default.
# This is a provider hint, not a hard cap; the ledger bills observed tokens.
# GEMINI_THINKING_BUDGET overrides the per-call request when explicitly set.
RECAP_DECISION_THINKING_BUDGET = int(os.getenv("RECAP_DECISION_THINKING_BUDGET", "1024"))

# --- RECAP reuse-path feature flags (default to the SAFE/validated setting) ---
def _envflag(name, default):
    return os.getenv(name, default) not in ("0", "false", "False")
# index verify-reject fallback plans? OFF by default: label-free admission can
# store a wrong-but-successful fallback and poison future reuse (needs a quality
# gate / provisional index + ablation before enabling for a publication run).
RECAP_INDEX_FALLBACKS = _envflag("RECAP_INDEX_FALLBACKS", "0")
# certified substituted replay of value-variants. OFF by default: the guard is a
# tool-name denylist (no compute steps), NOT true output->argument provenance, so
# it cannot certify a plan where one tool's output feeds a later tool's argument
# (e.g. account_lookup(id_from_prior_lookup) stays stale after the entity swaps).
# Value-variants therefore route to the guided agent (which re-derives bindings)
# unless this is explicitly enabled for an ablation with a provenance-aware guard.
RECAP_CERTIFIED_REPLAY = _envflag("RECAP_CERTIFIED_REPLAY", "0")
# pass the judge's value bindings into the guided demonstration (ablation flag).
RECAP_GUIDED_BINDINGS = _envflag("RECAP_GUIDED_BINDINGS", "1")
# Primary-study admission is clean-only.  Salvaged max-budget traces can be
# enabled explicitly for the planned admission-policy sensitivity ablation, but
# are not admitted by default because direct reuse has no verifier.
RECAP_ADMIT_SALVAGE = _envflag("RECAP_ADMIT_SALVAGE", "0")

# ---- data sources ----

# sql-api-bench: spider SQLite databases
SQL_API_BENCH_DATA_DIR = os.getenv("SQL_API_BENCH_DATA_DIR", "./datasets/sql_api_bench_databases")

# financebench: loaded directly from huggingface
FINANCEBENCH_HF_DATASET = "PatronusAI/financebench"

# ---- gaia ----

# gaia: loaded from huggingface (gated dataset, requires HF_TOKEN)
GAIA_HF_DATASET = "gaia-benchmark/GAIA"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# web search provider for gaia tools: "serper", "serpapi", or "google"
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
# google custom search (closed to new customers as of 2025)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("GOOGLE_CX", "")
