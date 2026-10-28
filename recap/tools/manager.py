"""Tool registration for REUSE-QA's FinanceBench, SQL-API-Bench, and GAIA tasks."""

from recap import config
from recap.tools.shared import Calculator, Finish


# ---- sql-api-bench ----

def get_sql_api_tools(data_dir=None):
    """
    get tools for sql-api-bench.
    data_dir: the directory created by the setup script (contains spider/ subdirectory).
    """
    from recap.tools.sql_api_bench import (
        find_database_dir,
        GetSchema, SQLExecute, CountSyllables,
        IsPrime, IsFibonacci, IsPerfectSquare, IsEven, IsDivisibleBy,
        Geocode, GeoDistance,
    )

    data_dir = data_dir or config.SQL_API_BENCH_DATA_DIR
    database_dir = find_database_dir(data_dir)
    if database_dir is None:
        raise FileNotFoundError(
            f"could not find spider database directory in {data_dir}. "
            "run the setup script first."
        )

    geocode = Geocode()

    return [
        GetSchema(str(database_dir)),
        SQLExecute(str(database_dir)),
        CountSyllables(),
        IsPrime(), IsFibonacci(), IsPerfectSquare(), IsEven(), IsDivisibleBy(),
        geocode, GeoDistance(geocode),
        Calculator(),
        Finish(),
    ]


# ---- financebench ----

def get_financebench_tools(evidence_store, metadata_store,
                           *, include_metadata=True, observation_max_chars=None):
    """
    get tools for financebench.
    pass in the evidence and metadata stores for the current questions.
    """
    from recap.tools.financebench import DocSearch, LLMAnalyze, GetFilingMetadata

    result = [
        DocSearch(evidence_store, observation_max_chars=observation_max_chars),
        LLMAnalyze(),
        Calculator(),
    ]
    if include_metadata:
        result.append(GetFilingMetadata(metadata_store))
    result.append(Finish())
    return result


# ---- gaia ----

def get_gaia_tools():
    """
    get the 9 text-only tools for the GAIA benchmark (GAIA-Text evaluation,
    inspired by the optimized toolset from the octotools paper):
    web_search, web_fetch, wikipedia_search, llm_reason,
    python_exec, arxiv_search, pubmed_search, calculator, finish.
    """
    from recap.tools.gaia import (
        WebSearch, WebFetch, WikipediaSearch, LLMReasoner,
        ArxivSearch, PubMedSearch,
    )
    from recap.tools.python_exec import PythonInterpreter

    return [
        WebSearch(),
        WebFetch(),
        WikipediaSearch(),
        LLMReasoner(),
        PythonInterpreter(),
        ArxivSearch(),
        PubMedSearch(),
        Calculator(),
        Finish(),
    ]
