<div align="center">
  <img src="https://huggingface.co/spaces/RECAP-data/README/resolve/main/recap-banner.png" alt="RECAP: capture, compile, retrieve, judge, and execute reusable agent plans" width="100%">
  <p><strong>Retrieval and Execution of Compiled Agent Plans</strong></p>
  <p><a href="#quick-start">Quick start</a> · <a href="#run-an-experiment">Experiments</a> · <a href="#use-recap-with-an-agent">Agent integration</a> · <a href="#datasets-and-tools">Datasets</a></p>
</div>

# RECAP

RECAP turns prior tool-using agent executions into structured plans that can be replayed, adapted, or rejected for a new request. Plans retain successful tool calls, their arguments, and their intended goals. Reuse executes tools again to obtain fresh observations.

The library includes ReAct and Plan-and-Act agents, deterministic plan compilation, embedding retrieval, an applicability judge, and optional answer verification. It accompanies *RECAP: Structured Plan Reuse for Efficient Tool-Using Agents* and the [REUSE-QA benchmark](https://huggingface.co/RECAP-data).

## How RECAP works

**Capture → Compile → Retrieve → Judge → Execute**

1. The larger agent states each tool call's goal during a fresh solve *(self-annotation)*.
2. Compilation keeps successful, nonterminal calls in order, without another model call.
3. An embedding index retrieves candidate plans; a smaller model judges applicability.
4. RECAP replays eligible calls, guides a smaller agent through adaptation, or solves afresh.
5. With verification enabled, rejected reuse falls back to the larger agent.

**RECAP-verified** checks reused answers. **RECAP-lite** omits verification and uses a formatter after replay. These are different accuracy–cost operating points; verification offers better robustness.

## Quick start

Install the library from this directory:

```bash
pip install .
```

For the benchmark loader and experiment runner, install the dataset extra:

```bash
pip install ".[datasets]"
```

Copy `.env.example` to `.env` in your working directory and configure the provider keys **before importing `recap`**. You can select another environment file with `RECAP_ENV_FILE`; existing environment variables take precedence.

The default model configuration is:

```dotenv
LLM_BIG=gemini-3.5-flash
LLM_BIG_PROVIDER=gemini
LLM_SMALL=gemini-3.1-flash-lite
LLM_SMALL_PROVIDER=gemini
```

Set the corresponding provider key, such as `GEMINI_API_KEY`, in the same environment. The LLM wrapper also supports OpenAI, DeepSeek, and OpenRouter. Embeddings default to local `all-MiniLM-L6-v2` with exact FAISS retrieval; the embedding model is downloaded on first use.

Model costs use the rates in `recap.config.MODEL_PRICING`; update them when using other models or rates.

## Run an experiment

One entry point runs one agent and RECAP configuration over an ordered stream:

```bash
python -m recap.experiment --dataset financebench --agent react --mode verified --groups 5 --output results/financebench.jsonl
python -m recap.experiment --dataset sql_api_bench --agent plan_and_act --mode lite --groups 5 --sql-data-dir /path/to/prepared/spider --output results/sql.jsonl
python -m recap.experiment --dataset gaia_l1 --agent react --mode verified --groups 5 --output results/gaia.jsonl
```

The installed command `recap-experiment` accepts the same arguments. Available splits are `financebench`, `sql_api_bench`, `gaia_l1`, and `gaia_l2`. Omit `--groups`, or use 0, to select all families. Additional controls are `--tau`, `--top-k`, `--max-steps`, `--seed`, and `--revision`.

The runner defaults to `--order family`: an original immediately precedes its variants, matching the existing experiment loader. Use `--order file` to retain the published file order. Each run starts with an empty plan library. Record the order and dataset revision when comparing runs.

Results are written incrementally as one JSON object per question, including the final answer, executed steps, execution path, model-call ledger, token usage, model cost, and elapsed time. Known benchmarks receive a deterministic final-answer score after execution. `success` records agent completion, not answer correctness. The final console summary reports model cost and, when scored, accuracy.

## Use RECAP with an agent

The public API uses the same engine as the experiment runner. This example selects one FinanceBench family:

```python
from recap import RecapEngine, run_with_recap
from recap.datasets import load_benchmark

questions, tools, preamble = load_benchmark("financebench", groups=1)
engine = RecapEngine(
    tools=tools, dataset="financebench", preamble=preamble,
    agent_type="react", config_mode="verified",
)
for question in questions:
    result = run_with_recap(question, engine)
    print(result["agent_answer"])
```

For Plan-and-Act with RECAP-lite, create a fresh engine and tool set:

```python
questions, tools, preamble = load_benchmark("financebench", groups=1)
engine = RecapEngine(
    tools=tools, dataset="financebench", preamble=preamble,
    agent_type="plan_and_act", config_mode="direct",
)
for question in questions:
    result = run_with_recap(question, engine)
    print(result["agent_answer"])
```

The CLI's `lite` mode maps to the engine's `config_mode="direct"`; `verified` uses `config_mode="verified"`. `load_benchmark` preserves file order by default; pass `order="family"` for the runner's default ordering.

Defaults are `tau=0.5`, `top_k=3`, and an agent reasoning-call budget of `max_steps=15`. Use one engine per sequential stream; its per-question state is shared within that engine.

### Integrate another agent

Subclass the engine's construction hook to supply an agent that follows the same interface:

```python
from recap import RecapEngine
from my_project import MyAgent

class MyRecapEngine(RecapEngine):
    def _make_agent(self, config):
        return MyAgent(tools=self.tools, config=config)
```

`MyAgent.run(query, context=None)` must return an `AgentResult`, with `StepRecord` tool records and a `LLMCallRecord` ledger. These types are exported from `recap`. Follow the supplied `AgentConfig`, including model selection, `self_annotate` on fresh solves, and `demonstration` on guided execution. Record each tool call's goal, arguments, and result; successful calls use `result["status"] == "success"`. Reset per-question tool state. The engine's replay, applicability decisions, compilation, and verification remain shared.

### Use another dataset

A custom experiment factory should return `(questions, tools, preamble)`. Questions need `id`, `question`, and optional `context`; reference answers may use `answer` or `expected`. For related families, supply `variant_type` and `parent_id`. The runner copies records and maps `answer` to `expected`; only the question and runtime context reach the agent.

For example, place this factory in an importable `my_experiment` module, with your own `my_tools.make_tools` implementation:

```python
import json
from my_tools import make_tools

def build():
    with open("questions.jsonl", encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle]
    return questions, make_tools(), "Instructions for this task."

def score(result, question):
    return result["agent_answer"].strip() == question["expected"].strip()
```

```bash
python -m recap.experiment --factory my_experiment:build --scorer my_experiment:score --dataset-name my_tasks --agent react --output results/custom.jsonl
```

Custom scorers receive `(result, question)` after execution and return a boolean or `(boolean, reason)`. Without `--scorer`, custom datasets receive no correctness metric. Tools follow `BaseTool`: provide a name, description, `execute`, and any needed `reset` method. A tool result is a dictionary containing `status`, `output`, and `output_type`.

## Datasets and tools

| Source | REUSE-QA location | Execution setup |
| --- | --- | --- |
| FinanceBench | [Main repository](https://huggingface.co/datasets/RECAP-data/reuse-qa), `financebench` | Supplied evidence is installed in the document tool; each full record ID becomes `question_id`. |
| SQL-API-Bench | [Main repository](https://huggingface.co/datasets/RECAP-data/reuse-qa), `sql_api_bench` | Run the preparation command below; set `SQL_API_BENCH_DATA_DIR` or pass `--sql-data-dir`. |
| GAIA | [Gated companion](https://huggingface.co/datasets/RECAP-data/reuse-qa-gaia), `gaia_l1` / `gaia_l2` | Accept the repository's conditions and authenticate or set `HF_TOKEN`; configure a search provider and its key. |

Prepare SQL databases from this checkout, or point the script at an existing Spider copy with `--spider-dir`:

```bash
pip install gdown
python scripts/prepare_sql_api_bench.py --work-dir ./sql_api_bench_data
```

The published records have eight fields: `id`, `parent_id`, `dataset`, `variant_type`, `question`, `answer`, `answer_type`, and `context`. Runtime records use `expected` for the reference answer. References are used for scoring. FinanceBench uses oracle evidence.

The supplied GAIA tools support text, web search, page fetching, reasoning, and computation. 

## Citation

Please cite the accompanying manuscript, *RECAP: Structured Plan Reuse for Efficient Tool-Using Agents*, and the source benchmarks used in your evaluation. Dataset provenance, access conditions, and source citations are documented in the two REUSE-QA dataset cards linked above.

The GAIA tools include adaptations from [OctoTools](https://github.com/octotools/octotools). 
