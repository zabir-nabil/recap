"""
reuse/plan.py — the compiled linear plan and the deterministic COMPILE step.

A Plan is the ordered sequence of an agent's SUCCESSFUL tool calls, compiled
from an AgentResult with zero LLM calls. Each step keeps its
"Goal:" provenance line, the concrete tool call, and a deterministic summary of
the tool's output. The plan is the reusable artifact retrieved, judged, and
replayed for similar future questions.
"""

from dataclasses import dataclass, field


# result summaries are guidance only (never returned as an answer), so a tight
# per-field cap keeps a plan at ~100-200 tokens.
_MAX_SUMMARY_CHARS = 200


@dataclass
class PlanStep:
    """one compiled tool call: its intent, the concrete call, and a summary."""

    goal: str            # the step's "Goal:" provenance line ("" if the model omitted it)
    tool_name: str
    tool_args: dict      # concrete args as recorded; used verbatim by deterministic REPLAY
    action_str: str      # human-readable tool call, for prompts/display
    result_summary: str  # deterministic summary of the tool output (never the raw answer)

    def to_dict(self):
        return {
            "goal": self.goal,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "action_str": self.action_str,
            "result_summary": self.result_summary,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            goal=d.get("goal", ""),
            tool_name=d.get("tool_name", ""),
            tool_args=dict(d.get("tool_args", {})),
            action_str=d.get("action_str", ""),
            result_summary=d.get("result_summary", ""),
        )


@dataclass
class Plan:
    """a compiled, re-runnable plan (model-agnostic; embeddings are in the index)."""

    question: str        # the original question this plan solved
    dataset: str         # scorer/dataset key (used for retrieval filtering)
    steps: list          # list[PlanStep], in execution order, finish/error steps removed
    # Deprecated in-memory compatibility field for older tests/artifacts.
    # New plans leave it empty, serialization omits it, and routing never reads it.
    answer: str = ""
    aliases: list = field(default_factory=list)   # other questions that map to this plan (dedup)
    metadata: dict = field(default_factory=dict)

    @property
    def tool_sequence(self):
        return [s.tool_name for s in self.steps]

    def to_dict(self):
        return {
            "question": self.question,
            "dataset": self.dataset,
            "steps": [s.to_dict() for s in self.steps],
            "aliases": list(self.aliases),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            question=d.get("question", ""),
            dataset=d.get("dataset", ""),
            steps=[PlanStep.from_dict(s) for s in d.get("steps", [])],
            aliases=list(d.get("aliases", [])),
            metadata=dict(d.get("metadata", {})),
        )


def _truncate(value, n=_MAX_SUMMARY_CHARS):
    s = str(value)
    return s if len(s) <= n else s[:n] + "…"


def _summarize_result(result):
    """deterministic, no-LLM summary of a tool result's output.

    numeric -> the number; boolean -> the value; text -> head slice + length;
    structured -> shape + first row/item. Always returns a plain, JSON-safe
    string (str() collapses numpy/NaN), so a plan is serializable regardless of
    the raw output's type. Large outputs are truncated, never LLM-summarized —
    replay regenerates the real output anyway.
    """
    otype = result.get("output_type", "") or ""
    output = result.get("output")

    if otype == "numeric":
        return f"numeric: {_truncate(output)}"
    if otype == "boolean":
        return f"boolean: {output}"
    if otype == "text":
        if isinstance(output, list):
            n = len(output)
            if n == 0:
                return "text: 0 items"
            return f"text: {n} item(s); first: {_truncate(output[0])}"
        s = str(output)
        return f"text[len={len(s)}]: {_truncate(s)}"
    if otype == "structured":
        if isinstance(output, list):
            n = len(output)
            if n == 0:
                return "structured: 0 rows"
            return f"structured: {n} row(s); first: {_truncate(output[0])}"
        if isinstance(output, dict):
            return f"structured: keys={list(output.keys())}; {_truncate(output)}"
        return f"structured: {_truncate(output)}"
    # defensive fallback for any unexpected output_type
    return f"{otype or 'output'}: {_truncate(output)}"


def _format_action(tool_name, tool_args):
    """reconstruct a human-readable tool call string from recorded args."""
    parts = []
    for k, v in tool_args.items():
        v_str = str(v).replace('"', '\\"')
        parts.append(f'{k}="{v_str}"')
    return f"{tool_name}({', '.join(parts)})"


def compile_plan(agent_result, dataset):
    """COMPILE: deterministic AgentResult -> Plan, zero LLM.

    Keeps only successful, non-finish steps in execution order; each surviving
    step stores its Goal line, the concrete call, and a deterministic result
    summary. Returns None when no real tool step survives (nothing to reuse).

    This is a pure transform. It does not check agent_result.success or gold
    correctness — the engine gates on those before indexing.
    """
    steps = []
    for s in getattr(agent_result, "steps", None) or []:
        result = s.result or {}
        if result.get("status") != "success":
            continue
        # drop the terminal finish (guard on both the tool name and output_type)
        if s.tool_name == "finish" or result.get("output_type") == "final":
            continue
        args = dict(s.tool_args or {})
        steps.append(PlanStep(
            goal=(getattr(s, "goal", "") or "").strip(),
            tool_name=s.tool_name,
            tool_args=args,
            action_str=_format_action(s.tool_name, args),
            result_summary=_summarize_result(result),
        ))

    if not steps:
        return None

    return Plan(
        question=agent_result.query,
        dataset=dataset,
        steps=steps,
        metadata={
            "num_recorded_steps": len(agent_result.steps or []),
            "source_success": bool(getattr(agent_result, "success", False)),
            "source_error": str(getattr(agent_result, "error", "") or ""),
            "source_salvaged": any(
                str(getattr(call, "role", "")).lower() == "salvage"
                for call in (getattr(agent_result, "llm_calls", None) or [])
            ),
        },
    )
