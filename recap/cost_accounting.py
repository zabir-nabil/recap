"""
cost_accounting: compute exact USD cost and breakdowns from the per-call LLM ledger.

usage:
    from recap.cost_accounting import compute_cost_breakdown
    breakdown = compute_cost_breakdown(result.llm_calls, result.steps)
"""

from collections import defaultdict
from recap import config


def compute_cost_breakdown(llm_calls, steps):
    """compute exact cost and structured breakdown from an agent's LLM call ledger.

    args:
        llm_calls: list of LLMCallRecord (from AgentResult.llm_calls)
        steps: list of StepRecord (from AgentResult.steps)

    returns:
        dict with:
            usd_cost: float — exact total USD cost
            num_llm_calls: int — total LLM invocations (agent + tool-internal)
            num_tool_calls: int — tool dispatches by agent (excludes finish)
            tool_call_counts: dict — per-tool-name histogram
            token_breakdown: dict with "by_role" and "by_model" sub-dicts
            cost_breakdown: dict with "by_role" and "by_model" sub-dicts
    """
    # exact USD: price each call at its own model's rate
    usd_cost = 0.0
    for entry in llm_calls:
        usd_cost += config.get_token_cost(
            entry.model,
            entry.input_tokens,
            entry.output_tokens + entry.reasoning_tokens,
        )

    # num_tool_calls: tool dispatches excluding finish
    num_tool_calls = sum(
        1 for s in steps if s.tool_name and s.tool_name != "finish"
    )

    # tool_call_counts: per-tool histogram (excludes finish, so the counts sum
    # to num_tool_calls — finish is a control action, not a data tool call)
    tool_call_counts = defaultdict(int)
    for s in steps:
        if s.tool_name and s.tool_name != "finish":
            tool_call_counts[s.tool_name] += 1

    # breakdowns by role and by model
    by_role = defaultdict(lambda: {
        "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
        "calls": 0, "usd": 0.0,
    })
    by_model = defaultdict(lambda: {
        "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
        "calls": 0, "usd": 0.0,
    })

    for entry in llm_calls:
        entry_usd = config.get_token_cost(
            entry.model,
            entry.input_tokens,
            entry.output_tokens + entry.reasoning_tokens,
        )

        r = by_role[entry.role]
        r["input_tokens"] += entry.input_tokens
        r["output_tokens"] += entry.output_tokens
        r["reasoning_tokens"] += entry.reasoning_tokens
        r["calls"] += 1
        r["usd"] += entry_usd

        m = by_model[entry.model]
        m["input_tokens"] += entry.input_tokens
        m["output_tokens"] += entry.output_tokens
        m["reasoning_tokens"] += entry.reasoning_tokens
        m["calls"] += 1
        m["usd"] += entry_usd

    return {
        "usd_cost": round(usd_cost, 8),
        "num_llm_calls": len(llm_calls),
        "num_tool_calls": num_tool_calls,
        "tool_call_counts": dict(tool_call_counts),
        "token_breakdown": {
            "by_role": {k: dict(v) for k, v in by_role.items()},
            "by_model": {k: dict(v) for k, v in by_model.items()},
        },
        "cost_breakdown": {
            "by_role": {k: round(v["usd"], 8) for k, v in by_role.items()},
            "by_model": {k: round(v["usd"], 8) for k, v in by_model.items()},
        },
    }


def tier_breakdown(llm_calls):
    """split a ledger into big-tier vs small-tier tokens and USD.

    Classifies each LLMCallRecord by its model: config.LLM_BIG -> "big",
    config.LLM_SMALL -> "small", anything else -> "other" (so an unexpected
    model is visible, not silently mis-bucketed). Tokens/USD are compared across
    methods on a fair basis: raw token counts are NOT comparable across tiers
    (a big token costs ~6x a small one).
    """
    tiers = {t: {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                 "calls": 0, "usd": 0.0} for t in ("big", "small", "other")}
    for e in llm_calls:
        if e.model == config.LLM_BIG:
            t = "big"
        elif e.model == config.LLM_SMALL:
            t = "small"
        else:
            t = "other"
        billable_out = e.output_tokens + e.reasoning_tokens
        tiers[t]["input_tokens"] += e.input_tokens
        tiers[t]["output_tokens"] += e.output_tokens
        tiers[t]["reasoning_tokens"] += e.reasoning_tokens
        tiers[t]["calls"] += 1
        tiers[t]["usd"] += config.get_token_cost(e.model, e.input_tokens, billable_out)
    for t in tiers.values():
        t["total_tokens"] = t["input_tokens"] + t["output_tokens"] + t["reasoning_tokens"]
        t["usd"] = round(t["usd"], 8)
    return tiers


def analysis_fields(llm_calls, steps):
    """consistent per-question analysis block shared by every method's entry dict.

    Reports per-tier (big/small) tokens and USD, tool counts, and call counts.
    Correctness, wall-clock, and the raw answer are
    added by the caller (they need gold / timing the ledger does not carry).
    """
    breakdown = compute_cost_breakdown(llm_calls, steps)
    tiers = tier_breakdown(llm_calls)
    return {
        "usd_cost": round(breakdown["usd_cost"], 6),
        "big_llm_cost_usd": tiers["big"]["usd"],
        "small_llm_cost_usd": tiers["small"]["usd"],
        "other_llm_cost_usd": tiers["other"]["usd"],
        "big_llm_tokens": tiers["big"]["total_tokens"],
        "small_llm_tokens": tiers["small"]["total_tokens"],
        "other_llm_tokens": tiers["other"]["total_tokens"],
        "big_llm_calls": tiers["big"]["calls"],
        "small_llm_calls": tiers["small"]["calls"],
        "num_llm_calls": breakdown["num_llm_calls"],
        "num_tool_calls": breakdown["num_tool_calls"],
        "tool_call_counts": breakdown["tool_call_counts"],
        "tier_breakdown": tiers,
        "token_breakdown": breakdown["token_breakdown"],
        "cost_breakdown": breakdown["cost_breakdown"],
    }


class _LedgerView:
    """lightweight adapter so tier_breakdown can read a serialized llm_call dict
    (from an already-built entry) with the same attribute access as LLMCallRecord."""
    __slots__ = ("model", "input_tokens", "output_tokens", "reasoning_tokens")

    def __init__(self, d):
        self.model = d.get("model", "")
        self.input_tokens = int(d.get("input_tokens", 0) or 0)
        self.output_tokens = int(d.get("output_tokens", 0) or 0)
        self.reasoning_tokens = int(d.get("reasoning_tokens", 0) or 0)


def add_tier_analysis(entry):
    """augment an already-built method entry dict with the big/small tier split.

    Reads entry["llm_calls"] (serialized dicts) and adds big_llm_*/small_llm_*
    token and USD fields. Idempotent. Returns entry.
    """
    calls = [_LedgerView(c) for c in entry.get("llm_calls", [])]
    tiers = tier_breakdown(calls)
    entry["big_llm_tokens"] = tiers["big"]["total_tokens"]
    entry["small_llm_tokens"] = tiers["small"]["total_tokens"]
    entry["other_llm_tokens"] = tiers["other"]["total_tokens"]
    entry["big_llm_cost_usd"] = tiers["big"]["usd"]
    entry["small_llm_cost_usd"] = tiers["small"]["usd"]
    entry["other_llm_cost_usd"] = tiers["other"]["usd"]
    entry["big_llm_calls"] = tiers["big"]["calls"]
    entry["small_llm_calls"] = tiers["small"]["calls"]
    entry["tier_breakdown"] = tiers
    return entry
