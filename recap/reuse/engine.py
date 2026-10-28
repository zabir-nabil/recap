"""
reuse/engine.py — RECAP orchestration: deterministic REPLAY (part 1) and the
RecapEngine / run_with_recap retrieve->judge->route->verify driver (part 2).

REPLAY re-executes a plan's stored tool calls verbatim. By default it is used
when the tools are available and applying the judge's substitutions changes no
concrete argument. A separate optional flag permits constrained substitutions. Context
identifiers (db_name / question_id / doc_name) are always refreshed from the NEW
question's context, because they are supplied externally, not derived by the
model — replaying a stale question_id would silently fetch the wrong evidence.

Replay runs tools through a BaseAgent subclass so tool dispatch, error handling,
tool.reset(), and tool-internal LLM-token absorption behave exactly as in a
normal agent run — keeping the cost ledger faithful.

The RecapEngine wires A_big (indexing + fallback) and A_small (judge, guided
agent, format+verify) with the plan index. Retrieval returns the top-k plans and
the judge scores them in similarity order (first "full" wins; else the highest
"full"-less "partial"; else all-none -> big agent), so a question can incur 1..k
judge calls. Every LLM call on a question — each judge call, replay tool-internal,
guided-agent, format+verify, big fallback — lands in a single per-question ledger
passed once to compute_cost_breakdown, so cost is a faithful straight sum. Cache
admission is LABEL-FREE: no gold answer and no scorer is ever consulted inside a
RECAP run. Gold is retained for scoring by the experiment runner.
"""

import re
import time
import logging

from recap import config
from recap.agents import AgentConfig, LLMCallRecord
from recap.agents.base import BaseAgent
from recap.agents.react import ReActAgent
from recap.agents.plan_and_act import PlanAndActAgent
from recap.cost_accounting import compute_cost_breakdown, analysis_fields
from recap.llm import call_llm

from recap.reuse.plan import compile_plan
from recap.reuse.index import PlanIndex
from recap.reuse.prompts import (
    build_judge_prompt, parse_judge_output,
    build_format_verify_prompt, parse_format_verify_output,
    build_direct_format_prompt, parse_direct_format_output,
    build_demonstration, _extract_json_object,
)

logger = logging.getLogger(__name__)


# context keys that identify WHICH data a tool reads; on replay these come from
# the new question's context.
CONTEXT_ID_KEYS = ("db_name", "question_id", "doc_name", "graph_name")

# compute/derived-input tools: their args are typically computed FROM earlier
# tool outputs, so certified substituted replay (which swaps only input literals)
# would leave stale downstream inputs -> a plan containing any of these is NOT
# dataflow-safe for replay_subst and routes to the guided agent instead.
_COMPUTE_TOOLS = frozenset({
    "calculator", "python_exec", "is_prime", "is_fibonacci", "is_perfect_square",
    "is_even", "is_divisible_by", "count_syllables", "geo_distance", "geocode",
    "llm_analyze", "llm_reason",
})


# ---------------------------------------------------------------------------
# ledger helper — convert a standalone call_llm() response into a ledger entry
# ---------------------------------------------------------------------------

def record_from_response(response, role, model, provider):
    """build an LLMCallRecord from an LLMResponse (for judge / format+verify).

    always sets all four token fields (reasoning defaults to 0 if absent) so
    cost_accounting never hits a missing-attribute error.
    """
    return LLMCallRecord(
        role=role,
        model=model,
        provider=provider,
        input_tokens=getattr(response, "input_tokens", 0) or 0,
        output_tokens=getattr(response, "output_tokens", 0) or 0,
        reasoning_tokens=getattr(response, "reasoning_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# substitution application
# ---------------------------------------------------------------------------

def apply_substitutions(tool_args, substitutions, new_context=None):
    """return a new args dict with judge substitutions and context-ids applied.

    - substitutions: {past_value -> new_value|None} from the judge. Each string
      arg value has any past_value substring replaced by its new_value. A None
      (null) new_value means the new question does not supply the value; it is
      left unchanged here (the caller must not deterministically replay when any
      substitution is null — that routes to the guided agent).
    - new_context: the NEW question's context dict; any arg whose key is a
      context id (db_name/question_id/doc_name/graph_name) is overwritten with
      the new context's value so replay reads the correct data.
    """
    new_context = new_context or {}
    out = {}
    for key, val in tool_args.items():
        # context ids are refreshed from the new question, not substituted
        if key in CONTEXT_ID_KEYS and key in new_context:
            out[key] = str(new_context[key])
            continue
        if isinstance(val, str):
            new_val = val
            for past_value, new_value in substitutions.items():
                if new_value is None or past_value == "":
                    continue
                if past_value in new_val:
                    new_val = new_val.replace(past_value, new_value)
            out[key] = new_val
        else:
            out[key] = val
    return out


def substitutions_change_args(plan, substitutions, new_context=None):
    """True if applying substitutions/context would change ANY concrete arg.

    This is what decides deterministic-replay eligibility, rather than "is the
    substitution dict non-empty": the judge often returns identical mappings for
    a paraphrase (e.g. {"2023":"2023"}), which must NOT force the guided path.
    A None (null) substitution always counts as a change (value unavailable).
    """
    if any(v is None for v in substitutions.values()):
        return True
    for step in plan.steps:
        applied = apply_substitutions(step.tool_args, substitutions, new_context)
        if applied != step.tool_args:
            return True
    return False


def is_certified_substitution(plan, substitutions, new_context=None, tools_by_name=None):
    """Eligibility for the optional conservative whole-value substitution path."""
    tools_by_name = tools_by_name or {}
    if any(v is None for v in substitutions.values()):
        return False
    for step in plan.steps:
        t = tools_by_name.get(step.tool_name)
        if t is None or not getattr(t, "is_deterministic", False):
            return False
        if step.tool_name in _COMPUTE_TOOLS:      # derived-input step -> not dataflow-safe
            return False
    changed = False
    for step in plan.steps:
        for key, val in step.tool_args.items():
            if key in CONTEXT_ID_KEYS:
                continue
            if not isinstance(val, str):
                continue
            sval = val.strip()
            hit = [pk for pk in substitutions if pk and pk in val]
            if not hit:
                continue                       # arg untouched by any substitution
            nv = substitutions.get(sval)
            if hit == [sval] and nv is not None and str(nv) != sval:
                changed = True                 # clean whole-value swap
            else:
                return False                   # embedded / multi-key -> unsafe
    return changed


# ---------------------------------------------------------------------------
# replay-safety predicate
# ---------------------------------------------------------------------------

def _ended_working(agent_result):
    """True if the trace ended in a working state (RECAP's admission gate).

    "Working" = the last recorded step is a successful real tool call, or a
    successful terminal finish whose immediately-preceding real step succeeded.
    Mid-trace errors are allowed: an agent that retries and recovers may produce a
    fine strategy. A trace whose LAST action errored generally does not.
    """
    steps = list(getattr(agent_result, "steps", None) or [])
    if not steps:
        return False

    # walk back past the terminal finish (a control action, not a data step)
    for s in reversed(steps):
        result = s.result or {}
        if s.tool_name == "finish" or result.get("output_type") == "final":
            if result.get("status") != "success":
                return False        # the finish itself failed
            continue
        return result.get("status") == "success"   # last real tool call
    return False                    # finish-only trace: no real tool call


def is_replay_safe(plan, tools_by_name):
    """True if every plan step calls a tool that is PRESENT in the current set.

    Determinism is intentionally NOT required. Replay RE-EXECUTES the stored
    calls (it does not return cached outputs), and it only fires on the
    no-argument-actually-changes path (a pure paraphrase, `substitutions_change_args`
    is False), so a non-deterministic tool simply re-runs the SAME call and the
    downstream FORMAT+VERIFY gate + the failed-replay->guided fallback guard the
    result. The one hard requirement is that every referenced
    tool still exists in this tool set; a missing tool cannot be replayed.
    """
    return all(tools_by_name.get(step.tool_name) is not None for step in plan.steps)


# ---------------------------------------------------------------------------
# deterministic REPLAY
# ---------------------------------------------------------------------------

class ReplayExecutor(BaseAgent):
    """re-executes a plan's stored tool calls, reusing BaseAgent tool dispatch.

    subclassing BaseAgent gives tool lookup, exception-safe execute(), and
    _absorb_tool_tokens (so any tool-internal LLM cost lands in the ledger)
    identically to a normal agent run.
    """

    def run(self, query, context=None):  # not used; replay drives execution directly
        raise NotImplementedError("ReplayExecutor.replay() drives execution, not run()")

    def replay(self, plan, substitutions=None, new_context=None):
        """re-run the plan's steps in order. returns (ok, executed, llm_calls).

        ok: True only if every step succeeded with a non-empty output.
        executed: list of (goal, action_str, output) per step (for format+verify).
        llm_calls: ledger entries accrued (tool-internal LLM tokens), popped fresh.

        On the first error/empty output, stops and returns ok=False with the
        steps executed so far (caller routes a failed replay to the guided agent).
        """
        substitutions = substitutions or {}
        self._reset()   # resets counters AND calls tool.reset() on every tool

        executed = []
        ok = True
        for step in plan.steps:
            args = apply_substitutions(step.tool_args, substitutions, new_context)
            step_id = self._next_step_id()
            result = self._execute_tool(step.tool_name, args)   # absorbs tool-internal tokens
            self._record_step(step_id, step.tool_name, args, result, goal=step.goal)
            output = result.get("output", "")
            action_str = _format_action(step.tool_name, args)
            executed.append((step.goal, action_str, _truncate_output(output)))
            if result.get("status") == "error" or _is_empty_output(output):
                ok = False
                break
        return ok, executed, list(self._llm_calls), list(self._steps)


def _is_empty_output(output):
    """replay-failure emptiness check (aligned with how tools denote 'no result').

    treats None, empty string, and the common 'no rows'/'[]'/'none' textual
    markers as empty. a legitimate empty result set expressed as [] is empty for
    replay purposes — the caller falls back to the guided agent, which is safe.
    """
    if output is None:
        return True
    if isinstance(output, (list, dict, str)) and len(output) == 0:
        return True
    if isinstance(output, str):
        return output.strip().lower() in ("[]", "none", "no rows", "no results", "empty")
    return False


def _truncate_output(output, n=500):
    """render a tool output for the verifier, capped at ~n chars, KEEPING BOTH
    THE HEAD AND THE TAIL.

    Reasoning/compute tools (llm_reason, python_exec) put the final answer at the
    END of a long output; a head-only cut hides it, so the verifier sees an
    unfinished analysis and wrongly judges the step "incomplete/unsound". Keeping 
    the first and last halves preserves both the setup and the final result.
    """
    s = str(output)
    if len(s) <= n:
        return s
    half = max(1, n // 2)
    return f"{s[:half]}… [truncated: {_shape_of(output)}, {len(s)} chars total] …{s[-half:]}"


def _shape_of(output):
    """one-phrase shape description used in the truncation marker."""
    if isinstance(output, list):
        return f"{len(output)} item(s)"
    if isinstance(output, dict):
        return f"dict with {len(output)} key(s)"
    return f"{type(output).__name__}"


_ANCHOR_WORD = re.compile(r"[A-Za-z]{4,}")


def _evidence_view(output, query, candidate, cap=2000):
    """ANSWER/QUERY-anchored view of ONE tool output for the verifier.

    Blind head/tail truncation drops mid-document values (a doc's line item, a
    discography row), which makes the verifier reject a correct reuse for lack of
    evidence. This keeps the evidence that actually supports the answer within a
    fixed char budget:
      - structured rows (list): keep the first rows + a total-row-count marker
        (ROW-based, not char-based) so counts / row-set answers stay intact.
      - long text (documents / web pages): keep the single most relevant
        CONTIGUOUS block (a de-facto section) — the window that maximizes matches
        of the candidate value, query keywords, and numbers — plus the first line
        for context, with "…" marking every elision. 
      - short output: returned as-is.
    """
    if isinstance(output, list):
        n = len(output)
        if n == 0:
            return "[] (0 rows)"
        s = str(output[:25])
        marker = "" if n <= 25 else f" … [{n} rows total]"
        return (s[:cap] + marker) if len(s) > cap else (s + marker)
    if isinstance(output, dict):
        return str(output)[:cap]

    s = str(output)
    if len(s) <= cap:
        return s

    lines = s.split("\n")
    qtok = {w.lower() for w in _ANCHOR_WORD.findall(query or "")}
    ctok = [t for t in re.split(r"[\s,]+", str(candidate or ""))
            if len(t) >= 1 and t.lower() not in ("none", "null", "no_answer", "")]

    def _score(ln):
        low = ln.lower()
        sc = 0.0
        if any(t and t in ln for t in ctok):
            sc += 3.0
        if any(t in low for t in qtok):
            sc += 2.0
        if re.search(r"\d", ln):
            sc += 0.5
        return sc

    scores = [_score(ln) for ln in lines]
    lens = [len(ln) + 1 for ln in lines]
    # sliding window over lines: best contiguous block within `cap` chars
    best_lo = best_hi = 0
    best_sc = -1.0
    lo = 0
    cur_len = 0.0
    cur_sc = 0.0
    for hi in range(len(lines)):
        cur_len += lens[hi]
        cur_sc += scores[hi]
        while cur_len > cap and lo < hi:
            cur_len -= lens[lo]
            cur_sc -= scores[lo]
            lo += 1
        if cur_sc > best_sc:
            best_sc, best_lo, best_hi = cur_sc, lo, hi

    parts = []
    if best_lo > 0:
        parts.append(lines[0])          # title/context line
        if best_lo > 1:
            parts.append("…")
    parts.extend(lines[best_lo:best_hi + 1])
    if best_hi < len(lines) - 1:
        parts.append("…")
    return "\n".join(parts)


def _llm_output_marker(tool):
    """ ' [LLM reasoning output]' for helper-LLM tools (llm_analyze/llm_reason),
    else ''. Lets the verifier weigh primary retrieved/computed outputs over a
    helper model's own reasoning (which it otherwise over-trusts)."""
    return " [LLM reasoning output]" if getattr(tool, "llm_needed_output", False) else ""


def _format_action(tool_name, tool_args, max_val=120):
    """render a tool call for the verifier's executed-step view.

    Argument VALUES are truncated (max_val chars): a verbose arg like
    llm_reason(prompt="<the whole question restated>") or python_exec(code=...)
    otherwise restates the question inside the step and bloats the verify prompt,
    which pushed the small model into "solve it" prose. The verifier judges the
    OUTPUT of each step, not the agent's input args, so trimming args is lossless
    for its purpose. (StepRecords keep full args for the trace log.)
    """
    parts = []
    for k, v in tool_args.items():
        s = str(v)
        if len(s) > max_val:
            s = s[:max_val] + "…"
        parts.append(f'{k}="{s}"')
    return f"{tool_name}({', '.join(parts)})"


def make_small_replay_executor(tools):
    """build a ReplayExecutor bound to the given tools (config is irrelevant to
    replay — no agent LLM call is made; only tools execute)."""
    return ReplayExecutor(tools=tools, config=AgentConfig())


# ===========================================================================
# part 2 — RecapEngine + run_with_recap (retrieve -> judge -> route -> verify)
# ===========================================================================

# recap_path values kept in metadata for analysis (the binary hit/miss label,
PATH_NONE_PREFILTER = "none_prefilter"   # no candidate >= tau -> big agent
PATH_NONE_JUDGE = "none_judge"           # every top-k candidate judged none -> big agent
PATH_REPLAY = "replay"                   # deterministic replay (no arg change) -> verify accepted
PATH_REPLAY_SUBST = "replay_subst"       # certified WHOLE-VALUE substituted replay -> verify accepted
PATH_GUIDED_FULL = "guided_full"         # full + change / not replay-safe -> guided
PATH_GUIDED_PARTIAL = "guided_partial"   # partial coverage -> guided (continue-after)
PATH_REPLAY_THEN_GUIDED = "replay_then_guided"   # replay failed -> guided


class RecapEngine:
    """RECAP plan-reuse engine: indexes A_big plans and reuses them via A_small.

    A_big  (planner/executor models) — solves novel questions and cold/verify
            fallbacks; run with self_annotate=True so traces compile to plans.
    A_small (small model) — carries the reuse path: JUDGE, guided agent, and
            FORMAT+VERIFY.

    All three agents share the SAME tools object (so per-question tool state,
    e.g. FinanceBench evidence_store, is consistent across paths).
    """

    def __init__(self, tools, dataset, preamble="", agent_type="react",
                 tau=None, top_k=3, max_steps=15, index=None, dedup=False,
                 config_mode="verified"):
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.dataset = dataset
        self.preamble = preamble
        self.agent_type = agent_type
        # operating point:
        #   "verified" (default, Config B): reuse -> format+verify gate -> A_big
        #     fallback on unsound/empty. Verification adds a judgment call and can trigger fallback.
        #   "direct"   (Config A): return a guided answer directly, or derive a
        #     replay answer with one format-only call — no soundness gate and no
        #     fallback. Lower cost, no safety net.
        if config_mode not in ("verified", "direct"):
            raise ValueError(f"config_mode must be 'verified' or 'direct', got {config_mode!r}")
        self.config_mode = config_mode
        # tau=None -> resolve RECAP's retrieval pre-filter per dataset from config
        # (default 0.5; per-dataset overrides via RECAP_TAU_* env). 
        self.tau = tau if tau is not None else config.recap_tau(dataset)
        self.top_k = top_k
        self.max_steps = max_steps
        self.index = index if index is not None else PlanIndex(dedup=dedup)

        self._assert_small_is_cheaper()

    def _assert_small_is_cheaper(self):
        """the reuse path must run on the small model, not the big one. fail if they are equal."""
        if config.LLM_SMALL == config.LLM_BIG:
            raise ValueError(
                "RECAP reuse path would run on the big model: "
                f"LLM_SMALL == LLM_BIG == {config.LLM_BIG!r}. "
                "Set a cheaper LLM_SMALL so reuse actually saves cost."
            )

    # ------------------------------------------------------------------
    # agent construction
    # ------------------------------------------------------------------

    def _build_big_agent(self, self_annotate=True):
        """A_big: the expensive solver, on the LLM_BIG tier (planner+executor).

        self_annotate=True on a fresh-question solve so its trace compiles into
        an indexable plan; self_annotate=False on a fallback after a reuse
        attempt already failed (no plan is indexed then — skip the annotation
        cost and keep the fallback clean).
        """
        if self.agent_type == "react":
            cfg = AgentConfig(
                model=config.LLM_BIG, provider=config.LLM_BIG_PROVIDER,
                max_steps=self.max_steps, max_tokens=config.MAX_TOKENS,
                system_preamble=self.preamble, self_annotate=self_annotate,
            )
        else:
            cfg = AgentConfig(
                planner_model=config.LLM_BIG, planner_provider=config.LLM_BIG_PROVIDER,
                executor_model=config.LLM_BIG, executor_provider=config.LLM_BIG_PROVIDER,
                max_steps=self.max_steps, max_steps_per_plan_step=5, max_tokens=config.MAX_TOKENS,
                system_preamble=self.preamble, self_annotate=self_annotate,
            )
        return self._make_agent(cfg)

    def _build_small_agent(self, demonstration):
        """A_small guided agent: small model, with the plan injected as a demo.

        Built fresh per reuse so a prior question's demonstration can never leak.
        planner AND executor use the small model (guided reuse is fully A_small).
        """
        if self.agent_type == "react":
            cfg = AgentConfig(
                model=config.LLM_SMALL, provider=config.LLM_SMALL_PROVIDER,
                max_steps=self.max_steps, max_tokens=config.MAX_TOKENS,
                system_preamble=self.preamble, demonstration=demonstration,
            )
        else:
            cfg = AgentConfig(
                planner_model=config.LLM_SMALL, planner_provider=config.LLM_SMALL_PROVIDER,
                executor_model=config.LLM_SMALL, executor_provider=config.LLM_SMALL_PROVIDER,
                max_steps=self.max_steps, max_steps_per_plan_step=5, max_tokens=config.MAX_TOKENS,
                system_preamble=self.preamble, demonstration=demonstration,
            )
        return self._make_agent(cfg)

    def _make_agent(self, cfg):
        if self.agent_type == "react":
            return ReActAgent(tools=self.tools, config=cfg)
        return PlanAndActAgent(tools=self.tools, config=cfg)

    # ------------------------------------------------------------------
    # A_small standalone calls (judge, format+verify) — ledgered
    # ------------------------------------------------------------------

    def _small_call(self, prompt, role, ledger):
        """one A_small call for judge / format+verify; appends a ledger entry."""
        response = call_llm(
            [{"role": "user", "content": prompt}],
            model=config.LLM_SMALL, provider=config.LLM_SMALL_PROVIDER,
            temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
            # JUDGE / FORMAT+VERIFY are short JSON verdicts.
            thinking_budget=config.RECAP_DECISION_THINKING_BUDGET,
        )
        ledger.append(record_from_response(
            response, role, config.LLM_SMALL, config.LLM_SMALL_PROVIDER))
        return response.text

    def judge(self, new_question, plan, ledger, similarity=None, rank=None):
        prompt = build_judge_prompt(new_question, plan, list(self.tools_by_name.keys()))
        raw = self._small_call(prompt, "judge", ledger)
        verdict, substitutions = parse_judge_output(raw)
        parsed = _extract_json_object(raw)
        record = {
            "rank": rank,
            "similarity": similarity,
            "plan_question": plan.question,
            "verdict": verdict,
            "substitutions": substitutions,
            "parse_ok": parsed is not None,
            "raw": raw or "",
        }
        self._judge_records.append(record)
        if parsed is None:
            self._runtime_errors.append(
                f"judge parse failure for candidate rank {rank}"
            )
        return verdict, substitutions

    def format_verify(self, new_question, executed, ledger):
        """VERIFY the reuse and produce the final answer (one A_small call).

        `executed` is (goal, action_str, FULL_output) per step. We build the
        verifier's evidence by ANSWER/QUERY-anchored truncation of each output
        (so the value the answer needs survives — blind truncation dropped it).
        The guided agent's own answer (self._cand_answer, set by _run_guided) is
        passed as a CANDIDATE for the verifier to CHECK rather than re-derive; on
        pure replay there is no candidate (None) so the verifier derives from the
        visible tool outputs, which ARE the result.
        """
        candidate = getattr(self, "_cand_answer", None)
        view = [
            (goal, action, _evidence_view(output, new_question, candidate))
            for (goal, action, output) in executed
        ]
        prompt = build_format_verify_prompt(
            new_question, view, dataset=self.dataset, candidate_answer=candidate)
        raw = self._small_call(prompt, "format_verify", ledger)
        answer, sound = parse_format_verify_output(raw)
        # capture for post-hoc debugging: parse_ok distinguishes a JSON-format
        # slip (parse failure) from a genuine "unsound" verdict, and we keep the
        # candidate + the answer the verifier produced (even when rejected).
        self._last_verify = {
            "parse_ok": _extract_json_object(raw) is not None,
            "candidate": candidate,
            "answer": answer,
            "raw": raw or "",
        }
        return answer, sound

    def format_direct(self, new_question, executed, ledger):
        """Config A formatter only: no soundness judgment and no fallback."""
        candidate = getattr(self, "_cand_answer", None)
        view = [
            (goal, action, _evidence_view(output, new_question, candidate))
            for (goal, action, output) in executed
        ]
        prompt = build_direct_format_prompt(
            new_question, view, dataset=self.dataset,
        )
        raw = self._small_call(prompt, "format_direct", ledger)
        answer = parse_direct_format_output(raw)
        self._last_direct_format = {
            "parse_ok": _extract_json_object(raw) is not None,
            "answer": answer,
            "raw": raw or "",
        }
        if self._last_direct_format["parse_ok"] is False:
            self._runtime_errors.append("direct formatter parse failure")
        return answer

    # ------------------------------------------------------------------
    # reuse execution paths
    # ------------------------------------------------------------------

    def _replay(self, plan, subs, ctx):
        """deterministic replay via a ReplayExecutor bound to this engine's tools.

        returns (ok, executed, tool_calls, steps): executed is the per-step
        (goal, action, FULL_output) list for format+verify (anchor-truncated
        later in format_verify); tool_calls are ledger entries (tool-internal LLM
        tokens); steps are StepRecords for the cost histogram. Replay has no agent
        answer, so _cand_answer=None -> format+verify DERIVES from the outputs.
        """
        self._cand_answer = None
        executor = make_small_replay_executor(self.tools)
        ok, _exec, tool_calls, steps = executor.replay(plan, substitutions=subs, new_context=ctx)
        # rebuild executed from the recorded steps with FULL outputs (the verifier
        # anchor-truncates), tagging any helper-LLM tool's output.
        executed = []
        for s in steps:
            if s.tool_name == "finish" or (s.result or {}).get("output_type") == "final":
                continue
            action = _format_action(s.tool_name, s.tool_args or {}) + _llm_output_marker(self.tools_by_name.get(s.tool_name))
            executed.append((getattr(s, "goal", ""), action, (s.result or {}).get("output", "")))
        return ok, executed, tool_calls, steps

    def _run_guided(self, query, ctx, plan, partial, ledger, subs=None):
        """run the guided A_small agent with the plan injected as a demonstration.

        returns (executed, steps): executed is the (goal, action, FULL_output)
        list from the agent's non-finish steps (anchor-truncated later in
        format_verify); steps are the agent's StepRecords. The agent's own final
        answer is stashed as self._cand_answer so format+verify CHECKS it rather
        than re-deriving. The agent's llm_calls are merged into `ledger` here.
        """
        demonstration = build_demonstration(
            plan, partial=partial,
            substitutions=(subs if config.RECAP_GUIDED_BINDINGS else None))
        agent = self._build_small_agent(demonstration)
        result = agent.run(query, context=ctx)
        ledger.extend(result.llm_calls)
        if result.error:
            self._runtime_errors.append(f"guided agent: {result.error}")
        self._cand_answer = result.answer   # the candidate the verifier will check
        executed = []
        for s in result.steps:
            if s.tool_name == "finish" or (s.result or {}).get("output_type") == "final":
                continue
            action = _format_action(s.tool_name, s.tool_args or {}) + _llm_output_marker(self.tools_by_name.get(s.tool_name))
            executed.append((getattr(s, "goal", ""), action, (s.result or {}).get("output", "")))
        return executed, list(result.steps)

    def index_from_big_run(self, big_result):
        """compile + admission-gate + index a plan from an A_big run."""
        if not _ended_working(big_result):
            return False
        if not str(getattr(big_result, "answer", "") or "").strip():
            return False
        if not config.RECAP_ADMIT_SALVAGE:
            if not bool(getattr(big_result, "success", False)):
                return False
            if str(getattr(big_result, "error", "") or "").strip():
                return False
            if any(
                str(getattr(call, "role", "")).lower() == "salvage"
                for call in (getattr(big_result, "llm_calls", None) or [])
            ):
                return False
        plan = compile_plan(big_result, self.dataset)
        if plan is None:
            return False
        plan.metadata["replay_safe"] = is_replay_safe(plan, self.tools_by_name)
        stored = self.index.add(plan)
        if stored:
            self._admitted_plan = plan.to_dict()
        return stored


def _steps_to_dicts(steps):
    """serialize StepRecords for JSONL output (preserves step order)."""
    return [
        {
            "step_id": s.step_id,
            "tool_name": s.tool_name,
            "tool_args": s.tool_args,
            "result_status": (s.result or {}).get("status", ""),
            # truncated output/error text for post-hoc diagnosis (network vs
            # model/sandbox failure); errors are short so 500 chars suffices.
            "result_output": str((s.result or {}).get("output", ""))[:500],
            "goal": getattr(s, "goal", ""),
            "thought": s.thought,
        }
        for s in steps
    ]


def _result_entry(engine, question, query, gold, *, agent_answer, success,
                  cache_decision, recap_path, verdict, sound, similarity,
                  steps, ledger, t0, error=""):
    """assemble the per-question result dict (per-question) from ONE merged ledger.

    the entire per-question cost is computed once, here, over `ledger` — the
    single concatenation of every LLM call made on this question (judge call(s),
    replay tool-internal, guided-agent, format+verify, big fallback). token totals
    are summed over that list, so sum(ledger) == reported totals by construction.
    The shared analysis_fields() block adds per-tier tokens, USD, and tool counts.
    """
    entry = {
        "question_id": question["id"],
        "question": query,
        "expected_answer": gold,
        "agent_answer": agent_answer,
        "success": success,
        "cache_decision": cache_decision,          # "hit" | "miss" (binary, for the runners)
        "steps": _steps_to_dicts(steps),
        "num_steps": len(steps),
        "input_tokens": sum(c.input_tokens for c in ledger),
        "output_tokens": sum(c.output_tokens for c in ledger),
        "reasoning_tokens": sum(c.reasoning_tokens for c in ledger),
        "total_tokens": sum(c.input_tokens + c.output_tokens + c.reasoning_tokens for c in ledger),
        "elapsed_seconds": round(time.time() - t0, 2),
        "error": error,
        "llm_calls": [c.to_dict() for c in ledger],
        "metadata": {
            "source": "recap",
            "recap_path": recap_path,      # replay / guided_full / guided_partial / none_* / *_verify_reject
            "verdict": verdict,            # judge verdict, or None if the judge did not run
            "sound": sound,                # format+verify soundness, or None if it did not run
            "similarity": similarity,      # SELECTED candidate's retrieval score
                                           # (the top-1 score only on the none_* paths), or None
            # debug (per-question, reset each run): lets us root-cause routing
            # decisions from the jsonl instead of re-instrumenting a run.
            "substitutions": getattr(engine, "_last_subs", None),   # judge's value bindings for the selected plan
            "verify": getattr(engine, "_last_verify", None),        # {parse_ok, answer, raw} of the format+verify call
            "direct_format": getattr(engine, "_last_direct_format", None),
            "retrieval_candidates": getattr(engine, "_retrieval_candidates", []),
            "judge_outputs": getattr(engine, "_judge_records", []),
            "matched_plan_query": getattr(engine, "_matched_plan_query", None),
            "admitted_plan": getattr(engine, "_admitted_plan", None),
            "index_size_before": getattr(engine, "_index_size_before", None),
            "index_size_after": engine.index.size,
            "runtime_errors": getattr(engine, "_runtime_errors", []),
        },
    }
    entry.update(analysis_fields(ledger, steps))   # usd_cost, big/small tokens+cost, tool counts, breakdowns
    return entry


def run_with_recap(question, engine, index_on_miss=True):
    """run one question through RECAP (retrieve -> judge -> route -> verify).

    question: dict with at least {id, question, expected, context}.
    engine: a configured RecapEngine.

    Routing:
      no candidate >= tau            -> big agent            (miss, none_prefilter)
      all top-k judged none          -> big agent            (miss, none_judge)
      full + no arg change + safe    -> deterministic replay -> format+verify
      full + change / not safe       -> guided small agent   -> format+verify
      partial                        -> guided small agent   -> format+verify
      verify sound=false / no answer -> big-agent fallback   (miss, *_verify_reject)

    Candidate selection: the retrieved top-k (>= tau) are JUDGED in similarity
    order. The first "full" is used immediately (early exit); if none is "full",
    the highest-similarity "partial" is used; if all are "none", the big agent
    runs. This recovers reuses where the true match is retrieved but not top-1
    (a different-group original out-scored it) at the cost of up to k judge
    calls (all A_small; the common case is a single call).

    cache_decision is "hit" iff a reuse path produced a verify-accepted answer;
    every fallback (prefilter, judge=none, verify reject) is "miss".

    A_big runs in two modes:
      - fresh solve (cold: no candidate / judge=none) — self_annotate=True and
        the resulting plan is admission-gated (label-free) and indexed (this
        populates the library).
      - fallback (a reuse attempt already failed verify) — self_annotate=False
        and NOT indexed: minimize cost after a failure and avoid re-indexing an
        already-clustered question with a goal-less trace.
    Reuse answers are NEVER indexed (only A_big traces populate the library).
    Returns an per-question result dict.
    """
    t0 = time.time()
    query = question["question"]
    gold = question.get("expected", "")
    ctx = question.get("context") or {}
    ledger = []   # the single per-question ledger; every LLM call appends here
    # per-question debug/reuse state (reset here so a prior question never leaks);
    # populated by judge/_run_guided/_replay/format_verify and logged into the
    # entry metadata. _cand_answer carries the guided agent's answer into the
    # verifier; it MUST be reset per question so a stale candidate can't bleed in.
    engine._last_subs = None
    engine._last_verify = None
    engine._last_direct_format = None
    engine._cand_answer = None
    engine._retrieval_candidates = []
    engine._judge_records = []
    engine._matched_plan_query = None
    engine._admitted_plan = None
    engine._runtime_errors = []
    engine._index_size_before = engine.index.size

    def runtime_error():
        return "; ".join(engine._runtime_errors)

    def run_big_and_index(recap_path, prior_steps=None, prior_error="", is_fallback=False):
        """run A_big, merge its ledger/steps. On a fresh solve (not a fallback)
        annotate and admission-gate-index the plan; on a fallback do neither."""
        big = engine._build_big_agent(self_annotate=not is_fallback)
        try:
            big_result = big.run(query, context=ctx)
        except Exception as e:
            # the agent may already have made (and billed) LLM/tool calls before
            # raising; pull its partial ledger/steps off the instance so incurred
            # cost is not silently dropped (keeps sum(ledger) faithful to spend).
            ledger.extend(getattr(big, "_llm_calls", []) or [])
            steps = list(prior_steps or []) + list(getattr(big, "_steps", []) or [])
            return _result_entry(
                engine, question, query, gold,
                agent_answer="", success=False, cache_decision="miss",
                recap_path=recap_path, verdict=_verdict_holder.get("v"), sound=_verdict_holder.get("s"),
                similarity=_verdict_holder.get("sim"), steps=steps, ledger=ledger, t0=t0,
                error="; ".join(filter(None, [
                    prior_error, runtime_error(), f"{type(e).__name__}: {e}",
                ])),
            )
        ledger.extend(big_result.llm_calls)
        # cold solves always populate the index. Verify-reject FALLBACKS are
        # indexed ONLY when RECAP_INDEX_FALLBACKS is enabled (default OFF): admission
        # is label-free, so a wrong-but-structurally-successful fallback would be
        # stored and could POISON future reuse (a wrong-answer probe grew the index
        # 1->2 with the wrong plan).
        if index_on_miss and (not is_fallback or getattr(config, "RECAP_INDEX_FALLBACKS", False)):
            engine.index_from_big_run(big_result)
        steps = list(prior_steps or []) + list(big_result.steps)
        return _result_entry(
            engine, question, query, gold,
            agent_answer=big_result.answer, success=big_result.success, cache_decision="miss",
            recap_path=recap_path, verdict=_verdict_holder.get("v"), sound=_verdict_holder.get("s"),
            similarity=_verdict_holder.get("sim"), steps=steps, ledger=ledger, t0=t0,
            error="; ".join(filter(None, [
                prior_error, runtime_error(), big_result.error,
            ])),
        )

    # holds judge/verify signals so a later big fallback can report them
    _verdict_holder = {"v": None, "s": None, "sim": None}

    # ---- retrieve ----
    hits = engine.index.search(query, engine.dataset, top_k=engine.top_k, tau=engine.tau)
    engine._retrieval_candidates = [
        {
            "rank": rank,
            "score": similarity,
            "plan_question": candidate.question,
            "tool_sequence": candidate.tool_sequence,
        }
        for rank, (candidate, similarity) in enumerate(hits, 1)
    ]
    if not hits:
        return run_big_and_index(PATH_NONE_PREFILTER)
    # record the top-1 similarity up front (used by fallback entries / analysis)
    _verdict_holder["sim"] = hits[0][1]

    # ---- judge the retrieved candidates in similarity order ----
    # The embedding pre-filter is a coarse filter: the top-1 by cosine is not
    # always the reusable plan (a different-group original can out-score the true
    # match). So we JUDGE candidates in rank order and select:
    #   - the FIRST candidate judged "full"  -> use it immediately (early exit;
    #     since hits are similarity-sorted, this is the most-similar full).
    #   - else the FIRST candidate judged "partial" (highest similarity, because
    #     we scan in descending-sim order and keep the first partial seen).
    #   - else (all "none") -> big agent.
    # Each judge is one A_small call; early-exit keeps the common case at 1 call.
    plan = similarity = subs = verdict = None
    first_partial = None   # (plan, similarity, subs) of the highest-sim partial
    for rank, (cand_plan, cand_sim) in enumerate(hits, 1):
        try:
            cand_verdict, cand_subs = engine.judge(
                query, cand_plan, ledger, similarity=cand_sim, rank=rank,
            )
        except Exception as e:
            # a transient A_small failure must NOT abort the whole RECAP run.
            logger.warning("judge call failed for a candidate (%s: %s); treating as none",
                           type(e).__name__, e)
            engine._runtime_errors.append(
                f"judge provider failure rank {rank}: {type(e).__name__}: {e}"
            )
            engine._judge_records.append({
                "rank": rank,
                "similarity": cand_sim,
                "plan_question": cand_plan.question,
                "verdict": "none",
                "substitutions": {},
                "parse_ok": False,
                "raw": "",
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        if cand_verdict == "full":
            plan, similarity, subs, verdict = cand_plan, cand_sim, cand_subs, "full"
            break
        if cand_verdict == "partial" and first_partial is None:
            first_partial = (cand_plan, cand_sim, cand_subs)
        # "none" -> keep scanning to the next candidate

    if verdict is None:   # no "full" found across all candidates
        if first_partial is not None:
            plan, similarity, subs = first_partial
            verdict = "partial"
        else:
            # every retrieved candidate judged "none" -> big agent
            _verdict_holder["v"] = "none"
            _verdict_holder["sim"] = hits[0][1]
            return run_big_and_index(PATH_NONE_JUDGE)

    # the selected candidate's verdict/similarity carry into any later fallback
    _verdict_holder["v"] = verdict
    _verdict_holder["sim"] = similarity
    engine._last_subs = subs   # log the judge's value bindings for the selected plan
    engine._matched_plan_query = plan.question

    # ---- route ----
    changes = substitutions_change_args(plan, subs, ctx)
    safe = is_replay_safe(plan, engine.tools_by_name)
    # certified substituted replay: a `full` verdict whose changes are a
    # provably-safe WHOLE-VALUE re-binding into deterministic tools -> re-execute
    # the stored calls with the new values, bypassing the unreliable guided agent.
    # Any unsafe/embedded/non-deterministic change falls through to guided as before.
    certified = (config.RECAP_CERTIFIED_REPLAY and changes and verdict == "full"
                 and is_certified_substitution(plan, subs, ctx, engine.tools_by_name))

    reuse_steps = []
    if verdict == "full" and safe and (not changes or certified):
        ok, executed, tool_calls, rsteps = engine._replay(plan, subs, ctx)
        ledger.extend(tool_calls)
        reuse_steps = rsteps
        if ok:
            recap_path = PATH_REPLAY_SUBST if certified else PATH_REPLAY
        else:
            # failed replay -> guided small agent
            engine._runtime_errors.append("replay tool failure or empty output")
            recap_path = PATH_REPLAY_THEN_GUIDED
            executed, gsteps = engine._run_guided(query, ctx, plan, partial=False, ledger=ledger, subs=subs)
            reuse_steps = reuse_steps + gsteps
    else:
        partial = (verdict == "partial")
        recap_path = PATH_GUIDED_PARTIAL if partial else PATH_GUIDED_FULL
        executed, gsteps = engine._run_guided(query, ctx, plan, partial=partial, ledger=ledger, subs=subs)
        reuse_steps = gsteps

    # ---- Config A (direct mode): return the reuse answer directly ----
    # No format+verify GATE and no A_big fallback (verifier-free by construction).
    # Three sources of the answer, in order of how the reuse was served:
    #   guided  -> the guided agent's own final answer (_cand_answer).
    #   replay -> derive the answer from the newly executed outputs with ONE
    #             format-only small call.  Plans never store source answers.
    #             The formatter's answer is returned without a soundness verdict
    #             or A_big fallback.
    # The last executed step's raw output is NOT the answer in general (a plan may
    # end on a validation/boolean step), so we never use str(executed[-1]).
    if engine.config_mode == "direct":
        direct_ans = engine._cand_answer
        if direct_ans is None:              # pure replay (no guided candidate)
            engine._last_verify = None
            try:
                direct_ans = engine.format_direct(query, executed, ledger)
            except Exception as e:
                engine._runtime_errors.append(
                    f"direct formatter provider failure: {type(e).__name__}: {e}"
                )
                direct_ans = None
        return _result_entry(
            engine, question, query, gold,
            agent_answer=direct_ans,
            success=bool(direct_ans and str(direct_ans).strip()),
            cache_decision="hit", recap_path=recap_path, verdict=verdict,
            sound=None, similarity=similarity, steps=reuse_steps, ledger=ledger, t0=t0,
            error=runtime_error(),
        )

    # ---- format + verify (on every reuse path) ----
    # a reuse that executed NOTHING has no evidence to verify: the guided agent
    # answered from memory (or aborted) without a tool call. Skip the verify
    # call (it would be asked to judge an empty step list) and fall back. Clear
    # _last_verify explicitly: format_verify never runs on this branch, so
    # without this the logged verify state would be stale (None from the top of
    # the run; but guard against any future reordering leaking a prior value).
    if not executed:
        engine._last_verify = None
        return run_big_and_index(recap_path + "_verify_reject", prior_steps=reuse_steps,
                                 is_fallback=True)

    try:
        answer, sound = engine.format_verify(query, executed, ledger)
    except Exception as e:
        # a transient A_small failure on the verifier must not abort the run.
        # conservatively fall back to A_big. Accrued reuse ledger/steps are carried in.
        logger.warning("format_verify call failed (%s: %s); falling back to A_big",
                       type(e).__name__, e)
        engine._runtime_errors.append(
            f"format_verify provider failure: {type(e).__name__}: {e}"
        )
        return run_big_and_index(recap_path + "_verify_reject", prior_steps=reuse_steps,
                                 is_fallback=True)
    _verdict_holder["s"] = sound
    if getattr(engine, "_last_verify", {}).get("parse_ok") is False:
        engine._runtime_errors.append("format_verify parse failure")

    if sound and answer is not None:
        return _result_entry(
            engine, question, query, gold,
            agent_answer=answer, success=True, cache_decision="hit",
            recap_path=recap_path, verdict=verdict, sound=True, similarity=similarity,
            steps=reuse_steps, ledger=ledger, t0=t0,
            error=runtime_error(),
        )

    # verify rejected -> big-agent fallback. The reuse steps and
    # ledger so far are carried in so their cost is not lost. this is a fallback,
    # so A_big runs without self-annotation and its plan is not indexed.
    return run_big_and_index(recap_path + "_verify_reject", prior_steps=reuse_steps,
                             is_fallback=True)
