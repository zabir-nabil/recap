"""
reuse/prompts.py — the two standalone A_small prompts and the guided-agent
plan demonstration.

Both standalone prompts return a single JSON object."""

import json
import logging

from recap.agents.prompts import answer_format_instruction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JUDGE + binding — one A_small call: coverage + value binding
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """A cached plan (an ordered list of "goal -> tool" steps) solved a past question. Can it be reused for the NEW question?

NEW QUESTION: {new_question}
PAST QUESTION: {past_question}
PLAN STEPS:
{plan_steps}
AVAILABLE TOOLS: {tool_names}

Reply with one JSON object, no prose:
{{"verdict": "full" | "partial" | "none",
 "substitutions": {{"<value in the past question>": "<corresponding value in the new question, or null>"}}}}

verdict:
  full    = the plan's steps produce everything the new question asks for; only input values may differ.
  partial = the plan covers only part of the new question — either it answers a sub-part, OR the new question adds requirement(s) or constraint(s) the plan's steps do not address. More steps are needed.
  none    = a different approach is needed, or it needs unavailable tools.
substitutions: for each concrete value in the PAST QUESTION (entity, number, date, name) that has a counterpart in the NEW QUESTION, map past->new; use null if the new question doesn't supply one; map an unchanged value to itself. Empty {{}} when verdict is "none"."""


def build_judge_prompt(new_question, plan, tool_names):
    """render the judge prompt for a (new question, candidate plan) pair."""
    step_lines = "\n".join(
        f"- {(s.goal or s.action_str)} -> {s.tool_name}" for s in plan.steps
    )
    return JUDGE_PROMPT.format(
        new_question=new_question,
        past_question=plan.question,
        plan_steps=step_lines,
        tool_names=", ".join(tool_names),
    )


def parse_judge_output(text):
    """parse the judge JSON. FAILS CLOSED to ('none', {}) on any problem.

    returns (verdict, substitutions):
      verdict in {"full","partial","none"}
      substitutions: dict[str -> str|None]; JSON null -> Python None
    """
    obj = _extract_json_object(text)
    if obj is None:
        return "none", {}

    verdict = obj.get("verdict")
    if verdict not in ("full", "partial", "none"):
        return "none", {}

    subs = obj.get("substitutions")
    if not isinstance(subs, dict):
        subs = {}
    # keys must be strings; values are str or None (JSON null)
    clean = {}
    for k, v in subs.items():
        clean[str(k)] = None if v is None else str(v)
    return verdict, clean


# ---------------------------------------------------------------------------
# FORMAT + VERIFY — one A_small call: produce answer + soundness
# ---------------------------------------------------------------------------

_VERIFY_RULES = """- Return ONLY the value the NEW QUESTION asks for. Follow the answer-format rule below: give a bare number without a unit label unless the question asks for one, or give a short phrase, yes/no, or a comma-separated list as appropriate.
- Use the numerical unit requested by the question. If the evidence uses a different unit, CONVERT the value. Put NO thousands separators in the numeric part.
- [LLM reasoning output] lines are a helper model's reasoning, not retrieved facts — weigh the primary retrieved/computed outputs more heavily.
- Do NOT reject merely because an output was shortened (shown by "…"); judge only whether the VISIBLE evidence supports the answer.
- {answer_format_rule}
- Use ONLY the outputs below — no outside knowledge."""

VERIFY_CANDIDATE_PROMPT = """You are validating a REUSED plan's ANSWER. A candidate final answer is given. Using ONLY the executed steps and their outputs below, decide whether the candidate answer is correct for the NEW QUESTION, and return it in the exact format the question requires.

Output EXACTLY ONE JSON object and nothing else — no text before or after:
{{"reasoning": "<one short sentence>", "answer": "<the final answer in the required format, or NO_ANSWER>", "sound": true or false}}

CANDIDATE ANSWER: {candidate}

""" + _VERIFY_RULES + """
- If the outputs are about the RIGHT SUBJECT and SUPPORT the candidate: return the correctly-formatted answer and sound=true. If they are about the wrong subject, contradict the candidate, or you cannot produce a well-formed answer from the visible evidence: answer="NO_ANSWER" and sound=false.

NEW QUESTION:
{new_question}

EXECUTED STEPS AND OUTPUTS:
{executed_steps}"""

VERIFY_DERIVE_PROMPT = """You are validating a REUSED plan. Using ONLY the executed steps and their outputs below, (a) produce the final answer to the NEW QUESTION, and (b) judge whether those steps actually answer it.

Output EXACTLY ONE JSON object and nothing else — no text before or after:
{{"reasoning": "<one short sentence>", "answer": "<final answer in the required format, or NO_ANSWER if the outputs do not answer the question>", "sound": true or false}}

""" + _VERIFY_RULES + """
- sound = true ONLY if the outputs are about the right subject AND contain the information the question needs; otherwise answer="NO_ANSWER" and sound=false.

NEW QUESTION:
{new_question}

EXECUTED STEPS AND OUTPUTS:
{executed_steps}"""


FORMAT_DIRECT_PROMPT = """Format the answer to the NEW QUESTION using the executed tool outputs below.

Output EXACTLY ONE JSON object and nothing else:
{{"answer": "<final answer in the required format, or NO_ANSWER if no answer can be formatted>"}}

This is a formatting task only.
- {answer_format_rule}
- Use only the visible outputs below.

NEW QUESTION:
{new_question}

EXECUTED STEPS AND OUTPUTS:
{executed_steps}"""


def build_format_verify_prompt(new_question, executed, dataset=None, candidate_answer=None):
    """render the format+verify prompt.

    executed: list of (goal, action_str, output_str) per executed step, in order
    (already anchor/section-truncated by the caller; a step whose tool is a
    helper-LLM tool has " [LLM reasoning output]" appended to its action_str by
    the caller). A step with an empty goal renders without a goal prefix.

    candidate_answer: the guided agent's own answer to CHECK (guided paths). When
    None (pure replay, no agent answer), the DERIVE framing is used and the
    verifier produces the answer from the visible tool outputs. The verifier
    never sees the cached plan's original question/answer.

    dataset: selects the short, format-only per-dataset answer contract appended
    to the general ANSWER_FORMAT_RULE (no agent-workflow prose).
    """
    lines = []
    for goal, action, output in executed:
        g = str(goal).strip()
        prefix = f"{g} -> " if g else ""
        lines.append(f"- {prefix}{action} -> {output}")
    fmt = answer_format_instruction(dataset)
    if candidate_answer is not None and str(candidate_answer).strip():
        return VERIFY_CANDIDATE_PROMPT.format(
            candidate=candidate_answer,
            new_question=new_question,
            executed_steps="\n".join(lines),
            answer_format_rule=fmt,
        )
    return VERIFY_DERIVE_PROMPT.format(
        new_question=new_question,
        executed_steps="\n".join(lines),
        answer_format_rule=fmt,
    )


def build_direct_format_prompt(new_question, executed, dataset=None):
    """Build Config A's formatter-only prompt; it contains no verifier task."""
    lines = []
    for goal, action, output in executed:
        goal = str(goal).strip()
        prefix = f"{goal} -> " if goal else ""
        lines.append(f"- {prefix}{action} -> {output}")
    fmt = answer_format_instruction(dataset)
    return FORMAT_DIRECT_PROMPT.format(
        new_question=new_question,
        executed_steps="\n".join(lines),
        answer_format_rule=fmt,
    )

_NO_ANSWER_SENTINELS = frozenset({"no_answer", "null", "n/a", "na"})

def parse_format_verify_output(text):
    """parse the format+verify JSON. FAILS CLOSED to (None, False) on any problem.

    returns (answer, sound):
      answer: str or None (None when the model says the plan does not address it,
              canonicalized from JSON null / the NO_ANSWER / "null" / "n/a" sentinels)
      sound: bool — True only when explicitly true AND a real answer is present
    """
    obj = _extract_json_object(text)
    if obj is None:
        return None, False

    sound = obj.get("sound")
    if not isinstance(sound, bool):
        # non-boolean / missing soundness -> treat as unsound (conservative)
        return None, False

    answer = obj.get("answer")
    if answer is None:                       # JSON null
        return None, False
    answer = str(answer)

    # an empty/whitespace answer, or a canonical no-answer sentinel, is not a real
    # answer -> unsound. (Leave "none"/"[]" alone: legitimate empty-set answers.)
    if not answer.strip() or answer.strip().lower() in _NO_ANSWER_SENTINELS:
        return None, False

    # sound is authoritative; a true verdict with a real answer is accepted
    return (answer, True) if sound else (answer, False)


def parse_direct_format_output(text):
    """Parse Config A formatting output without reading any soundness signal."""
    obj = _extract_json_object(text)
    if obj is None:
        return None
    answer = obj.get("answer")
    if answer is None:
        return None
    answer = str(answer).strip()
    if not answer or answer.lower() in _NO_ANSWER_SENTINELS:
        return None
    return answer


# ---------------------------------------------------------------------------
# guided-agent demonstration — injected via AgentConfig.demonstration
# ---------------------------------------------------------------------------

_DEMO_HEADER_VALUE = (
    "A plan that solved a SIMILAR question is shown below. Follow the same "
    "strategy, adapt the values to your question, and adapt if a step misbehaves. "
    "It is a guide, not a solution to copy — verify each step against your own question."
)

_DEMO_HEADER_PARTIAL = (
    "A plan that solved a similar question is shown below, but it does not cover "
    "everything your question asks. Use it to see the approach, then work out the "
    "full solution your own question needs — adding or changing steps where "
    "required. Verify each step against your own question."
)


def _render_plan_demonstration(plan):
    lines = [f'Past question: "{plan.question}"', "Plan steps:"]
    for i, s in enumerate(plan.steps, 1):
        goal = f" — {s.goal}" if s.goal else ""
        lines.append(f"  {i}. {s.action_str}{goal}  (result: {s.result_summary})")
    return "\n".join(lines)


def _render_bindings(substitutions):
    """render the judge's value bindings as explicit substitution instructions.

    The judge already computes {past_value -> new_value}; previously this was
    discarded on the guided path, forcing the small agent to re-derive the
    mapping blind (and often ignore it). Surfacing only the CHANGED, non-null
    bindings tells the agent exactly which values to swap for its question.
    """
    if not substitutions:
        return ""
    pairs = [(k, v) for k, v in substitutions.items()
             if v is not None and str(k).strip() and str(v) != str(k)]
    if not pairs:
        return ""
    lines = "\n".join(f'  "{k}" -> "{v}"' for k, v in pairs)
    return ("\n\nVALUE BINDINGS for YOUR question (the plan solved a DIFFERENT "
            "question; substitute the RIGHT-hand value wherever the plan used the "
            "LEFT-hand value — do NOT reuse the left-hand past values):\n" + lines)


def build_demonstration(plan, partial=False, substitutions=None):
    """build the demonstration block injected into the guided A_small agent.

    partial=False -> value-differs / replay-failed framing;
    partial=True  -> partial-coverage "use as a start, then continue" framing.
    both say SIMILAR (never identical): the judge is fallible and may over-label.

    substitutions (optional): the judge's {past_value -> new_value} bindings.
    When provided, the CHANGED bindings are appended as explicit substitution
    instructions so the agent applies the computed mapping instead of guessing.
    """
    header = _DEMO_HEADER_PARTIAL if partial else _DEMO_HEADER_VALUE
    bindings = _render_bindings(substitutions)
    return (f"=== REFERENCE PLAN ===\n{header}\n\n"
            f"{_render_plan_demonstration(plan)}{bindings}\n======================")


# ------------------------
# shared JSON extraction
# ------------------------

def _extract_json_object(text):
    """extract the first well-formed top-level JSON object from model output.

    tolerates ```json fences and surrounding prose. returns a dict or None.

    FAILS CLOSED on a payload whose top-level JSON value is NOT an object — a
    JSON array (``[{...}, ...]``), string, or number degrades to None rather
    than silently unwrapping an inner object (e.g. an array's first element),
    which would let the parser echo the model's own verdict/soundness instead
    of the conservative fallback the callers rely on.
    """
    if not text or not str(text).strip():
        return None
    s = _strip_code_fences(str(text))

    # First try the WHOLE payload as one JSON value. This is the common case
    # (the model returns exactly one JSON object) and it is what makes a
    # top-level array fail closed: json.loads succeeds, the value is a list, and
    # we return None instead of reaching the brace-scan (which would unwrap the
    # array's first object). A dict is returned directly; anything else -> None.
    try:
        whole = json.loads(s.strip())
    except json.JSONDecodeError:
        whole = None
    else:
        return whole if isinstance(whole, dict) else None

    start = s.find("{")
    if start == -1:
        return None
    # payload is not pure JSON (surrounding prose): scan for the first top-level
    # object. Guard against an array embedded in prose ("result: [{...}]") by
    # rejecting a leading '{' that is immediately preceded by '[' — that first
    # object is an array element, not a top-level object, so fail closed.
    if s[:start].rstrip().endswith("["):
        return None
    # scan for the matching closing brace, respecting string literals so a
    # brace inside a quoted value does not end the object prematurely.
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text
