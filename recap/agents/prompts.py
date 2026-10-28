"""
prompt construction for ReAct and Plan-and-Act agents.

Design principles:
- Tool descriptions are auto-generated from BaseTool instances
- Both agents share the same grounding rules (no internal knowledge, tool-first)
- The Thought/Action format permits optional content between them (annotation-ready)
- Follows the ReAct paradigm (Yao et al., 2023) for the reasoning loop
"""

import inspect


# ---------------------------------------------------------------------------
# tool formatting
# ---------------------------------------------------------------------------

def format_tool_signatures(tools):
    """generate tool descriptions from BaseTool instances.

    inspects execute() signature to extract parameter names automatically.
    output looks like:
        - doc_search(question_id, query): search a financial document...
        - calculator(expression): evaluate a math expression...
    """
    lines = []
    for tool in tools:
        params = _get_tool_params(tool)
        param_str = ", ".join(params)
        lines.append(f"- {tool.name}({param_str}): {tool.description}")
    return "\n".join(lines)


def _get_tool_params(tool):
    """extract parameter names from tool.execute() signature, skipping 'self'."""
    try:
        sig = inspect.signature(tool.execute)
        return [
            p.name for p in sig.parameters.values()
            if p.name != "self"
        ]
    except (ValueError, TypeError):
        return ["**kwargs"]


def format_tool_names(tools):
    """return comma-separated tool names for use in prompts."""
    return ", ".join(t.name for t in tools)


# ---------------------------------------------------------------------------
# shared grounding rules (identical for ReAct and Plan-and-Act executor)
# ---------------------------------------------------------------------------

# These rules implement "tool-augmented generation" grounding: the agent must
# use tools rather than parametric knowledge for factual claims. This follows
# the methodology of Schick et al. (2023) "Toolformer" and Yao et al. (2023)
# "ReAct" where the model is trained/prompted to defer factual lookups to tools.

# Answer-format contract, shared by every agent AND the RECAP format+verify step
# (imported by reuse/prompts.py) so the final answer is produced in the exact
# form the question asks for — matched identically across all methods.
ANSWER_FORMAT_RULE = (
    'Give ONLY the answer the question asks for, in its exact unit and format: a '
    'bare number (no commas or units unless asked), a short word/phrase, or a '
    'comma-separated list; convert units when asked (e.g. "thousand hours" -> '
    'divide hours by 1000). For query/table results, give the rows themselves '
    '(e.g. a JSON array of row objects). Add no explanation.'
)

# Share the final-answer contract between agents and RECAP's formatter/verifier.
# These snippets specify output shape, without workflow instructions or evidence.
ANSWER_FORMAT_BY_DATASET = {
    "sql_api_bench": "For query/table results, return the rows themselves as a JSON array of row objects.",
    "financebench": "Return a bare number in the unit the question asks (e.g. USD millions), or yes/no.",
    "gaia": "Return a number (no commas/units unless asked), a few words (no city abbreviations, no articles), or a comma-separated list.",
    "toolqa": "Return the bare value — a number or a short string.",
}


def answer_format_instruction(dataset=None):
    """Shared final-answer rule plus the dataset's short output contract."""
    if dataset in ("gaia_l1", "gaia_l2"):
        dataset = "gaia"
    dataset_rule = ANSWER_FORMAT_BY_DATASET.get(dataset, "")
    return f"{ANSWER_FORMAT_RULE} {dataset_rule}" if dataset_rule else ANSWER_FORMAT_RULE

GROUNDING_RULES = f"""Grounding requirement: This is a tool-grounded task. Your final answer must be derived from tool Observations obtained during this session. You may use your own knowledge to decide WHICH tools to call and HOW to interpret their output, but every factual claim in your answer must be traceable to an Observation.

Procedure:
1. Your FIRST action must be an information-gathering tool call. Never call finish() before you have made at least one successful tool call.
2. Call exactly ONE tool per step, then wait for its Observation.
3. When a question requires checking a property (prime, fibonacci, even, etc.) across multiple values, call the checking tool once per value.
4. Call finish() only when the Observations collected so far contain the information needed to answer.

Format rules:
- Always quote string parameter values.
- Use the Observation from prior steps to inform later steps.
- If a tool returns an error, reason about what went wrong and try a different approach.
- {ANSWER_FORMAT_RULE}"""


# ---------------------------------------------------------------------------
# RECAP self-annotation (opt-in via AgentConfig.self_annotate)
# ---------------------------------------------------------------------------

# The step-format skeleton. When self_annotate is on, a "Goal:" line is part of
# the format itself (between Thought and Action) rather than a separate trailing
# directive — so placement is unambiguous and the skeleton is shown once, not
# duplicated. The Thought -> Goal -> Action order is mandated so the Goal line
# lands inside the text captured by _extract_thought (Thought: up to the first
# Action:) and never interferes with the last-Action parser.
# ReAct format skeleton (plain vs annotated). 
_FORMAT_PLAIN = (
    'Thought: <your reasoning about what to do next>\n'
    'Action: <tool_name>(<param1>="<value1>", <param2>="<value2>")'
)
_FORMAT_ANNOTATED = (
    'Thought: <your reasoning about what to do next>\n'
    'Goal: <what this step is for and where its inputs come from>\n'
    'Action: <tool_name>(<param1>="<value1>", <param2>="<value2>")'
)

# Plan-and-Act executor format skeleton.
_EXEC_FORMAT_PLAIN = (
    'Thought: <your reasoning about how to accomplish the current plan step>\n'
    'Action: <tool_name>(<param1>="<value1>", <param2>="<value2>")'
)
_EXEC_FORMAT_ANNOTATED = (
    'Thought: <your reasoning about how to accomplish the current plan step>\n'
    'Goal: <what this step is for and where its inputs come from>\n'
    'Action: <tool_name>(<param1>="<value1>", <param2>="<value2>")'
)

# The Goal-line content rule (A_big only). Emitted during indexing so COMPILE
# can recover each step's intent and input provenance. Values derived from
# earlier steps or the data are described by their SOURCE, not their concrete
# value, so a compiled plan generalizes to a new question rather than baking in
# one question's literals.
GOAL_RULE = """Goal line: state what this step is for and where its key inputs come from — from the question, from an earlier step's output, or from your own reasoning or calculation. If you compute or decide a value yourself (a formula, a choice, an index), say so briefly. Describe a value that comes from an earlier step or the data by its SOURCE, not its concrete value — write "each organization id from the previous query" or "the total assets found earlier", NOT the literal number. Only values that come straight from the question may appear literally. Write one line only; do not restate the tool's output."""


# ---------------------------------------------------------------------------
# ReAct prompts
# ---------------------------------------------------------------------------

REACT_SYSTEM_TEMPLATE = """You are a tool-augmented reasoning agent. Answer the user's question by interleaving reasoning (Thought) with tool use (Action), following the ReAct framework.

Available tools:
{tool_signatures}
{preamble}
Format for each step:
{format_spec}

After each Action you will receive an Observation with the tool's output.

When you have gathered sufficient information to answer, use:
Action: finish(answer="<your final answer>")

{grounding_rules}{goal_rule}{demonstration}"""


def _format_spec(self_annotate, executor=False):
    """the step-format skeleton; includes the Goal line when self-annotating.

    executor=True selects the Plan-and-Act executor's wording (its Thought line
    differs from ReAct's), keeping both plain variants byte-identical to the
    pre-RECAP prompts.
    """
    if executor:
        return _EXEC_FORMAT_ANNOTATED if self_annotate else _EXEC_FORMAT_PLAIN
    return _FORMAT_ANNOTATED if self_annotate else _FORMAT_PLAIN


def _goal_rule_block(self_annotate):
    """trailing Goal-line content rule; empty (no-op) unless self-annotating."""
    return f"\n\n{GOAL_RULE}" if self_annotate else ""


def _demonstration_block(demonstration):
    """trailing plan-demonstration block; empty (no-op) unless provided."""
    return f"\n\n{demonstration}" if demonstration else ""


def build_react_system_prompt(tools, preamble="", self_annotate=False, demonstration=""):
    """build the full system prompt for the ReAct agent.

    self_annotate / demonstration are RECAP hooks; with both off the output is
    byte-identical to the pre-RECAP prompt.
    """
    tool_sigs = format_tool_signatures(tools)
    preamble_block = f"\n{preamble}\n" if preamble else ""
    return REACT_SYSTEM_TEMPLATE.format(
        tool_signatures=tool_sigs,
        preamble=preamble_block,
        format_spec=_format_spec(self_annotate),
        grounding_rules=GROUNDING_RULES,
        goal_rule=_goal_rule_block(self_annotate),
        demonstration=_demonstration_block(demonstration),
    )


# ---------------------------------------------------------------------------
# Plan-and-Act prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_TEMPLATE = """You are a planning agent. Given a user query and available tools, generate a structured step-by-step plan.

Available tools:
{tool_signatures}
{preamble}
Output a numbered plan. Each step must follow this format:
## Step N
Reasoning: <why this step is needed and how it connects to previous steps>
Step: <concise description of what to accomplish>

Guidelines:
- Each step should represent one logical sub-goal, not a low-level tool call.
- Be specific about values: say "extract total current assets for Costco" not "extract the relevant metric."
- Minimize steps by grouping related sub-goals.
- If a step depends on a prior step's output, state the dependency explicitly.
- The final step should always be returning the answer to the user.
- Do NOT specify which tool to use — the executor decides that.
- Do NOT rely on your internal knowledge to pre-fill answers. The executor will use tools to retrieve all factual information.{demonstration}"""


EXECUTOR_SYSTEM_TEMPLATE = """You are a tool-augmented executor agent. Follow the plan below and use tools to complete the current step.

Available tools:
{tool_signatures}
{preamble}
=== FULL PLAN ===
{plan_text}
=================

=== CURRENT STEP ===
{current_step}
====================

Format for each turn:
{format_spec}

When the current plan step is complete, indicate it in your Thought.
When all plan steps are done and you have the final answer, use:
Action: finish(answer="<your final answer>")

{grounding_rules}{goal_rule}{demonstration}"""


REPLAN_SYSTEM_TEMPLATE = """You are a planning agent revising an existing plan based on execution progress.

Available tools:
{tool_signatures}

=== ORIGINAL PLAN ===
{original_plan}
=====================

=== EXECUTION SO FAR ===
{execution_summary}
========================

Generate a REVISED plan for the remaining work. Keep completed steps as-is and adjust or add steps as needed. Use the same format:
## Step N
Reasoning: <why>
Step: <what to do>"""


def build_planner_system_prompt(tools, preamble="", demonstration=""):
    """build the system prompt for the Plan-and-Act planner.

    the planner does not call tools, so it takes only the demonstration hook
    (no self-annotation). with demonstration empty the output is unchanged.
    """
    tool_sigs = format_tool_signatures(tools)
    preamble_block = f"\n{preamble}\n" if preamble else ""
    return PLANNER_SYSTEM_TEMPLATE.format(
        tool_signatures=tool_sigs,
        preamble=preamble_block,
        demonstration=_demonstration_block(demonstration),
    )


def build_executor_system_prompt(tools, plan_text, current_step, preamble="",
                                 self_annotate=False, demonstration=""):
    """build the system prompt for the Plan-and-Act executor.

    self_annotate / demonstration are RECAP hooks; with both off the output is
    byte-identical to the pre-RECAP prompt.
    """
    tool_sigs = format_tool_signatures(tools)
    preamble_block = f"\n{preamble}\n" if preamble else ""
    return EXECUTOR_SYSTEM_TEMPLATE.format(
        tool_signatures=tool_sigs,
        plan_text=plan_text,
        current_step=current_step,
        preamble=preamble_block,
        format_spec=_format_spec(self_annotate, executor=True),
        grounding_rules=GROUNDING_RULES,
        goal_rule=_goal_rule_block(self_annotate),
        demonstration=_demonstration_block(demonstration),
    )


def build_replan_prompt(tools, original_plan, execution_summary):
    """build the prompt for dynamic replanning."""
    tool_sigs = format_tool_signatures(tools)
    return REPLAN_SYSTEM_TEMPLATE.format(
        tool_signatures=tool_sigs,
        original_plan=original_plan,
        execution_summary=execution_summary,
    )
