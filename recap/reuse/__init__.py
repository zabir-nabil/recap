"""RECAP: Retrieval and Execution of Compiled Agent Plans.

Compile an agent's successful tool calls into structured plans, retrieve them
for later questions, and choose replay or guided execution with optional answer
verification. Verification is a model judgment, not a correctness guarantee.
"""

from recap.reuse.plan import Plan, PlanStep, compile_plan
from recap.reuse.index import PlanIndex
from recap.reuse.prompts import (
    build_judge_prompt, parse_judge_output,
    build_format_verify_prompt, parse_format_verify_output,
    build_demonstration,
)
from recap.reuse.engine import RecapEngine, run_with_recap

__all__ = [
    "Plan", "PlanStep", "compile_plan", "PlanIndex",
    "build_judge_prompt", "parse_judge_output",
    "build_format_verify_prompt", "parse_format_verify_output",
    "build_demonstration",
    "RecapEngine", "run_with_recap",
]
