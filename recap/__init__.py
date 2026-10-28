"""RECAP: structured plan reuse for tool-using agents.

Use RecapEngine with run_with_recap for online reuse. AgentConfig and the
agent/result types support custom tools and integrations; compiled plans can
be serialized through Plan and PlanIndex without changing their execution.
"""

from recap.agents import AgentConfig, AgentResult, StepRecord, LLMCallRecord
from recap.agents.base import BaseAgent
from recap.agents.react import ReActAgent
from recap.agents.plan_and_act import PlanAndActAgent
from recap.reuse.plan import Plan, PlanStep, compile_plan
from recap.reuse.index import PlanIndex
from recap.reuse.engine import RecapEngine, run_with_recap
from recap.tools import BaseTool

__all__ = [
    "RecapEngine", "run_with_recap", "PlanIndex", "Plan", "PlanStep", "compile_plan",
    "AgentConfig", "AgentResult", "StepRecord", "LLMCallRecord", "BaseAgent",
    "ReActAgent", "PlanAndActAgent", "BaseTool",
]
