"""
agents: ReAct and Plan-and-Act agents for the RECAP project.

usage:
    from recap.agents import AgentConfig
    from recap.agents.react import ReActAgent
    from recap.agents.plan_and_act import PlanAndActAgent
"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """configuration for both agent types."""

    # --- llm settings (used by ReAct, and as defaults for Plan-and-Act) ---
    model: str = ""            # defaults to config.PLANNER_MODEL at runtime
    provider: str = ""         # defaults to config.PLANNER_PROVIDER at runtime
    # temperature=None -> call_llm falls back to config.TEMPERATURE (the single
    # source of truth, itself None = provider default). Left as None so agents
    # never pin a temperature the cache methods lack; see config.TEMPERATURE.
    temperature: float = None
    max_tokens: int = 4096

    # --- plan-and-act: separate planner / executor models ---
    planner_model: str = ""    # defaults to model (or config.PLANNER_MODEL)
    planner_provider: str = ""
    executor_model: str = ""   # defaults to model (or config.EXECUTOR_MODEL)
    executor_provider: str = ""

    # --- agent behavior ---
    max_steps: int = 15                  # agent reasoning-call budget per run
    max_retries: int = 2                 # retries on parse failure before giving up

    # --- plan-and-act specifics ---
    max_steps_per_plan_step: int = 5     # executor budget per planner step
    enable_replan: bool = False          # dynamic replanning
    replan_interval: int = 3             # replan after this many executor actions

    # --- dataset-specific prompt guidance ---
    system_preamble: str = ""    # injected into system prompt before tool descriptions

    # --- RECAP plan-reuse hooks (opt-in; empty/false = unchanged prompts) ---
    self_annotate: bool = False  # A_big: emit a "Goal:" line before each Action (for COMPILE)
    demonstration: str = ""      # A_small guided reuse: a cached plan injected as a demonstration

    def resolve(self):
        """fill in defaults from config.py for any empty fields."""
        from recap import config as cfg
        if not self.model:
            self.model = cfg.PLANNER_MODEL
        if not self.provider:
            self.provider = cfg.PLANNER_PROVIDER
        if not self.planner_model:
            self.planner_model = self.model
        if not self.planner_provider:
            self.planner_provider = self.provider
        if not self.executor_model:
            self.executor_model = getattr(cfg, "EXECUTOR_MODEL", self.model)
        if not self.executor_provider:
            self.executor_provider = getattr(cfg, "EXECUTOR_PROVIDER", self.provider)
        return self


@dataclass
class LLMCallRecord:
    """one LLM invocation — the ground-truth ledger entry for cost accounting.

    every call_llm() and every tool-internal LLM call produces one of these.
    exact USD cost is computed per-entry at its own model's rate.
    """

    role: str                           # "planner", "executor", "replan", "agent", "tool:<name>"
    model: str                          # resolved model name (e.g. "gemini-2.5-flash")
    provider: str                       # "gemini", "openai", "deepseek", "openrouter"
    input_tokens: int = 0
    output_tokens: int = 0              # generated text only (excludes reasoning)
    reasoning_tokens: int = 0           # thinking / reasoning tokens

    def to_dict(self):
        """serialize for JSON output."""
        return {
            "role": self.role,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class StepRecord:
    """one tool invocation in the agent trace."""

    step_id: str                        # "S1", "S2", ...
    tool_name: str                      # which tool was called
    tool_args: dict                     # arguments passed to tool.execute()
    result: dict                        # raw tool return dict {status, output, output_type}
    thought: str = ""                   # the llm's reasoning text before this action
    goal: str = ""                      # RECAP "Goal:" annotation line (empty unless self_annotate)


@dataclass
class AgentResult:
    """complete result of an agent run."""

    query: str                                  # original user query
    answer: str = ""                            # final answer (from finish tool or last output)
    success: bool = False                       # whether finish was called cleanly
    steps: list[StepRecord] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)   # full message history
    total_input_tokens: int = 0
    total_output_tokens: int = 0                # generated text tokens only (excludes reasoning)
    total_reasoning_tokens: int = 0             # thinking tokens (Gemini 2.5 / o-series)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)  # per-call ledger
    metadata: dict = field(default_factory=dict)       # agent-specific (e.g. planner's plan)
    error: str = ""                             # error message if agent failed
