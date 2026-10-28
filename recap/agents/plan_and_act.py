"""
Plan-and-Act agent: planner generates high-level plan, executor carries it out.

architecture:
    phase 1 - planner llm generates a structured multi-step plan
    phase 2 - executor llm runs a mini ReAct loop per plan step
    optional - dynamic replanning after every N executor actions
"""

import re
import logging
from dataclasses import dataclass
from recap.agents import AgentConfig, AgentResult
from recap.agents.base import BaseAgent
from recap.agents.prompts import (
    build_planner_system_prompt,
    build_executor_system_prompt,
    build_replan_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """one step in the planner's output."""
    step_num: int
    reasoning: str
    instruction: str
    completed: bool = False
    executor_summary: str = ""


class PlanAndActAgent(BaseAgent):
    """Plan-and-Act agent: planner generates plan, executor runs it step by step."""

    def run(self, query, context=None):
        """execute the full plan-and-act pipeline.

        args:
            query: the user's question
            context: optional dict of extra context

        returns:
            AgentResult with answer, steps, trace, plan in metadata
        """
        self._reset()
        self._executor_action_count = 0

        # ---- phase 1: planning ----
        plan_steps, plan_text, plan_error = self._run_planner(query, context)
        if plan_error:
            return self._build_result(query, error=plan_error)

        logger.info(f"planner generated {len(plan_steps)} steps")

        # ---- phase 2: execution ----
        answer, success, exec_error = self._run_executor(
            query, context, plan_steps, plan_text
        )

        return self._build_result(
            query,
            answer=answer,
            success=success,
            error=exec_error,
            metadata={
                "plan": plan_text,
                "plan_steps": [
                    {
                        "step_num": ps.step_num,
                        "instruction": ps.instruction,
                        "reasoning": ps.reasoning,
                        "completed": ps.completed,
                        "executor_summary": ps.executor_summary,
                    }
                    for ps in plan_steps
                ],
            },
        )

    # ------------------------------------------------------------------
    # phase 1: planner
    # ------------------------------------------------------------------

    def _run_planner(self, query, context=None):
        """invoke the planner llm to generate a structured plan.

        returns (plan_steps, plan_text, error).
        """
        system_prompt = build_planner_system_prompt(
            self.tool_list, preamble=self.config.system_preamble,
            demonstration=self.config.demonstration,
        )

        user_msg = query
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            user_msg = f"{query}\n\nContext:\n{context_str}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self._call_llm(
                messages,
                model=self.config.planner_model,
                provider=self.config.planner_provider,
                role="planner",
            )
        except Exception as e:
            logger.error(f"planner llm failed: {e}")
            return [], "", f"planner LLM call failed: {e}"

        plan_text = response.text
        self._trace.extend(messages)
        self._trace.append({"role": "assistant", "content": plan_text, "_phase": "planner"})

        plan_steps = self._parse_plan(plan_text)
        if not plan_steps:
            return [], plan_text, "planner produced no parseable steps"

        return plan_steps, plan_text, ""

    def _parse_plan(self, plan_text):
        """parse ## Step N blocks from planner output."""
        steps = []
        parts = re.split(r'##\s*Step\s+(\d+)', plan_text)
        # parts = [preamble, "1", content1, "2", content2, ...]
        i = 1
        while i < len(parts) - 1:
            step_num = int(parts[i])
            content = parts[i + 1].strip()

            reasoning = ""
            instruction = ""

            r_match = re.search(
                r'Reasoning:\s*(.+?)(?=\nStep:|$)', content, re.DOTALL
            )
            if r_match:
                reasoning = r_match.group(1).strip()

            s_match = re.search(r'Step:\s*(.+?)$', content, re.DOTALL)
            if s_match:
                instruction = s_match.group(1).strip()
            elif not reasoning:
                instruction = content

            steps.append(PlanStep(
                step_num=step_num,
                reasoning=reasoning,
                instruction=instruction,
            ))
            i += 2

        return steps

    # ------------------------------------------------------------------
    # phase 2: executor
    # ------------------------------------------------------------------

    def _run_executor(self, query, context, plan_steps, plan_text):
        """execute plan steps one by one using a mini ReAct loop per step.

        returns (answer, success, error).
        """
        user_context = ""
        if context:
            user_context = "\n".join(f"{k}: {v}" for k, v in context.items())

        executor_messages = []
        exec_finish_rejects = 0   # finish-first grounding rejects, across all plan steps
        _FINISH_REJECT_CAP = 3

        for ps_idx, plan_step in enumerate(plan_steps):
            current_step_text = (
                f"## Step {plan_step.step_num}\n"
                f"Reasoning: {plan_step.reasoning}\n"
                f"Step: {plan_step.instruction}"
            )

            system_prompt = build_executor_system_prompt(
                self.tool_list,
                plan_text=plan_text,
                current_step=current_step_text,
                preamble=self.config.system_preamble,
                self_annotate=self.config.self_annotate,
                demonstration=self.config.demonstration,
            )

            if ps_idx == 0:
                user_msg = f"Query: {query}"
                if user_context:
                    user_msg += f"\n\nContext:\n{user_context}"
                user_msg += f"\n\nBegin executing Step {plan_step.step_num}."

                executor_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ]
                self._trace.extend(executor_messages)
            else:
                executor_messages[0] = {"role": "system", "content": system_prompt}
                transition = (
                    f"Step {plan_steps[ps_idx - 1].step_num} is complete. "
                    f"Now execute Step {plan_step.step_num}: {plan_step.instruction}"
                )
                transition_msg = {"role": "user", "content": transition}
                executor_messages.append(transition_msg)
                self._trace.append(transition_msg)

            # mini ReAct loop for this plan step
            step_budget = self.config.max_steps_per_plan_step
            consecutive_failures = 0
            step_signaled_complete = False

            for _ in range(step_budget):
                # unified step budget: cap on agent-reasoning LLM calls
                # (planner + executor + replan), identical to ReAct's max_steps.
                if self._reasoning_call_count >= self.config.max_steps:
                    salvaged = self._salvage_answer(
                        executor_messages, model=self.config.executor_model,
                        provider=self.config.executor_provider)
                    return (salvaged or self._last_output(), False,
                            f"max reasoning calls ({self.config.max_steps}) exhausted")

                try:
                    response = self._call_llm(
                        executor_messages,
                        model=self.config.executor_model,
                        provider=self.config.executor_provider,
                        role="executor",
                    )
                except Exception as e:
                    logger.error(f"executor llm failed: {e}")
                    return "", False, f"executor LLM call failed: {e}"

                llm_text = response.text

                assistant_msg = {"role": "assistant", "content": llm_text}
                executor_messages.append(assistant_msg)
                self._trace.append(assistant_msg)

                # parse action
                tool_name, tool_args = self._parse_action(llm_text)

                if tool_name is None:
                    consecutive_failures += 1
                    if consecutive_failures > self.config.max_retries:
                        return self._last_output(), False, (
                            f"executor parse failed {consecutive_failures} times "
                            f"consecutively on plan step {plan_step.step_num}"
                        )
                    retry_msg = (
                        "Could not parse your action. Please respond with exactly:\n"
                        "Thought: <reasoning>\n"
                        "Action: <tool_name>(<param>=\"<value>\")"
                    )
                    executor_messages.append({"role": "user", "content": retry_msg})
                    self._trace.append({"role": "user", "content": retry_msg})
                    continue

                consecutive_failures = 0

                # reject an EMPTY finish answer: re-prompt for the value rather
                # than recording a "successful" finish with no answer. bounded by
                # the per-step budget and max_steps. genuinely-empty results are
                # expressed as "[]"/"none", so this does not reject those.
                if tool_name == "finish" and not str(tool_args.get("answer", "")).strip():
                    empty_msg = (
                        "REJECTED: finish() was called with an empty answer. Provide "
                        "the actual answer value derived from the Observations, e.g. "
                        "finish(answer=\"<the value>\")."
                    )
                    executor_messages.append({"role": "user", "content": empty_msg})
                    self._trace.append({"role": "user", "content": empty_msg})
                    continue

                # reject finish() before ANY data tool has run (grounding guard,
                # mirrors ReActAgent) — an ungrounded first-action answer is not
                # acceptable. bounded by _FINISH_REJECT_CAP so it cannot loop.
                if tool_name == "finish" and not any(s.tool_name != "finish" for s in self._steps):
                    exec_finish_rejects += 1
                    if exec_finish_rejects >= _FINISH_REJECT_CAP:
                        return self._last_output(), False, (
                            f"executor attempted finish-first {exec_finish_rejects} times "
                            f"without calling a data tool"
                        )
                    reject_msg = (
                        "REJECTED: you called finish() without retrieving any data. "
                        "Call a data-gathering tool first, examine its Observation, "
                        "then call finish() with the verified result."
                    )
                    executor_messages.append({"role": "user", "content": reject_msg})
                    self._trace.append({"role": "user", "content": reject_msg})
                    continue

                step_id = self._next_step_id()
                thought = self._extract_thought(llm_text)
                goal = self._extract_goal(llm_text)
                self._executor_action_count += 1

                # finish ends entire execution
                if tool_name == "finish":
                    answer = tool_args.get("answer", "")
                    result = {"status": "success", "output": answer, "output_type": "final"}
                    self._record_step(step_id, tool_name, tool_args, result, thought=thought, goal=goal)
                    plan_step.completed = True
                    plan_step.executor_summary = f"completed with finish: {answer}"
                    return answer, True, ""

                # execute tool
                result = self._execute_tool(tool_name, tool_args)
                obs_text = self._format_observation(result)
                self._record_step(step_id, tool_name, tool_args, result, thought=thought, goal=goal)

                obs_msg = {"role": "user", "content": obs_text}
                executor_messages.append(obs_msg)
                self._trace.append(obs_msg)

                # dynamic replanning check
                if (self.config.enable_replan
                        and self._executor_action_count % self.config.replan_interval == 0
                        and ps_idx < len(plan_steps) - 1):
                    new_plan_text = self._replan(query, plan_text, plan_steps, ps_idx)
                    if new_plan_text:
                        new_steps = self._parse_plan(new_plan_text)
                        if new_steps:
                            plan_text = new_plan_text
                            remaining = [s for s in new_steps if s.step_num > plan_step.step_num]
                            if remaining:
                                plan_steps[ps_idx + 1:] = remaining
                                logger.info(f"replanned: {len(remaining)} remaining steps")

                # check if executor signaled plan step completion
                if self._step_complete_signal(llm_text, plan_step.step_num):
                    step_signaled_complete = True
                    break

            plan_step.completed = step_signaled_complete
            last_output = self._steps[-1].result.get("output", "") if self._steps else ""
            plan_step.executor_summary = str(last_output)[:200]
            if not step_signaled_complete:
                logger.info(
                    f"plan step {plan_step.step_num} exhausted budget ({step_budget}) "
                    f"without explicit completion signal; advancing to next step"
                )

        # all plan steps done but no finish() called: salvage a best-effort answer
        # from the executor's Observations.
        salvaged = self._salvage_answer(
            executor_messages, model=self.config.executor_model,
            provider=self.config.executor_provider) if executor_messages else ""
        return (salvaged or self._last_output(), False,
                "all plan steps completed but finish() was not called")

    def _step_complete_signal(self, llm_text, current_step_num):
        """detect if the executor indicated the current plan step is done.

        scoped to the Thought block to avoid false positives from Observation content.
        """
        text_to_check = self._extract_thought(llm_text) or llm_text
        patterns = [
            rf'step\s+{current_step_num}(?!\d)\s+(?:is\s+)?(?:complete|done|finished)',
            rf'(?:completed|finished|done\s+with)\s+step\s+{current_step_num}(?!\d)',
            rf'(?:moving|proceed|move)\s+(?:on\s+)?to\s+step\s+{current_step_num + 1}(?!\d)',
            rf'this\s+completes?\s+step\s+{current_step_num}(?!\d)',
        ]
        for p in patterns:
            if re.search(p, text_to_check, re.IGNORECASE):
                return True
        return False

    # ------------------------------------------------------------------
    # dynamic replanning
    # ------------------------------------------------------------------

    def _replan(self, query, original_plan, plan_steps, current_idx):
        """invoke the planner to revise the remaining plan."""
        completed = plan_steps[:current_idx + 1]
        summary_lines = []
        for ps in completed:
            status = "completed" if ps.completed else "in progress"
            summary_lines.append(
                f"Step {ps.step_num} [{status}]: {ps.instruction}\n"
                f"  Result: {ps.executor_summary}"
            )

        recent_steps = self._steps[-3:] if len(self._steps) >= 3 else self._steps
        for s in recent_steps:
            summary_lines.append(
                f"  Tool: {s.tool_name}({s.tool_args}) -> {s.result.get('status')}: "
                f"{str(s.result.get('output', ''))[:100]}"
            )

        execution_summary = "\n".join(summary_lines)
        replan_prompt = build_replan_prompt(
            self.tool_list, original_plan, execution_summary
        )

        messages = [
            {"role": "system", "content": replan_prompt},
            {"role": "user", "content": f"Original query: {query}\n\nRevise the plan for remaining steps."},
        ]

        try:
            response = self._call_llm(
                messages,
                model=self.config.planner_model,
                provider=self.config.planner_provider,
                role="replan",
            )
            self._trace.append({"role": "assistant", "content": response.text, "_phase": "replan"})
            return response.text
        except Exception as e:
            logger.warning(f"replanning failed: {e}")
            return None

    # ------------------------------------------------------------------
    # result construction
    # ------------------------------------------------------------------

    def _last_output(self):
        """return the last tool output as a fallback answer."""
        if self._steps:
            return str(self._steps[-1].result.get("output", ""))
        return ""

    def _build_result(self, query, answer="", success=False, error="", metadata=None):
        """construct the final AgentResult."""
        return AgentResult(
            query=query,
            answer=answer,
            success=success,
            steps=list(self._steps),
            trace=list(self._trace),
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_reasoning_tokens=self._total_reasoning_tokens,
            llm_calls=list(self._llm_calls),
            metadata=metadata or {},
            error=error,
        )
