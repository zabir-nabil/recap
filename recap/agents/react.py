"""
ReAct agent: single-loop reasoning and acting.

loop:
    1. LLM receives (system_prompt + conversation_history) -> outputs Thought + Action
    2. agent parses Action -> dispatches tool -> gets result
    3. agent formats Observation -> appends to conversation
    4. if Action was finish(), return answer
    5. repeat until max_steps
"""

import logging
from recap.agents import AgentConfig, AgentResult
from recap.agents.base import BaseAgent
from recap.agents.prompts import build_react_system_prompt

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """ReAct agent: single reasoning-and-acting loop."""

    def run(self, query, context=None):
        """execute the agent on a query.

        args:
            query: the user's question (string)
            context: optional dict of extra context injected into the user message
                     (e.g. {"db_name": "wrestler", "question_id": "fb_03029"})

        returns:
            AgentResult with answer, steps, trace, token counts, llm_calls ledger
        """
        self._reset()

        system_prompt = build_react_system_prompt(
            self.tool_list,
            preamble=self.config.system_preamble,
            self_annotate=self.config.self_annotate,
            demonstration=self.config.demonstration,
        )

        # build user message with optional context
        user_msg = query
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            user_msg = f"{query}\n\nContext:\n{context_str}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        self._trace = list(messages)

        # main loop
        consecutive_failures = 0
        consecutive_finish_rejects = 0
        consecutive_empty_finish = 0
        _FINISH_REJECT_CAP = 3
        _EMPTY_FINISH_CAP = 3

        for step_num in range(self.config.max_steps):
            # call llm
            try:
                response = self._call_llm(messages)
            except Exception as e:
                logger.error(f"LLM call failed at step {step_num + 1}: {e}")
                return self._build_result(query, error=f"LLM call failed: {e}")

            llm_text = response.text

            # append assistant message
            assistant_msg = {"role": "assistant", "content": llm_text}
            messages.append(assistant_msg)
            self._trace.append(assistant_msg)

            # parse action
            tool_name, tool_args = self._parse_action(llm_text)

            if tool_name is None:
                consecutive_failures += 1
                if consecutive_failures > self.config.max_retries:
                    return self._build_result(
                        query,
                        error=f"action parse failed {consecutive_failures} times consecutively",
                        answer=self._last_output(),
                    )
                retry_msg = (
                    "Could not parse your action. Please respond with exactly:\n"
                    "Thought: <reasoning>\n"
                    "Action: <tool_name>(<param>=\"<value>\")"
                )
                messages.append({"role": "user", "content": retry_msg})
                self._trace.append({"role": "user", "content": retry_msg})
                continue

            consecutive_failures = 0

            # reject finish-as-first-action (prevents hallucinated answers)
            if tool_name == "finish" and len(self._steps) == 0 and step_num < self.config.max_steps - 1:
                consecutive_finish_rejects += 1
                if consecutive_finish_rejects >= _FINISH_REJECT_CAP:
                    logger.warning(
                        "LLM emitted finish() %d times without calling any tool; aborting",
                        consecutive_finish_rejects,
                    )
                    return self._build_result(
                        query,
                        error=f"agent attempted finish-first {consecutive_finish_rejects} times without calling a tool",
                        answer="",
                    )
                reject_msg = (
                    "REJECTED: you called finish() without retrieving any data. "
                    "You have NOT seen the documents, database contents, or web pages "
                    "needed to answer this question. Call a data-gathering tool first "
                    "(doc_search / get_schema / wikipedia_search / web_search), examine "
                    "its Observation, then call finish() with the verified result."
                )
                messages.append({"role": "user", "content": reject_msg})
                self._trace.append({"role": "user", "content": reject_msg})
                continue

            consecutive_finish_rejects = 0

            # reject an EMPTY finish answer: an empty/whitespace answer is not a
            # real answer, so re-prompt for the value instead of recording a
            # "successful" finish. bounded by _EMPTY_FINISH_CAP (and max_steps as
            # a backstop). a genuinely-empty result set is expressed as "[]" /
            # "none", never as an empty string, so this does not reject those.
            if tool_name == "finish" and not str(tool_args.get("answer", "")).strip():
                consecutive_empty_finish += 1
                if consecutive_empty_finish >= _EMPTY_FINISH_CAP:
                    logger.warning("LLM emitted empty finish() %d times; aborting",
                                   consecutive_empty_finish)
                    return self._build_result(
                        query,
                        error=f"agent called finish() with an empty answer {consecutive_empty_finish} times",
                        answer=self._last_output(),
                    )
                empty_msg = (
                    "REJECTED: finish() was called with an empty answer. Provide the "
                    "actual answer value derived from the Observations, e.g. "
                    "finish(answer=\"<the value>\")."
                )
                messages.append({"role": "user", "content": empty_msg})
                self._trace.append({"role": "user", "content": empty_msg})
                continue

            consecutive_empty_finish = 0

            # allocate step_id after rejection check
            step_id = self._next_step_id()
            thought = self._extract_thought(llm_text)
            goal = self._extract_goal(llm_text)

            # handle finish
            if tool_name == "finish":
                answer = tool_args.get("answer", "")
                result = {"status": "success", "output": answer, "output_type": "final"}
                self._record_step(step_id, tool_name, tool_args, result, thought=thought, goal=goal)
                return self._build_result(query, answer=answer, success=True)

            # execute tool
            result = self._execute_tool(tool_name, tool_args)
            obs_text = self._format_observation(result)
            self._record_step(step_id, tool_name, tool_args, result, thought=thought, goal=goal)

            # append observation
            obs_msg = {"role": "user", "content": obs_text}
            messages.append(obs_msg)
            self._trace.append(obs_msg)

        # max steps exhausted: salvage a best-effort answer from the Observations
        # (the agent often found/computed the answer but never called finish()).
        logger.warning(f"max steps ({self.config.max_steps}) reached without finish()")
        salvaged = self._salvage_answer(messages)
        return self._build_result(
            query,
            error=f"max steps ({self.config.max_steps}) exhausted without calling finish()",
            answer=salvaged or self._last_output(),
        )

    def _last_output(self):
        """return the last tool output as a fallback answer."""
        if self._steps:
            return str(self._steps[-1].result.get("output", ""))
        return ""

    def _build_result(self, query, answer="", success=False, error=""):
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
            error=error,
        )
