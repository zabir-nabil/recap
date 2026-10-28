"""
base agent: shared infrastructure for ReAct and Plan-and-Act.

handles:
  - tool registration and dispatch
  - action string parsing (robust regex-based)
  - trace recording and token tracking (per-call LLM ledger)
  - llm wrapper
"""

import re
import logging
from recap.agents import AgentConfig, StepRecord, AgentResult, LLMCallRecord

logger = logging.getLogger(__name__)


class BaseAgent:
    """base class for ReAct and Plan-and-Act agents."""

    def __init__(self, tools, config=None):
        self.config = (config or AgentConfig()).resolve()
        self.tools = {t.name: t for t in tools}
        self.tool_list = list(tools)

        # per-run state (reset on each run() call)
        self._step_counter = 0
        self._trace = []
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._llm_calls = []
        # count of AGENT-reasoning LLM calls (agent/planner/executor/replan) — the
        # single, shared step budget: both ReAct and Plan-and-Act are bounded to
        # config.max_steps of these. Tool-internal LLM calls (llm_analyze /
        # llm_reason) go through _absorb_tool_tokens.
        self._reasoning_call_count = 0

    def _reset(self):
        """reset per-run state (agent counters AND per-question tool state)."""
        self._step_counter = 0
        self._trace = []
        self._steps = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._llm_calls = []
        self._reasoning_call_count = 0
        # clear any mutable tool state so it never leaks across questions
        for tool in self.tool_list:
            try:
                tool.reset()
            except Exception as e:
                logger.warning("tool %s reset() failed: %s", getattr(tool, "name", "?"), e)

    # ------------------------------------------------------------------
    # llm wrapper
    # ------------------------------------------------------------------

    def _call_llm(self, messages, model=None, provider=None, role="agent"):
        """call the llm and track tokens.

        args:
            messages: chat messages list
            model: model name override (default: self.config.model)
            provider: provider override (default: self.config.provider)
            role: caller identity for cost ledger — one of:
                  "agent" (ReAct main loop),
                  "planner" (Plan-and-Act planning phase),
                  "executor" (Plan-and-Act execution phase),
                  "replan" (Plan-and-Act dynamic replanning)
        """
        from recap.llm import call_llm

        resolved_model = model or self.config.model
        resolved_provider = provider or self.config.provider

        response = call_llm(
            messages,
            model=resolved_model,
            provider=resolved_provider,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens
        self._total_reasoning_tokens += response.reasoning_tokens
        self._llm_calls.append(LLMCallRecord(
            role=role,
            model=resolved_model,
            provider=resolved_provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
        ))
        self._reasoning_call_count += 1
        return response

    def _salvage_answer(self, messages, model=None, provider=None):
        """best-effort final answer when the step budget is exhausted without a
        clean finish(). Makes ONE extra LLM call asking for the answer derivable
        from the Observations gathered so far, so a run that FOUND or COMPUTED the
        answer but never called finish() is not discarded (otherwise its raw last
        tool output — often a search dump — would be reported as the answer).

        Returns the salvaged answer string, or "" if none was produced (the caller
        then falls back to the last tool output). The extra call is recorded in the LLM ledger.
        """
        if not messages:
            return ""
        salvage_prompt = (
            "You have reached the step limit and can no longer call any tool. "
            "Using ONLY the Observations gathered above, give your single best "
            "final answer now. Respond with exactly one line:\n"
            'Action: finish(answer="<your final answer>")'
        )
        try:
            response = self._call_llm(
                messages + [{"role": "user", "content": salvage_prompt}],
                model=model, provider=provider, role="salvage",
            )
        except Exception as e:
            logger.warning("salvage-answer call failed: %s", e)
            return ""
        tool_name, tool_args = self._parse_action(response.text)
        if tool_name == "finish":
            return str((tool_args or {}).get("answer", "")).strip()
        return ""

    # ------------------------------------------------------------------
    # action parsing
    # ------------------------------------------------------------------

    # matches: Action: tool_name(param1="value1", param2="value2")
    _ACTION_RE = re.compile(
        r'Action:\s*(\w+)\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*$',
        re.MULTILINE
    )

    # matches individual key=value pairs. triple-quoted values come FIRST so a
    # multi-line code/text argument (e.g. python_exec(code=\"\"\"...\"\"\")) is
    # captured whole — otherwise the leading \"\"\" reads as an empty \"\" and the
    # code body's own `name = value` lines get mis-parsed as extra kwargs.
    # re.DOTALL lets the triple-quoted (.*?) span newlines.
    _PARAM_RE = re.compile(
        r'(\w+)\s*=\s*'
        r'(?:'
        r'"""(.*?)"""|'             # triple double-quoted (multi-line)
        r"'''(.*?)'''|"             # triple single-quoted (multi-line)
        r'"((?:[^"\\]|\\.)*)"|'     # double-quoted value
        r"'((?:[^'\\]|\\.)*)'|"     # single-quoted value
        r'([^,\s]+)'                # unquoted value
        r')',
        re.DOTALL,
    )

    @staticmethod
    def _param_value(pm):
        """extract a parameter value from a _PARAM_RE match, picking whichever
        quote style matched. triple-quoted content is taken verbatim (triple
        strings are literal); single/double-quoted content is unescaped."""
        if pm.group(2) is not None:            # triple double-quoted
            return pm.group(2)
        if pm.group(3) is not None:            # triple single-quoted
            return pm.group(3)
        if pm.group(4) is not None:            # double-quoted
            return pm.group(4).replace('\\"', '"').replace("\\'", "'")
        if pm.group(5) is not None:            # single-quoted
            return pm.group(5).replace('\\"', '"').replace("\\'", "'")
        return pm.group(6)                     # unquoted

    @staticmethod
    def _parse_params(params_str):
        """split a param string into {key: value}, respecting triple-quotes,
        quotes, and []/{} nesting so a structured value is kept WHOLE.

        Fixes the truncation where an unquoted JSON value — e.g.
        finish(answer=[{"id": 4, "type": "x"}]) — was cut at the first comma by
        the `[^,\\s]+` regex, corrupting SQL row-set and list answers."""
        args = {}
        i, n = 0, len(params_str)
        while i < n:
            while i < n and params_str[i] in ", \t\r\n":
                i += 1
            eq = params_str.find("=", i)
            if eq == -1:
                break
            key = params_str[i:eq].strip()
            j = eq + 1
            while j < n and params_str[j] in " \t\r\n":
                j += 1
            if j >= n:
                args[key] = ""
                break
            # triple-quoted value (verbatim)
            if params_str.startswith('"""', j) or params_str.startswith("'''", j):
                q = params_str[j:j + 3]
                end = params_str.find(q, j + 3)
                end = n if end == -1 else end
                args[key] = params_str[j + 3:end]
                i = end + 3
            elif params_str[j] in "\"'":            # single/double-quoted
                q = params_str[j]
                k, buf = j + 1, []
                while k < n:
                    ch = params_str[k]
                    if ch == "\\" and k + 1 < n:
                        buf.append(params_str[k + 1]); k += 2; continue
                    if ch == q:
                        k += 1; break
                    buf.append(ch); k += 1
                args[key] = "".join(buf)
                i = k
            elif params_str[j] in "[{":             # balanced bracketed value (kept whole)
                depth, instr, k = 0, None, j
                while k < n:
                    ch = params_str[k]
                    if instr:
                        if ch == "\\":
                            k += 2; continue
                        if ch == instr:
                            instr = None
                    elif ch in "\"'":
                        instr = ch
                    elif ch in "[{":
                        depth += 1
                    elif ch in "]}":
                        depth -= 1
                        if depth == 0:
                            k += 1; break
                    k += 1
                args[key] = params_str[j:k]
                i = k
            else:                                   # unquoted scalar up to top-level comma
                k = j
                while k < n and params_str[k] != ",":
                    k += 1
                args[key] = params_str[j:k].strip()
                i = k
        return args

    def _parse_action(self, text):
        """parse an Action: line from recap.llm output.

        returns (tool_name, args_dict) or (None, None) on failure.
        takes the LAST real Action: line (in case the model outputs multiple).
        """
        # anchor on a line-start "Action:"
        m = list(re.finditer(r'(?m)^[ \t]*Action:\s*(\w+)\s*\(', text))
        if not m:
            return self._parse_action_fallback(text)
        last = m[-1]
        tool_name = last.group(1)
        # take everything after the opening '(' up to the LAST ')' in the text
        open_paren = text.index("(", last.start())
        close_paren = text.rfind(")")
        if close_paren <= open_paren:
            return self._parse_action_fallback(text)
        params_str = text[open_paren + 1:close_paren].strip()
        if not params_str:
            return tool_name, {}
        return tool_name, self._parse_params(params_str)

    def _parse_action_fallback(self, text):
        """fallback parser: isolate the LAST line-start Action: and parse its
        balanced-paren argument span."""
        m = list(re.finditer(r'(?m)^[ \t]*Action:\s*(\w+)\s*\(', text))
        if not m:
            # no line-start action; last-resort rfind
            last_pos = text.rfind("Action:")
            if last_pos == -1:
                return None, None
            mt = re.search(r'Action:\s*(\w+)\((.*)\)\s*$', text[last_pos:], re.DOTALL)
            if not mt:
                return None, None
            tool_name, inner = mt.group(1), mt.group(2).strip()
            return tool_name, (self._parse_params(inner) if inner else {})
        last = m[-1]
        tool_name = last.group(1)
        open_paren = text.index("(", last.start())
        close_paren = text.rfind(")")
        if close_paren <= open_paren:
            return tool_name, {}
        inner = text[open_paren + 1:close_paren].strip()
        return tool_name, (self._parse_params(inner) if inner else {})

    # ------------------------------------------------------------------
    # tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name, args):
        """look up tool by name and call execute(**args).

        returns the tool's result dict {status, output, output_type}.
        """
        if tool_name not in self.tools:
            return {
                "status": "error",
                "output": f"unknown tool: {tool_name}. available: {list(self.tools.keys())}",
                "output_type": "text",
            }

        tool = self.tools[tool_name]
        try:
            result = tool.execute(**args)
            self._absorb_tool_tokens(result, tool_name=tool_name)
            return result
        except TypeError as e:
            expected = _get_tool_params_str(tool)
            return {
                "status": "error",
                "output": f"wrong arguments for {tool_name}({expected}): {e}",
                "output_type": "text",
            }
        except Exception as e:
            return {
                "status": "error",
                "output": f"tool execution error: {type(e).__name__}: {e}",
                "output_type": "text",
            }

    def _absorb_tool_tokens(self, result, tool_name=""):
        """absorb tool-internal LLM tokens into the agent's running counters
        and append to the per-call ledger. No-op for tools that don't call LLMs."""
        if not isinstance(result, dict):
            return
        tokens = result.get("tokens")
        if not isinstance(tokens, dict):
            return
        inp = int(tokens.get("input", 0) or 0)
        out = int(tokens.get("output", 0) or 0)
        reason = int(tokens.get("reasoning", 0) or 0)
        self._total_input_tokens += inp
        self._total_output_tokens += out
        self._total_reasoning_tokens += reason
        from recap import config as cfg
        self._llm_calls.append(LLMCallRecord(
            role=f"tool:{tool_name}" if tool_name else "tool:unknown",
            model=tokens.get("model", "") or cfg.SMALL_LLM_MODEL,
            provider=tokens.get("provider", "") or cfg.SMALL_LLM_PROVIDER,
            input_tokens=inp,
            output_tokens=out,
            reasoning_tokens=reason,
        ))

    # ------------------------------------------------------------------
    # observation formatting
    # ------------------------------------------------------------------

    def _format_observation(self, tool_result):
        """format a tool result as the Observation text shown to the agent."""
        raw_output = tool_result.get("output", "")
        output_str = str(raw_output)

        # A tool may request a larger cap when its supplied evidence is the
        # complete source document. Other tools keep the historical 2k cap.
        max_chars = int(tool_result.get("observation_max_chars", 2000))
        if len(output_str) > max_chars:
            output_str = output_str[:max_chars] + "\n... [truncated]"

        return f"Observation: {output_str}"

    # ------------------------------------------------------------------
    # step recording
    # ------------------------------------------------------------------

    def _next_step_id(self):
        """increment and return the next step id."""
        self._step_counter += 1
        return f"S{self._step_counter}"

    def _record_step(self, step_id, tool_name, tool_args, result, thought="", goal=""):
        """record a step in the trace."""
        record = StepRecord(
            step_id=step_id,
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
            thought=thought,
            goal=goal,
        )
        self._steps.append(record)
        return record

    # ------------------------------------------------------------------
    # thought extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_thought(text):
        """extract the Thought: content from recap.llm output."""
        match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # matches a single-line RECAP 'Goal:' annotation
    _GOAL_RE = re.compile(r'(?mi)^[ \t]*Goal:[ \t]*(.+?)[ \t]*$')

    @classmethod
    def _last_action_pos(cls, text):
        """offset of the 'Action:' the action parser will execute, or None.

        mirrors _parse_action / _parse_action_fallback, both of which take the
        LAST Action:, so goal extraction anchors on the same step.
        """
        matches = list(cls._ACTION_RE.finditer(text))
        if matches:
            return matches[-1].start()
        pos = text.rfind("Action:")
        return pos if pos != -1 else None

    @classmethod
    def _extract_goal(cls, text):
        """extract the RECAP 'Goal:' line belonging to the EXECUTED action.
        """
        matches = list(cls._GOAL_RE.finditer(text))
        if not matches:
            return ""
        action_pos = cls._last_action_pos(text)
        if action_pos is not None:
            before = [m for m in matches if m.start() < action_pos]
            if before:
                return before[-1].group(1).strip()
        return matches[-1].group(1).strip()


def _get_tool_params_str(tool):
    """helper: get parameter names as string for error messages."""
    import inspect
    try:
        sig = inspect.signature(tool.execute)
        params = [p.name for p in sig.parameters.values() if p.name != "self"]
        return ", ".join(params)
    except (ValueError, TypeError):
        return "**kwargs"
