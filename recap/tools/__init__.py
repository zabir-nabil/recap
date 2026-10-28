class BaseTool:
    name = ""
    description = ""
    is_deterministic = True
    llm_needed_output = False
    has_side_effect = False

    def execute(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        """reset any per-question mutable state. no-op by default.

        the agent calls this on every tool at the start of each run() so that
        state accumulated for one question (e.g. a filtered DataFrame) never
        leaks into the next question. stateless tools need not override it.
        """
        pass

    def __repr__(self):
        return f"<Tool: {self.name}>"
