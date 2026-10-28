import os
import math
import signal
import threading
from recap.tools import BaseTool


class _CalcTimeout(BaseException):
    """raised by the calculator wall-clock alarm; subclasses BaseException so a
    stray `except Exception` inside eval'd content cannot swallow it."""


class Calculator(BaseTool):
    name = "calculator"
    description = (
        "evaluate a math expression. "
        "supports: +, -, *, /, **, %, round(), abs(), min(), max(), sqrt(). "
        "example: round(23485 / 23237, 2)"
    )

    SAFE_NAMES = {
        "round": round, "abs": abs, "min": min, "max": max,
        "int": int, "float": float, "sum": sum,
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "pi": math.pi, "e": math.e,
    }

    def execute(self, expression):
        # block attribute traversal attacks and code constructs
        import re
        if '__' in expression:
            return {"status": "error", "output": "double underscores not allowed in expressions", "output_type": "text"}
        if re.search(r'\blambda\b', expression):
            return {"status": "error", "output": "lambda not allowed in expressions", "output_type": "text"}

        # wall-clock timeout: eval() advertises `**`, so a runaway exponentiation
        # like `9**9**9` computes a ~369M-digit int and hangs the (single-threaded)
        # benchmark run forever with no backstop. Mirror python_exec's SIGALRM
        # guard (main-thread only; the runners dispatch tools on the main thread).
        # Override with CALC_TIMEOUT_S (default 5s; <=0 disables).
        try:
            timeout_s = int(float(os.environ.get("CALC_TIMEOUT_S", "5")))
        except ValueError:
            timeout_s = 5
        use_alarm = (
            timeout_s > 0
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        old_handler = None

        def _on_timeout(signum, frame):
            raise _CalcTimeout()

        try:
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, _on_timeout)
                signal.alarm(timeout_s)
            result = eval(expression, {"__builtins__": {}}, self.SAFE_NAMES)
            return {"status": "success", "output": result, "output_type": "numeric"}
        except _CalcTimeout:
            return {"status": "error",
                    "output": f"calculation timed out after {timeout_s}s", "output_type": "text"}
        except Exception as e:
            return {"status": "error", "output": str(e), "output_type": "text"}
        finally:
            if use_alarm:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)


class Finish(BaseTool):
    name = "finish"
    description = "return the final answer to the user"

    def execute(self, answer):
        return {"status": "success", "output": answer, "output_type": "final"}
