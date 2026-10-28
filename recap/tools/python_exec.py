"""Pure-computation Python tool with the original isolated worker policy."""

import json
import os
import subprocess
import sys
import tempfile

from recap.tools import BaseTool


def _run_python_exec_isolated(code):
    from recap.tools.python_exec_worker import validate_code

    problem = validate_code(code)
    if problem:
        return {"status": "error", "output": problem, "output_type": "text"}
    try:
        timeout_s = max(1, int(float(os.environ.get("PYTHON_EXEC_TIMEOUT_S", "15"))))
    except ValueError:
        timeout_s = 15
    worker = os.path.join(os.path.dirname(__file__), "python_exec_worker.py")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="python_exec_") as tmpdir:
        try:
            child = subprocess.run(
                [sys.executable, "-I", "-S", worker], input=code,
                text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=tmpdir, env={}, timeout=timeout_s,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error",
                    "output": f"execution timed out after {timeout_s}s",
                    "output_type": "text"}
        except OSError as exc:
            return {"status": "error", "output": f"worker launch failed: {exc}",
                    "output_type": "text"}
    if child.returncode:
        return {"status": "error", "output": "python_exec worker failed",
                "output_type": "text"}
    try:
        return json.loads(child.stdout)
    except (ValueError, TypeError):
        return {"status": "error", "output": "python_exec worker returned invalid data",
                "output_type": "text"}


class PythonInterpreter(BaseTool):
    name = "python_exec"
    description = "execute Python for pure computation only and return stdout output."
    is_deterministic = False

    def execute(self, code):
        return _run_python_exec_isolated(code)
