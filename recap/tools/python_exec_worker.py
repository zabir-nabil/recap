"""Isolated worker for the benchmark's pure-computation python_exec tool.

This is a restricted execution policy, not an operating-system security jail.
The parent launches it with an empty environment and a disposable working
directory, validates code before launch, and enforces a process timeout.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import sys


ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "statistics", "decimal", "fractions", "random",
    "itertools", "functools", "collections", "collections.abc", "re",
    "datetime", "heapq", "bisect", "json", "numbers",
    "unicodedata",
})
ALLOWED_BUILTINS = frozenset({
    "bool", "int", "float", "complex", "str", "bytes", "bytearray",
    "list", "tuple", "dict", "set", "frozenset", "slice", "range",
    "len", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "iter", "next", "all", "any", "sum", "min", "max", "abs", "round",
    "pow", "divmod", "repr", "format", "hash", "ord", "chr", "hex",
    "oct", "bin", "ascii", "print", "Exception", "ValueError",
    "TypeError", "KeyError", "IndexError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "ArithmeticError", "OverflowError",
    "AssertionError", "NotImplementedError", "LookupError", "NameError",
    "FloatingPointError", "RecursionError", "MemoryError",
})
FORBIDDEN_ATTRIBUTES = frozenset({"format", "format_map", "mro"})
MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 16_000


def validate_code(code: str) -> str | None:
    if not isinstance(code, str) or len(code) > MAX_CODE_CHARS:
        return "code must be text of at most 20000 characters"
    if "__" in code:
        return "double underscores are not allowed in python_exec"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                return f"attribute {node.attr!r} is not allowed in python_exec"
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return "private names are not allowed in python_exec"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names]
                     if isinstance(node, ast.Import) else [node.module or ""])
            if isinstance(node, ast.ImportFrom) and node.level:
                return "relative imports are not allowed in python_exec"
            if any(name not in ALLOWED_IMPORTS for name in names):
                return "module is not allowed in python_exec"
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global and nonlocal declarations are not allowed in python_exec"
    return None


def safe_builtins() -> dict:
    output = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in ALLOWED_IMPORTS:
            raise ImportError("module is not allowed in python_exec")
        return builtins.__import__(name, globals, locals, fromlist, level)

    output["__import__"] = safe_import
    return output


class LimitedOutput(io.StringIO):
    def write(self, text):
        remaining = MAX_OUTPUT_CHARS - self.tell()
        if remaining > 0:
            super().write(str(text)[:remaining])
        return len(text)


def run(code: str) -> dict:
    problem = validate_code(code)
    if problem:
        return {"status": "error", "output": problem, "output_type": "text"}
    capture = LimitedOutput()
    sandbox = {"__builtins__": safe_builtins()}
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(compile(code, "<python_exec>", "exec"), sandbox)
        output = capture.getvalue().strip()
        if not output:
            output = str(sandbox.get("result", "executed successfully"))[:MAX_OUTPUT_CHARS]
        return {"status": "success", "output": output, "output_type": "text"}
    except BaseException as exc:
        return {"status": "error", "output": f"{type(exc).__name__}: {exc}"[:MAX_OUTPUT_CHARS],
                "output_type": "text"}


def main() -> None:
    code = sys.stdin.read(MAX_CODE_CHARS + 1)
    print(json.dumps(run(code), ensure_ascii=False))


if __name__ == "__main__":
    main()
