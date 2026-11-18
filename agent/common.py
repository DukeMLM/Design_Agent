# common.py
"""Small helpers shared by the forward loop, the code modifier and the inverse design."""

import ast
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def has_header(path: str) -> bool:
    """
    True when the first row of a CSV file is a header and not data.

    The raw dataset files have no header. The files written to ./input by
    generate_dataset() have the header pandas writes ("0,1,2,...").
    """
    with open(path, "r") as f:
        first = f.readline().strip().split(",")
    try:
        values = [float(v) for v in first]
    except ValueError:
        return True
    return len(values) > 1 and values == [float(i) for i in range(len(values))]


def read_table(path: str) -> np.ndarray:
    """Read a numeric CSV file, with or without a header row, as a float32 array."""
    header = 0 if has_header(path) else None
    return pd.read_csv(path, header=header).values.astype("float32")


def strip_code_fences(text: str) -> str:
    """Remove a Markdown code fence around a script returned by an LLM."""
    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```$", text, flags=re.DOTALL)
    return match.group(1) if match else text


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_script(code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Run `code` as a Python script from the current directory and capture its output.

    Returns {"ok", "returncode", "stdout", "stderr"}. "ok" is True only when the
    script exits with code 0 inside the time limit.
    """
    # AIDE-generated scripts save their predictions to ./working/submission.csv
    os.makedirs("./working", exist_ok=True)

    tf = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tf.write(code)
        tf.close()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, tf.name],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                check=False,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            returncode = -1
            stdout = _as_text(e.stdout)
            stderr = _as_text(e.stderr) + f"\nTimeoutError: the script ran longer than {timeout} s"
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass

    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def last_line(text: str, default: str = "") -> str:
    """Last non-empty line of `text` (the error line of a traceback)."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return default


def check_script_structure(code: str) -> Optional[str]:
    """
    Check, without running it, that a refactored training script can be imported
    by the inverse design. Returns a description of the problem, or None.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno}). Return plain Python source only."

    has_class = any(
        isinstance(node, ast.ClassDef) and node.name == "forward_model" for node in tree.body
    )
    if not has_class:
        return "The script must define a class named `forward_model` at module level."

    def is_main_guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            return False
        names = [node.test.left] + list(node.test.comparators)
        has_name = any(isinstance(n, ast.Name) and n.id == "__name__" for n in names)
        has_main = any(isinstance(n, ast.Constant) and n.value == "__main__" for n in names)
        return has_name and has_main

    if not any(is_main_guard(node) for node in tree.body):
        return (
            'Data loading, training, saving and printing must be inside `if __name__ == "__main__":` '
            "so that `from modified import forward_model` does not start a training run."
        )
    return None
