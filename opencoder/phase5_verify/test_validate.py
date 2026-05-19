"""Phase V, Step 12: Test & Validate.

Executes the generated code against any provided unit tests in a
subprocess with a wall-clock timeout. Returns pass/fail + stdout/stderr.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class TestReport:
    passed: bool
    stdout: str
    stderr: str
    returncode: int


def run_tests(generated_code: str, test_code: str | None, timeout: int = 30) -> TestReport:
    if not test_code:
        return TestReport(passed=True, stdout="", stderr="(no tests provided)", returncode=0)
    with tempfile.TemporaryDirectory() as d:
        gen_path = os.path.join(d, "solution.py")
        test_path = os.path.join(d, "test_solution.py")
        with open(gen_path, "w") as f:
            f.write(generated_code)
        with open(test_path, "w") as f:
            f.write("import sys\nsys.path.insert(0, '.')\nfrom solution import *\n\n" + test_code)
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                cwd=d,
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return TestReport(
                passed=(r.returncode == 0),
                stdout=r.stdout,
                stderr=r.stderr,
                returncode=r.returncode,
            )
        except subprocess.TimeoutExpired:
            return TestReport(passed=False, stdout="", stderr="timeout", returncode=-1)
        except FileNotFoundError:
            # pytest not installed; fall back to plain exec.
            try:
                ns: dict = {}
                exec(compile(generated_code + "\n" + test_code, "gen", "exec"), ns)
                return TestReport(passed=True, stdout="exec-ok", stderr="", returncode=0)
            except Exception as e:
                return TestReport(passed=False, stdout="", stderr=repr(e), returncode=-1)
