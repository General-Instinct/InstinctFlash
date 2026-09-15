#!/usr/bin/env python3
"""Run script-style and pytest tests in separate subprocesses.

    python tests/run_tests.py

Two conventions coexist in this directory and both are supported:

  * script-style — the file does its work under `if __name__ == "__main__"` and reports by
    exit code. This is what most of the suite uses.
  * pytest-style — pytest collects functions, fixtures and test classes. This includes
    files whose legacy `__main__` only calls `run_module_tests(globals())`.

Every file is run as a SUBPROCESS. That is not incidental: most of this suite needs the
upstream lingbot-va or cosmos-framework trees, and importing those in-process means one
missing checkout takes down the whole run instead of skipping one file.

Known optional model dependencies may SKIP. Missing core dependencies, local adapters,
unknown imports and pytest itself FAIL. Pytest must report actual tests or explicit skips;
merely importing a file successfully is never enough to pass it.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from xml.etree import ElementTree

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MISSING_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'$")
# Exact top-level names only: torch.some_missing_api is an incompatible installation,
# not an absent optional dependency. Keep local source (including *_iwm) out of this set.
OPTIONAL_MODULES = frozenset({
    "PIL", "accelerate", "cosmos_framework", "cv2", "diffusers", "easydict",
    "einops", "flash_attn", "flash_rt", "ftfy", "gr00t", "groot", "hydra",
    "imageio", "instinct_pdd", "lerobot", "libero", "modules", "msgpack",
    "numpy", "omegaconf", "openpi_client", "rerun", "safetensors", "scipy",
    "torch", "torchvision", "transformers", "triton", "utils", "websockets",
})


def _main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    pair = (test.left, test.comparators[0])
    return any(
        isinstance(name, ast.Name) and name.id == "__name__"
        and isinstance(value, ast.Constant) and value.value == "__main__"
        for name, value in (pair, pair[::-1])
    )


def execution_mode(path: Path) -> str:
    """Inspect source without importing model stacks or executing test code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guards = [node for node in tree.body if _main_guard(node)]
    if not guards:
        return "pytest"
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test_")]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name.startswith("Test")]
    for guard in guards:
        calls = [node.func for node in ast.walk(guard) if isinstance(node, ast.Call)]
        if any((isinstance(fn, ast.Name) and fn.id == "run_module_tests")
               or (isinstance(fn, ast.Attribute) and fn.attr == "run_module_tests") for fn in calls):
            return "pytest"
        # Some older files inline the same globals()/test_* loop instead of
        # importing run_module_tests. Its text summary loses per-test errors.
        if functions and any(isinstance(fn, ast.Name) and fn.id == "globals" for fn in calls):
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "startswith" and node.args
                   and isinstance(node.args[0], ast.Constant) and node.args[0].value == "test_"
                   for node in ast.walk(guard)):
                return "pytest"
        # An early main misses tests defined later in the module. Pytest also
        # gives standard unittest suites per-case skip/error results.
        if any(node.lineno > guard.lineno for node in functions + classes):
            return "pytest"
        if any(isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
               and fn.value.id == "unittest" and fn.attr == "main" for fn in calls):
            return "pytest"
    if any(fn.decorator_list or len(fn.args.posonlyargs) + len(fn.args.args) > len(fn.args.defaults)
           or any(default is None for default in fn.args.kw_defaults) for fn in functions):
        return "pytest"
    return "script"


def run_module_tests(namespace: dict) -> int:
    """Run every `test_*` callable in `namespace`. Returns a process exit code.

    Used by the function-style files so they behave like the script-style ones when executed
    directly, while staying collectable by pytest.
    """
    failures = 0
    executed = 0
    for name in sorted(namespace):
        if not name.startswith("test_"):
            continue
        fn = namespace[name]
        if not callable(fn):
            continue
        executed += 1
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print(f"FAIL {name}")
            failures += 1
        else:
            print(f"ok   {name}")
    if not executed:
        print("FAIL no test functions found")
    return 1 if failures or not executed else 0


def _missing_module(message: str) -> str | None:
    """Only accept the exception itself, never text embedded in an assertion."""
    lines = message.strip().splitlines()
    if not lines:
        return None
    line = re.sub(r"^E\s+", "", lines[-1].strip())
    match = MISSING_RE.fullmatch(line)
    return match.group(1) if match else None


def _failure_detail(proc: subprocess.CompletedProcess) -> str:
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return tail[-1] if tail else f"exit {proc.returncode}"


def _classify_pytest(proc: subprocess.CompletedProcess, report: Path) -> tuple[str, str]:
    try:
        cases = ElementTree.parse(report).getroot().findall(".//testcase")
    except (OSError, ElementTree.ParseError):
        return "FAIL", f"pytest did not produce a test report: {_failure_detail(proc)}"
    passed = skipped = 0
    optional = set()
    failures = []
    for case in cases:
        problems = list(case.findall("failure")) + list(case.findall("error"))
        if problems:
            for problem in problems:
                missing = (_missing_module(problem.get("message", ""))
                           or _missing_module(problem.text or ""))
                if missing in OPTIONAL_MODULES:
                    optional.add(missing)
                else:
                    failures.append(f"cannot import {missing}" if missing
                                    else problem.get("message") or "test error")
        elif case.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    if failures:
        return "FAIL", failures[0].splitlines()[0]
    # Exit 2 can be a missing optional import at collection. Other interruptions,
    # usage/internal errors and missing pytest are never a successful test run.
    if proc.returncode not in (0, 1, 2, 5) or (proc.returncode in (1, 2) and not optional):
        return "FAIL", _failure_detail(proc)
    if optional:
        return "SKIP", f"{passed} passed; needs {', '.join(sorted(optional))}"
    if not cases or (proc.returncode == 5 and not skipped):
        return "FAIL", "pytest collected no tests"
    if not passed:
        return "SKIP", f"{skipped} tests skipped"
    return "PASS", f"{passed} tests passed" + (f", {skipped} skipped" if skipped else "")


def classify(proc: subprocess.CompletedProcess, *, pytest_report: Path | None = None) -> tuple[str, str]:
    """-> (verdict, detail)."""
    if pytest_report is not None:
        return _classify_pytest(proc, pytest_report)
    if proc.returncode == 0:
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if lines and all(re.match(r"^SKIP(?=$|[:\s])", line) for line in lines):
            reasons = [re.sub(r"^SKIP[:\s]*", "", line) for line in lines]
            return "SKIP", "; ".join(reasons) or "script reported skip"
        return "PASS", ""
    missing = _missing_module(proc.stderr or "")
    if missing:
        if missing in OPTIONAL_MODULES and (proc.stderr or "").count("Traceback (most recent call last):") <= 1:
            return "SKIP", f"needs {missing}"
        return "FAIL", f"cannot import {missing}"
    detail = _failure_detail(proc)
    if detail.startswith(("Skipped: ", "unittest.case.SkipTest: ")):
        return "SKIP", detail
    return "FAIL", detail


def run_test(path: Path, *, timeout: float = 600) -> tuple[str, str, subprocess.CompletedProcess]:
    """Run one file in isolation, keeping the report outside the checkout."""
    path = path.resolve()
    try:
        mode = execution_mode(path)
    except (OSError, SyntaxError) as exc:
        proc = subprocess.CompletedProcess([], 1, "", str(exc))
        return "FAIL", str(exc), proc
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # A directly executed script starts sys.path in tests/, whereas `-m pytest`
    # starts in the checkout. Both modes must be able to import local source.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), str(TESTS_DIR), env.get("PYTHONPATH"))))
    with tempfile.TemporaryDirectory(prefix="instinctflash-tests-") as tmp:
        report = Path(tmp) / "result.xml" if mode == "pytest" else None
        if report is None:
            command = [sys.executable, "-B", str(path)]
        else:
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            # Ambient selection/exit options must not make this file disappear.
            env.pop("PYTEST_ADDOPTS", None)
            command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--junitxml", str(report), str(path)]
        try:
            proc = subprocess.run(command, cwd=REPO_ROOT, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            proc = subprocess.CompletedProcess(command, 124, decoded(exc.stdout), decoded(exc.stderr))
            return "FAIL", f"timed out after {timeout:g}s", proc
        verdict, detail = classify(proc, pytest_report=report)
        return verdict, detail, proc


def main() -> int:
    results = []
    paths = sorted(TESTS_DIR.glob("test_*.py"))
    if not paths:
        print(f"FAIL no test files found in {TESTS_DIR}")
        return 1
    for path in paths:
        verdict, detail, proc = run_test(path)
        results.append((path.name, verdict, detail, proc))
        print(f"{verdict:5s} {path.name:34s} {detail}")

    failed = [r for r in results if r[1] == "FAIL"]
    for name, _, _, proc in failed:
        print("\n" + "=" * 72)
        print(name)
        print((proc.stdout or "") + (proc.stderr or ""))

    passed = sum(1 for r in results if r[1] == "PASS")
    skipped = sum(1 for r in results if r[1] == "SKIP")
    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
