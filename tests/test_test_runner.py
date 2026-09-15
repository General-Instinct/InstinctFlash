"""Exercise suite dispatch and verdicts with tiny, isolated CPU-only test files."""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from tests import run_tests as runner


def write_module(tmp_path: Path, source: str, name: str = "test_example.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_function_failure_is_executed(tmp_path):
    path = write_module(tmp_path, """
        def test_failure():
            assert False, 'the function actually ran'
    """)
    verdict, detail, proc = runner.run_test(path)
    assert verdict == "FAIL"
    assert "the function actually ran" in detail
    assert "1 failed" in proc.stdout


def test_fixtures_and_all_parameter_cases_execute(tmp_path):
    marker = tmp_path / "executed.txt"
    path = write_module(tmp_path, f"""
        from pathlib import Path
        import pytest

        @pytest.fixture
        def number():
            return 10

        @pytest.mark.parametrize('value', [1, 2])
        def test_parameters(number, value, tmp_path):
            assert tmp_path.is_dir()
            with Path({str(marker)!r}).open('a') as stream:
                stream.write(str(number + value) + '\\n')
    """)
    verdict, detail, _ = runner.run_test(path)
    assert (verdict, detail) == ("PASS", "2 tests passed")
    assert marker.read_text().splitlines() == ["11", "12"]


def test_generic_main_does_not_hide_later_fixture_tests(tmp_path):
    path = write_module(tmp_path, """
        def test_first():
            pass

        if __name__ == '__main__':
            from tests.run_tests import run_module_tests
            raise SystemExit(run_module_tests(globals()))

        def test_later(tmp_path):
            assert tmp_path.is_dir()
            assert False, 'later fixture test ran'
    """)
    verdict, detail, proc = runner.run_test(path)
    assert verdict == "FAIL" and "later fixture test ran" in detail
    assert "1 failed, 1 passed" in proc.stdout


@pytest.mark.parametrize("body, verdict", [
    ("raise ModuleNotFoundError(\"No module named 'torch'\")", "SKIP"),
    ("assert False, 'real regression'", "FAIL"),
])
def test_inline_function_harness_keeps_per_case_results(tmp_path, body, verdict):
    path = write_module(tmp_path, f"""
        def test_ok():
            pass
        def test_other():
            {body}
        if __name__ == '__main__':
            failures = 0
            for name, fn in sorted(globals().items()):
                if name.startswith('test_') and callable(fn):
                    try:
                        fn()
                    except Exception:
                        failures += 1
            raise SystemExit(bool(failures))
    """)
    assert runner.run_test(path)[0] == verdict


def test_mixed_unittest_main_collects_free_functions(tmp_path):
    path = write_module(tmp_path, """
        import unittest
        class TestLegacy(unittest.TestCase):
            def test_ok(self):
                self.assertTrue(True)

        if __name__ == '__main__':
            unittest.main()

        def test_extra(tmp_path):
            assert False, 'free function ran'
    """)
    verdict, detail, proc = runner.run_test(path)
    assert verdict == "FAIL" and "free function ran" in detail
    assert "1 failed, 1 passed" in proc.stdout


@pytest.mark.parametrize("body, expected", [
    ("raise ModuleNotFoundError(\"No module named 'torch'\")", ("SKIP", "1 passed; needs torch")),
    ("self.fail('real unittest regression')", ("FAIL", "AssertionError: real unittest regression")),
])
def test_unittest_cases_have_individual_results(tmp_path, body, expected):
    path = write_module(tmp_path, f"""
        import unittest
        class LegacyChecks(unittest.TestCase):
            def test_ok(self):
                self.assertTrue(True)
            def test_other(self):
                {body}
        if __name__ == '__main__':
            unittest.main()
    """)
    assert runner.run_test(path)[:2] == expected


def test_custom_script_main_and_process_isolation_are_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv("INSTINCTFLASH_RUNNER_ISOLATION_TEST", raising=False)
    first = write_module(tmp_path, """
        import os
        def test_helper():
            pass
        def main():
            os.environ['INSTINCTFLASH_RUNNER_ISOLATION_TEST'] = 'child only'
            print('custom script check ran')
            return 7
        if '__main__' == __name__:
            raise SystemExit(main())
    """, "test_first.py")
    second = write_module(tmp_path, """
        import os
        def test_isolation():
            assert 'INSTINCTFLASH_RUNNER_ISOLATION_TEST' not in os.environ
    """, "test_second.py")
    verdict, _, proc = runner.run_test(first)
    assert verdict == "FAIL" and proc.returncode == 7
    assert proc.stdout.strip() == "custom script check ran"
    assert runner.run_test(second)[0] == "PASS"


def test_script_can_import_checkout_source_without_an_editable_install(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "local_source.py").write_text("VALUE = 42\n")
    test_dir = checkout / "tests"
    test_dir.mkdir()
    path = write_module(test_dir, """
        from local_source import VALUE
        if __name__ == '__main__':
            assert VALUE == 42
    """)
    monkeypatch.setattr(runner, "REPO_ROOT", checkout)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert runner.run_test(path)[0] == "PASS"


def test_installed_tests_package_cannot_shadow_repository_tests(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    shadow = installed / "tests"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise AssertionError('external tests package imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(installed))
    path = write_module(tmp_path, f"""
        from pathlib import Path
        from tests import run_tests
        if __name__ == '__main__':
            assert Path(run_tests.__file__).resolve() == Path({str(Path(runner.__file__).resolve())!r})
    """)
    verdict, _, proc = runner.run_test(path)
    assert verdict == "PASS", proc.stdout + proc.stderr


def test_empty_module_is_not_a_pass(tmp_path):
    path = write_module(tmp_path, "VALUE = 42\n")
    verdict, detail, _ = runner.run_test(path)
    assert verdict == "FAIL" and "no tests" in detail
    assert runner.run_module_tests({"VALUE": 42}) == 1


@pytest.mark.parametrize("source", [
    "import pytest\npytest.skip('optional environment', allow_module_level=True)\n",
    "import pytest\n@pytest.mark.skip(reason='optional environment')\ndef test_skipped(): pass\n",
    "import pytest\nif __name__ == '__main__':\n    pytest.skip('optional environment')\n",
    "import unittest\n@unittest.skip('optional environment')\nclass Checks(unittest.TestCase):\n    def test_skipped(self): pass\nif __name__ == '__main__': unittest.main()\n",
])
def test_explicit_skips_are_not_passes(tmp_path, source):
    path = write_module(tmp_path, source)
    assert runner.run_test(path)[0] == "SKIP"


@pytest.mark.parametrize("output, code, expected", [
    ("SKIP: needs CUDA", 0, "SKIP"),
    ("SKIP: needs CUDA\nSKIP optional kernel", 0, "SKIP"),
    ("CPU metadata passed\nSKIP: GPU kernel", 0, "PASS"),
    ("SKIP: needs CUDA", 1, "FAIL"),
])
def test_script_skip_exit_code_convention(tmp_path, output, code, expected):
    path = write_module(tmp_path, f"""
        if __name__ == '__main__':
            print({output!r})
            raise SystemExit({code})
    """)
    assert runner.run_test(path)[0] == expected


@pytest.mark.parametrize("source, passed", [
    ("raise ModuleNotFoundError(\"No module named 'torch'\")\n", 0),
    ("def test_ok(): pass\ndef test_optional():\n    raise ModuleNotFoundError(\"No module named 'torch'\")\n", 1),
])
def test_only_optional_import_errors_can_skip(tmp_path, source, passed):
    path = write_module(tmp_path, source)
    verdict, detail, _ = runner.run_test(path)
    assert verdict == "SKIP"
    assert f"{passed} passed" in detail and "needs torch" in detail


def test_optional_dependency_failure_does_not_hide_assertion_failure(tmp_path):
    path = write_module(tmp_path, """
        def test_optional():
            raise ModuleNotFoundError("No module named 'torch'")
        def test_broken():
            assert False, 'real regression'
    """)
    verdict, detail, _ = runner.run_test(path)
    assert verdict == "FAIL" and "real regression" in detail


@pytest.mark.parametrize("module", [
    "instinctflash", "instinctflash.runtime.missing", "dreamzero_iwm", "cosmos3_iwm",
    "pi05_iwm", "groot_n17_iwm", "lingbot_vla_iwm", "lingbot_vla_v2_iwm",
    "benchmarks", "huggingface_hub", "yaml", "pytest", "torch.missing_api", "typo_dependency",
])
def test_missing_core_adapter_tooling_or_unknown_import_fails(module):
    proc = subprocess.CompletedProcess([], 1, "", f"ModuleNotFoundError: No module named '{module}'\n")
    assert runner.classify(proc)[0] == "FAIL"


def test_pytest_collection_error_in_adapter_is_failure(tmp_path):
    path = write_module(tmp_path, "raise ModuleNotFoundError(\"No module named 'dreamzero_iwm'\")\n")
    assert runner.run_test(path)[0] == "FAIL"


def test_assertion_text_cannot_impersonate_optional_import(tmp_path):
    path = write_module(tmp_path, """
        def test_failure():
            assert False, "ModuleNotFoundError: No module named 'torch'"
    """)
    assert runner.run_test(path)[0] == "FAIL"
    proc = subprocess.CompletedProcess([], 1, "", "AssertionError: ModuleNotFoundError: No module named 'torch'\n")
    assert runner.classify(proc)[0] == "FAIL"


def test_missing_pytest_or_test_report_is_failure(tmp_path):
    for code, error in [(1, '/python: No module named pytest'), (0, '')]:
        proc = subprocess.CompletedProcess([], code, "", error)
        assert runner.classify(proc, pytest_report=tmp_path / "missing.xml")[0] == "FAIL"


def test_ambient_pytest_selection_does_not_omit_tests(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nonexistent_test_name")
    path = write_module(tmp_path, "def test_runs(): pass\n")
    assert runner.run_test(path)[:2] == ("PASS", "1 tests passed")


def test_timeout_is_a_file_failure(tmp_path):
    path = write_module(tmp_path, """
        if __name__ == '__main__':
            import time
            time.sleep(30)
    """)
    verdict, detail, _ = runner.run_test(path, timeout=0.1)
    assert verdict == "FAIL" and "timed out" in detail


def test_suite_continues_after_syntax_failure(tmp_path, monkeypatch, capsys):
    write_module(tmp_path, "def test_broken(: pass\n", "test_first.py")
    write_module(tmp_path, "def test_ok(): pass\n", "test_second.py")
    monkeypatch.setattr(runner, "TESTS_DIR", tmp_path)
    assert runner.main() == 1
    output = capsys.readouterr().out
    assert "1 passed, 0 skipped, 1 failed" in output
    assert "test_second.py" in output


def test_empty_suite_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TESTS_DIR", tmp_path)
    assert runner.main() == 1
