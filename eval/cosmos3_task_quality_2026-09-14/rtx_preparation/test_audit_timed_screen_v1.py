"""CPU-only adversarial checks for independent SCREEN counting and read-only guards."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from instinctflash.verify.certify import ONE_SIDED_95_Z, _tango_paired_score_bounds  # noqa: E402

spec = importlib.util.spec_from_file_location("screen_audit_under_test", HERE / "audit_timed_screen_v1.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
API = SimpleNamespace(ONE_SIDED_95_Z=ONE_SIDED_95_Z, _tango_paired_score_bounds=_tango_paired_score_bounds)


class ScreenAuditTests(unittest.TestCase):
    def setUp(self):
        self.plan = audit.read(HERE / "fixed_smoke_screen_plan_v1.json")
        self.rows = [{"family": family, "arm": arm, "pair_id": episode["pair_id"],
                      "task_id": episode["task_id"], "success": False}
                     for family in audit.FAMILIES for arm in audit.ARMS for episode in self.plan["selection"]["episodes"]]

    def test_all_failures_do_not_certify(self):
        report = audit.statistics(self.rows, "edge", self.plan, API, complete=True)
        self.assertEqual(report["paired_counts"]["both_failure"], 6)
        self.assertAlmostEqual(report["lower_confidence_bound"], -0.3107839822769165)
        self.assertFalse(report["fixed_family_gate_passed"])
        self.assertFalse(report["task_quality_validated"])
        self.assertIsNone(report["certificate"])

    def test_even_large_gain_remains_screen(self):
        for row in self.rows:
            row["success"] = row["arm"] == "candidate"
        report = audit.statistics(self.rows, "edge", self.plan, API, complete=True)
        self.assertTrue(report["fixed_family_gate_passed"])
        self.assertFalse(report["task_quality_validated"])
        self.assertIsNone(report["certificate"])

    def test_partial_has_no_completed_ci(self):
        report = audit.statistics(self.rows[:7], "edge", self.plan, API)
        self.assertEqual(report["absolute_counts"]["baseline"]["completed"], 6)
        self.assertEqual(report["matched_pairs"], 1)
        self.assertFalse(report["fixed_family_gate_evaluated"])
        self.assertIsNone(report["lower_confidence_bound"])
        with self.assertRaises(ValueError):
            audit.statistics(self.rows[:7], "edge", self.plan, API, complete=True)

    def test_matched_absolute_and_per_task_delta(self):
        self.rows[0]["success"] = True
        self.rows[7]["success"] = True
        self.rows[9]["success"] = True
        report = audit.statistics(self.rows[:12], "edge", self.plan, API, complete=True)
        self.assertEqual(report["paired_counts"], {"both_success": 0, "baseline_only_success": 1,
                                                  "candidate_only_success": 2, "both_failure": 3})
        self.assertAlmostEqual(report["matched_candidate_minus_baseline"], 1 / 6)
        self.assertEqual(report["per_task"]["BananaInBowlTask"]["matched_candidate_minus_baseline"], 0)
        self.assertAlmostEqual(report["per_task"]["RubiksCubeAndBananaTask"]["matched_candidate_minus_baseline"], 1 / 3)

    def test_family_does_not_wait_on_nano(self):
        report = audit.statistics(self.rows[:12], "edge", self.plan, API, complete=True)
        self.assertEqual(report["matched_pairs"], 6)

    def test_boolean_outcomes_only(self):
        with self.assertRaises(ValueError):
            audit.count_pairs([(0, False)])

    def test_csv_has_per_task_and_no_partial_ci(self):
        report = audit.statistics(self.rows[:7], "edge", self.plan, API)
        report["audit_status"] = "PARTIAL"
        import csv
        import io
        rows = list(csv.DictReader(io.StringIO(audit.csv_text([report]))))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["display_label"] == "SMOKE SCREEN" for row in rows))
        self.assertTrue(all(row["family_lower_one_sided95_pp"] == "" for row in rows))

    def test_readonly_anchor_never_writes_and_restores(self):
        calls = []
        formal = SimpleNamespace(durable=lambda *args, **kwargs: calls.append(args))
        original = formal.durable
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "anchor.json"
            path.write_text('{"x": 1}')
            before = path.read_bytes(), path.stat().st_mtime_ns
            with audit.readonly_anchor(formal, path):
                formal.durable(path, {"x": 1})
                with self.assertRaises(ValueError):
                    formal.durable(path, {"x": 2})
                with self.assertRaises(ValueError):
                    formal.durable(path.with_name("new.json"), {"x": 1})
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        self.assertIs(formal.durable, original)
        self.assertEqual(calls, [])

    def test_actual_spec_only_allows_smaller_timeout(self):
        expected = {"max_seconds": 100, "command": ["native"], "output": "/fixed"}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "spec.json"
            for value, passed in ((dict(expected, max_seconds=70), True), (dict(expected, max_seconds=101), False),
                                  (dict(expected, command=["changed"]), False), (dict(expected, max_seconds=True), False)):
                path.write_text(json.dumps(value))
                if passed:
                    self.assertEqual(audit.actual_spec(path, expected), value)
                else:
                    with self.assertRaises(ValueError):
                        audit.actual_spec(path, expected)

    def test_write_once_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "receipt.json"
            audit.write_once(path, {"one": 1})
            with self.assertRaises(FileExistsError):
                audit.write_once(path, {"two": 2})
            self.assertEqual(audit.read(path), {"one": 1})

    def test_server_budget_is_exactly_coupled_to_outer_timeout(self):
        expected = {"max_seconds": 14520, "grace_seconds": 120,
                    "command": ["native", "--max-seconds", "14400", "--identity", "fixed"]}
        actual = dict(expected, max_seconds=3720, command=["native", "--max-seconds", "3600", "--identity", "fixed"])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "spec.json"
            path.write_text(json.dumps(actual))
            self.assertEqual(audit.actual_spec(path, expected, server=True), actual)
            for bad in (dict(actual, max_seconds=3719), dict(actual, command=["native", "--max-seconds", "03600", "--identity", "fixed"]),
                        dict(actual, command=["native", "--max-seconds", "3600", "--identity", "changed"])):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    audit.actual_spec(path, expected, server=True)

    def test_published_false_certificate_rejected(self):
        report = audit.statistics(self.rows, "edge", self.plan, API, complete=True)
        published = copy.deepcopy(report)
        published.update(protocol_sha256="p", screen_plan_sha256="s", completed_baseline=6, completed_candidate=6,
                         per_task={task: data["absolute_counts"] for task, data in report["per_task"].items()})
        audit.compare_report(report, published, "p", "s")
        published["task_quality_validated"] = True
        with self.assertRaises(ValueError):
            audit.compare_report(report, published, "p", "s")

    def test_real_fixed_deadline(self):
        self.assertEqual(audit.DEADLINE, "2026-09-15T03:15:43Z")

    def test_terminal_publication_gets_final_new_family_readiness_pass(self):
        owner = self

        class FakeAudit:
            plan, api, sources = owner.plan, API, {}
            published = False
            calls = []

            def records(self):
                return owner.rows

            def ready(self, family):
                return self.published

            def terminal(self):
                self.published = True
                return {"status": "completed_fixed_SCREEN"}

            def family(self, family, rows):
                self.calls.append(family)
                report = audit.statistics(rows, family, self.plan, self.api, complete=True)
                report.update(audit_status="PASSED_NATIVE_CLOSED_REPLAY", native_records_audited=12)
                return {"report": report}

        with tempfile.TemporaryDirectory() as name:
            fake = FakeAudit()
            fake.root = Path(name)
            result = audit.observe(fake, Path(name) / "new")
            self.assertEqual(result["status"], "PASSED_ALL_24_CLOSED_SCREEN")
            self.assertEqual(fake.calls, ["edge", "nano"])

    def test_family_readiness_refreshes_older_records_snapshot(self):
        owner = self

        class FakeAudit:
            plan, api, sources = owner.plan, API, {}
            published = False

            def records(self):
                return owner.rows if self.published else []

            def ready(self, family):
                self.published = True
                return True

            def terminal(self):
                return {"status": "completed_fixed_SCREEN"}

            def family(self, family, rows):
                report = audit.statistics(rows, family, self.plan, self.api, complete=True)
                report.update(audit_status="PASSED_NATIVE_CLOSED_REPLAY", native_records_audited=12)
                return {"report": report}

        with tempfile.TemporaryDirectory() as name:
            fake = FakeAudit()
            fake.root = Path(name)
            result = audit.observe(fake, Path(name) / "new")
            self.assertEqual(result["status"], "PASSED_ALL_24_CLOSED_SCREEN")

    def test_dead_malformed_terminal_is_failure(self):
        with tempfile.TemporaryDirectory() as name:
            value = audit.Audit.__new__(audit.Audit)
            value.root = Path(name)
            value.launch = {"pid": 999999999, "proc_start_ticks": "1"}
            (value.root / "completion.json").write_text('{"partial":')
            with self.assertRaises(ValueError):
                value.terminal()

    def test_only_live_bound_producer_terminal_tail_may_wait(self):
        import os
        with tempfile.TemporaryDirectory() as name:
            value = audit.Audit.__new__(audit.Audit)
            value.root = Path(name)
            fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
            value.launch = {"pid": os.getpid(), "proc_start_ticks": fields[19]}
            (value.root / "completion.json").write_text('{"partial":')
            self.assertIsNone(value.terminal())
            value.launch["proc_start_ticks"] = "1"
            with self.assertRaises(ValueError):
                value.terminal()


if __name__ == "__main__":
    unittest.main()
