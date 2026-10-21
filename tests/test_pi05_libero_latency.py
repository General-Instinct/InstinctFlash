"""CPU-only checks for the live LIBERO latency example; no weights or simulator required."""

import argparse
import ast
import json
import math
import statistics
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def load_helpers():
    """Load the example's pure helpers without its optional LeRobot/CUDA imports."""
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/pi05_vla/reproduce_sample_actions.py"
    )
    names = {
        "latency_summary",
        "require_capture",
        "save_report",
        "capture_evidence",
        "parse_args",
        "episode_success",
        "timed_sample_actions",
    }
    tree = ast.parse(source.read_text(), filename=str(source))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "argparse": argparse,
        "json": json,
        "math": math,
        "statistics": statistics,
        "Path": Path,
        "torch": torch,
        "np": np,
        "CKPT": "lerobot/pi05_libero_finetuned_v044",
        "DEV": "cuda:0",
        "time": SimpleNamespace(perf_counter=None),
    }
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102 — trusted repository source
    return SimpleNamespace(**{name: namespace[name] for name in names}), namespace


helpers, namespace = load_helpers()


class LatencySummaryTests(unittest.TestCase):
    def test_summary_uses_all_raw_samples(self):
        result = helpers.latency_summary([100, 200, 300, 400])
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["median_ms"], 250)
        self.assertEqual(result["mean_ms"], 250)
        self.assertEqual(result["min_ms"], 100)
        self.assertEqual(result["max_ms"], 400)
        self.assertAlmostEqual(
            result["stdev_ms"], statistics.stdev([100, 200, 300, 400])
        )

    def test_single_sample_has_zero_deviation(self):
        self.assertEqual(helpers.latency_summary([123])["stdev_ms"], 0)

    def test_invalid_samples_are_rejected(self):
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                helpers.latency_summary(values)


class ArgumentTests(unittest.TestCase):
    def test_defaults(self):
        args = helpers.parse_args([])
        self.assertEqual(
            (args.suite, args.task_id, args.episodes), ("libero_object", 0, 5)
        )
        self.assertEqual((args.warmup_chunks, args.if_warmup_chunks), (2, 3))
        self.assertEqual(
            (args.seed, args.device, args.output_dir), (1000, "cuda:0", None)
        )

    def test_task_alias_and_output_path(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "new-run"
            args = helpers.parse_args(["--task-id", "3", "--output-dir", str(path)])
            self.assertEqual(args.task_id, 3)
            self.assertEqual(args.output_dir, path)

    def test_invalid_arguments_are_rejected(self):
        cases = [
            ["--episodes", "0"],
            ["--warmup-chunks", "0"],
            ["--if-warmup-chunks", "-1"],
            ["--task", "10"],
            ["--suite", "unknown"],
            ["--seed", "-1"],
            ["--seed", str(2**32 - 1)],
            ["--device", "cpu"],
        ]
        for argv in cases:
            with (
                self.subTest(argv=argv),
                patch("sys.stderr"),
                self.assertRaises(SystemExit) as error,
            ):
                helpers.parse_args(argv)
            self.assertEqual(error.exception.code, 2)

    def test_existing_output_directory_is_rejected(self):
        with (
            TemporaryDirectory() as directory,
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            helpers.parse_args(["--output-dir", directory])


class CaptureTests(unittest.TestCase):
    def test_accepted_capture_must_have_exactness_and_replay(self):
        helpers.require_capture(
            SimpleNamespace(rejected=False, replays=10, self_check={"bitexact": True})
        )

    def test_missing_rejected_unchecked_or_unused_capture_fails(self):
        cases = [None]
        for rejected, replays, check in (
            (True, 10, {"bitexact": True}),
            (False, 0, {"bitexact": True}),
            (False, 10, None),
            (False, 10, {"bitexact": False}),
        ):
            cases.append(
                SimpleNamespace(rejected=rejected, replays=replays, self_check=check)
            )
        for driver in cases:
            with self.subTest(driver=driver), self.assertRaises(RuntimeError):
                helpers.require_capture(driver)

    def test_capture_evidence_records_actual_mode(self):
        driver = SimpleNamespace(
            _step_tables=True,
            _full_chunk=False,
            _prefix_graph_enabled=False,
            replays=20,
            chunk_replays=0,
            prefix_replays=0,
            rejected=False,
            self_check={"bitexact": True},
        )
        evidence = helpers.capture_evidence(driver)
        self.assertEqual(evidence["denoise_replays"], 20)
        self.assertTrue(evidence["step_tables"])
        self.assertFalse(evidence["full_chunk_graph"])
        json.dumps(evidence, allow_nan=False)


class ReportTests(unittest.TestCase):
    def test_report_is_updated_and_temporary_file_removed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            helpers.save_report(path, {"status": "running"})
            helpers.save_report(path, {"status": "complete", "latency_ms": [123]})
            self.assertEqual(
                json.loads((path / "results.json").read_text())["status"], "complete"
            )
            self.assertFalse((path / "results.json.tmp").exists())

    def test_invalid_report_preserves_previous_result(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            helpers.save_report(path, {"status": "running"})
            with self.assertRaises(ValueError):
                helpers.save_report(path, {"latency_ms": float("nan")})
            self.assertEqual(
                json.loads((path / "results.json").read_text()), {"status": "running"}
            )
            self.assertFalse((path / "results.json.tmp").exists())

    def test_terminal_only_needs_no_output_directory(self):
        helpers.save_report(None, {"status": "complete"})


class TimingTests(unittest.TestCase):
    def test_synchronization_surrounds_only_sample_actions(self):
        calls = []
        output = object()

        def sample_actions(*args, **kwargs):
            calls.append(("sample_actions", args, kwargs))
            return output

        def counter():
            calls.append("clock")
            return next(ticks)

        ticks = iter([1.0, 1.125])
        tokens = torch.tensor([[1]])
        model = SimpleNamespace(sample_actions=sample_actions)
        with (
            patch.object(
                torch.cuda,
                "synchronize",
                side_effect=lambda device: calls.append("sync"),
            ),
            patch.object(namespace["time"], "perf_counter", side_effect=counter),
        ):
            elapsed, actions = helpers.timed_sample_actions(model, [], [], tokens, None)
        self.assertEqual(elapsed, 125)
        self.assertIs(actions, output)
        self.assertEqual(
            [call if isinstance(call, str) else call[0] for call in calls],
            ["sync", "clock", "sample_actions", "sync", "clock"],
        )
        self.assertEqual(calls[2][2], {"noise": None})


class EpisodeSuccessTests(unittest.TestCase):
    def test_direct_and_terminal_success_metadata(self):
        self.assertTrue(helpers.episode_success({"is_success": np.array([True])}))
        self.assertTrue(helpers.episode_success({"final_info": {"is_success": [True]}}))
        self.assertTrue(helpers.episode_success({"final_info": [{"is_success": True}]}))
        self.assertFalse(helpers.episode_success({}))
        self.assertFalse(helpers.episode_success({"is_success": [False]}))


if __name__ == "__main__":
    unittest.main()
