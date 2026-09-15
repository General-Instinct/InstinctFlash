"""Public schedule evaluation must work without the training packages."""

from pathlib import Path
import subprocess
import sys
import textwrap


def test_public_schedule_ranking_without_training_imports(tmp_path):
    script = textwrap.dedent('''
        import copy
        import importlib.abc
        import sys

        sys.path.insert(0, sys.argv[1])
        forbidden = ("instinctflash.distill", "instinctflash.train")
        attempts = []

        def blocked(name):
            return any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)

        class RejectTraining(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if blocked(fullname):
                    attempts.append(fullname)
                    raise AssertionError("Public ranking imported " + fullname)

        assert not any(blocked(name) for name in sys.modules)
        sys.meta_path.insert(0, RejectTraining())

        from benchmarks.vla.schedule_sweep import best_untrained_per_schedule, schedule_key
        from instinctflash.verify.ranking import rank_candidates

        def point(name, nfe, scale, batch2):
            return {"id": name, "nfe": {"action": nfe},
                    "guidance": {"action": {"mode": "cfg", "scale": scale}},
                    "forwards_per_cycle": nfe + 1,
                    "batch2_forwards_per_cycle": batch2}

        spec = {"baseline": point("highest", 4, 5.0, 4), "points": [
            point("lower_scale", 4, 1.0, 0),
            point("higher_scale", 4, 3.0, 0),
            point("outside_tie", 4, 1.0, 0),
            point("unmeasured", 4, 1.0, 0),
            point("other_schedule", 2, 1.0, 0),
        ]}
        successes = {name: {"success": rate, "n_pairs": 100} for name, rate in [
            ("highest", 0.90), ("lower_scale", 0.89), ("higher_scale", 0.895),
            ("outside_tie", 0.88), ("other_schedule", 0.70),
        ]}
        original = copy.deepcopy((spec, successes))
        groups = best_untrained_per_schedule(spec, successes)
        four = groups[schedule_key({"action": 4})]
        assert four["best"]["point"] == "lower_scale"
        assert [row["point"] for row in rank_candidates(four["candidates"])] == [
            "lower_scale", "higher_scale", "highest", "outside_tie"]
        assert groups[schedule_key({"action": 2})]["best"]["point"] == "other_schedule"
        assert (spec, successes) == original
        assert not attempts, attempts
        assert not any(blocked(name) for name in sys.modules)
    ''')
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parents[1])],
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
