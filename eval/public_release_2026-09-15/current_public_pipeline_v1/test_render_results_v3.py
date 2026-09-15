"""Prospective RST mapping and provenance checks using synthetic summaries only."""
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load("render_results_v3")


def representative():
    s = load("test_render_readme_v2").representative_summary()
    for family, row in s["main"].items():
        root = m.STUDY / "qualification" / family
        for name, path in (("plan", root / "run/plan.json"), ("run", root / "run/run.json"),
                           ("replay_report", m.HERE / "snapshot_v5" / (family + "_paired_replay.json"))):
            row["paired"][name] = {"path": str(path), "sha256": "1" * 64}
        row["historical_equivalence"]["arrays"] = [
            {"cell": cell["id"], "historical_actions_match_bytes": True}
            for cell in row["paired"]["cells"]]
        for ws in row["serving"]["results"]:
            ws["receipt"] = {"path": str(root / "serving/receipt.json"), "sha256": "2" * 64}
    history = s["main"]["dreamzero"]["historical_equivalence"]["arrays"]
    history[0]["historical_actions_match_bytes"] = False
    history[2].update(historically_comparable=False, historical_actions_match_bytes=None)
    for name, row in s["foreign"].items():
        root = m.STUDY / "qualification/foreign" / name
        row["capture"] = {"path": str(root / "capture.json"), "sha256": "3" * 64}
        row["published_report"] = {"path": str(root / "report.json"), "sha256": "4" * 64}
    return s


def test_rst_keeps_current_selection_historical_drift_and_portable_links():
    text, rows = m.render(representative(), "5" * 64)
    assert '"DreamZero DROID","2/3","1","1","failed_historical_equivalence"' in text
    assert rows["dreamzero"]["published_cell"] == "dreamzero-dynamic-fp8"
    assert '"dreamzero-eager_native","Native reference; vendor encoder compilation","native","fixed 8/16 DiT","8000.00"' in text
    assert '"dreamzero-runtime_default","Runtime default control (BITEXACT)","native","fixed 8/16 DiT","4000.00"' in text
    assert '"dreamzero-runtime_selected","Native runtime-selected control (BITEXACT)","native","fixed 8/16 DiT","2000.00"' in text
    assert '"dreamzero-dynamic-fp8","Published and WebSocket route (BEHAVIORAL)","FP8","16 solver updates, dynamic cache","1200.00"' in text
    assert '"↳ LingBot-VA @2V/4A","Unmeasured","1171.23; native, 2V/4A"' in text
    assert "qualification/foreign/lerobot-pi05/report.json" in text
    assert "<../../REPRODUCE.rst>" in text
    assert "/home/ubuntu" not in text and "/dev/shm" not in text
    assert "No SDE1 value is inferred" in text and "negative Nano diagnostic" in text


def test_noncomparable_route_is_not_silently_called_exact():
    s = representative()
    s["main"]["dreamzero"]["historical_equivalence"]["arrays"][2]["historical_actions_match_bytes"] = True
    with pytest.raises(ValueError, match="non-comparable"):
        m.render(s, "5" * 64)


def test_external_path_is_labelled_provenance_and_not_linked():
    text = m.rst_link("stage", {"path": "/private/old_stage/manifest.json", "sha256": "6" * 64})
    assert "local-only provenance" in text and "/private/" not in text and "<" not in text
