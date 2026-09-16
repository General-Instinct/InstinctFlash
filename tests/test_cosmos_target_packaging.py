"""Apply both target patches to their complete hash-bound original metadata."""
import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("bootstrap_vendor", ROOT / "scripts/bootstrap_vendor.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


@pytest.mark.parametrize("family", ["edge", "nano"])
def test_cosmos_target_packaging_matches_selected_natten_and_preserves_every_other_byte(tmp_path, family):
    thor = bootstrap.load_profile(family)
    rtx = bootstrap.load_profile(family, target="rtx4090")
    old_patch = (ROOT / "release/vendor" / thor["packaging_patch"]["path"]).read_bytes()
    new_patch = (ROOT / "release/vendor" / rtx["packaging_patch"]["path"]).read_bytes()
    old_line = b'+  "natten==0.21.6.dev6+cu130.torch210.gb300",'
    new_line = b'+  "natten==0.21.6.dev6+cu130.torch210",'
    assert old_patch.count(old_line) == 1
    assert new_patch == old_patch.replace(old_line, new_line)
    assert thor["source"] == rtx["source"]
    # This patch contains one hunk covering the complete original pyproject;
    # reconstruct its original bytes and verify the frozen upstream digest.
    lines = old_patch.splitlines(keepends=True)
    match = re.fullmatch(rb"@@ -1,(\d+) \+1,\d+ @@\n", lines[2])
    assert match and all(line[:1] in (b" ", b"-", b"+") for line in lines[3:])
    original_lines = [line[1:] for line in lines[3:] if line[:1] in (b" ", b"-")]
    assert len(original_lines) == int(match[1])
    original = b"".join(original_lines)
    assert hashlib.sha256(original).hexdigest() == thor["packaging_patch"]["before"]["pyproject.toml"]
    results = []
    for target, profile, expected in (("thor", thor, "0.21.6.dev6+cu130.torch210.gb300"),
                                      ("rtx", rtx, "0.21.6.dev6+cu130.torch210")):
        folder = tmp_path / target
        folder.mkdir()
        metadata = folder / "pyproject.toml"
        metadata.write_bytes(original)
        patch = ROOT / "release/vendor" / profile["packaging_patch"]["path"]
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=folder, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=folder, check=True)
        assert bootstrap.sha(metadata) == profile["packaging_patch"]["after"]["pyproject.toml"]
        assert f'"natten=={expected}"' in metadata.read_text()
        assert ("natten-" + expected.replace("+", "%2B")) in profile["public_wheel_overrides"]["natten"]
        assert sorted(p.name for p in folder.iterdir()) == ["pyproject.toml"]
        results.append(metadata.read_bytes())
    assert results[1] == results[0].replace(b'"natten==0.21.6.dev6+cu130.torch210.gb300"',
                                         b'"natten==0.21.6.dev6+cu130.torch210"')
