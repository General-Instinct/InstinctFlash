"""Admission must compare original operators and undo partially installed patches."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/groot_n17'))
from groot_n17_iwm.full_capture import CaptureSafeBackbonePatches, StaticBackbone, _max_delta


def test_partial_patch_install_is_restored():
    patches = CaptureSafeBackbonePatches(None)
    owner = SimpleNamespace(forward='original')
    patches._replace(owner, 'forward', 'patched')
    assert not patches.installed  # installation failed before its final line
    patches.uninstall()
    assert owner.forward == 'original'
    assert patches._restore == []


def test_every_upstream_reference_restores_original_then_reinstalls():
    driver = StaticBackbone.__new__(StaticBackbone)
    calls = []
    patches = SimpleNamespace(installed=True)
    def uninstall():
        patches.installed = False
        calls.append('uninstall')
    def install():
        patches.installed = True
        calls.append('install')
    patches.uninstall, patches.install = uninstall, install
    driver.patches = patches
    driver._restore_norm = lambda: calls.append('restore_norm')
    driver._install_norm_recorder = lambda: calls.append('record_norm')
    driver._feature = lambda values: values
    def original(values):
        assert not patches.installed
        calls.append('original')
        return values
    driver.original = original
    for value in [1, 2, 3]:
        assert driver._upstream_features(value) == value
    assert calls == ['restore_norm', 'uninstall', 'original', 'install', 'record_norm'] * 3


def test_exact_admission_rejects_structure_and_signed_zero_changes():
    assert _max_delta([1], [1, 2]) != 0
    assert _max_delta({'x': 1}, {'x': 1, 'y': 2}) != 0
    assert _max_delta(torch.tensor([0.]), torch.tensor([-0.])) != 0
    assert _max_delta(torch.tensor([float('nan')]), torch.tensor([float('nan')])) != 0
