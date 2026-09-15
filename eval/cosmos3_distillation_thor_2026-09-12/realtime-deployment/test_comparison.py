"""Receipt validation must reject wrong CFG execution and stale evidence."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from realtime_adapter import GRIDS
from runtime_assets import WAN_BYTES, WAN_SHA256


def pair(directory, steps=1, guidance=1.):
    for arm in ('native', 'cudnn'):
        archive = directory / f'{arm}.npz'
        np.savez_compressed(archive, actions=np.zeros((16, 32, 8), dtype=np.float32))
        branches = 1 if guidance == 1. else 2
        latency = 200. if arm == 'native' else 100.
        report = dict(external_runtime_assets=[dict(bytes=WAN_BYTES,sha256=WAN_SHA256)], ok=True, attention=arm, qualification_only=True,
            declared_execution=dict(sigmas=GRIDS[steps], steps=steps, guidance=guidance,
                branches_per_callback=branches, action_padding='zero'),
            first_request_native_branch_clocks=[1000.*s for s in GRIDS[steps][:-1] for _ in range(branches)],
            actions_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            calls=[dict(i=i, ms=latency, phase='warmup' if i<6 else 'measured') for i in range(16)],
            backend_stats=dict(sampling=dict(sigmas=GRIDS[steps]), guidance=guidance,
                action_steps=steps, action_chunk_size=32, action_padding='zero', padding_hooks_restored=True,
                padding_projection_calls=16, padding_projected_velocity_branches=16*steps*branches,
                sampler_calls=16, velocity_evaluations=16*steps,
                numeric_attention=dict(eligible_python_calls=1 if arm=='cudnn' else 0)),
            execution_policy=dict(precision='native', changed_schedule=False,
                category='NUMERIC' if arm=='cudnn' else 'BITEXACT'),
            checkpoint_manifest_sha256='synthetic', benchmark_sha256='synthetic', fixture_sha256='synthetic',
            torch='test', cuda='test', cudnn='test', sources={'/fixture/realtime_adapter.py': 'synthetic'},
            p50_ms=latency, p95_ms=latency)
        (directory/f'{arm}.json').write_text(json.dumps(report))


def compare(directory, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['compare.py', str(directory)])
    runpy.run_path(str(Path(__file__).with_name('compare.py')), run_name='__main__')


@pytest.mark.parametrize('steps,guidance', [(1,1.), (1,4.), (2,1.), (2,4.), (4,1.), (4,4.)])
def test_accept_complete_declared_pairs(tmp_path, monkeypatch, steps, guidance):
    pair(tmp_path, steps, guidance)
    compare(tmp_path, monkeypatch)
    r=json.loads((tmp_path/'comparison.json').read_text())
    assert r['validation_complete'] and r['speedup']==2.
    assert not r['task_quality_certified']


@pytest.mark.parametrize('bad', ['branches','clocks','latency','padding','source','manifest','nonfinite'])
def test_reject_corrupt_pair_and_invalidate_stale_pass(tmp_path, monkeypatch, bad):
    pair(tmp_path)
    compare(tmp_path, monkeypatch)
    path=tmp_path/'cudnn.json'; r=json.loads(path.read_text())
    if bad=='branches': r['backend_stats']['padding_projected_velocity_branches']=32
    if bad=='clocks': r['first_request_native_branch_clocks']=[1000.,750.]
    if bad=='latency': r['p50_ms']=1.
    if bad=='padding': r['backend_stats']['padding_hooks_restored']=False
    if bad=='source': r['sources']['/fixture/realtime_adapter.py']='changed'
    if bad=='manifest': r['checkpoint_manifest_sha256']='changed'
    if bad=='nonfinite':
        archive=tmp_path/'cudnn.npz'
        np.savez_compressed(archive,actions=np.full((16,32,8),np.nan,dtype=np.float32))
        r['actions_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    path.write_text(json.dumps(r))
    with pytest.raises(AssertionError): compare(tmp_path, monkeypatch)
    assert not json.loads((tmp_path/'comparison.json').read_text())['validation_complete']
