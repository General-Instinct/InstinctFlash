"""Rebuild source-contract and historical-evidence inventory; no GPU required."""
import hashlib
import json
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

def read(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), data=json.loads(path.read_text()))

records = {family: read(repo / 'eval/results_2026-08_thor_h100' / (family + '.json')) for family in ('pi05', 'vla4b', 'vla2', 'groot')}
historical = dict(status='historical reports; no automatic certification of current runtime or new calibration', latency=records)
for key, path in {
    'pi05_500_pairs': '/home/ubuntu/iwm_distill/thor_t3/cert/ext_pairing.json',
    'v2_1100_pairs': repo / 'eval/results_2026-08_thor_h100/m3_certificate_pooled.json',
}.items():
    historical[key] = read(path)
(out / 'historical-evidence.json').write_text(json.dumps(historical, indent=2) + '\n')

run_file=Path('/home/ubuntu/iwm_distill/thor_t3/cert/ext_paired_runs.json')
runs=json.loads(run_file.read_text())
arm_a={r['task']:r for r in runs['armA']}
arm_b={r['task']:r for r in runs['armB']}
assert set(arm_a)==set(arm_b)==set(range(10))
pairs=[]
for task in sorted(arm_a):
    x,y=arm_a[task],arm_b[task]
    assert x['ok'] and y['ok'] and len(x['successes'])==len(y['successes'])==50
    pairs.extend(zip(x['successes'],y['successes']))
recount=dict(source=str(run_file),source_sha256=hashlib.sha256(run_file.read_bytes()).hexdigest(),n=len(pairs),
    original_successes=sum(bool(x) for x,y in pairs),candidate_successes=sum(bool(y) for x,y in pairs),
    candidate_only=sum(not x and bool(y) for x,y in pairs),original_only=sum(bool(x) and not y for x,y in pairs),
    scope='Recount from retained task-run arrays in their recorded episode order; no new scene/reset/profile certification')
(out/'pi05-historical-recount.json').write_text(json.dumps(recount,indent=2)+'\n')

import numpy as np
from instinctflash.runtime.engine_backend import _map_observation
from instinctflash.passes.generic.engine_offload import EngineOffloadApplicable
from types import SimpleNamespace
# Tiny labelled arrays expose mapping; do not masquerade as model evaluation.
observation = {
    'observation.images.image': np.full((3, 4, 4), 11, np.uint8),
    'observation.images.image2': np.full((3, 4, 4), 22, np.uint8),
    'observation.images.empty_camera_0': np.full((3, 4, 4), 0, np.uint8),
    'observation.state': np.arange(8, dtype=np.float32),
}
mapped = _map_observation(observation)
audit = dict(
    source_sha256=hashlib.sha256((repo / 'instinctflash/runtime/engine_backend.py').read_bytes()).hexdigest(),
    source_hashes={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in (
        'instinctflash/runtime/engine_backend.py', 'instinctflash/runtime/facade.py',
        'instinctflash/passes/generic/engine_offload.py')},
    pi05_camera_mapping={k: int(v.flat[0]) for k, v in mapped.items()},
    expected_two_real_cameras=dict(image=11, wrist_image=22),
    mapping_matches_two_real_cameras=bool(mapped['image'].flat[0] == 11 and mapped['wrist_image'].flat[0] == 22),
    state_present_after_mapping='observation.state' in mapped or 'state' in mapped,
    chunk50_geometry_guard_reason=EngineOffloadApplicable._action_geometry_problem(SimpleNamespace(notes={'backbone':'pi05','action_dim':'7','chunk_size':'50'})),
    horizon_gate_scope='The current geometry guard checks action dimension, not action horizon. The separate operating-point guard checks denoise NFE/guidance. Neither is a verified chunk-50-to-chunk-10 refusal.',
    action_contract='Native pi05 _Pi05Loop uses select_action and returns one decoded action (buffered across calls); EngineBackend returns a full 10x7 chunk each call. Runtime method names alone do not unify this controller cadence.',
    runtime_scope='Current generic engine reset receives the episode prompt; predict does not run the checkpoint state-to-prompt processor. T3 specialized server instead receives per-call token IDs and applies the native postprocessor externally.',
    decision='Do not label the staged engine benchmark an interchangeable pi05 robot-facing Runtime API benchmark.',
    families={
        'pi05': 'Engine supports FP16 control and FP8; fixed chunk 10. Generic Runtime contract has state/prompt and camera mapping gaps for standard v044 observations.',
        'vla4b': 'Standalone staged engine available, FP16 vision plus FP8 LM/expert. No full-engine FP16 switch. Historical gate inputs are synthetic; not task-quality evidence.',
        'vla2': 'Historical complete engine uses additional repacked artifacts/custom MoE wiring. Generic staged frontend defaults to routed_moe_stub. Earlier qualification observed intermittent native NUMERIC capture admission; the new profile is measured separately. Historical and current sources are not interchangeable.',
        'groot': 'Standalone engine consumes backbone auxiliary features prepared outside each prediction; old 122/42 ratio excludes repeated vision only on engine side. No matched full-policy speed ratio established.',
        'wan_va': 'Experimental build-specific engine; no unified FP8 Runtime route. Different NFE/CFG and ring history require their own comparison.',
        'cosmos_dreamzero': 'No supported unified FP8 comparison established by existing records.',
    },
)
(out / 'contract-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
