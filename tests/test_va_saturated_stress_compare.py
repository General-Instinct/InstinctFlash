import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

source = Path(__file__).resolve().parents[1]/'eval/thor_precision_completion_2026-09-09/compare_va_saturated_stress.py'
spec = importlib.util.spec_from_file_location('va_stress_compare', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def arm(tmp_path, precision, *, tamper=None, family='libero'):
    shape, frame_chunk, video_steps, capacity, initial, increment, action_tokens = {
        'libero': ((7,4,4),4,20,2160,128,144,16),
        'robotwin': ((16,2,16),2,25,9792,240,272,32),
    }[family]
    arrays, calls = {}, []
    for episode in range(2):
        for cycle in range(48):
            label = f'episode_{episode}_cycle_{cycle}'
            arrays[label] = np.full(shape, cycle, dtype=np.float32)
            live = min(initial+cycle*increment, capacity-action_tokens)
            head = max(0,initial+cycle*increment-live)
            if precision == 'fp8':
                live += action_tokens
            cache = dict(capacity=capacity,live=live)
            cache = {'pos':cache} if precision=='native' else dict(cache,head=head,tail=head+live,frame=cycle*frame_chunk)
            calls.append(dict(label=label, episode=episode,cycle=cycle,phase='warmup' if episode==0 else 'measured',frame=cycle*frame_chunk,ms=2 if precision=='native' else 1,cache=cache))
    if tamper=='action': arrays['episode_1_cycle_40'][0,0,0] += 1
    if tamper=='position': calls[80]['frame'] += 4
    if tamper=='occupancy': calls[80]['cache']['head'] = 0
    if tamper=='partial': calls.pop()
    path = tmp_path/f'{precision}.json'
    np.savez(path.with_suffix('.npz'),**arrays)
    path.write_text(json.dumps(dict(ok=True,family=family,precision=precision,nfe={'video':video_steps,'action':50},boot_id='same',torch='same',source_sha256={'source':'same'},calls=calls,actions_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())))
    return path


@pytest.mark.parametrize('family,first_cycle', [('libero',15),('robotwin',36)])
def test_post_restore_saturation_compares(tmp_path,family,first_cycle):
    result = module.compare(arm(tmp_path,'native',family=family),arm(tmp_path,'fp8',family=family),family)
    assert result['saturated_speedup']==2
    assert result['saturated_cycles_zero_based']==list(range(first_cycle,48))
    assert set(result['episode_latency'])=={'first_exposure','shape_warm_repeat'}


def test_first_exposure_cost_is_not_hidden_by_warm_result(tmp_path):
    native, fp8 = arm(tmp_path,'native'), arm(tmp_path,'fp8')
    report = json.loads(fp8.read_text())
    for row in report['calls'][:48]:
        row['ms'] = 10
    fp8.write_text(json.dumps(report))
    result = module.compare(native,fp8,'libero')
    assert result['saturated_speedup']==2
    first = result['episode_latency']['first_exposure']
    assert first['saturated_speedup']==.2
    assert first['arms']['fp8']['first_cycle_ms']==10


@pytest.mark.parametrize('tamper',['action','position','occupancy','partial'])
def test_semantic_corruption_with_matching_artifact_hash_refused(tmp_path,tamper):
    with pytest.raises(ValueError):
        module.compare(arm(tmp_path,'native'),arm(tmp_path,'fp8',tamper=tamper),'libero')
