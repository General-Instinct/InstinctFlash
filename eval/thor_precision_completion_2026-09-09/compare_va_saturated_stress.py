"""Verify completed native/FP8 stress artifacts before reporting saturation gains."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(path, family, precision):
    path = Path(path)
    report = json.loads(path.read_text())
    if report.get('ok') is not True or report.get('family') != family or report.get('precision') != precision:
        raise ValueError('incomplete or incorrectly identified arm')
    geometry = {'libero': ((7, 4, 4), 4, 20), 'robotwin': ((16, 2, 16), 2, 25)}
    shape, frame_chunk, video_steps = geometry[family]
    if report['nfe'] != {'video': video_steps, 'action': 50}:
        raise ValueError('schedule differs from checkpoint')
    archive = path.with_suffix('.npz')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != report['actions_sha256']:
        raise ValueError('action artifact hash mismatch')
    labels = [f'episode_{e}_cycle_{c}' for e in range(2) for c in range(48)]
    calls = report['calls']
    if [c['label'] for c in calls] != labels:
        raise ValueError('missing, duplicate or reordered calls')
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != set(labels):
            raise ValueError('action population mismatch')
        actions = {label: data[label].copy() for label in labels}
    occupancy = []
    for row in calls:
        episode, cycle = divmod(len(occupancy), 48)
        if (row['episode'], row['cycle'], row['phase'], row['frame']) != (
                episode, cycle, 'warmup' if episode == 0 else 'measured', cycle * frame_chunk):
            raise ValueError('episode, phase or history position mismatch')
        if not np.isfinite(row['ms']) or row['ms'] <= 0:
            raise ValueError('invalid latency')
        action = actions[row['label']]
        if action.shape != shape or not np.isfinite(action).all():
            raise ValueError('invalid action shape or values')
        cache = row['cache']
        if precision == 'native':
            if set(cache) != {'pos'}:
                raise ValueError('unexpected native cache layout')
            cache = cache['pos']
        else:
            if cache['frame'] != row['frame'] or cache['tail']-cache['head'] != cache['live']:
                raise ValueError('FP8 slab position mismatch')
        capacity, live = cache['capacity'], cache['live']
        if type(capacity) is not int or type(live) is not int or not 0 < live <= capacity:
            raise ValueError('invalid cache occupancy')
        if capacity != {'libero': 2160, 'robotwin': 9792}[family]:
            raise ValueError('cache capacity differs from checkpoint attention window')
        # Native terminal elision defers provisional-action allocation until
        # clear_pred_cache. FP8 allocates those never-read slots immediately.
        # Normalize this known representation difference before comparing
        # post-call occupancy; neither record alone proves token identities.
        action_tokens = {'libero': 16, 'robotwin': 32}[family]
        normalized_live = live-action_tokens if precision == 'fp8' else live
        if normalized_live <= 0:
            raise ValueError('invalid normalized cache occupancy')
        occupancy.append((capacity, normalized_live))
    if len({x[0] for x in occupancy}) != 1 or occupancy[:48] != occupancy[48:]:
        raise ValueError('cache geometry or reset occupancy changed')
    for cycle in range(48):
        a, b = (actions[f'episode_{e}_cycle_{cycle}'] for e in range(2))
        if a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise ValueError('reset did not restore action bytes')
    # The terminal transient action append evicts old tokens even though its
    # slots are restored before returning. End-of-call occupancy is therefore
    # capacity minus action tokens, not capacity. Use a conservative interior
    # eviction regime: the no-eviction growth exceeds capacity and the measured
    # occupancy reaches the allocator's post-restore plateau.
    video_tokens, action_tokens = {'libero': (128, 16), 'robotwin': (240, 32)}[family]
    saturated = [i for i, (capacity, live) in enumerate(occupancy[48:])
                 if occupancy[0][1] + i*(video_tokens+action_tokens) > capacity
                 and live == capacity-action_tokens]
    if not saturated:
        raise ValueError('cache saturation was not observed')
    if precision == 'fp8' and any(calls[48+i]['cache']['head'] <= 0 for i in saturated):
        raise ValueError('FP8 slab has not evicted history')
    return report, actions, occupancy, saturated


def compare(native, fp8, family):
    n, na, nc, ns = verify(native, family, 'native')
    f, fa, fc, fs = verify(fp8, family, 'fp8')
    for key in ('boot_id', 'torch', 'source_sha256', 'nfe'):
        if n[key] != f[key]:
            raise ValueError(f'paired {key} differs')
    if nc != fc or ns != fs:
        raise ValueError('native/FP8 cache occupancy differs')
    medians = {p: float(np.median([r['calls'][48+i]['ms'] for i in ns]))
               for p, r in (('native', n), ('fp8', f))}
    episodes = {}
    for episode, name in enumerate(('first_exposure', 'shape_warm_repeat')):
        arms = {}
        for precision, report in (('native', n), ('fp8', f)):
            rows = report['calls'][48*episode:48*(episode+1)]
            arms[precision] = dict(first_cycle_ms=rows[0]['ms'])
            for regime, indices in (('all_cycles', range(48)), ('saturated', ns)):
                times = [rows[i]['ms'] for i in indices]
                arms[precision][regime] = dict(p50_ms=float(np.median(times)), max_ms=max(times))
        episodes[name] = dict(arms=arms, saturated_speedup=
            arms['native']['saturated']['p50_ms']/arms['fp8']['saturated']['p50_ms'])
    diffs = np.concatenate([(na[k].astype(np.float64)-fa[k].astype(np.float64)).ravel() for k in na])
    return dict(family=family, verified_calls_per_arm=96, saturated_cycles_zero_based=ns,
                saturated_p50_ms=medians, saturated_speedup=medians['native']/medians['fp8'],
                episode_latency=episodes,
                setup_seconds={p:r.get('setup_seconds') for p,r in (('native',n),('fp8',f))},
                action_mae=float(np.abs(diffs).mean()), action_max_abs=float(np.abs(diffs).max()),
                repeated_episode_byte_equal=True, paired_normalized_cache_occupancy_equal=True,
                cache_comparison_scope='FP8 occupancy minus immediately reserved provisional action slots versus native occupancy before its deferred reservation; validates counts, not token identities.',
                saturation_criterion='Conservative post-eviction regime: no-eviction token growth exceeds capacity, live occupancy equals capacity minus restored action-token slots, and FP8 slab head advances.',
                arm_sha256={p: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                            for p, v in (('native', native), ('fp8', fp8))},
                scope='One measured 48-cycle episode per arm after one warmup episode, repeated recorded history. No physical episode, task-quality, isolated FP8 arithmetic or sustained-tail certificate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True, type=Path)
    parser.add_argument('--fp8', required=True, type=Path)
    parser.add_argument('--family', required=True, choices=['libero', 'robotwin'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.native, args.fp8, args.family)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')
