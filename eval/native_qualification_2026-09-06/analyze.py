"""Recompute admission and fixed-input diagnostics; never produces a quality certificate."""
import argparse
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def delta(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'valid': False, 'bitexact': False, 'max_abs_delta': None}
    return {'valid': True, 'bitexact': bool(np.array_equal(a, b)),
            'max_abs_delta': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))}


def analyze(root):
    result = {'schema_version': 1, 'scope': 'Startup and fixed-input diagnostics only; no closed-loop certificate.',
              'phase1': {}, 'final_startups': {}, 'setup_failures': []}
    for device, folder in [('h100', root), ('thor', root / 'thor-results')]:
        trials = []
        for p in sorted(folder.glob(f'{device}.*.json')):
            if re.fullmatch(rf'{device}\.(stock|normal|diagnostic|fault)\.\d+\.json', p.name):
                trials.append((p, json.loads(p.read_text())))
        normal = [d for p, d in trials if '.normal.' in p.name]
        diags = [d for p, d in trials if '.diagnostic.' in p.name]
        velocity = []
        for stage in ('captured-chunk', 'refilled'):
            rows = [r for d in diags for r in d['velocity_diagnostics'] if r['stage'] == stage]
            velocity.append({'stage': stage, 'comparisons': len(rows), **{
                k: max((r[k]['max_abs_delta'] for r in rows), default=None)
                for k in ('aa', 'ab', 'bb', 'a_static')}})
        groups = {'stock': [], 'accepted': [], 'rejected_fallback': []}
        for p, d in trials:
            if '.stock.' in p.name:
                group = 'stock'
            elif '.normal.' in p.name:
                group = 'accepted' if d['graph_stats'].get('captured') else 'rejected_fallback'
            else:
                continue
            with np.load(str(p) + '.npz') as arrays:
                actions = {k: arrays[k].copy() for k in arrays.files if k.startswith('action_')}
            groups[group].append((p.name, actions))
        comparisons = {}
        for label, ga, gb in [('AA', 'stock', 'stock'), ('AB', 'stock', 'accepted'),
                              ('BB', 'accepted', 'accepted'), ('A_fallback', 'stock', 'rejected_fallback')]:
            pairs = (itertools.combinations(groups[ga], 2) if ga == gb else
                     itertools.product(groups[ga], groups[gb]))
            rows = []
            for (pa, aa), (pb, bb) in pairs:
                if aa.keys() != bb.keys():
                    raise ValueError('fixed-input action keys differ')
                checks = [delta(aa[k], bb[k]) for k in aa]
                rows.append({'a': pa, 'b': pb, 'actions': len(checks),
                             'valid': all(c['valid'] for c in checks),
                             'bitexact': all(c['bitexact'] for c in checks),
                             'max_abs_delta': max((c['max_abs_delta'] for c in checks
                                                   if c['valid']), default=None)})
            comparisons[label] = rows
        result['phase1'][device] = {'normal_attempts': len(normal),
            'normal_accepted': sum(bool(d['graph_stats'].get('captured')) for d in normal),
            'diagnostic_processes': len(diags), 'velocity_max_abs_delta': velocity,
            'fixed_recorded_noise_action_comparisons': comparisons}
    for folder in (root, root / 'thor-results'):
        for p in sorted(folder.glob('*.receipt.json')):
            if not p.name.startswith(('final.', 'corrected-env.', 'strict.', 'va-final.')):
                continue
            d = json.loads(p.read_text())
            startup = d.get('startup', {})
            if not startup:
                continue
            key = '|'.join([d['hardware']['gpu_name'], d['model_id'],
                            d['execution'].get('checkpoint_subdir') or '-', d['execution']['mode'],
                            str(d['execution'].get('tier_ceiling')),
                            'fault' if startup.get('fault_injection') else 'ordinary'])
            group = result['final_startups'].setdefault(key, [])
            checks = [t['params']['self_check'] for t in d['execution']['transforms']
                      if 'self_check' in t.get('params', {})]
            group.append({'receipt': p.name, 'attempt_id': startup['attempt_id'],
                          'status': startup['status'], 'graph_stats': d['execution']['graph_stats'],
                          'self_checks': checks})
    for p in sorted(root.glob('final.h100.pi05*.launch.json')):
        if not p.with_name(p.name.replace('.launch.json', '.receipt.json')).exists():
            result['setup_failures'].append(p.name)
    result['va_preliminary_failures'] = [p.name for p in sorted(root.glob('va.h100.*.launch.json'))
        if not p.with_name(p.name.replace('.launch.json', '.receipt.json')).exists()]
    return finite_json(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite analysis')
    args.output.write_text(json.dumps(analyze(args.root), indent=2, allow_nan=False) + '\n')
