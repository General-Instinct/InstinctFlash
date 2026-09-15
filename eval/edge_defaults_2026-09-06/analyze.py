"""Summarize raw native Thor timings and action bytes without inferring task success."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0 or not np.isfinite(xs).all():
        raise ValueError('empty/nonfinite timings')
    return dict(n=len(xs), p50=float(np.percentile(xs, 50)),
                p95=float(np.percentile(xs, 95)), p99=float(np.percentile(xs, 99)),
                max=float(xs.max()))


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('incompatible/nonfinite action records')
    return dict(n=len(a), exact=sum(x.tobytes() == y.tobytes() for x, y in zip(a,b)),
                max_abs=float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max()))


def summarize(root):
    root = Path(root)
    hashes = {}
    def load(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    v2, arrays = {}, {}
    for path in sorted(root.glob('*.json')):
        if path.name.split('.')[0] not in ('stock','current','capture','final_numeric','final_strict'):
            continue
        doc = load(path)
        v2[path.stem] = dict(latency_ms=stats(doc['samples_ms']),
                            numeric_environment=doc['numeric_environment'],
                            checkpoint_sha256=doc['checkpoint_sha256'],
                            input_cases=doc['input_cases'])
        ap = Path(str(path)+'.actions.npz')
        if ap.exists():
            hashes[ap.name] = hashlib.sha256(ap.read_bytes()).hexdigest()
            a = np.load(ap, allow_pickle=False)['actions']
            if len(a) != 1 + doc['warmup'] + doc['iterations']:
                raise ValueError('action sample count differs from timings')
            arrays[path.stem] = a[1+doc['warmup']:]
    pairs = {}
    for x,y in [('stock.0','capture.0'),('stock.1','capture.1'),('stock.0','stock.1'),
                ('capture.0','capture.1'),('current.0','current.1'),
                ('stock.0','current.0'),('stock.1','current.1'),
                ('final_strict.0','final_numeric.0'),('stock.1','final_numeric.0')]:
        if x in arrays and y in arrays:
            pairs[x+' vs '+y] = compare(arrays[x], arrays[y])
    va, va_arrays = {}, {}
    for name in ('va_base','va_terminal','va_action_graph'):
        directory = root/name
        if not (directory/'complete.json').exists():
            continue
        metadata = load(directory/'complete.json')
        path = directory/'cycles.jsonl'
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != metadata['cycles']*metadata['runs']:
            raise ValueError('incomplete episode')
        timings = {}
        for regime, lo, hi in [('early',1,8),('saturated',36,metadata['cycles'])]:
            kept = [r for r in rows if r['run']>0 and not r['instrumented'] and lo <= r['cycle'] < hi]
            timings[regime] = {key:stats([r[key] for r in kept]) for key in ('infer_ms','commit_ms','total_ms')}
        ap = directory/'actions.npz'
        hashes[str(ap.relative_to(root))] = hashlib.sha256(ap.read_bytes()).hexdigest()
        actions = np.load(ap, allow_pickle=False)['actions']
        if len(actions) != len(rows):
            raise ValueError('VA action count differs from cycles')
        expected_order = [(run, cycle) for run in range(metadata['runs'])
                          for cycle in range(metadata['cycles'])]
        if [(r['run'], r['cycle']) for r in rows] != expected_order:
            raise ValueError('VA episode/cycle order differs from the protocol')
        if any(hashlib.sha256(a.tobytes()).hexdigest() != r['action_sha256']
               for a, r in zip(actions, rows)):
            raise ValueError('VA saved actions differ from the per-cycle hashes')
        va_arrays[name] = actions
        n = metadata['cycles']
        va[name] = dict(metadata=metadata, latency_ms=timings,
                       repeatability=compare(actions[:n],actions[n:2*n]),
                       phases=[r for r in rows if r['instrumented']])
    if len(va) > 1:
        reference = va['va_base']['metadata']
        for name, item in va.items():
            for key in ('input_sha256', 'schedule', 'guidance', 'numeric_environment',
                        'cycles', 'runs', 'torch', 'gpu'):
                if item['metadata'][key] != reference[key]:
                    raise ValueError(f'VA paired protocol mismatch: {name}/{key}')
    va_pair = compare(va_arrays['va_base'],va_arrays['va_terminal']) if {'va_base','va_terminal'} <= va_arrays.keys() else None
    graph_pair = compare(va_arrays['va_terminal'],va_arrays['va_action_graph']) if {'va_terminal','va_action_graph'} <= va_arrays.keys() else None
    full_pair = compare(va_arrays['va_base'],va_arrays['va_action_graph']) if {'va_base','va_action_graph'} <= va_arrays.keys() else None
    return dict(v2=v2,v2_action_comparisons=pairs,va=va,va_action_comparison=va_pair,
                va_graph_action_comparison=graph_pair,va_combined_action_comparison=full_pair,
                evidence_sha256=hashes,
                limitations=['Fixed-input systems evidence, not task success or a closed-loop certificate.',
                             'Small samples: reported p99 is descriptive, not a long-run tail guarantee.',
                             'V2 retains native numeric settings; action deltas are not accuracy-loss percentages.',
                             'VA cycles are zero-indexed; CUDA event spans include GPU idle intervals.',
                             'VA discards the first full episode and all instrumented cycles for timing.'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.root),indent=2)+'\n')
