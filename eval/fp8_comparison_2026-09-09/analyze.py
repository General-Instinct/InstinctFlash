"""Summarize retained runs without pooling failed admission or changing scope."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = {}
for path in sorted((a.root / 'results').glob('*.json')):
    if path.name.startswith(('pilot.', 'vla4.', 'v2.')):
        continue
    data = json.loads(path.read_text())
    values = path.with_suffix('.npz')
    if hashlib.sha256(values.read_bytes()).hexdigest() != data['actions_sha256']:
        raise ValueError(f'action archive hash mismatch: {values}')
    records.setdefault(data['arm'], []).append((data, np.load(values)))

def delta(x, y):
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite action evidence')
    bx,by=np.broadcast_arrays(x,y)
    byte_equal=bool(bx.dtype==by.dtype and np.ascontiguousarray(bx).tobytes()==np.ascontiguousarray(by).tobytes())
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                p99_abs=float(np.percentile(d, 99)), exact=byte_equal,
                value_equal=bool(np.all(d==0)),bitwise_equal=byte_equal)

summary = dict(scope='Staged computation benchmark; not full Runtime API parity or closed-loop task quality',
               startup_attempts=json.loads((a.root / 'status.json').read_text()), arms={}, comparisons={}, quality_status='diagnostic only')
for arm, items in records.items():
    repeats=[d['repeat'] for d,_ in items]
    if len(set(repeats))!=len(repeats):raise ValueError(f'duplicate pi05 arm/repeat: {arm}')
    mids = [d['latency_ms']['p50'] for d, _ in items]
    tails = [d['latency_ms']['p99'] for d, _ in items]
    summary['arms'][arm] = dict(
        successful_runs=len(items), p50_ms_median=float(np.median(mids)),
        p50_ms_range=[min(mids), max(mids)], p99_ms_median=float(np.median(tails)),
        p99_ms_range=[min(tails), max(tails)],
        load_calibrate_s=[d['load_calibrate_s'] for d, _ in items],
        warmup_s=[d['warmup_s'] for d, _ in items],
        captured=[d['graph_captured'] for d, _ in items],
        within_process_null=[delta(v['null'][0], v['null'][1:]) for _, v in items],
        between_process_actions=[delta(x[1]['actions'], y[1]['actions']) for x, y in itertools.combinations(items, 2)],
        memory=[{k: d[k] for k in ('torch_peak_allocated_bytes', 'torch_peak_reserved_bytes', 'device_free_total_bytes', 'memory_caveat')} for d, _ in items],
    )
for original, candidate in [('stock10','capture10'), ('stock10','engine16'), ('stock10','engine8'), ('capture10','engine8'), ('engine16','engine8'), ('stock50','capture50')]:
    if original not in records or candidate not in records:
        continue
    pairs = []
    for d, x in records[original]:
        for e, y in records[candidate]:
            if d['repeat'] != e['repeat']:
                continue
            for field in ('input_sha256', 'assets_sha256', 'chunk', 'steps', 'views', 'prompt_tokens', 'iterations'):
                if d[field] != e[field]:
                    raise ValueError(f'incomparable {field}')
            if not np.array_equal(x['noises'], y['noises']):
                raise ValueError('noise mismatch')
            if x['actions'].shape != y['actions'].shape:
                raise ValueError('action geometry mismatch')
            pairs.append(dict(repeat=d['repeat'], speedup_p50=d['latency_ms']['p50']/e['latency_ms']['p50'], action_delta=delta(x['actions'], y['actions'])))
    ratio = summary['arms'][original]['p50_ms_median'] / summary['arms'][candidate]['p50_ms_median']
    summary['comparisons'][original + '_to_' + candidate] = dict(speedup=ratio, latency_reduction_percent=100*(1-1/ratio), paired_runs=pairs)
a.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: {x:y for x,y in v.items() if x in ('p50_ms_median','p99_ms_median','captured','successful_runs')} for k,v in summary['arms'].items()}, indent=2))

for family in ('vla4', 'v2'):
    groups={}
    repaired=(a.root/'cwd-repair').is_dir()
    measured_root=a.root/'cwd-repair' if repaired and family=='v2' else a.root
    paths=list((measured_root/'results').glob(f'{family}.*.json'))
    if repaired and family=='vla4':
        paths+=list((a.root/'cwd-repair/results').glob('vla4.*.json'))
    if family=='v2':
        paths+=list((a.root/'receipt-repair/results').glob('v2.*.json'))
    for path in sorted(paths):
        d=json.loads(path.read_text())
        values=path.with_suffix('.npz')
        if hashlib.sha256(values.read_bytes()).hexdigest()!=d['actions_sha256']:
            raise ValueError('action hash mismatch')
        groups.setdefault(d['arm'],[]).append((d,np.load(values)))
    if not groups:
        continue
    result=dict(scope=next(iter(groups.values()))[0][0]['scope'],quality_status='diagnostic only; no new closed-loop certificate',arms={},comparisons={},
                startup_attempts=json.loads((measured_root/f'{family}-status.json').read_text()))
    vendor_root=a.root/'cwd-repair' if repaired else a.root
    if (vendor_root/'vendor-status.json').exists():
        result['startup_attempts'] += [x for x in json.loads((vendor_root/'vendor-status.json').read_text()) if x['label'].startswith(family+'.')]
    if repaired and family=='v2':
        receipt_status=a.root/'receipt-repair/status.json'
        if receipt_status.exists():result['startup_attempts']+=json.loads(receipt_status.read_text())
        original_status=a.root/'v2-status.json'
        interruption=a.root/'interrupted-setup.json'
        result['excluded_pre_repair_attempts']=json.loads(original_status.read_text()) if original_status.exists() else []
        result['setup_interruption']=json.loads(interruption.read_text()) if interruption.exists() else None
        result['setup_amendment']='SETUP_AMENDMENT.md: corrected native working directory; complete replacement arm set'
    for arm,items in groups.items():
        repeats=[d['repeat'] for d,_ in items]
        if len(set(repeats))!=len(repeats):raise ValueError(f'duplicate {family} arm/repeat: {arm}')
        mids=[d['latency_ms']['p50'] for d,_ in items]
        tails=[d['latency_ms']['p99'] for d,_ in items]
        result['arms'][arm]=dict(successful_runs=len(items),p50_ms_median=float(np.median(mids)),p50_ms_range=[min(mids),max(mids)],
            p99_ms_median=float(np.median(tails)),p99_ms_range=[min(tails),max(tails)],captured=[d['captured'] for d,_ in items],
            load_calibrate_s=[d['load_calibrate_s'] for d,_ in items],warmup_s=[d['warmup_s'] for d,_ in items],
            within_process_null=[delta(v['null'][0],v['null'][1:]) for _,v in items],
            between_process_actions=[delta(x[1]['actions'],y[1]['actions']) for x,y in itertools.combinations(items,2)])
        result['arms'][arm]['expected_runs']=1 if arm.endswith('_vendor') else 3
        result['arms'][arm]['numeric_environments']=[d.get('numeric_environment') for d,_ in items]
        result['arms'][arm]['observed_capture_states']={}
        for state in (False,True):
            subset=[d for d,_ in items if d['captured']==state]
            if subset:
                result['arms'][arm]['observed_capture_states'][str(state).lower()]=dict(
                    n=len(subset),repeats=[d['repeat'] for d in subset],
                    p50_ms_median=float(np.median([d['latency_ms']['p50'] for d in subset])),
                    p99_ms_median=float(np.median([d['latency_ms']['p99'] for d in subset])))
    for original,candidate in [('stock','capture'),('stock','engine8'),('capture','engine8'),
                               ('stock_vendor','capture_vendor'),('stock_vendor','engine8'),
                               ('capture_vendor','engine8'),('stock','stock_vendor'),('capture','capture_vendor')]:
        if original not in groups or candidate not in groups:continue
        pairs=[]
        for d,x in groups[original]:
            for e,y in groups[candidate]:
                if d['repeat']!=e['repeat']:continue
                for field in ('input_sha256','chunk','steps','views','iterations'):
                    if d[field]!=e[field]:raise ValueError(f'{family}: incomparable {field}')
                if 'noises' in x and not np.array_equal(x['noises'],y['noises']):raise ValueError('noise mismatch')
                if x['actions'].shape!=y['actions'].shape:raise ValueError('action geometry mismatch')
                pairs.append(dict(repeat=d['repeat'],speedup_p50=d['latency_ms']['p50']/e['latency_ms']['p50'],action_delta=delta(x['actions'],y['actions'])))
        ratio=result['arms'][original]['p50_ms_median']/result['arms'][candidate]['p50_ms_median']
        result['comparisons'][original+'_to_'+candidate]=dict(speedup=ratio,latency_reduction_percent=100*(1-1/ratio),paired_runs=pairs)
    dest=a.output.with_name(f'{family}-summary.json')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({family:{k: {x:y for x,y in v.items() if x in ('p50_ms_median','p99_ms_median','captured','successful_runs')} for k,v in result['arms'].items()}},indent=2))
