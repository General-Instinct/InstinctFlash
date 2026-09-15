"""Check the compile-shape ablation against its own BF16 numerical reference."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np

root = Path(sys.argv[1]); family = sys.argv[2]
variants = ('dynamic_a','static','dynamic_b')
reports = {v: json.loads((root/f'{family}-{v}.json').read_text()) for v in variants}
arrays = {}
for v, r in reports.items():
    assert r['ok'] and not r['competing_gpu_processes'], v
    for field in ('family','revision','precision','torch','device','input_archive_sha256',
                  'reference_constructor_sha256','cases','effective_schedule','numeric_environment'):
        assert r[field] == reports['dynamic_a'][field], (v,field)
    setup = r['audit']['effective_setup']
    assert setup['use_torch_compile'] and not setup['use_cuda_graphs'] and not setup['diffusion_cache']
    assert setup['compile_dynamic'] == (v != 'static')
    assert r['audit']['engine_attention']['calls'] > 0
    assert r['audit']['engine_attention']['library_sha256'] == reports['dynamic_a']['audit']['engine_attention']['library_sha256']
    path = root/f'{family}-{v}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['actions_sha256']
    with np.load(path) as data:
        arrays[v] = data['actions'].copy()
    assert np.isfinite(arrays[v]).all()
reference = arrays['dynamic_a']
rows=[]
for v,r in reports.items():
    a=arrays[v]
    assert a.shape==reference.shape and a.dtype==reference.dtype
    rows.append(dict(variant=v,p50_ms=statistics.median(c['ms'] for c in r['calls'] if c['phase']=='measured'),
        byte_equal_to_dynamic_a=np.array_equal(a.view(np.uint8),reference.view(np.uint8)),
        max_abs_delta=float(np.max(np.abs(a.astype(float)-reference.astype(float)))),
        mean_abs_delta=float(np.mean(np.abs(a.astype(float)-reference.astype(float)))),
        shape=list(a.shape),attention_calls=r['audit']['engine_attention']['calls']))
result=dict(family=family,qualification='Comparison within the experimental BF16 numeric attention lane; no upstream BITEXACT or task-quality certificate',
    rows=rows,speedup_vs_a=rows[0]['p50_ms']/rows[1]['p50_ms'],speedup_vs_b=rows[2]['p50_ms']/rows[1]['p50_ms'])
(root/f'{family}-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if not rows[2]['byte_equal_to_dynamic_a']:
    raise SystemExit('Dynamic reference did not reproduce its action bytes')
