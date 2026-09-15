"""Update only the matched Thor column after complete validated measurements."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();raw=a.root.resolve();dest=Path(__file__).resolve().parent;repo=dest.parents[1]
subprocess.run([sys.executable,str(dest/'summarize.py'),str(raw),'--output',str(raw/'comparison.json')],check=True)
report=json.loads((raw/'comparison.json').read_text());rows={r['family']:r for r in report['rows']}
labels={'va':'LingBot-VA','va_2v4a':'LingBot-VA @ 2V/4A','vla4':'LingBot-VLA-4B','vla2':'LingBot-VLA-V2-6B','edge':'Cosmos3-Edge-Policy','nano':'Cosmos3-Nano-Policy','pi05':'pi05','groot':'GR00T-N1.7-3B','dreamzero':'DreamZero-DROID'}
assert set(rows)==set(labels)
for file in raw.glob('*.json'):shutil.copy2(file,dest/file.name)
for family in rows:
 for precision in ('native','fp8'):shutil.copy2(raw/f'{family}-{precision}.npz',dest/f'{family}-{precision}.npz')
readme=repo/'README.md';text=readme.read_text();lines=[];updated=set()
for line in text.splitlines():
 for family,label in labels.items():
  if line.startswith('| **'+label+'**'):
   assert family not in updated;parts=line.split('|');assert len(parts)==8
   row=rows[family];parts[4]=f" {row['native_p50_ms']:.0f}&nbsp;→&nbsp;{row['fp8_p50_ms']:.0f}&nbsp;ms,&nbsp;**{row['speedup']:.2f}×** "
   line='|'.join(parts);updated.add(family);break
 lines.append(line)
assert updated==set(labels)
text='\n'.join(lines)+'\n'
# The actual DROID checkpoint's audited FFNs alone exceed five billion weights.
dz=json.loads((raw/'dreamzero-fp8.json').read_text())['backend_stats']['fp8_recipe']
ffn_weights=sum(p['in_features']*p['out_features'] for p in dz['projections'] if '.ffn.' in p['path'])
assert ffn_weights>5_000_000_000
text=text.replace('**DreamZero-DROID** (Wan2.2-5B WAM)', '**DreamZero-DROID** (WAM)')
text=text.replace('The new Thor Runtime column reports matched native/FP8 p50 latency;', 'The Thor Runtime column reports the refreshed matched native/FP8 p50 latency;')
text=text.replace('[Protocol and results](eval/thor_fp8_2026-09-10/README.md).', '[Protocol and results](eval/thor_fp8_complete_2026-09-10/README.md).')
text=text.replace('[matched Thor speed results](eval/thor_fp8_2026-09-10/README.md)', '[matched Thor speed results](eval/thor_fp8_complete_2026-09-10/README.md)')
needle='FP8 is lossy NUMERIC execution; task accuracy is evaluated separately.'
text=text.replace(needle, 'FP8 is lossy NUMERIC execution; expanded recipes require their own task-quality evaluation.')
body=['# Refreshed Thor FP8 Runtime results','', 'All nine matched pairs use full public Runtime generation timing on one Jetson Thor. Native remains the default; select `precision="fp8"` explicitly. FP8 includes retained higher-precision components and does not promise faster execution for every model.','', '| Model | Native p50 | FP8 p50 | Native / FP8 |','|---|---:|---:|---:|']
for family in labels:
 row=rows[family];body.append(f"| {labels[family]} | {row['native_p50_ms']:.2f} ms | {row['fp8_p50_ms']:.2f} ms | {row['speedup']:.3f}× |")
body+=['', '**Below 1× means slower.** Twelve measured calls per stateless arm, nine early-history cycles per VA/DreamZero arm. These are short-run p50 measurements, not sustained real-time qualification. pi05 uses two active cameras with normalized float32 inputs. Historical H100/Thor cells in the main README retain their original scopes.','', 'The changes add exact-admitted native BF16 vision graphs to VLA-4B/V2, reduce shared FP8 activation-packing overhead, and expand Cosmos dense MLP, GR00T text attention/MLP and DreamZero FFN coverage. VA and pi05 retain their existing fused recipes. See the [per-family coverage and protocol](PROTOCOL.md).','', 'Expanded numerical recipes have **no inherited closed-loop quality certificate**. Action-array deltas in the comparison are numerical screens only. Earlier simulator results remain attached to their original checkpoint/source/recipe.','', 'Evidence: [88 regression checks](regression.json), [validated comparison](comparison.json), [hardware](hardware.json), [final source verification](source-final.json), and per-arm `<family>-<precision>.json` / `.npz` receipts. Frozen source and failed attempts remain in the raw archive named in the protocol. [Earlier sweep and audit](../thor_fp8_2026-09-10/README.md).','']
(dest/'README.md').write_text('\n'.join(body));readme.write_text(text)
print('Published nine refreshed matched pairs; historical columns retained.')
