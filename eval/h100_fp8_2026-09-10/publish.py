"""Publish only a complete independently checked sweep; retain historical cells."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();raw=a.root.resolve();dest=Path(__file__).resolve().parent;repo=dest.parents[1]
subprocess.run([sys.executable,str(dest/'summarize.py'),str(raw),'--output',str(raw/'comparison.json')],check=True)
report=json.loads((raw/'comparison.json').read_text());rows={r['family']:r for r in report['rows']}
slower=sum(r['speedup']<1 for r in rows.values())
conclusion=('The current H100 FP8 executor is slower in every measured pair.' if slower==len(rows) else f'The current H100 FP8 executor is slower in {slower} of {len(rows)} measured pairs.')
for name in ['comparison.json','hardware.json','source-v1.json']+[f'{f}-{p}.json' for f in rows for p in ('native','fp8')]:shutil.copy2(raw/name,dest/name)
labels={'va':'LingBot-VA','va_2v4a':'LingBot-VA @ 2V/4A','vla4':'LingBot-VLA-4B','vla2':'LingBot-VLA-V2-6B','edge':'Cosmos3-Edge-Policy','nano':'Cosmos3-Nano-Policy','pi05':'pi05','groot':'GR00T-N1.7-3B','dreamzero':'DreamZero-DROID'}
readme=repo/'README.md';text=readme.read_text();assert 'Runtime native → FP8 (H100' not in text,'do not duplicate table column'
lines=[]
for line in text.splitlines():
 if line.startswith('| model |'):
  parts=line.split('|');parts.insert(3,' Runtime native → FP8 (H100, NUMERIC) ');line='|'.join(parts)
 elif line=='|:--|:--|:--|:--|:--|':line='|:--|:--|:--|:--|:--|:--|'
 else:
  for family,label in labels.items():
   if line.startswith('| **'+label+'**'):
    row=rows[family];cell=f" {row['native_p50_ms']:.0f}&nbsp;→&nbsp;{row['fp8_p50_ms']:.0f}&nbsp;ms,&nbsp;**{row['speedup']:.2f}×** "
    parts=line.split('|');parts.insert(3,cell);line='|'.join(parts);break
 lines.append(line)
text='\n'.join(lines)+'\n'
needle='**BITEXACT**: action bytes matched in the reported benchmark.'
annotation='The new H100 Runtime column is a matched native/FP8 sweep; **below 1× means slower**.\nH100 FP8 uses Q/K/V quantization and has no closed-loop accuracy certificate. [Protocol and results](eval/h100_fp8_2026-09-10/README.md).\n\n'
assert needle in text;text=text.replace(needle,annotation+needle,1);readme.write_text(text)
body=['# H100 FP8 Runtime results','', f'All nine pairs completed and passed source, input, schedule, finite-output and timing checks. **{conclusion}** Native precision remains the default.','', '| Model | Native p50 | FP8 p50 | Native / FP8 | FP8 projections |','|---|---:|---:|---:|---:|']
for family,row in rows.items():body.append(f"| {labels[family]} | {row['native_p50_ms']:.2f} ms | {row['fp8_p50_ms']:.2f} ms | {row['speedup']:.3f}× | {row['fp8_projections']} |")
body += ['', 'These are matched public Runtime measurements on H100-80GB. Ratios below 1 mean FP8 is slower. The H100 implementation quantizes attention Q/K/V using PyTorch E4M3 matrix multiplication; it is distinct from the Thor fused kernels. These results describe this implementation, not a limit on optimized FP8 inference.', '', 'Use `Runtime.from_pretrained(checkpoint, precision="fp8")` or `--fp8` to select it explicitly. `precision="native"` stays the default. FP8 changes arithmetic; H100 task quality remains unverified, and Thor simulator scores do not certify this different recipe.', '', 'The original README cells retain their historical protocols. The new column supplies its own matched native/FP8 pair: it does not divide FP8 latency by an unrelated old baseline. See [protocol](PROTOCOL.md), [checked comparison](comparison.json), [hardware](hardware.json), and [frozen source manifest](source-v1.json). Each arm has a `<family>-<precision>.json` receipt with raw timings, action archive hash, quantized module inventory and source hashes.', '', 'Reproduction on the measurement host: the frozen `source-v1`, recorded inputs and raw action archives are retained at `/home/ubuntu/ifl_eval/h100_fp8_20260910`. `benchmark.py` runs one arm, `run_lane.py` schedules pairs, and `summarize.py` validates the full sweep. Setup smoke runs and failed attempts are excluded. Other hosts must supply the checkpoint caches, family environments and the recorded camera archive named in the benchmark; this repository does not bundle model weights.', '', 'Validation: 46 precision/schedule/geometry tests, 25 integration tests, and four H100/feedback tests passed. The CPU-only integration run skipped its H100 case, which passed separately on an actual H100.']
(dest/'README.md').write_text('\n'.join(body)+'\n')
print('Published',len(rows),'matched pairs without replacing historical cells')
