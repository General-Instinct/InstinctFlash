"""Publish only a complete independently checked sweep; retain historical cells."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();raw=a.root.resolve();dest=Path(__file__).resolve().parent;repo=dest.parents[1]
subprocess.run([sys.executable,str(dest/'summarize.py'),str(raw),'--output',str(raw/'comparison.json')],check=True)
report=json.loads((raw/'comparison.json').read_text())
from validate_pi05_correction import validate
correction=dest/'pi05-two-camera'
if not correction.exists():correction=raw.parent/'thor_fp8_audit_20260910/pi05-two-camera-normalized'
assert correction.exists(),'The two-camera normalized pi05 correction is required; do not republish the old fixture.'
corrected_pi05=validate(correction)
correction_dest=dest/'pi05-two-camera';correction_dest.mkdir(exist_ok=True)
if correction.resolve()!=correction_dest.resolve():
 for source in correction.glob('pi05-two-camera-*'):
  if source.suffix in ('.json','.npz','.py'):shutil.copy2(source,correction_dest/source.name)
(correction_dest/'pi05-two-camera-comparison.json').write_text(json.dumps(corrected_pi05,indent=2)+'\n')
report['rows']=[corrected_pi05 if row['family']=='pi05' else row for row in report['rows']]
report['pi05_correction']='pi05-two-camera/pi05-two-camera-comparison.json'
report.pop('superseded_rows',None)
rows={r['family']:r for r in report['rows']}
faster=sum(r['speedup']>1.01 for r in rows.values());slower=sum(r['speedup']<0.99 for r in rows.values());flat=len(rows)-faster-slower
conclusion=f'{faster} pairs measured more than 1% faster, {slower} more than 1% slower, and {flat} within 1% of native latency. This is a descriptive band, not a statistical significance test.'
for name in ['comparison.json','hardware.json','source-v1.json','source-v2.json','source-v3.json','progress-recovery-1.json','va-cache-repair.json','dreamzero-v1-cancelled.json','progress-initial.json','compiled-libraries.json','source-final.json','progress.json']+[f'{f}-{p}.json' for f in rows for p in ('native','fp8')]:shutil.copy2(raw/name,dest/name)
(dest/'current-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
labels={'va':'LingBot-VA','va_2v4a':'LingBot-VA @ 2V/4A','vla4':'LingBot-VLA-4B','vla2':'LingBot-VLA-V2-6B','edge':'Cosmos3-Edge-Policy','nano':'Cosmos3-Nano-Policy','pi05':'pi05','groot':'GR00T-N1.7-3B','dreamzero':'DreamZero-DROID'}
readme=repo/'README.md';text=readme.read_text();assert 'Runtime native → FP8 (Thor' not in text,'do not duplicate table column'
lines=[]
for line in text.splitlines():
 if line.startswith('| model |'):
  parts=line.split('|');parts.insert(4,' Runtime native → FP8 (Thor, NUMERIC) ');line='|'.join(parts)
 elif line=='|:--|:--|:--|:--|:--|':line='|:--|:--|:--|:--|:--|:--|'
 else:
  for family,label in labels.items():
   if line.startswith('| **'+label+'**'):
    row=rows[family];cell=f" {row['native_p50_ms']:.0f}&nbsp;→&nbsp;{row['fp8_p50_ms']:.0f}&nbsp;ms,&nbsp;**{row['speedup']:.2f}×** "
    parts=line.split('|');parts.insert(4,cell);line='|'.join(parts);break
 lines.append(line)
text='\n'.join(lines)+'\n'
needle='**BITEXACT**: action bytes matched in the reported benchmark.'
annotation='The new Thor Runtime column reports matched native/FP8 p50 latency; **below 1× means slower**.\nFP8 is lossy NUMERIC execution; task accuracy is evaluated separately. The pi05 pair uses two-camera LIBERO v044 with float32 images in [0,1]. [Protocol and results](eval/thor_fp8_2026-09-10/README.md).\n\n'
assert needle in text;text=text.replace(needle,annotation+needle,1)
text=text.replace('For current native/FP8 Runtime comparisons, see the [Thor study](eval/thor_precision_completion_2026-09-09/COMPARISON.md); its scopes differ from these historical rows.', 'The new column uses the [matched Runtime protocol](eval/thor_fp8_2026-09-10/PROTOCOL.md); historical columns retain their original measurement scopes.')
text=text.replace('[speed and quality comparison](eval/thor_precision_completion_2026-09-09/COMPARISON.md)', '[matched Thor speed results](eval/thor_fp8_2026-09-10/README.md),\n[checkpoint-specific quality screens](eval/thor_precision_completion_2026-09-09/COMPARISON.md)')
text=text.replace('| H100 tier | Thor tier |','| H100 baseline tier | Thor baseline tier |').replace('Remeasuring *','— *')
text=text.replace('The new column uses the [matched Runtime protocol](eval/thor_fp8_2026-09-10/PROTOCOL.md); historical columns retain their original measurement scopes.\n','')
text=text.replace('† GR00T Thor excludes per-call vision processing; no comparable end-to-end speedup is established.','† The historical GR00T Thor cell excludes per-call vision processing; the new Runtime pair includes it.')
text=text.replace('model, and full qualification is still in progress. See the checkpoint-specific','model, and full qualification is still in progress. See the')
readme.write_text(text)
body=['# Thor FP8 Runtime results','', f'All nine current pairs passed their checks; pi05 uses the separately validated two-camera float32 input correction. **{conclusion}** Native precision remains the default.','', '| Model | Native p50 | FP8 p50 | Native / FP8 | E4M3 tensors |','|---|---:|---:|---:|---:|']
for family,row in rows.items():body.append(f"| {labels[family]} | {row['native_p50_ms']:.2f} ms | {row['fp8_p50_ms']:.2f} ms | {row['speedup']:.3f}× | {row['e4m3_tensors']} |")
body += ['', 'These are matched public Runtime measurements on Jetson Thor. The pi05 pair uses `lerobot/pi05_libero_finetuned_v044`; GR00T, Cosmos and DreamZero use their DROID checkpoints. Ratios below 1 mean FP8 is slower. Each model uses its existing Thor FP8 implementation, including retained higher-precision components; VLA-4B vision stays BF16. The E4M3 inventory includes buffers and potentially unused frontend weights, not executed quantization coverage. See the [execution audit](ENGINE_AUDIT.md).', '', 'Use `Runtime.from_pretrained(checkpoint, precision="fp8")` or `--fp8` to select it explicitly. `precision="native"` stays the default. FP8 is lossy arithmetic; speed does not certify task success. Existing simulator screens are separately scoped to their tested checkpoint, source and protocol. See [VA RoboTwin](../thor_precision_completion_2026-09-09/VA_ROBOTWIN_SCREEN.md), [VLA-4B/V2](../thor_precision_completion_2026-09-09/VLA_JOINT_SCREEN.md), [pi05 LIBERO](../thor_precision_completion_2026-09-09/PI05_PUBLIC_SIM_SMOKE.md) and [GR00T LIBERO fine-tune](../thor_precision_completion_2026-09-09/GROOT_LIBERO_SCREEN.md). The GR00T DROID, Cosmos and DreamZero speed rows have no corresponding closed-loop certificate here.', '', 'The original README cells retain their historical protocols. The new column supplies its own matched native/FP8 pair: it does not divide FP8 latency by an unrelated old baseline. See [protocol](PROTOCOL.md), [current checked comparison](current-comparison.json), [hardware](hardware.json), and [frozen source manifest](source-v1.json). Each arm has a `<family>-<precision>.json` receipt with raw timings, action archive hash, quantized module inventory and source hashes.', '', 'Reproduction on the measurement host: the frozen `source-v1`, `source-v2` and `source-v3`, compiled libraries and raw action archives are retained at `/home/guanming/ifl_eval/thor_fp8_20260910` on Thor, with receipts mirrored under `/home/ubuntu/ifl_eval/thor_fp8_20260910`. `benchmark.py` runs one arm, `run_sweep.py` schedules pairs under an exclusive GPU lock, and `summarize.py` validates the full sweep. Setup smoke runs and failed attempts are excluded. Other hosts must supply the checkpoint caches, family environments and the recorded camera archive named in the benchmark; this repository does not bundle model weights.', '', 'Validation: all 18 arm receipts, source/library hashes, matched inputs/schedules, finite actions and timing summaries checked. Shared runtime regression checks passed before the sweep. This remains a short latency test, not sustained thermal or real-time qualification.']
(dest/'README.md').write_text('\n'.join(body)+'\n')
print('Published',len(rows),'matched pairs without replacing historical cells')
