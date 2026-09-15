"""Publish validated normal-input pi05 data without rewriting retained original receipts."""
import json,shutil,sys
from pathlib import Path
from validate_pi05_correction import validate
raw=Path(sys.argv[1]);dest=Path(__file__).resolve().parent;repo=dest.parents[1]
row=validate(raw);target=dest/'pi05-two-camera';target.mkdir(exist_ok=True)
for path in raw.glob('pi05-two-camera-*'):
 if path.suffix in ('.json','.npz','.py'):shutil.copy2(path,target/path.name)
(target/'pi05-two-camera-comparison.json').write_text(json.dumps(row,indent=2)+'\n')
d=json.loads((dest/'comparison.json').read_text());d['rows']=[row if r['family']=='pi05' else r for r in d['rows']];d['pi05_correction']='pi05-two-camera/pi05-two-camera-comparison.json';(dest/'current-comparison.json').write_text(json.dumps(d,indent=2)+'\n')
p=repo/'README.md';lines=p.read_text().splitlines();updated=0
for i,line in enumerate(lines):
 if line.startswith('| **pi05**'):
  fields=line.split('|');fields[4]=f" {row['native_p50_ms']:.0f}&nbsp;→&nbsp;{row['fp8_p50_ms']:.0f}&nbsp;ms,&nbsp;**{row['speedup']:.2f}×** ";lines[i]='|'.join(fields);updated+=1
assert updated==1
text='\n'.join(lines)+'\n';text=text.replace('The new pi05 pair uses LIBERO v044.', 'The pi05 pair uses two-camera LIBERO v044 with float32 images in [0,1].');p.write_text(text)
p=dest/'README.md';text=p.read_text().replace('All nine pairs completed and passed source, input, schedule, finite-output and timing checks.', 'All nine current pairs passed their checks; pi05 uses the separately validated two-camera float32 input correction.')
lines=text.splitlines()
for i,line in enumerate(lines):
 if line.startswith('| pi05 |'):lines[i]=f"| pi05 | {row['native_p50_ms']:.2f} ms | {row['fp8_p50_ms']:.2f} ms | {row['speedup']:.3f}× | {row['e4m3_tensors']} |"
text='\n'.join(lines)+'\n';text=text.replace('[checked comparison](comparison.json)','[current checked comparison](current-comparison.json)')
text+='\nThe original pi05 7.41× cell used an unrecognized second camera key and unscaled uint8 images. It is superseded by the [validated two-camera correction](pi05-two-camera/pi05-two-camera-comparison.json); reproduce it with [the corrected benchmark](pi05-two-camera/pi05-two-camera-benchmark.py). The [initial sweep comparison](comparison.json) and receipts are retained for audit, not as the current pi05 result.\n'
p.write_text(text)
print('Published corrected pi05',row['native_p50_ms'],'->',row['fp8_p50_ms'],row['speedup'])
