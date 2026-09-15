"""Final source/library and completion check on the actual Thor host."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
p=json.loads((r/'progress.json').read_text())
assert p['status']=='complete' and len(p['jobs'])==18 and all(j['status']=='complete' and j['exit_code']==0 for j in p['jobs'])
m=json.loads((r/'source-v1.json').read_text())['files'];libs=json.loads((r/'compiled-libraries.json').read_text())
for name,digest in {**m,**libs}.items():assert hashlib.sha256((r/'source-v1'/name).read_bytes()).hexdigest()==digest,name
m2=json.loads((r/'source-v2.json').read_text())['files']
for name,digest in {**m2,**libs}.items():assert hashlib.sha256((r/'source-v2'/name).read_bytes()).hexdigest()==digest,name
m3=json.loads((r/'source-v3.json').read_text())['files']
for name,digest in {**m3,**libs}.items():assert hashlib.sha256((r/'source-v3'/name).read_bytes()).hexdigest()==digest,name
h=json.loads((r/'hardware.json').read_text());assert h['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
output=r/'source-final.json';assert not output.exists()
output.write_text(json.dumps({'ok':True,'source_files':len(m),'source_v3_files':len(m3),'source_v3_manifest_sha256':hashlib.sha256((r/'source-v3.json').read_bytes()).hexdigest(),'source_v2_files':len(m2),'source_v2_manifest_sha256':hashlib.sha256((r/'source-v2.json').read_bytes()).hexdigest(),'compiled_libraries':len(libs),'boot_id':h['boot_id'],'progress_sha256':hashlib.sha256((r/'progress.json').read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((r/'source-v1.json').read_bytes()).hexdigest(),'compiled_manifest_sha256':hashlib.sha256((r/'compiled-libraries.json').read_bytes()).hexdigest()},indent=2)+'\n')
print('FINAL SOURCE CHECK PASS')
