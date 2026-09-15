"""On Thor, validate immutable sources/libraries after the exclusive sweep."""
import argparse,fcntl,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root.resolve()
lock=Path('/tmp/thor_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
progress=json.loads((r/'progress.json').read_text())
assert progress['status']=='complete' and len(progress['jobs'])==18
manifest=json.loads((r/'source-v1.json').read_text())['files'];libs=json.loads((r/'compiled-libraries.json').read_text())
versions={}
for path in sorted(r.glob('source-v*.json')):
 version=path.stem;files=json.loads(path.read_text())['files']
 for name,digest in {**files,**libs}.items():assert sha(r/version/name)==digest,(version,name)
 versions[version]=dict(manifest_sha256=sha(path),source_files=len(files))
hardware=dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit','--format=csv'],text=True),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),l4t=Path('/etc/nv_tegra_release').read_text())
(r/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
(r/'source-final.json').write_text(json.dumps(dict(ok=True,source_versions=versions,source_files=len(manifest),compiled_libraries=len(libs),source_manifest_sha256=sha(r/'source-v1.json'),compiled_manifest_sha256=sha(r/'compiled-libraries.json'),progress_sha256=sha(r/'progress.json'),hardware_sha256=sha(r/'hardware.json')),indent=2)+'\n')
print('Verified',len(manifest),'sources and',len(libs),'libraries')
