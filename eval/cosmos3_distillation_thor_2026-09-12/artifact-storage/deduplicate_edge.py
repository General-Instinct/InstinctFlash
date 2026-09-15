"""Deduplicate only hash-verified frozen Edge safetensors, retaining every path."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('study_root',type=Path);p.add_argument('receipt',type=Path)
p.add_argument('--apply',action='store_true')
a=p.parse_args()
if a.receipt.exists():p.error('Use a fresh receipt')
packages=[f'student-v16-cfg{cfg}-seed{seed}' for cfg in (1,4) for seed in (12031,12032)]
packages += [f'student-v17-cfg1-seed{seed}-v{version}' for version in (2,3) for seed in (12031,12032)]
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()
def free_bytes():
    stat=os.statvfs(a.study_root);return stat.f_bavail*stat.f_frsize
report=dict(status='running',apply=a.apply,source_sha256=digest(Path(__file__)),packages=[],groups=[],replacements=[],started_unix=time.time())
def save():
    temp=a.receipt.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.receipt)
with open('/tmp/thor_gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    report['free_bytes_before']=free_bytes()
    groups=defaultdict(list)
    manifests=[]
    for name in packages:
        root=a.study_root/name
        manifest=root/'instinctcompress_manifest.json'
        data=json.loads(manifest.read_text());manifests.append((manifest,digest(manifest)))
        report['packages'].append(dict(path=str(root),manifest_sha256=manifests[-1][1]))
        for row in data['files']:
            relative=Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path=root/relative
            if path.suffix!='.safetensors':continue
            assert not path.is_symlink() and path.is_file()
            st=path.stat();assert st.st_size==row['bytes']
            groups[(row['sha256'],row['bytes'])].append(path)
    # Rehash each existing inode before any replacement. No manifest alone is
    # taken as proof that two files still contain identical bytes.
    verified={}
    for (sha,size),paths in groups.items():
        for path in paths:
            st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            if key not in verified:verified[key]=digest(path)
            assert verified[key]==sha,path
        unique={(path.stat().st_dev,path.stat().st_ino) for path in paths}
        report['groups'].append(dict(sha256=sha,bytes=size,paths=[str(p) for p in paths],unique_inodes_before=len(unique)))
    save()
    try:
        if a.apply:
            for (sha,size),paths in groups.items():
                source=paths[0]
                for target in paths[1:]:
                    ss,ts=source.stat(),target.stat()
                    if (ss.st_dev,ss.st_ino)==(ts.st_dev,ts.st_ino):continue
                    # Scope is immutable qualification weights. Configs,
                    # manifests, logs and producer training outputs are untouched.
                    assert ss.st_dev==ts.st_dev
                    temp=target.with_name(target.name+f'.dedup-{os.getpid()}')
                    assert not temp.exists()
                    os.link(source,temp)
                    try:os.replace(temp,target)
                    finally:
                        if temp.exists():temp.unlink()
                    report['replacements'].append(dict(source=str(source),target=str(target),sha256=sha,bytes=size))
                    save()
        post_verified={}
        for (sha,size),paths in groups.items():
            for path in paths:
                st=path.stat();assert st.st_size==size
                key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
                if key not in post_verified:post_verified[key]=digest(path)
                assert post_verified[key]==sha,path
        for manifest,sha in manifests:assert digest(manifest)==sha
        report.update(status='success',all_weight_paths_verified=sum(len(p) for p in groups.values()),
                      unique_inodes_before=len(verified),unique_inodes_after=len(post_verified),
                      manifests_unchanged=True,free_bytes_after=free_bytes(),finished_unix=time.time())
        save();print(json.dumps({k:v for k,v in report.items() if k not in ('packages','groups','replacements')}),flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error));save();raise
