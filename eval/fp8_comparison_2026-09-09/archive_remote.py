"""Archive measured sources and outputs after all GPU processes finish.

Large external weights/repack inputs are inventoried, not copied into the archive.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import shutil

root=Path(sys.argv[1]).resolve()
completion=('complete.json','vla4-complete.json','cwd-repair/v2-complete.json','cwd-repair/vendor-complete.json') if (root/'cwd-repair').is_dir() else ('complete.json','vla4-complete.json','v2-complete.json','vendor-complete.json')
if (root/'receipt-repair').is_dir():completion+=('receipt-repair/complete.json',)
for name in completion:
    if not (root/name).exists():raise RuntimeError(f'incomplete campaign: {name}')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()

home=Path.home()
shutil.copytree(home/'ifl_eval/native_qualification_20260906/final-execution/instinctflash',
                root/'reference-runtime/instinctflash',ignore=shutil.ignore_patterns('__pycache__'))
external={}
dirs=[
    home/'thor_t2v2/repack_v2_out',home/'thor_t2v2/calib',
    home/'ifl_eval/native_qualification_20260906/final-execution/instinctflash',
    home/'ifl_eval/next_steps_20260906/qwen-config',
    home/'.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3',
]
for model,rev in [
    ('lerobot--pi05_libero_finetuned_v044','8e174154ef5f6c60a8da12ae99c303d8963138c1'),
    ('robbyant--lingbot-vla-4b-posttrain-robotwin','fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e'),
    ('robbyant--lingbot-vla-v2-6b-robotwin','0451855729ec904f970600e0aec8b84661423afe'),
]:
    dirs.append(home/f'.cache/huggingface/hub/models--{model}/snapshots/{rev}')
for folder in dirs:
    if not folder.is_dir():raise RuntimeError(f'missing external directory: {folder}')
    for path in sorted(folder.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            external[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
for path in [home/'thor_t2v2/m2_eval_obs.npz',home/'thor_t2v2/p1_dumps/obs0.pt',home/'thor_t2v2/p1_dumps/obs8.pt',home/'thor_t2v2/p1_dumps/vis_tables.pt',home/'ifl/t3_assets/thor_assets.json',home/'ifl/t3_assets/calib_obs.npz']:
    external[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
norm=home/'.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json'
if norm.exists():external[str(norm)]=dict(bytes=norm.stat().st_size,sha256=sha(norm))
(root/'external-artifacts.json').write_text(json.dumps(external,indent=2)+'\n')
environments={}
code='import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":{k:m.version(k) for k in ("torch","transformers","numpy","safetensors")}}))'
for name,exe in [('pi05',home/'frt_env/bin/python'),('vla4',home/'venv_vla4b/bin/python'),('v2',home/'ifl_eval/next_steps_20260906/run-env/bin/python')]:
    environments[name]=json.loads(subprocess.check_output([str(exe),'-c',code],text=True))
    environments[name]['installed_distributions']=json.loads(subprocess.check_output(
        [str(exe),'-c','import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata["Name"],d.version) for d in m.distributions())))'],text=True))
(root/'environments.json').write_text(json.dumps(environments,indent=2)+'\n')
files={}
for path in sorted(root.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.gz','.pyc'):continue
    if path.name in ('archive-index.json','archive-files.json'):continue
    files[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=sha(path))
(root/'archive-files.json').write_text(json.dumps(files,indent=2)+'\n')
archive=root/'evidence.tar.gz'
if archive.exists():raise RuntimeError('refusing archive overwrite')
with tarfile.open(archive,'w:gz',dereference=True) as tar:
    for name in files:tar.add(root/name,arcname=name,recursive=False)
    tar.add(root/'archive-files.json',arcname='archive-files.json')
with tarfile.open(archive,'r:gz') as tar:
    for name,record in files.items():
        stream=tar.extractfile(name)
        assert stream is not None and hashlib.sha256(stream.read()).hexdigest()==record['sha256']
index=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),verified_files=len(files),
           external_artifacts='external-artifacts.json',scope='Raw measured sources, input-only VLA4 fixtures, results, failures and environment metadata. External model/repack weights are referenced by digest, not bundled.')
(root/'archive-index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(index),flush=True)
