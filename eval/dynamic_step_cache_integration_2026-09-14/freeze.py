"""Read-only source/dependency inventory, without importing CUDA libraries."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'source/benchmarks/regression'))
from runtime_bundle import build_manifest

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    path = root/name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+'\n')

native = Path('/home/guanming/thorcol/dreamzero')
head = native/'groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py'
assert digest(head) == '7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845'
external = {str(native/name): value for name, value in build_manifest(native)['files'].items()}
fixture = Path('/home/guanming/ifl_eval/four_frameworks_20260913/source/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')
external[str(fixture)] = digest(fixture)
write('source_manifest.json', build_manifest(root/'source'))
write('external_manifest.json', external)
write('driver_manifest.json', {name: digest(root/name) for name in
    ('benchmark.py', 'run.py', 'load_memory.py', 'freeze.py', 'protocol.json')})
packages = {}
for name in ('torch', 'torchvision', 'transformers', 'diffusers', 'triton', 'numpy',
             'safetensors', 'huggingface-hub', 'hydra-core', 'omegaconf', 'flash-attn'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
write('environment.json', dict(python=sys.version, packages=packages,
    manifest_scope='source files and native binaries in isolated Runtime source and external DreamZero tree; fixture; drivers',
    weights='Pinned HF revision; existing full checkpoint loader inventory retained in each runtime receipt'))
print(json.dumps(dict(source_files=len(build_manifest(root/'source')['files']), external_files=len(external))))
