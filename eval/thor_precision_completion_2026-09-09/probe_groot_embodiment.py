"""Public Runtime contract probe for a checkpoint-declared GR00T embodiment.

Recorded cameras and synthetic states; no task-quality or timing certificate.
Each invocation owns a new package view and output, preserving all attempts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from instinctflash import Runtime
from instinctflash.descriptors.known import KNOWN_DECLARATIONS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--model-id', default='nvidia/GR00T-N1.7-3B')
parser.add_argument('--frames', type=Path, required=True)
parser.add_argument('--embodiment', required=True)
parser.add_argument('--precision', choices=['native', 'fp8'], required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
package = args.output / 'package'
package.mkdir()
checkpoint = args.checkpoint.resolve()
for entry in checkpoint.iterdir():
    if entry.name != 'instinctflash.json':
        (package / entry.name).symlink_to(entry)
doc = copy.deepcopy(KNOWN_DECLARATIONS['nvidia/GR00T-N1.7-3B'])
doc['execution']['model_id'] = args.model_id
doc['execution']['base_weights'] = str(checkpoint)
doc['execution']['embodiment_tag'] = args.embodiment.upper()
(package / 'instinctflash.json').write_text(json.dumps(doc, indent=2))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
api = Runtime.from_pretrained(package, precision=args.precision,
                              device='cuda:0', placement='in_process')
actions, labels = [], []
try:
    # Native execution constructs the policy lazily at the first reset.
    api.reset(prompt='pick up the cup')
    loop = api._backend._loop if args.precision == 'fp8' else api._backend._impl
    assert loop._policy.embodiment_tag.value == args.embodiment
    cfg = loop._policy.modality_configs
    keys = list(cfg['video'].modality_keys)
    frames = np.load(args.frames)
    def image(index):
        key = 'image' if index == 0 else 'wrist_image'
        return np.clip((frames[key][3].astype(np.float32)+1)*127.5, 0, 255).astype(np.uint8)
    state = {k: np.zeros(dim, dtype=np.float32) for k, dim in loop._state_dims.items()}
    # Reference poses need a valid rotation, regardless of the embodiment's
    # field naming (R1 uses wrist_eef, G1 uses wrist_eef_9d).
    for item in cfg['action'].action_configs or ():
        if getattr(item.format, 'name', None) == 'XYZ_ROT6D':
            value = state[item.state_key]
            if value.shape == (9,):
                value[3:9] = [1, 0, 0, 0, 1, 0]
    obs = {'video': {k: image(i) for i, k in enumerate(keys)}, 'state': state}
    def predict(label, observation):
        torch.manual_seed(191)
        result = api.predict(observation)
        a = np.asarray(result['action'])
        assert a.ndim == 2 and a.shape[0] == len(cfg['action'].delta_indices) and np.isfinite(a).all()
        assert list(result['actions']) == list(cfg['action'].modality_keys)
        actions.append(a.copy()); labels.append(label)
        np.savez(args.output / 'actions.npz', actions=np.stack(actions))
        print(label, a.shape, flush=True)
        return a
    api.reset(prompt='pick up the cup')
    baseline = predict('baseline', obs)
    assert np.array_equal(baseline, predict('repeat', obs))
    for key in keys:
        changed = {**obs, 'video': {**obs['video'], key: np.flip(obs['video'][key], axis=0).copy()}}
        assert not np.array_equal(baseline, predict('camera:'+key, changed)), key
    assert not np.array_equal(baseline, predict('state', {**obs, 'state': {k:v+.1 for k,v in state.items()}}))
    assert not np.array_equal(baseline, predict('prompt', {**obs, 'prompt':'move the cup to the plate'}))
    api.reset(prompt='pick up the cup')
    assert np.array_equal(baseline, predict('reset_restored', obs))
    stats = loop.backend_stats
    if args.precision == 'fp8':
        assert loop._runner.replays > 0
        assert all(w.dtype == torch.float8_e4m3fn for w in loop._frontend._vlsa_q_w)
        slot = loop._frontend._embodiment_id
        assert slot == json.loads((checkpoint/'embodiment_id.json').read_text())[args.embodiment]
    else:
        assert not any('float8' in str(p.dtype) for p in loop._policy.model.parameters())
        slot = None
    sources = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith(('instinctflash.', 'flash_rt.', 'groot_n17_iwm.')) and path and path.endswith('.py'):
            sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(model_id=args.model_id, embodiment=args.embodiment, precision=args.precision, checkpoint=str(checkpoint),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        declaration_sha256=hashlib.sha256((package/'instinctflash.json').read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256(args.frames.read_bytes()).hexdigest(),
        video_keys=keys, state_dimensions=loop._state_dims, labels=labels, action_shape=list(baseline.shape),
        backend_stats=stats, slot=slot, source_sha256=sources,
        actions_sha256=hashlib.sha256((args.output/'actions.npz').read_bytes()).hexdigest(),
        scope='Public API recorded-camera/synthetic-state checks, not task success or latency qualification.')
finally:
    api.close()
report['closed'] = True
(args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
