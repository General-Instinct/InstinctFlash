"""Capture two real prefill MLP inputs in each of Edge layers 0 and 14."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from instinctflash import Runtime

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--fixture', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
with np.load(a.fixture, allow_pickle=True) as data:
    frame = np.asarray(Image.open(io.BytesIO(bytes(data['jpeg_0'][0][0]))).convert('RGB').resize((640, 540)))
api = Runtime.from_pretrained('nvidia/Cosmos3-Edge-Policy-DROID', precision='native', tier_ceiling='numeric')
api.reset(prompt='pick up the object')  # Materialize the lazy backend before installing capture wrappers.
layers = api._backend._impl._service.model.net.language_model.model.layers
payload = {}
patches = []
for index in (0, 14):
    layer = layers[index]
    mlp = layer.mlp_moe_gen
    assert getattr(mlp, 'gate_proj', None) is None
    assert mlp.up_proj.bias is None and mlp.down_proj.bias is None
    state = dict(up_weight=mlp.up_proj.weight.detach().cpu(), down_weight=mlp.down_proj.weight.detach().cpu(), samples=[])
    payload[str(index)] = state
    norm = layer.post_attention_layernorm_moe_gen
    original = norm.forward
    def make_forward(original, state):
        def forward(hidden):
            result = original(hidden)
            if len(state['samples']) < 2:
                state['samples'].append(dict(x=result.detach().cpu(), residual=hidden.detach().cpu()))
            return result
        return forward
    patches.append((norm, original))
    norm.forward = make_forward(original, state)
try:
    api.reset(prompt='pick up the object')
    action = np.asarray(api.predict(dict(image=frame, state=np.zeros(8, np.float32), prompt='pick up the object'))['action'])
    assert action.shape == (32, 8) and np.isfinite(action).all()
    for index, state in payload.items():
        assert len(state['samples']) == 2
        assert state['up_weight'].shape == (9216, 2048)
        assert state['down_weight'].shape == (2048, 9216)
        assert all(s['x'].shape == s['residual'].shape == (3093, 2048) for s in state['samples'])
        with torch.inference_mode():
            module = layers[int(index)].mlp_moe_gen
            for sample in state['samples']:
                x, residual = sample['x'].cuda(), sample['residual'].cuda()
                expected = residual + module(x)
                reconstructed = residual + torch.nn.functional.linear(
                    torch.relu(torch.nn.functional.linear(x, module.up_proj.weight)).square(), module.down_proj.weight)
                assert torch.isfinite(expected).all()
                assert torch.equal(expected.view(torch.uint8), reconstructed.view(torch.uint8))
                sample['native_output'] = expected.cpu()
    torch.save(payload, a.output)
    report = dict(ok=True, scope='Real native prefill operands, two branches each at layers 0 and 14; not a quality screen',
                  payload_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), native_reconstruction_byte_equal=True,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
                  execution_policy=api.execution_policy,
                  sources={str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                    for name, m in list(sys.modules.items()) if name.startswith(('instinctflash', 'cosmos3_iwm', 'cosmos_framework'))
                    and getattr(m, '__file__', None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()})
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
finally:
    for norm, original in reversed(patches):
        norm.forward = original
    api.close()
