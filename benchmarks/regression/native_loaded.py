"""Before timing, bind all released tensors to the actual native eager model."""
import hashlib
import json
from pathlib import Path


def verify_loaded(policy, checkpoint, *, output_dir):
    import torch
    from safetensors import safe_open

    assert policy.eval_bf16 is True, 'The original policy must itself select BF16'
    model = policy.trained_model
    assert type(model).__module__ == 'groot.vla.model.dreamzero.base_vla'
    state = model.state_dict()
    root = Path(checkpoint)
    index = root / 'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    checked = {}
    for shard in shards:
        with safe_open(str(root / shard), framework='pt', device='cpu') as archive:
            for name in archive.keys():
                assert name in state, name
                actual = state[name].detach().cpu()
                expected = archive.get_tensor(name).to(dtype=actual.dtype)
                assert actual.shape == expected.shape and torch.isfinite(actual).all(), name
                a = actual.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                b = expected.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                assert a == b, name
                checked[name] = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                                     sha256=hashlib.sha256(a).hexdigest())
                del actual, expected, a, b
    assert checked and all(not p.is_meta for p in model.parameters())
    dit = {name for name in state if name.startswith('action_head.model.')}
    assert dit and dit <= checked.keys()
    assert all(state[name].dtype == torch.bfloat16 for name in dit)
    result = dict(status='passed', native_eval_bf16=True, tensors=checked,
                  dit_tensors=len(dit), checkpoint=str(root),
                  scope='All stored tensors equal native destination-dtype values; unchanged eager forward; no task-quality claim')
    output = Path(output_dir) / 'native_loaded_values.json'
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return dict(status='passed', tensors=len(checked), dit_tensors=len(dit),
                receipt=output.name, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
