"""Verify the loaded pi05 tensors; upstream may swallow state-dict loading errors."""
from pathlib import Path
from .util import ConfigurationError, sha256_file


def verify_loaded_weights(policy, snapshot):
    import torch
    from safetensors import safe_open
    path = Path(snapshot) / 'model.safetensors'
    state = policy.state_dict()
    aliases = {}
    for name, tensor in state.items():
        key = (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        aliases.setdefault(key, []).append(name)
    tied = {}
    with safe_open(str(path), framework='pt', device='cpu') as file:
        names = set(file.keys())
        if names - set(state): raise ConfigurationError('checkpoint contains unaccounted model keys')
        checked = set()
        for name, tensor in state.items():
            if name in names: source = name
            else:
                key = (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
                candidates = [n for n in aliases[key] if n in names]
                if not candidates: raise ConfigurationError(f'unloaded checkpoint tensor: {name}')
                source = candidates[0]; tied[name] = source
            if source in checked: continue
            reference = file.get_tensor(source).to(dtype=tensor.dtype)
            actual = tensor.detach().cpu().contiguous()
            if actual.shape != reference.shape or not torch.equal(actual.reshape(-1).view(torch.uint8), reference.contiguous().reshape(-1).view(torch.uint8)):
                raise ConfigurationError(f'loaded checkpoint tensor differs: {name}')
            checked.add(source)
    return {'checkpoint_sha256': sha256_file(path), 'checked_tensors': len(checked), 'verified_aliases': tied}
