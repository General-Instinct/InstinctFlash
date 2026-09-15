import json

import pytest
import torch
from safetensors.torch import save_file
from instinctflash.runtime.dreamzero_checkpoint import (
    DIT_TARGET, prepare_full_checkpoint, read_tensor_headers, validate_dit_coverage,
)


class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)


def checkpoint(tmp_path):
    config = {'action_head_cfg': {'config': {'train_architecture': 'full',
              'diffusion_model_cfg': {'_target_': DIT_TARGET}}}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    save_file({'action_head.model.proj.weight': torch.ones(2,2,dtype=torch.bfloat16),
               'action_head.model.proj.bias': torch.zeros(2,dtype=torch.bfloat16),
               'action_head.vae.other': torch.zeros(1)}, str(tmp_path/'model.safetensors'))
    return config


def test_view_changes_only_preload_flag_and_keeps_original_weights(tmp_path):
    config = checkpoint(tmp_path)
    original = (tmp_path/'config.json').read_bytes()
    rng = torch.get_rng_state().clone()
    view, receipt = prepare_full_checkpoint(tmp_path, TinyDiT)
    from pathlib import Path
    path = Path(view.name)
    try:
        changed = json.loads((path/'config.json').read_text())
        assert changed['action_head_cfg']['config'].pop('skip_component_loading') is True
        assert changed == config
        assert (path/'model.safetensors').resolve() == tmp_path/'model.safetensors'
        assert receipt['verified_dit_tensors'] == 2
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        view.cleanup()
    assert not path.exists() and (tmp_path/'model.safetensors').exists()
    assert (tmp_path/'config.json').read_bytes() == original


@pytest.mark.parametrize('bad', ['missing', 'extra', 'shape', 'dtype', 'lora'])
def test_incomplete_or_incompatible_full_checkpoint_cannot_skip_base(tmp_path,bad):
    config = checkpoint(tmp_path)
    headers = read_tensor_headers(tmp_path)
    key = 'action_head.model.proj.weight'
    if bad == 'missing': del headers[key]
    elif bad == 'extra': headers['action_head.model.extra'] = headers[key]
    elif bad == 'shape': headers[key]['shape'] = [3,2]
    elif bad == 'dtype': headers[key]['dtype'] = 'F32'
    else: config['action_head_cfg']['config']['train_architecture'] = 'lora'
    with pytest.raises(ValueError):
        validate_dit_coverage(config, headers, TinyDiT)


def test_index_cannot_hide_missing_tensors(tmp_path):
    checkpoint(tmp_path)
    headers = read_tensor_headers(tmp_path)
    index = {'weight_map': {key:'model.safetensors' for key in headers}}
    index['weight_map']['absent'] = 'model.safetensors'
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='absent'):
        read_tensor_headers(tmp_path)


def test_direct_factory_rejects_changed_geometry_before_loading(tmp_path,monkeypatch):
    from instinctflash.runtime import dreamzero_checkpoint as loader
    checkpoint(tmp_path)
    monkeypatch.setattr(loader, 'load_full_dit_bf16', lambda _: pytest.fail('loaded mismatched configuration'))
    with pytest.raises(ValueError, match='disagrees'):
        loader.load_full_dit_for_native(str(tmp_path), dim=999)


def test_direct_view_records_constructor_change(tmp_path):
    from pathlib import Path
    checkpoint(tmp_path)
    view, receipt = prepare_full_checkpoint(tmp_path,TinyDiT,direct_bf16=True)
    try:
        cfg = json.loads((Path(view.name)/'config.json').read_text())
        dit = cfg['action_head_cfg']['config']['diffusion_model_cfg']
        assert dit['_target_'].endswith('.load_full_dit_for_native')
        assert dit['checkpoint_path'] == str(tmp_path)
        assert receipt['direct_bf16_dit'] is True
        assert 'RNG consumption differs' in receipt['scope']
    finally:
        view.cleanup()
