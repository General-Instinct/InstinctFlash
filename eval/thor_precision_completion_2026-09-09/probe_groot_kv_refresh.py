"""Verify live cross-KV refresh through actual GR00T DiT graphs on Thor.

Uses captured setup auxiliary tensors and synthetic backbone perturbations.
This is not a complete camera-to-action FP8 execution or quality benchmark.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
import flash_rt.core.quant.calibrator as calibrator

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
calibrator.CACHE_DIR=out.parent/'groot-refresh-calibration'
checkpoint=Path.home()/'.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495'
frontend=GrootN17TorchFrontendThor(str(checkpoint),num_views=4)
native_tensors_verified=0
if '--verify-native-dit' in sys.argv:
    from flash_rt.executors.torch_weights import MultiSafetensorsSource
    source=MultiSafetensorsSource(sorted(str(p) for p in checkpoint.glob('model-*.safetensors')),device='cpu')
    names={'q':'attn1.to_q','k':'attn1.to_k','v':'attn1.to_v','o':'attn1.to_out.0',
           'ada':'norm1.linear','ff_proj':'ff.net.0.proj','ff_down':'ff.net.2'}
    for layer in range(32):
        for attr,key in names.items():
            for suffix,extension in [('w','weight'),('b','bias')]:
                original=source.get(f'action_head.model.transformer_blocks.{layer}.{key}.{extension}')
                actual=getattr(frontend,f'_dit_{attr}_{suffix}')[layer]
                assert original.dtype==torch.bfloat16 and actual.dtype==torch.bfloat16
                expected=original.T.contiguous() if suffix=='w' else original
                assert torch.equal(actual.cpu(),expected),f'changed checkpoint tensor {layer}.{key}.{extension}'
                native_tensors_verified+=1
aux=torch.load(Path.home()/'thorcol/results/groot_aux.pt',map_location='cpu',weights_only=False)
frontend.set_prompt(aux=aux,prompt='recorded setup')
base=frontend._backbone_features.clone();mask=frontend._visual_pos_masks.clone()
assert torch.isfinite(base).all()
torch.manual_seed(191)
noise=torch.randn(1,40,132,device='cuda',dtype=torch.bfloat16)
state=torch.zeros(1,1,132,device='cuda')
def infer():
    result=frontend.infer(state,initial_noise=noise)
    assert torch.isfinite(result).all()
    return result.cpu().numpy().copy()
actions=[infer()]
graphs=frontend._dit_graphs
pointers=[t.data_ptr() for t in frontend._dit_cross_K+frontend._dit_cross_V]
changed=base.clone();changed[:,mask]*=0.5
frontend.update_backbone_features(changed,mask)
assert frontend._dit_graphs is graphs
assert pointers==[t.data_ptr() for t in frontend._dit_cross_K+frontend._dit_cross_V]
actions.append(infer());assert not np.array_equal(actions[0],actions[1]),'captured DiT ignored new image features'
frontend.update_backbone_features(base,mask)
actions.append(infer());assert np.array_equal(actions[0],actions[2]),'original features did not restore actions'
# A longer text prefix requires fresh attention descriptors and graphs.
longer=torch.cat([base,base[:,:1]],dim=1)
newmask=torch.cat([mask,torch.zeros(1,dtype=torch.bool,device=mask.device)])
frontend.update_backbone_features(longer,newmask)
assert not hasattr(frontend,'_dit_graphs') and not hasattr(frontend,'_dit_attn')
actions.append(infer());assert frontend._dit_graphs is not graphs
np.savez(out.with_suffix('.npz'),actions=np.stack(actions))
result=dict(same_geometry_keeps_kv_addresses=True,same_geometry_reuses_graphs=True,
            native_dit_checkpoint_tensors_verified=native_tensors_verified,
            image_feature_change_affects_actions=True,restored_actions_byte_equal=True,
            changed_geometry_rebuilds_graphs=True,
            artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
            frontend_sha256=hashlib.sha256(Path(inspect.getfile(GrootN17TorchFrontendThor)).read_bytes()).hexdigest(),
            scope='Actual BF16 DiT graphs; captured auxiliary setup and synthetic backbone perturbations; no full FP8 Runtime or task-quality claim')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
