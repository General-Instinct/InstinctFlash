"""Real FP8 VLSA + native-weight BF16 DiT on Thor, using staged LLM features."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
from flash_rt.models.groot_n17.vlsa_runner import GrootN17VlsaRunner
from flash_rt.models.groot_n17 import calibration as cal

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
checkpoint=Path.home()/'.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495'
fe=GrootN17TorchFrontendThor(str(checkpoint),num_views=4)
aux=torch.load(Path.home()/'thorcol/results/groot_aux.pt',map_location='cpu',weights_only=False)
cos,sin=cal.build_vit_rope_tables(aux['grid_thw'].tolist(),head_dim=64,theta=10000.0,spatial_merge_size=2,device='cuda')
vit=cal.calibrate_vit(fe,aux['pixel_features'].cuda().float(),cos.float(),sin.float(),num_views=4)
ds=cal.calibrate_deepstack(fe,vit['deepstack_taps'])
llm=cal.calibrate_llm(fe,aux['llm_input_embeds'].cuda().float(),
    aux['rope_cos'][0].cuda().float(),aux['rope_sin'][0].cuda().float(),
    aux['visual_pos_masks'][0].cuda(),ds['features'])['llm_final']
mask=aux['visual_pos_masks'][0].cuda()
reference=cal.calibrate_vlsa(fe,llm)['backbone_features']
runner=GrootN17VlsaRunner(fe)
features=[];actions=[]
torch.manual_seed(191)
noise=torch.randn(1,40,132,device='cuda',dtype=torch.bfloat16)
state=torch.zeros(1,1,132,device='cuda')
changed=llm.clone();changed[:,mask]*=0.5
for value in (llm,changed,llm):
    features.append(runner(value,mask).float().cpu().numpy().copy())
    action=fe.infer(state,initial_noise=noise)
    assert torch.isfinite(action).all()
    actions.append(action.cpu().numpy().copy())
assert np.array_equal(features[0],features[2]) and np.array_equal(actions[0],actions[2])
assert not np.array_equal(features[0],features[1]) and not np.array_equal(actions[0],actions[1])
graph=runner.graph
longer=torch.cat([llm,llm[:,:1]],dim=1)
long_mask=torch.cat([mask,torch.zeros(1,dtype=torch.bool,device='cuda')])
runner(longer,long_mask);assert runner.graph is not graph
assert torch.isfinite(fe.infer(state,initial_noise=noise)).all()
ref=reference.float().cpu().numpy()
delta=features[0]-ref
np.savez(out.with_suffix('.npz'),features=np.stack(features),actions=np.stack(actions),shadow_reference=ref)
result=dict(vlsa_weight_dtype='float8_e4m3fn',dit_weights_and_compute='checkpoint BF16',
    actual_vlsa_graph_replays=runner.replays,changed_input_affects_features_and_actions=True,
    restored_features_and_actions_byte_equal=True,changed_token_count_rebuilds_graph=True,
    vs_shadow_maxabs=float(np.abs(delta).max()),vs_shadow_rmse=float(np.sqrt(np.mean(delta**2))),
    artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256(Path(inspect.getfile(GrootN17VlsaRunner)).read_bytes()).hexdigest(),
    scope='Actual FP8 VLSA and BF16 DiT; staged shadow-LLM features from recorded auxiliary setup; synthetic feature change; not live-camera Runtime or quality certificate')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
