"""Exercise odd token counts and in-place state-token updates on Thor."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

out=Path(sys.argv[1])
if out.exists():raise RuntimeError('refusing overwrite')
home=Path.home()
checkpoint=home/'.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1'
frames=np.load(home/'ifl/t3_assets/calib_obs.npz')
obs=dict(image=frames['image'][3],wrist_image=frames['wrist_image'][3])
base=json.loads((home/'ifl/t3_assets/thor_assets.json').read_text())['synth_ids']['48']
fe=Pi05TorchFrontendThor(str(checkpoint),num_views=2,autotune=0,
    action_chunk=50,action_output='normalized')
records=[]
for n in (49,48):
    ids=(base+[base[-1]])[:n];changed=ids.copy();changed[-2]=base[1]
    fe.set_prompt(ids)
    assert fe.prefix_tokens==512+n and fe.Se==(512+n+15)//16*16
    def predict():
        np.random.seed(414)
        a=fe.infer(obs)['actions'];assert np.isfinite(a).all();return a
    predict();a=predict()
    if fe._S_lang > n:
        fe._lang_emb[n:].fill_(0.75)
        assert np.array_equal(a,predict()),'alignment rows leaked into attention'
        fe._lang_emb[n:].zero_()
    graph=fe._enc_ae_graph;lang_ptr=fe._lang_emb.data_ptr()
    fe.set_prompt(changed);b=predict()
    assert graph is fe._enc_ae_graph and lang_ptr==fe._lang_emb.data_ptr()
    assert not np.array_equal(a,b),'changed prompt was ignored'
    fe.set_prompt(ids);c=predict()
    assert np.array_equal(a,c),'restoring prompt failed to restore actions'
    records.append(dict(tokens=n,encoder_tokens=fe.Se,graph_reused=True,
        changed_tokens_change_actions=True,restored_tokens_restore_actions=True,
        alignment_rows_excluded=True,
        max_abs_change=float(np.max(np.abs(a-b)))))
out.write_text(json.dumps(dict(cases=records,scope='Token/graph contract check, not task-quality evidence',
    frontend_sha256=hashlib.sha256(Path(__import__('inspect').getfile(Pi05TorchFrontendThor)).read_bytes()).hexdigest()),indent=2)+'\n')
print(out.read_text())
