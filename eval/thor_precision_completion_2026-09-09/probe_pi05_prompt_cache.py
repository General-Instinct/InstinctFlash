"""Check graph/profile restoration against retained actions across token lengths."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',type=Path)
args=parser.parse_args();out=args.output
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
home=Path.home()
checkpoint=home/'.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/8e174154ef5f6c60a8da12ae99c303d8963138c1'
frames=np.load(home/'ifl/t3_assets/calib_obs.npz')
obs=dict(image=frames['image'][3],wrist_image=frames['wrist_image'][3])
base=json.loads((home/'ifl/t3_assets/thor_assets.json').read_text())['synth_ids']['48']
fe=Pi05TorchFrontendThor(str(checkpoint),num_views=2,autotune=0,
    action_chunk=50,action_output='normalized',prompt_cache_size=3)
records=[];references={};graphs={};all_actions=[]
for n in (49,48,37,49,37,48):
    ids=(base+[base[-1]])[:n]
    np.random.seed(414);torch.manual_seed(414)
    t0=time.perf_counter();fe.set_prompt(ids)
    a=fe.infer(obs)['actions'].copy()
    elapsed=(time.perf_counter()-t0)*1000
    assert np.isfinite(a).all() and a.shape==(50,32)
    assert fe.prefix_tokens==512+n and fe._real_data_calibrated
    if n in references:
        assert graphs[n] is fe._enc_ae_graph,'length revisit rebuilt graph'
        assert np.array_equal(references[n],a),'restored profile changed fixed-input actions'
    else:
        references[n]=a.copy();graphs[n]=fe._enc_ae_graph
    records.append(dict(tokens=n,set_prompt_and_infer_ms=elapsed,
                        hits=fe.prompt_cache_hits,misses=fe.prompt_cache_misses))
    all_actions.append(a)
assert fe.prompt_cache_hits==3 and fe.prompt_cache_misses==3
assert len(fe._prompt_profiles)<=3
# Same-length changed IDs must still take effect after restoring a cached graph.
ids=base.copy();ids[-2]=base[1]
fe.set_prompt(ids);np.random.seed(414)
changed=fe.infer(obs)['actions'].copy()
assert not np.array_equal(changed,references[48])
fe.set_prompt(base);np.random.seed(414)
assert np.array_equal(fe.infer(obs)['actions'],references[48])
# Force cache eviction, then check that all raw pointers are still usable.
fe._prompt_cache_size=1
for n in (46,49,48):
    fe.set_prompt((base+[base[-1]])[:n])
    assert np.isfinite(fe.infer(obs)['actions']).all()
    assert len(fe._prompt_profiles)<=1
np.savez(out.with_suffix('.npz'),actions=np.stack(all_actions),changed=changed)
result=dict(cases=records,cross_length_restore_byte_equal=True,
            graph_reused=True,same_length_updates_preserved=True,eviction_executes=True,
            frontend_sha256=hashlib.sha256(Path(inspect.getfile(Pi05TorchFrontendThor)).read_bytes()).hexdigest(),
            actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
            scope='Prompt profile restoration and eviction, not a task-quality certificate')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
