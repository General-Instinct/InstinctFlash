"""Pinned Omni action-policy timing. Run each model in a fresh, locked process."""
import argparse
import hashlib
import io
import json
import os
import time
import traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--deploy', required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iterations', type=int, default=30)
    a=p.parse_args()
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    import importlib.metadata as metadata
    path=snapshot_download(a.model,local_files_only=True)
    report={'framework':'vllm-omni','model':a.model,'checkpoint':path,
            'scope':'local Omni request to CPU actions, including worker IPC; no network',
            'packages':{n:metadata.version(n) for n in ['vllm','vllm-omni','torch','transformers']},
            'fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol':{'steps':4,'guidance':3.,'shift':5.,'history_length':1,'horizon':32,
                        'format_prompt_as_json':'Edge' in a.model,'precision':'bfloat16'},'calls':[]}
    engine=None
    actions=[]
    try:
        start=time.perf_counter()
        engine=Omni(model=path,deploy_config=a.deploy,enforce_eager=True,dtype='bfloat16')
        report['setup_seconds']=time.perf_counter()-start
        fixture=np.load(a.fixture,allow_pickle=True)
        for i in range(a.warmup+a.iterations):
            image=np.asarray(Image.open(io.BytesIO(bytes(fixture['jpeg_0'][i%12][0]))).convert('RGB').resize((640,540)))
            state=np.full(8,.01*(i%3),np.float32)
            prompt='pick up the object' if i<8 else 'place the object down'
            obs={'observation/image':image,'observation/joint_position':state[:7],
                 'observation/gripper_position':state[7:],'prompt':prompt}
            extra=dict(robot_obs=obs,num_steps=4,guidance=3.,shift=5.,history_length=1,
                       action_chunk_size=32,raw_action_dim=8,image_height=540,image_width=640,
                       conditioning_fps=15,domain_name='droid_lerobot',format_prompt_as_json='Edge' in a.model,
                       seed=int(np.random.default_rng(0).integers(0,2**31)))
            sp=OmniDiffusionSamplingParams(extra_args=extra)
            start=time.perf_counter()
            result=engine.generate(prompt,sampling_params_list=[sp])
            if not result:raise RuntimeError('empty Omni output')
            payload=result[0].multimodal_output
            if not isinstance(payload,dict):raise RuntimeError(f'non-dict action payload: {type(payload)}')
            value=payload.get('actions')
            if value is None and isinstance(payload.get('payload'),dict):value=payload['payload'].get('actions')
            if value is None:raise RuntimeError(f'no actions in payload: {list(payload)}')
            if isinstance(value,torch.Tensor):value=value.detach().float().cpu().numpy()
            action=np.asarray(value)
            ms=1000*(time.perf_counter()-start)
            assert action.shape==(32,8) and np.isfinite(action).all(),action.shape
            actions.append(action.copy())
            row={'i':i,'phase':'warmup' if i<a.warmup else 'measured','ms':ms,'shape':list(action.shape)}
            report['calls'].append(row);print(row,flush=True)
        samples=[x['ms'] for x in report['calls'] if x['phase']=='measured']
        report.update(ok=True,p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),p99_ms=float(np.percentile(samples,99)))
        np.savez_compressed(a.output.with_suffix('.npz'),actions=np.stack(actions))
    except Exception as e:
        report.update(ok=False,error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        if engine is not None:engine.close()
    if not report['ok']:raise SystemExit(1)

if __name__=='__main__':main()
