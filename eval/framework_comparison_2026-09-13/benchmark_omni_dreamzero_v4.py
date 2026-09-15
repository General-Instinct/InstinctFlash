"""Pinned Omni DreamZero action-policy timing. Run each model in a fresh, locked process."""
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
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--compile', action='store_true')
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
    report={'framework':'vllm-omni','compile_requested':a.compile,'model':a.model,'checkpoint':path,
            'scope':'local Omni request to CPU actions, including worker IPC; no network',
            'packages':{n:metadata.version(n) for n in ['vllm','vllm-omni','torch','transformers']},
            'fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol':{'schedule':'upstream DreamZero deploy defaults','steps':16,'cfg_scale':5.,'sigma_shift':5.,'step_cache':True,'seed':1140,'horizon':24,
                        'format_prompt_as_json':'Edge' in a.model,'precision':'bfloat16'},'calls':[]}
    engine=None
    actions=[]
    try:
        start=time.perf_counter()
        engine=Omni(model=path,deploy_config=a.deploy,enforce_eager=not a.compile,dtype='bfloat16',model_paths={'tokenizer':'/home/guanming/.cache/huggingface/hub/models--google--umt5-xxl/snapshots/66cb9e7e85526fe440a945569e42c72fb6cbc0ad'})
        report['setup_seconds']=time.perf_counter()-start
        fixture=np.load(a.fixture,allow_pickle=True)
        for i in range(a.warmup+a.iterations):
            episode,cycle=divmod(i,3)
            frames=[0] if cycle==0 else [1,2,3,4]
            prompt='pick up the object' if episode*3<8 else 'place the object down'
            obs={'observation/joint_position':np.zeros(7,np.float32),
                 'observation/cartesian_position':np.zeros(6,np.float32),
                 'observation/gripper_position':np.zeros(1,np.float32),
                 'prompt':prompt,'session_id':f'benchmark-{episode}'}
            for camera,key in enumerate(['observation/exterior_image_0_left','observation/exterior_image_1_left','observation/wrist_image_left']):
                imgs=[np.asarray(Image.open(io.BytesIO(bytes(fixture['frame0_0'][camera] if j==0 else fixture['jpeg_0'][j-1][camera]))).convert('RGB')).copy() for j in frames]
                obs[key]=imgs[0] if cycle==0 else np.stack(imgs)
            extra={'robot_obs':obs,'reset':cycle==0,'session_id':obs['session_id']}
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
            assert action.shape in [(24,8),(1,24,8)] and np.isfinite(action).all(),action.shape
            actions.append(action.copy())
            row={'i':i,'phase':'warmup' if i<a.warmup else 'measured','ms':ms,'shape':list(action.shape)}
            report['calls'].append(row);print(row,flush=True)
        samples=[x['ms'] for x in report['calls'] if x['phase']=='measured']
        if not samples:
            report.update(ok=True,smoke_only=True)
            return
        report.update(ok=True,p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),p99_ms=float(np.percentile(samples,99)))
        np.savez_compressed(a.output.with_suffix('.npz'),actions=np.stack(actions))
    except Exception as e:
        report.update(ok=False,error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        if engine is not None:engine.close()
    if not report['ok']:raise SystemExit(1)

if __name__=='__main__':main()
