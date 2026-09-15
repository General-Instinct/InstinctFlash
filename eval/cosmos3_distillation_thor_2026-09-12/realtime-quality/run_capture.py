"""Collect one complete frozen 128-request Thor deployed-quality variant."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--configuration',required=True)
    p.add_argument('--attention',choices=['native','cudnn'],required=True)
    a=p.parse_args()
    root=Path(__file__).parent
    plan=json.loads((root/'plan.json').read_text())
    assert all(digest(root/name)==sha for name,sha in plan['capture_sources'].items())
    assert digest(root/'protocol.json')==plan['protocol_sha256']
    protocol=json.loads((root/'protocol.json').read_text())
    assert {'configuration_id':a.configuration,'attention':a.attention} in plan['variants']
    assert digest(a.data)==plan['data']['sha256'] and a.data.stat().st_size==plan['data']['bytes']
    assert digest(root/'producer_capture.py')==plan['source_capture_sha256']
    config=next(c for c in protocol['configurations'] if c['id']==a.configuration)
    declaration=json.loads((a.checkpoint/'instinctflash.json').read_text())
    assert declaration['provenance']['arm_id']==config['arm_id']
    assert declaration['provenance']['state_key']=='student' and declaration['provenance']['student_updates']==64
    assert digest(a.checkpoint/'instinctcompress_manifest.json')==plan['packages'][a.configuration]['manifest_sha256']
    requests=[r for r in protocol['requests'] if r['configuration_id']==a.configuration]
    assert len(requests)==128
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope=plan['scope'],configuration_id=a.configuration,
        attention=a.attention,quality_certified=False,requests=[],protocol_sha256=plan['protocol_sha256'],
        plan_sha256=digest(root/'plan.json'),checkpoint_manifest_sha256=digest(a.checkpoint/'instinctcompress_manifest.json'))
    api=None
    with open('/tmp/thor_gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for name in list(os.environ):
                if name.startswith('IFL_COSMOS3_'):del os.environ[name]
            os.environ.update(IFL_COSMOS3_EXACT_POINTWISE='1',IFL_COSMOS3_LAYER_GRAPHS='1',IFL_COSMOS3_ATTENTION='native')
            import torch
            from instinctflash import Runtime,register
            from instinctflash.planners.planner import PassResult,Tier
            from instinct_compress.artifacts import verify_artifact
            from realtime_adapter import BACKBONE,Cosmos3RealtimeAdapter
            from runtime_assets import audit_vae_load
            from capture_request import capture_request
            assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(11,0)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            verify_artifact(a.checkpoint)
            register(BACKBONE,Cosmos3RealtimeAdapter)
            with audit_vae_load() as assets:
                api=Runtime.from_pretrained(a.checkpoint,strict=False,precision='native',placement='in_process',
                    tier_ceiling='numeric' if a.attention=='cudnn' else 'bitexact')
                api.reset(prompt='initialize quality worker')
            report['external_runtime_assets']=assets
            service=api._backend._impl._native_loop._service
            assert service.cfg.guidance==config['guidance'] and service.cfg.num_steps==1
            if a.attention=='cudnn':
                import cosmos_framework
                import cosmos_framework.model.generator.mot.attention as mot
                from cosmos3_iwm.conditioning_cache import SOURCE_HASHES
                from cosmos3_iwm.numeric_attention import NumericAttention
                vendor=Path(cosmos_framework.__file__).parent
                assert all(digest(vendor/name)==sha for name,sha in SOURCE_HASHES.items())
                assert torch.backends.cudnn.version()==91501
                owners=[layer.self_attn for layer in service.model.net.language_model.model.layers]
                assert len(owners)==28 and all(o.dispatch_attention_fn is mot.dispatch_attention for o in owners)
                service._ifl_numeric_attention=NumericAttention(mot,owners)
                api.plan.results.append(PassResult('experimental_student_cudnn',True,Tier.NUMERIC,'Frozen deployed quality capture; no certificate'))
            with np.load(a.data,allow_pickle=False) as archive:
                data={k:archive[k].copy() for k in archive.files}
            for index,request in enumerate(requests):
                others=subprocess.check_output(['/usr/sbin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
                assert not [pid for pid in others if int(pid)!=os.getpid()], 'GPU contention'
                i=request['source_index']
                assert (str(data['episode_id'][i]),int(data['frame_index'][i]))==(request['episode_id'],request['frame_index'])
                observation=dict(image=data['image'][i].copy(),state=np.asarray(data['state'][i],np.float32),prompt=str(data['prompt'][i]))
                api.reset(prompt=observation['prompt'])
                arrays,fields,trace=capture_request(api,observation,data['measured_action'][i],request['seed'])
                path=a.output/f'{index:04d}.npz'
                np.savez_compressed(path,**arrays,**fields)
                record=dict(request=request,arrays_sha256=digest(path),trace=trace)
                meta=path.with_suffix('.json');meta.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
                report['requests'].append(dict(index=index,json=meta.name,json_sha256=digest(meta),npz=path.name,npz_sha256=digest(path)))
                print(json.dumps(dict(completed=index+1,total=128,configuration=a.configuration,attention=a.attention)),flush=True)
            report.update(status='success',backend_stats=api._backend._impl.backend_stats(),execution_policy=api.execution_policy,
                torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        except BaseException:
            report.update(status='failed',error=traceback.format_exc())
            raise
        finally:
            report['sources']={str(Path(m.__file__).resolve()):digest(m.__file__) for name,m in list(sys.modules.items())
                if name.startswith(('instinctflash','instinct_compress','cosmos3_iwm','cosmos_framework','realtime_adapter','runtime_assets','request_seed','capture_request','producer_capture'))
                and getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
            report['script_sha256']=digest(__file__)
            if api is not None:api.close()
            (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
