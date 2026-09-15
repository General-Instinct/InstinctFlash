"""Isolated experiment: permit capture without changing the production planner."""
import argparse, dataclasses, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['stock','current','capture','capture_no_prefix','capture_cpu_preprocess'],required=True);p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=64);a=p.parse_args()
if a.arm.startswith('capture'):
    from instinctflash.passes.generic.graph_capture import GraphCaptureApplicable
    original=GraphCaptureApplicable.evaluate
    def experimental(self,spec,deployment):
        result=original(self,spec,dataclasses.replace(deployment,device=None))
        return dataclasses.replace(result, reason='EXPERIMENT ONLY: bypass device-wide capture rejection; '+result.reason)
    GraphCaptureApplicable.evaluate=experimental
if a.arm=='capture_no_prefix':os.environ['IFL_VLA2_PREFIX_GRAPH']='0'
if a.arm=='capture_cpu_preprocess':os.environ['IFL_VLA2_GPU_PREPROCESS']='0'
from benchmarks.vla.instinctflash_driver import RuntimeArm,STOCK_ARMS
outputs=[]
cls=STOCK_ARMS['lingbot_vla_v2'] if a.arm=='stock' else RuntimeArm
orig_predict=cls.predict
def predict(self,request):
    out=orig_predict(self,request);outputs.append(np.asarray(out).copy());return out
cls.predict=predict
from benchmarks.vla.latency_probe import measure
r=Path('/home/guanming/ifl_eval/next_steps_20260906')
result=measure(r/'input','robbyant/lingbot-vla-v2-6b-robotwin','0451855729ec904f970600e0aec8b84661423afe','stock' if a.arm=='stock' else 'runtime_default',a.output,warmup=8,iterations=a.iterations)
np.savez_compressed(a.output+'.actions.npz',actions=np.stack(outputs))
print(json.dumps({'arm':a.arm,'p50':float(np.median(result['samples_ms'])),'p99':float(np.percentile(result['samples_ms'],99))}),flush=True)
