"""Benchmark-only raw LIBERO bridge to the actual 50-action Runtime."""
import argparse, hashlib, inspect, json, random
from pathlib import Path
import numpy as np
import torch
from instinctflash import Runtime
from instinctflash.serving import WebsocketPolicyServer, default_metadata
def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
p=argparse.ArgumentParser(); p.add_argument('--precision',choices=['native','fp8'],required=True);p.add_argument('--port',type=int,default=19051);p.add_argument('--identity',type=Path,required=True);args=p.parse_args()
model='lerobot/pi05_libero_finetuned_v044';rev='8e174154ef5f6c60a8da12ae99c303d8963138c1'
rt=Runtime.from_pretrained(model,revision=rev,device='cuda:0',precision=args.precision,placement='in_process')
rt.reset(prompt='initializing benchmark')
cfg=json.loads((Path(rt._checkpoint.path)/'config.json').read_text())
assert cfg['n_action_steps']==cfg['chunk_size']==50
assert cfg['output_features']['action']['shape']==[7]
assert cfg['num_inference_steps']==10
identity={'protocol':'pi05-public-libero-50-v1','model_id':model,'model_revision':rev,'precision':rt.precision,'n_action_steps':50,'action_dim':7,'action_nfe':10,'observation_processor':'native LeRobot preprocess_observation + LiberoProcessorStep','seed_rule':'per request torch/numpy/python seed = episode_seed + action_index; no queue resets within episode','sources':{name:hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest() for name,obj in [('Runtime',Runtime),('LiberoProcessorStep',LiberoProcessorStep),('preprocess_observation',preprocess_observation)]}}
class Server(WebsocketPolicyServer):
    def _step(self, obs):
        if obs.get('reset'):
            if obs.get('benchmark_identity_sha256')!=sha256_json(identity):raise ValueError('benchmark identity mismatch')
            seed=obs.get('benchmark_seed')
            if type(seed) is not int:raise ValueError('integer benchmark seed required')
            result=super()._step(dict(obs));self.seed=seed;self.index=0
            result.update(benchmark_seed=seed,benchmark_identity_sha256=sha256_json(identity));return result
        if not hasattr(self,'seed'):raise ValueError('explicit benchmark reset required')
        raw=obs['libero_observation']
        batch={'pixels':{'image':np.asarray(raw['agentview_image']),'image2':np.asarray(raw['robot0_eye_in_hand_image'])},'robot_state':{'eef':{'pos':np.asarray(raw['robot0_eef_pos'])[None], 'quat':np.asarray(raw['robot0_eef_quat'])[None]},'gripper':{'qpos':np.asarray(raw['robot0_gripper_qpos'])[None]}}}
        processed=LiberoProcessorStep().observation(preprocess_observation(batch))
        observation={k:v.squeeze(0).numpy() for k,v in processed.items()};observation['prompt']=obs['prompt']
        if observation['observation.state'].shape!=(8,):raise ValueError('native state shape mismatch')
        seed=self.seed+self.index;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        result=super()._step(observation);self.index+=1
        if result['action'].shape!=(7,) or not np.isfinite(result['action']).all():raise ValueError('invalid action')
        return result
metadata=default_metadata(rt);metadata['benchmark_identity']=identity
args.identity.write_text(json.dumps(identity,indent=2)+'\n')
try:Server(rt,host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()
finally:rt.close()
