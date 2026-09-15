"""Expert-only diagnostic; never connects to a policy or contributes task scores."""
import argparse,json,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source-robotwin-v2'))
from benchmarks.vla import robotwin_driver as d
from benchmarks.vla.util import load_json,write_json_atomic,sha256_file
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(exist_ok=False)
request=load_json(ROOT/'robotwin-jobs/41744612719dd46890f65691.json')['request']
scene=load_json(ROOT/'robotwin-scenes.json')['scenes'][d.scene_key(request)]
client=d.import_client(Path('/home/ubuntu/RoboTwin'),Path('/home/ubuntu/lingbot-va'))
def probe(env,args,kwargs):
 settings=dict(args,eval_mode=True,render_freq=0,eval_video_log=False)
 settings.pop('eval_video_save_dir',None)
 env.setup_demo(now_ep_num=0,seed=scene['resolved_seed'],is_test=True,**settings)
 initial=d.initial_state(env.get_obs())
 episode=env.play_once()
 def capture():
  return dict(plan_success=bool(env.plan_success),success=bool(env.check_success()),deskbin=env.deskbin.get_pose().p.tolist(),garbage=[x.get_pose().p.tolist() for x in env.sphere_lst])
 before=capture();env.close_env();after=capture()
 report=dict(purpose=__doc__,seed=scene['resolved_seed'],initial=initial,expected_initial=scene['initial_state_sha256'],before_close=before,after_close=after,script_sha256=sha256_file(Path(__file__)))
 write_json_atomic(out/'report.json',report)
 print(json.dumps(report),flush=True)
 return {'success':before['success']}
with tempfile.TemporaryDirectory() as work:d.with_upstream_setup(client,request,probe,work)
