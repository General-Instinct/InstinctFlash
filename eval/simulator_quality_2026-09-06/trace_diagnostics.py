"""Descriptive controller-trace differences; later closed-loop actions are not matched-input kernel tests."""
import argparse,json,struct
from pathlib import Path
from benchmarks.vla.result import validate_result
from benchmarks.vla.util import load_json,write_json_atomic,sha256_file
root=Path(__file__).parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,default=root/'joint/vla2-run')
parser.add_argument('--output',type=Path,default=root/'vla2-trace-diagnostics.json')
args=parser.parse_args();run=args.run;plan=load_json(run/'plan.json');pairs={};count=0
for job in plan['jobs']:
 path=run/'results'/(job['job_id']+'.json')
 if not path.exists():continue
 result=load_json(path);validate_result(result,job);count+=1
 req=job['request'];pairs.setdefault(req['pair_id'],{})[req['arm']['role']]=(req,result)
rows=[]
for pair_id,pair in pairs.items():
 if set(pair)!={'control','treatment'}:continue
 (req,left),(_,right)=pair['control'],pair['treatment']
 if left['provenance']['scene_sha256']!=right['provenance']['scene_sha256']:raise RuntimeError('scene mismatch')
 width=req['adapter']['contract']['actions']['controller_dim'];a=left['metrics']['action_values'];b=right['metrics']['action_values']
 indices=[i for i,(x,y) in enumerate(zip(a,b)) if struct.pack('>d',float(x))!=struct.pack('>d',float(y))]
 first=indices[0] if indices else min(len(a),len(b)) if len(a)!=len(b) else None
 rows.append({'pair_id':pair_id,'suite_id':req['suite_id'],'task':req['task'],'resolved_seed':left['resolved_seed'],
 'control_success':left['metrics']['success'],'treatment_success':right['metrics']['success'],
 'control_steps':left['metrics']['executed_steps'],'treatment_steps':right['metrics']['executed_steps'],
 'first_different_controller_step':None if first is None else first//width,
 'first_different_controller_component':None if first is None else first%width,
 'first_different_control_value':None if first is None or first>=len(a) else a[first],
 'first_different_treatment_value':None if first is None or first>=len(b) else b[first],
 'first_step_max_abs_delta':max(abs(x-y) for x,y in zip(a[:width],b[:width])),
 'overlapping_steps_max_abs_delta':max(abs(x-y) for x,y in zip(a,b))})
value={'schema_version':1,'plan_id':plan['plan_id'],'complete':count==len(plan['jobs']),'results':count,'expected_results':len(plan['jobs']),
 'analysis_sha256':sha256_file(Path(__file__)),
 'scope':'Controller-space diagnostics on frozen paired scenes. The first action uses matched reset observations and declared seeds; upstream repeatability still needs separate evidence. After trajectories diverge, same-index actions do not have identical observations, so overlapping-step deltas are not isolated kernel error bounds.',
 'pairs':rows}
write_json_atomic(args.output,value)
print(json.dumps({'complete':value['complete'],'pairs':len(rows),'success_discordances':[r for r in rows if r['control_success']!=r['treatment_success']]},indent=2))
