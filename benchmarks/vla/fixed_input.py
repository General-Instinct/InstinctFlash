"""Replay real observations and actual sampled noise, independently of simulator feedback."""
from pathlib import Path
import hashlib
import numpy as np
from .policy_trace import read_trace
from .remote_policy import RemotePolicy
from .util import ConfigurationError,sha256_file,sha256_json,write_json_atomic

def compare_actions(a,b):
    a=np.asarray(a,dtype='>f8');b=np.asarray(b,dtype='>f8')
    if not a.size or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ConfigurationError('invalid replay actions')
    return {'identical':a.tobytes()==b.tobytes(),'max_abs':float(np.max(np.abs(a-b))),
            'reference_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256(b.tobytes()).hexdigest()}

def replay(traces,endpoint,identity,output,*,repeats=3):
    if type(repeats) is not int or repeats<2:raise ConfigurationError('replay needs at least two repetitions')
    output=Path(output)
    if output.exists():raise ConfigurationError('refusing to overwrite replay evidence')
    prepared=[]
    for path in traces:
        manifest,calls=read_trace(path)
        for key in ('model_id','model_revision','checkpoint_sha256','protocol'):
            if not identity.get(key) or manifest['identity'].get(key)!=identity[key]:raise ConfigurationError('replay checkpoint/protocol differs')
        if not calls[0][0].get('reset'):raise ConfigurationError('recording must begin with seeded reset')
        if not any(not request.get('reset') for request,response in calls):
            raise ConfigurationError('recording has no policy calls')
        if any('benchmark_noise' not in response for request,response in calls if not request.get('reset')):
            raise ConfigurationError('recording lacks actual initial noise tensors')
        prepared.append((path,manifest,calls))
    if not prepared:raise ConfigurationError('replay requires real recorded traces')
    rows=[];reference_repeats={}
    with_remote=RemotePolicy(endpoint,identity,timeout=300)
    try:
        for repeat in range(repeats):
            for path,manifest,calls in prepared:
                for index,(request,recorded) in enumerate(calls):
                    if request.get('reset'):
                        with_remote.reset_episode(request['prompt'],request['benchmark_seed']);continue
                    request=dict(request,benchmark_noise=recorded['benchmark_noise'])
                    response=with_remote.infer(request);action=response['action']
                    key=(str(path),index)
                    first=reference_repeats.setdefault(key,np.asarray(action).copy())
                    rows.append({'trace':str(path),'call':index,'repeat':repeat,
                        'against_recorded':compare_actions(recorded['action'],action),
                        'against_first_repeat':compare_actions(first,action),
                        'noise_identical':np.array_equal(recorded['benchmark_noise'],response['benchmark_noise']),
                        'action_values':np.asarray(action).tolist()})
    finally:with_remote.close()
    result={'schema_version':1,'synthetic':False,'scope':'Fixed real inputs and explicit initial noise; not fresh closed-loop quality evidence',
        'identity':identity,'repeats':repeats,'rows':rows,
        'traces':{str(p):sha256_file(Path(p)/'trace.json') for p,_,_ in prepared},
        'all_noise_identical':all(r['noise_identical'] for r in rows),
        'all_repeats_identical':all(r['against_first_repeat']['identical'] for r in rows),
        'all_recorded_actions_identical':all(r['against_recorded']['identical'] for r in rows)}
    result['sha256']=sha256_json(result);write_json_atomic(output,result);return result
