"""Measure a pinned live endpoint using real recorded observations; includes transport."""
from pathlib import Path
from .execution_evidence import execution_profile
from .policy_trace import read_trace
from .remote_policy import RemotePolicy
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic


def measure(trace, endpoint, receipt_path, output, *, warmup=8, iterations=128):
    if type(warmup) is not int or warmup<1 or type(iterations) is not int or iterations<1:
        raise ConfigurationError('positive warmup and iterations required')
    output=Path(output)
    if output.exists():raise ConfigurationError('refusing to overwrite endpoint timings')
    identity=load_json(Path(receipt_path));profile=execution_profile(identity)
    manifest,calls=read_trace(trace)
    for key in ('model_id','model_revision','checkpoint_sha256','upstream_sha256'):
        if manifest['identity'][key]!=identity[key]:raise ConfigurationError(f'trace differs in {key}')
    reset=next((request for request,response in calls if request.get('reset')),None)
    inputs=[{k:v for k,v in request.items() if k!='benchmark_noise'}
            for request,response in calls if not request.get('reset')]
    if not reset or not inputs:raise ConfigurationError('trace requires reset and real observations')
    if any(request.get('compute_kv_cache') or request.get('commit') for request in inputs):
        raise ConfigurationError('cache-commit protocols need their own latency driver; this probe measures inference-only chunks')
    remote=RemotePolicy(endpoint,identity)
    samples=[];digests=[]
    try:
        remote.reset_episode(reset['prompt'],reset['benchmark_seed'])
        for i in range(warmup+iterations):
            response=remote.infer(inputs[i%len(inputs)])
            if i>=warmup:
                samples.append(remote.timings[-1]['roundtrip_ms'])
                action=response['action'];digests.append(sha256_json(action.tolist()))
    finally:remote.close()
    result={'schema_version':1,'synthetic':False,'mode':identity['execution']['mode'],
            'execution_profile':profile,'hardware':identity['hardware'],
            'executed_actions':identity['execution']['action_shape'][0],
            'warmup':warmup,'iterations':iterations,'samples_ms':samples,'action_sha256':digests,
            'trace_sha256':sha256_file(Path(trace)/'trace.json'),
            'scope':'Observed endpoint roundtrip with real recorded observations, including serialization and transport. '
                    'Fixed-input replay, not a simulator success rate or a sustained controller guarantee. '
                    'Episode seed is reset once; recorded noise tensors are not injected.'}
    result['sha256']=sha256_json(result);write_json_atomic(output,result)
    return result
