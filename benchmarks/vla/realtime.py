"""Explicit deployment budgets and measured latency; no inference from simulator wall time."""
import math
from .util import ConfigurationError,sha256_json

def percentile(values,q):
    values=sorted(values);position=(len(values)-1)*q;i=int(position);fraction=position-i
    return values[i]*(1-fraction)+values[min(i+1,len(values)-1)]*fraction

def assess(measurement,budget):
    required={'control_hz','executed_actions','scheduling','max_deadline_miss_fraction','target_device'}
    if not required<=set(budget):raise ConfigurationError('budget lacks explicit control/schedule/device fields')
    hz=budget['control_hz'];actions=budget['executed_actions'];fraction=budget['max_deadline_miss_fraction']
    if isinstance(hz,bool) or not isinstance(hz,(int,float)) or not math.isfinite(hz) or hz<=0 or type(actions) is not int or actions<=0:
        raise ConfigurationError('control rate and executed action count must be positive')
    if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not math.isfinite(fraction) or not 0<=fraction<=1:
        raise ConfigurationError('invalid deadline miss budget')
    if budget['scheduling'] not in {'pipelined','blocking'}:raise ConfigurationError('unknown scheduling mode')
    samples=measurement.get('samples_ms',[])
    if not samples or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in samples):
        raise ConfigurationError('latency samples must be finite and nonnegative')
    if measurement.get('synthetic') is not False:raise ConfigurationError('real timing evidence required')
    device=measurement.get('hardware',{}).get('gpu_name','')
    target=budget['target_device']
    if not isinstance(target,str) or not target.strip():raise ConfigurationError('target device must be explicit')
    if not isinstance(device,str) or target.lower() not in device.lower():raise ConfigurationError('measurement device differs from budget target')
    if measurement.get('executed_actions')!=actions:raise ConfigurationError('measurement and budget execute different chunk lengths')
    # Pipelining can replenish a chunk during its execution. A blocking controller cannot.
    period=1000/hz;deadline=period*actions if budget['scheduling']=='pipelined' else period
    reaction=budget.get('max_observation_to_action_ms')
    if reaction is not None:
        if isinstance(reaction,bool) or not isinstance(reaction,(int,float)) or not math.isfinite(reaction) or reaction<=0:raise ConfigurationError('invalid reaction deadline')
        deadline=min(deadline,reaction)
    misses=sum(v>deadline for v in samples)
    result={'schema_version':1,'budget':budget,'measurement_sha256':sha256_json(measurement),
        'deadline_ms':deadline,'count':len(samples),'p50_ms':percentile(samples,.5),'p95_ms':percentile(samples,.95),
        'p99_ms':percentile(samples,.99),'max_ms':max(samples),'deadline_misses':misses,
        'miss_fraction':misses/len(samples),'meets_observed_budget':misses/len(samples)<=fraction,
        'p99_sample_warning':len(samples)<100,'deployment_certified':False,
        'scope':'Observed policy-call latency under this explicit scheduling scenario. Sensor/actuator latency and sustained controller scheduling require separate measurement.'}
    return result

def decision(assessments):
    """Preserve baseline semantics when sufficient; never authorize loss from speed alone."""
    if any(a['mode']=='stock' and a['assessment']['meets_observed_budget'] for a in assessments):
        return 'baseline_meets_observed_budget_no_lossy_optimization_needed'
    exact=[a for a in assessments if a.get('tier')=='BITEXACT' and a['assessment']['meets_observed_budget']]
    if exact:return 'check_target_device_equivalence_before_selecting_bitexact_candidate'
    return 'evaluate_minimum_quality_loss_only_if_no_semantics_preserving_option_meets_budget'
