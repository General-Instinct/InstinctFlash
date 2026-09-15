"""Select measured configurations under explicit precision, quality and control budgets."""
import math
from .execution_evidence import verify_record, HASH
from .realtime import assess
from .util import ConfigurationError, sha256_json

TIERS={'BITEXACT':0,'NUMERIC':1,'BEHAVIORAL':2}


def _tier(facts):
    ex=facts['execution']
    tiers=[r['tier'] for r in ex['transforms']]
    if ex['precision']=='fp8':tiers.append('NUMERIC')
    if any(t not in TIERS for t in tiers):raise ConfigurationError('unknown transformation tier')
    return max((TIERS[t] for t in tiers),default=0)


def _quality(record, baseline, policy):
    """A/A and B/B cannot authorize an A/B deployment; screening cannot certify it."""
    required=set(policy.get('required_suites',[]))
    if not required:return False,'quality scope not declared (required_suites)'
    tasks=policy.get('required_tasks',{})
    if any(not tasks.get(s) for s in required):return False,'quality task coverage not declared (required_tasks)'
    margin=policy.get('max_quality_loss')
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or not 0<=margin<=1:
        return False,'maximum permitted quality loss is not declared'
    registry=policy.get('quality_registry_sha256')
    if not registry:return False,'quality registry/protocol identity not declared'
    for q in record['quality']:
        if q['status']!='certified' or q.get('registry_sha256')!=registry:continue
        for comparison in q['comparisons']:
            treatment=comparison['treatment'];control=comparison['control']
            if treatment not in q['matched_arms']:continue
            if q['reference_profiles'].get(control,{}).get('profile_id')!=baseline['profile']['profile_id']:continue
            gate=q['success_gates'].get(treatment) or {}
            if 'margin' not in gate or gate['margin'] < -margin:continue
            rows={row['suite_id']:row for row in comparison['closed_loop']}
            coverage={row['suite_id']:set(row.get('per_task',{})) for row in comparison.get('paired_success',[])}
            if (required<=rows.keys() and all(rows[s]['verdict']=='PASS' for s in required)
                    and all(set(tasks[s])==coverage.get(s,set()) for s in required)):
                return True,'paired closed-loop gate matches baseline, execution and requested suites'
    return False,'no matching paired closed-loop certificate; screening/repeats do not authorize selection'


def select(records,budget,policy):
    allowed_keys={'baseline_profile_id','allowed_precisions','tier_ceiling','allow_step_change',
                  'required_suites','required_tasks','max_quality_loss','quality_registry_sha256'}
    if not isinstance(policy,dict) or set(policy)-allowed_keys:
        raise ConfigurationError('unknown selection policy fields')
    if not isinstance(policy.get('baseline_profile_id'),str):raise ConfigurationError('baseline_profile_id must be explicit')
    if not isinstance(policy.get('tier_ceiling','bitexact'),str):raise ConfigurationError('tier_ceiling must be a string')
    suites=policy.get('required_suites',[]);tasks=policy.get('required_tasks',{})
    if not isinstance(suites,list) or any(not isinstance(v,str) or not v for v in suites) or len(set(suites))!=len(suites):
        raise ConfigurationError('required_suites must name distinct suites')
    if (not isinstance(tasks,dict) or set(tasks)-set(suites)
            or any(not isinstance(v,list) or not v or any(not isinstance(t,str) or not t for t in v)
                   or len(set(v))!=len(v) for v in tasks.values())):
        raise ConfigurationError('required_tasks must explicitly list tasks within required suites')
    if 'quality_registry_sha256' in policy and (not isinstance(policy['quality_registry_sha256'],str) or not HASH.fullmatch(policy['quality_registry_sha256'])):
        raise ConfigurationError('quality_registry_sha256 must pin the benchmark registry/protocol')
    if 'max_quality_loss' in policy:
        loss=policy['max_quality_loss']
        if isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(loss) or not 0<=loss<=1:
            raise ConfigurationError('max_quality_loss must be a finite fraction in [0,1]')
    for record in records:verify_record(record)
    by_id={r['profile']['profile_id']:r for r in records}
    if len(by_id)!=len(records):raise ConfigurationError('duplicate execution profiles')
    baseline=by_id.get(policy.get('baseline_profile_id'))
    if baseline is None:raise ConfigurationError('an explicit measured baseline_profile_id is required')
    base=baseline['profile']['facts'];base_ex=base['execution']
    if base_ex['precision']!='native':raise ConfigurationError('baseline must preserve the selected checkpoint native precision')
    allowed=policy.get('allowed_precisions',['native'])
    if not isinstance(allowed,list) or not allowed or any(v not in ('native','fp8') for v in allowed):
        raise ConfigurationError('allowed_precisions must explicitly name native/fp8')
    ceiling=policy.get('tier_ceiling','bitexact').upper()
    if ceiling not in TIERS:raise ConfigurationError('unknown tier ceiling')
    if type(policy.get('allow_step_change',False)) is not bool:raise ConfigurationError('allow_step_change must be boolean')
    rows=[];eligible=[];baseline_ok=False
    for record in records:
        profile=record['profile'];facts=profile['facts'];ex=facts['execution'];reasons=[]
        is_base=profile['profile_id']==baseline['profile']['profile_id']
        if any(facts[k]!=base[k] for k in ('model_id','model_revision','checkpoint_sha256','upstream_sha256')):
            reasons.append('different checkpoint/reference implementation')
        if facts['hardware']!=base['hardware']:reasons.append('different hardware execution contract')
        if ex['precision'] not in allowed:reasons.append('precision not authorized')
        if not is_base and _tier(facts)>TIERS[ceiling]:reasons.append('transformation tier exceeds permission')
        if ex['declared'].get('nfe')!=base_ex['declared'].get('nfe') and not policy.get('allow_step_change',False):
            reasons.append('step change not authorized')
        if ex['declared'].get('guidance')!=base_ex['declared'].get('guidance'):
            reasons.append('guidance change not authorized')
        if ex['action_shape']!=base_ex['action_shape']:reasons.append('action geometry differs')
        timing=record['latency'];assessment=None
        if timing is None:reasons.append('no bound latency measurement')
        else:
            try:assessment=assess(timing,budget)
            except ConfigurationError as error:reasons.append(str(error))
            else:
                if not assessment['meets_observed_budget']:reasons.append('observed deadline budget missed')
        if not is_base:
            quality_ok,why=_quality(record,baseline,policy)
            if not quality_ok:reasons.append(why)
        row={'profile_id':profile['profile_id'],'precision':ex['precision'],
             'quality_status':record['quality_status'],'assessment':assessment,
             'eligible':not reasons,'reasons':reasons}
        rows.append(row)
        if not reasons:
            if is_base:baseline_ok=True
            else:eligible.append((ex['precision']!='native',_tier(facts),assessment['p99_ms'],profile['profile_id']))
    selected=baseline['profile']['profile_id'] if baseline_ok else min(eligible)[-1] if eligible else None
    reason=('native_baseline_meets_observed_budget' if baseline_ok else
            'eligible_configuration_within_explicit_permissions' if selected else
            'no_configuration_has_both_matching_evidence_and_budget_compliance')
    result={'schema_version':1,'selected_profile_id':selected,'reason':reason,'budget':budget,'policy':policy,
            'candidates':rows,'deployment_certified':False,
            'scope':'Offline evidence-based selection, not automatic model loading or an end-to-end realtime guarantee. '
                    'Prefers native precision and lower transformation tier; does not infer minimum true task loss from small samples.'}
    result['selection_sha256']=sha256_json(result)
    return result
