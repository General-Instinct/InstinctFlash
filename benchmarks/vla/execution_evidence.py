"""Bind measurements to an observed execution, never to a model family alone."""
from copy import deepcopy
from pathlib import Path
import re
import math
import tempfile

from .util import ConfigurationError, load_json, sha256_file, sha256_json

HASH = re.compile(r'^[0-9a-f]{64}$')
PROFILE_FIELDS = ('model_id','model_revision','checkpoint_sha256','upstream_sha256',
                  'runtime_source_sha256','pipeline_sha256','adapter_sha256',
                  'hardware','packages','numeric_environment','execution')


def execution_profile(receipt):
    """Normalize an endpoint's measured identity; replay counters are not configuration."""
    missing=[key for key in PROFILE_FIELDS if key not in receipt]
    if missing or receipt.get('synthetic') is not False:
        raise ConfigurationError(f'execution receipt lacks measured identity fields: {missing}')
    for key in ('checkpoint_sha256','upstream_sha256','runtime_source_sha256','pipeline_sha256'):
        if not isinstance(receipt[key],str) or not HASH.fullmatch(receipt[key]):
            raise ConfigurationError(f'invalid execution identity: {key}')
    if (not isinstance(receipt['model_id'],str) or not receipt['model_id']
            or not isinstance(receipt['model_revision'],str)
            or not re.fullmatch(r'[0-9a-f]{40,64}',receipt['model_revision'])):
        raise ConfigurationError('checkpoint identity must use a pinned commit/content digest')
    ex=receipt['execution']
    for key in ('mode','precision','backend','declared','transforms','action_shape'):
        if key not in ex:raise ConfigurationError(f'execution lacks {key}')
    if ex['precision'] not in ('native','fp8') or ex['backend'] not in ('stock','in_process','worker','engine'):
        raise ConfigurationError('unsupported explicit precision/backend')
    if ex['mode']!='stock' and (not isinstance(receipt['adapter_sha256'],str) or not HASH.fullmatch(receipt['adapter_sha256'])):
        raise ConfigurationError('Runtime receipt lacks adapter source identity')
    if not receipt['hardware'].get('gpu_name') or not receipt['packages'] or not receipt['numeric_environment']:
        raise ConfigurationError('hardware/software/numeric settings must be observed')
    shape=ex['action_shape']
    if not isinstance(shape,list) or len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):
        raise ConfigurationError('action shape must explicitly declare chunk and dimensions')
    if (not isinstance(ex['transforms'],list) or any(not isinstance(t,dict) or not t.get('name')
            or t.get('tier') not in ('BITEXACT','NUMERIC','BEHAVIORAL') or not isinstance(t.get('params'),dict)
            for t in ex['transforms'])):
        raise ConfigurationError('transforms must declare names, tiers and parameters')
    declared=ex['declared']
    if not declared.get('nfe') or not isinstance(declared.get('guidance'),dict):
        raise ConfigurationError('step schedule and guidance must be explicit')
    if ex['precision']=='fp8' or ex.get('graph_stats',{}).get('cuda_kernels'):
        artifacts=ex.get('artifacts',{})
        required=('engine_sha256','calibration_sha256') if ex['precision']=='fp8' else ('kernel_sha256',)
        if any(not isinstance(artifacts.get(k),str) or not HASH.fullmatch(artifacts[k]) for k in required):
            raise ConfigurationError('custom kernels/FP8 require binary and calibration artifact identities')
    facts={key:deepcopy(receipt[key]) for key in PROFILE_FIELDS}
    # Logs and changing counters can differ between fresh endpoints of the same execution.
    facts['execution'].pop('runtime_explanation',None)
    for transform in facts['execution']['transforms']:
        params=transform.get('params',{})
        params.pop('decision',None)
        if 'self_check' in params:
            params['self_check']={key:value for key,value in params['self_check'].items()
                                  if key in ('passed','n','tolerance','tolerance_provenance',
                                             'comparison_domain','tolerance_domain','calibration_status',
                                             'repeats','comparisons')}
    stats=facts['execution'].get('graph_stats',{})
    facts['execution']['graph_stats']={key:value for key,value in stats.items() if 'replay' not in key}
    profile={'schema_version':1,'facts':facts}
    profile['profile_id']=sha256_json(profile)
    return profile


def verify_profile(profile):
    unsigned=dict(profile);claimed=unsigned.pop('profile_id',None)
    if claimed!=sha256_json(unsigned) or profile.get('schema_version')!=1:
        raise ConfigurationError('execution profile identity changed')
    # Validate facts as strictly as an original receipt, not only its digest.
    receipt=dict(profile['facts'],synthetic=False)
    if execution_profile(receipt)!=profile:raise ConfigurationError('noncanonical execution profile')
    return profile


def _reference(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha256_file(path)}


def build_record(receipt_path, *, measurement_path=None, bundles=()):
    receipt_path=Path(receipt_path);receipt=load_json(receipt_path)
    profile=execution_profile(receipt)
    sources={'receipt':_reference(receipt_path),'measurement':None,'bundles':[]}
    timing=None
    if measurement_path is not None:
        measurement_path=Path(measurement_path);measurement=load_json(measurement_path)
        sources['measurement']=_reference(measurement_path)
        if measurement.get('synthetic') is not False or measurement.get('execution_profile')!=profile:
            raise ConfigurationError('latency is not bound to this exact execution profile')
        if measurement.get('hardware',{}).get('gpu_name')!=profile['facts']['hardware']['gpu_name']:
            raise ConfigurationError('latency hardware contradicts its execution profile')
        shape=profile['facts']['execution']['action_shape']
        samples=measurement.get('samples_ms',[])
        if (not samples or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in samples)
                or type(measurement.get('iterations')) is not int or measurement['iterations']!=len(samples)
                or type(measurement.get('warmup')) is not int or measurement['warmup']<1
                or measurement.get('executed_actions')!=shape[0]):
            raise ConfigurationError('invalid latency sample count, warmup or executed action geometry')
        if 'sha256' in measurement:
            unsigned=dict(measurement);claimed=unsigned.pop('sha256')
            if claimed!=sha256_json(unsigned):raise ConfigurationError('latency self-identity changed')
        timing=measurement
    quality=[]
    from .bundle import verify_bundle
    from .report import build_report
    from .registry import load_registry
    for bundle in bundles:
        root=Path(bundle).resolve();verified=verify_bundle(root)
        plan=load_json(root/'plan.json')
        matched=[]
        for arm in plan['arms']:
            identity=arm['operating_point'].get('remote',{}).get('identity')
            if identity is not None and execution_profile(identity)==profile:matched.append(arm['id'])
        if not matched:raise ConfigurationError('quality bundle belongs to another execution profile')
        # Recompute from hash-checked raw outcomes; a hand-edited summary is not a gate.
        with tempfile.TemporaryDirectory(prefix='ifl-execution-report-') as temporary:
            report=build_report(root,load_registry(root/'registry.json'),output=Path(temporary)/'report.json')
        comparisons=[]
        for comparison in report.get('comparisons',[]):
            if comparison.get('treatment') not in matched and comparison.get('control') not in matched:continue
            comparisons.append({key:comparison.get(key,[]) for key in
                                ('control','treatment','paired_success','closed_loop','closed_loop_actions')})
        status='certified' if (report.get('complete') is True and report.get('synthetic') is False
            and report.get('screening') is False and report.get('reportable') is True
            and report.get('gates_passed') is True and comparisons
            and all(c['treatment'] in matched and c['closed_loop'] and all(row.get('verdict')=='PASS' for row in c['closed_loop']) for c in comparisons)) else 'screen'
        quality.append({'status':status,'plan_id':plan['plan_id'],'registry_sha256':plan['registry_sha256'],'matched_arms':matched,
                        'comparisons':comparisons,'screening':report.get('screening'),
                        'reference_profiles':{arm['id']:execution_profile(arm['operating_point']['remote']['identity'])
                            for arm in plan['arms'] if 'remote' in arm['operating_point']},
                        'success_gates':{arm['id']:arm.get('gates',{}).get('success') for arm in plan['arms']},
                        'scope':'Only this execution, paired reference and declared task/seed/setting protocol.'})
        sources['bundles'].append({'path':str(root),'sha256':verified['bundle_sha256']})
    record={'schema_version':1,'profile':profile,'sources':sources,'latency':timing,
            'quality':quality,'quality_status':('certified' if any(q['status']=='certified' for q in quality)
                else 'screen' if quality else 'unmeasured'),
            'deployment_certified':False}
    record['record_sha256']=sha256_json(record)
    return record


def verify_record(record):
    unsigned=dict(record);claimed=unsigned.pop('record_sha256',None)
    if claimed!=sha256_json(unsigned):raise ConfigurationError('execution record changed')
    verify_profile(record['profile'])
    sources=record['sources']
    for ref in [sources['receipt'],sources['measurement']]:
        if ref and sha256_file(Path(ref['path']))!=ref['sha256']:raise ConfigurationError('execution evidence source changed')
    rebuilt=build_record(sources['receipt']['path'],
        measurement_path=sources['measurement']['path'] if sources['measurement'] else None,
        bundles=[ref['path'] for ref in sources['bundles']])
    if rebuilt!=record:raise ConfigurationError('execution record no longer matches recomputed evidence')
    return record
