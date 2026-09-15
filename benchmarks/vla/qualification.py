"""Checkpoint-specific qualification stages; historical family evidence never grants readiness."""
from copy import deepcopy
from pathlib import Path
import re
from .adapters import load_adapters
from .bundle import verify_bundle
from .coverage import run_evidence
from .execution_evidence import execution_profile,verify_record
from .util import ConfigurationError,load_json,sha256_file,sha256_json


def requested_profile_id(profile):
    """Group admission attempts by requested execution, retaining the declared guard strength."""
    facts=deepcopy(profile['facts']);ex=facts['execution']
    ex.pop('graph_stats',None)
    for transform in ex['transforms']:
        check=transform.get('params',{}).get('self_check')
        if check is not None:check.pop('passed',None)
    return sha256_json(facts)


def read_startup(path):
    receipt=load_json(Path(path));profile=execution_profile(receipt);startup=receipt.get('startup',{})
    protocol = startup.get('protocol')
    if protocol == 'seeded-eight-predict-seven-commit-v1':
        model = receipt['model_id']
        frames = ([12] + [16] * 6 if model == 'robbyant/lingbot-va-posttrain-libero-long' else
                  [4] + [8] * 6 if model == 'robbyant/lingbot-va-posttrain-robotwin' else None)
        if frames is None or startup.get('commit_calls') != 7 or startup.get('commit_frames') != frames:
            raise ConfigurationError('startup receipt lacks native checkpoint-specific commit history')
    elif protocol != 'seeded-eight-call-admission-v1':
        raise ConfigurationError('startup receipt lacks a reviewed admission protocol')
    if (startup.get('seed')!=0
            or startup.get('calls')!=8 or startup.get('status') not in ('ready','rejected')):
        raise ConfigurationError('startup receipt lacks the observed admission protocol')
    if not isinstance(startup.get('attempt_id'),str) or not re.fullmatch(r'[0-9a-f]{32}',startup['attempt_id']):
        raise ConfigurationError('startup receipt lacks a unique attempt identifier')
    if startup.get('fault_injection'):
        raise ConfigurationError('fault-injected trial is not ordinary startup admission evidence')
    ex=profile['facts']['execution']
    rejected=bool(ex.get('capture_required') and not ex.get('graph_stats',{}).get('captured'))
    if rejected!=(startup['status']=='rejected'):
        raise ConfigurationError('startup status contradicts observed required capture')
    return {'path':str(Path(path).resolve()),'sha256':sha256_file(Path(path)),
            'profile':profile,'requested_profile_id':requested_profile_id(profile),'status':startup['status'],
            'attempt_id':startup['attempt_id']}


def report(registry, *, target_device, bundles=(), records=(), startup_receipts=(), minimum_startups=3):
    if not isinstance(target_device,str) or not target_device.strip():raise ConfigurationError('target device is required')
    if type(minimum_startups) is not int or minimum_startups<1:raise ConfigurationError('minimum_startups must be positive')
    contracts=load_adapters();rows={};sources=[]
    def row(backbone,model,revision,subdir=None):
        key=(model,revision,subdir)
        if key not in rows:
            rows[key]={'backbone':backbone,'checkpoint':model,'revision':revision,'checkpoint_subdir':subdir,
                       'adapters':[],'historical_screening':[],'bound_executions':[]}
        elif rows[key]['backbone']!=backbone:raise ConfigurationError('conflicting backbone for checkpoint')
        return rows[key]
    for model in registry.models.values():row(model['backbone'],model['id'],model['revision'])
    for contract in contracts:
        for checkpoint in contract['checkpoints']:
            variants=contract.get('checkpoint_variants') or {None:None}
            for subdir,suite in variants.items():
                target=row(contract['backbone'],checkpoint['id'],checkpoint['revision'],subdir)
                target['adapters'].append({'id':contract['id'],'simulator':contract['simulator'],
                    'suites':[suite] if suite else contract['suite_ids']})
    for bundle in bundles:
        verified=verify_bundle(Path(bundle));sources.append({'path':str(Path(bundle).resolve()),'bundle_sha256':verified['bundle_sha256']})
        for evidence in run_evidence(bundle):
            target=row(evidence['backbone'],evidence['checkpoint'],evidence['revision'],evidence.get('checkpoint_subdir'))
            target['historical_screening'].append(dict(evidence,transferred_to_target_execution=False))
    attempts=[read_startup(path) for path in startup_receipts]
    if len({a['attempt_id'] for a in attempts})!=len(attempts):raise ConfigurationError('duplicate startup attempt supplied')
    seen=set()
    for record in records:
        verify_record(record);profile=record['profile'];facts=profile['facts'];pid=profile['profile_id']
        if pid in seen:raise ConfigurationError('duplicate execution record supplied')
        seen.add(pid)
        key=(facts['model_id'],facts['model_revision'],facts['execution'].get('checkpoint_subdir'))
        target=rows.get(key)
        if target is None:
            raise ConfigurationError('record checkpoint/variant has no registered qualification contract')
        rid=requested_profile_id(profile);matched=[a for a in attempts if a['requested_profile_id']==rid]
        rejected=sum(a['status']=='rejected' for a in matched)
        hardware_match=target_device.lower() in facts['hardware']['gpu_name'].lower()
        legacy_guard=any(t['params'].get('self_check',{}).get('calibration_status')=='legacy_cross_domain_guard'
                         for t in facts['execution']['transforms'])
        if not target['adapters']:stage='implement_native_simulator_contract'
        elif not hardware_match:stage='measure_target_device_execution'
        elif len(matched)<minimum_startups:stage='collect_fresh_startup_receipts'
        elif rejected:stage='investigate_startup_rejections'
        elif legacy_guard:stage='calibrate_same_domain_guard_without_relabeling_legacy_results'
        elif record['latency'] is None:stage='measure_bound_latency'
        elif record['quality_status']!='certified':stage='evaluate_budget_then_collect_needed_paired_quality'
        else:stage='apply_explicit_budget_and_quality_selection_policy'
        target['bound_executions'].append({'profile_id':pid,'requested_profile_id':rid,
            'hardware_matches_target':hardware_match,'precision':facts['execution']['precision'],
            'quality_status':record['quality_status'],'startup_attempts':len(matched),'startup_rejections':rejected,
            'latency_measured':record['latency'] is not None,'next_stage':stage,'deployment_certified':False})
    for target in rows.values():
        target['next_stage']=('implement_native_simulator_contract' if not target['adapters'] else
                              'measure_target_device_execution' if not target['bound_executions'] else 'inspect_execution_stages')
    result={'schema_version':1,'target_device':target_device,'minimum_startups':minimum_startups,
        'registry_sha256':registry.digest,'adapter_contracts_sha256':sha256_json(contracts),
        'checkpoints':sorted(rows.values(),key=lambda r:(r['backbone'],r['checkpoint'],r['checkpoint_subdir'] or '')),
        'source_bundles':sources,'startup_sources':[{k:v for k,v in a.items() if k!='profile'} for a in attempts],
        'protocol':['pin checkpoint, source, artifacts, schedule and numeric settings',
                    'fresh-process startup admission; retain rejected attempts and test fallback',
                    'same real observations and actual noise: AA, AB, BB; compare like tensor domains',
                    'separate target-device latency and explicit control/reaction budget',
                    'preserve baseline when sufficient; paired quality for permitted numerical/behavioral changes',
                    'bind exact task/scene/seed protocol and execution before selection'],
        'deployment_certified':False,
        'scope':'Qualification work inventory, not automatic execution or a release certificate. '
                'A finite startup sample is not a reliability-rate guarantee. Historical screening, '
                'another fine-tune/variant, or H100 evidence never grants Thor readiness. '
                'Renderer requirements remain those of each simulator adapter.'}
    result['report_sha256']=sha256_json(result);return result
