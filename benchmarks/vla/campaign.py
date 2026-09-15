"""Build portable screening plans from explicit installed simulator contracts."""
import copy
import importlib
import subprocess
from pathlib import Path

from .adapters import load_adapters
from .plan import build_plan
from .registry import Registry, load_registry, _validate
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic


SUPPORTED = {'lingbot_vla-robotwin-v1', 'lingbot_vla_v2-robotwin-v1',
             'groot-libero-v1', 'pi05-libero-schedule-v1'}


def build_campaign(spec, output, scenes=None):
    allowed = {'schema_version','adapter_id','checkpoint','driver_python','environment',
               'tasks','seeds','suites','candidate_tier','control','candidate',
               'comparison','task_names','record_inputs','checkpoint_subdir'}
    if set(spec) - allowed or spec.get('schema_version') != 1:
        raise ConfigurationError('unsupported campaign schema or unknown fields')
    adapter_id = spec.get('adapter_id')
    if adapter_id not in SUPPORTED:
        raise ConfigurationError('campaign builder supports joint RoboTwin, GR00T LIBERO and pi05 LIBERO contracts')
    contract = next(a for a in load_adapters() if a['id'] == adapter_id)
    checkpoint = spec.get('checkpoint', contract['checkpoints'][0])
    if checkpoint not in contract['checkpoints']:
        raise ConfigurationError('checkpoint ID/revision is not allowed by the adapter')
    for field in ('tasks','seeds'):
        if type(spec.get(field)) is not int or spec[field] <= 0:
            raise ConfigurationError(f'{field} must be a positive integer')
    comparison=spec.get('comparison','ab')
    if comparison not in {'aa','bb','ab'}:raise ConfigurationError('comparison must be aa, bb or ab')
    if adapter_id.startswith('pi05-') and comparison!='ab':
        raise ConfigurationError('pi05 capture campaigns currently implement only original-versus-capture')
    if 'record_inputs' in spec and type(spec['record_inputs']) is not bool:
        raise ConfigurationError('record_inputs must be a boolean')
    tier = spec.get('candidate_tier')
    if tier not in {'BITEXACT','NUMERIC','BEHAVIORAL'}:
        raise ConfigurationError('candidate_tier must explicitly declare BITEXACT, NUMERIC or BEHAVIORAL')
    if contract['backbone'] == 'lingbot_vla_v2' and tier == 'BITEXACT' and comparison != 'aa':
        raise ConfigurationError('the default V2 Runtime is NUMERIC; it cannot be labeled BITEXACT')
    environment = spec.get('environment')
    if not isinstance(environment, dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in environment.items()):
        raise ConfigurationError('environment must be an explicit string mapping')
    simulator_var = 'ROBOTWIN_ROOT' if contract['simulator'] == 'robotwin' else 'LIBERO_ROOT'
    if not environment.get(simulator_var):
        raise ConfigurationError(f'{simulator_var} is required')
    simulator_root = Path(environment[simulator_var]).expanduser().resolve()
    try:
        simulator_revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=simulator_root,text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError(f'cannot identify simulator checkout: {error}') from error
    interpreter = spec.get('driver_python')
    if not isinstance(interpreter,str) or not Path(interpreter).expanduser().is_file():
        raise ConfigurationError('driver_python must name the installed simulator interpreter')
    subdir=spec.get('checkpoint_subdir','libero_10')
    if 'checkpoint_subdir' in spec and not contract.get('checkpoint_variants'):
        raise ConfigurationError('this adapter does not support checkpoint subdirectories')
    default_suites=([contract['checkpoint_variants'].get(subdir)] if contract.get('checkpoint_variants') else contract['suite_ids'])
    suites = spec.get('suites', default_suites)
    if contract.get('checkpoint_variants') and suites!=default_suites:
        raise ConfigurationError('checkpoint subdirectory and task suite differ')
    if not isinstance(suites,list) or not suites or len(set(suites)) != len(suites) or not set(suites) <= set(contract['suite_ids']):
        raise ConfigurationError('suites must be a nonempty unique subset of the contract suites')
    raw = copy.deepcopy(load_registry().raw)
    model = next((m for m in raw['models'] if m['id'] == checkpoint['id']), None)
    if model is None:
        raw['models'].append({'id':checkpoint['id'],'revision':checkpoint['revision'],
            'backbone':contract['backbone'],'builtin':False,'source_env':[],
            'determinism':'bitexact'})
    elif model['revision'] != checkpoint['revision']:
        raise ConfigurationError('registry checkpoint revision conflicts with contract')
    dataset_id = adapter_id + '-simulator'
    raw['datasets'].append({'id':dataset_id,'kind':'simulator',
        'source': 'https://github.com/RoboTwin-Platform/RoboTwin' if contract['simulator']=='robotwin' else 'https://github.com/Lifelong-Robot-Learning/LIBERO',
        'revision':simulator_revision})
    if adapter_id == 'groot-libero-v1':
        if subdir not in contract['checkpoint_variants']:raise ConfigurationError('unsupported checkpoint subdirectory')
        raw['suites'].append({'id':contract['checkpoint_variants'][subdir],'kind':'closed_loop','dataset':dataset_id,
            'tasks':[f'{subdir}/{i}' for i in range(10)],'compatible_backbones':['groot_n17'],
            'seed_strategy':'fixed','seed_base':0,'required_metrics':['success','action_digest','finite'],
            'protocol':{}})
    for suite in raw['suites']:
        if suite['id'] not in suites:
            continue
        if 'task_names' in spec:
            names=spec['task_names']
            if not isinstance(names,list) or len(names)!=spec['tasks'] or len(set(names))!=len(names) or not set(names)<=set(suite['tasks']):
                raise ConfigurationError('task_names must explicitly select the requested number of valid tasks')
            suite['tasks']=names
        if spec['tasks'] > len(suite['tasks']):
            raise ConfigurationError('requested task count exceeds the suite; refusing silent truncation')
        suite.update(dataset=dataset_id, compatible_backbones=[contract['backbone']],model_ids=[checkpoint['id']])
        suite['protocol'].update(bridge=contract['protocol'],evaluation_mode='paused_simulation',
            screening=True,margin=-.05,interval='tango_one_sided95')
    raw['profiles']['simulator_screen']={'suites':suites,
        'limits':{'tasks':spec['tasks'],'seeds_per_task':{'closed_loop':spec['seeds']}},
        'arm_repeats':1,'latency':{'warmup':0,'iterations':1}}
    module = importlib.import_module(contract['driver_module'])
    driver_revision = module.driver_revision() if adapter_id.startswith('pi05-') else module.revision()
    if scenes and driver_revision.endswith('-dirty'):
        raise ConfigurationError('pi05 live plans require an isolated committed execution checkout')
    arms = {'schema_version':1,'control_arm':'stock','arms':[]}
    identities = []
    for role, name in [('control','stock'),('treatment','candidate')]:
        point = {'name':name,'tier':tier if comparison=='bb' or role=='treatment' else 'BITEXACT'}
        if contract.get('checkpoint_variants'):point['checkpoint_subdir']=subdir
        if spec.get('record_inputs'):
            if contract['simulator']!='robotwin':raise ConfigurationError('input recording currently supports joint RoboTwin')
            point['record_inputs']=True
        expected_mode='stock' if comparison=='aa' or (comparison=='ab' and role=='control') else 'runtime_default'
        if adapter_id.startswith('pi05-'):
            if tier != 'BITEXACT':
                raise ConfigurationError('pi05 capture comparison does not alter numeric settings or NFE')
            point.update(optimization='stock' if role=='control' else 'instinctflash_capture',
                         schedule={'nfe':{'action':10},'forwards_per_cycle':11})
        elif scenes:
            endpoint_spec = spec.get('control' if role=='control' else 'candidate', {})
            if not endpoint_spec.get('receipt') or not endpoint_spec.get('endpoint'):
                raise ConfigurationError('bound campaigns require both endpoint receipts and URLs')
            identity = load_json(Path(endpoint_spec['receipt']).expanduser())
            if (identity.get('model_id'),identity.get('model_revision'),identity.get('protocol')) != (checkpoint['id'],checkpoint['revision'],contract['protocol']):
                raise ConfigurationError('endpoint protocol/checkpoint differs from the contract')
            if identity.get('synthetic') is not False or identity['execution']['mode'] != expected_mode:
                raise ConfigurationError('endpoint does not implement the requested original/Runtime arm')
            if contract.get('checkpoint_variants') and identity['execution'].get('checkpoint_subdir','libero_10')!=subdir:
                raise ConfigurationError('endpoint loaded another checkpoint subdirectory')
            identities.append(identity)
            point.update(remote={'endpoint':endpoint_spec['endpoint'],'identity':identity},execution=identity['execution'])
        if scenes:
            scene_path = Path(scenes).expanduser().resolve()
            point['scene_manifest']={'path':str(scene_path),'sha256':sha256_file(scene_path)}
        arm = {'id':name,'role':role,'operating_point':point,
            'driver':{'command':[str(Path(interpreter).expanduser().absolute()),'-m',contract['driver_module']],
                'adapter_id':adapter_id,'environment':environment,'revision':driver_revision,'timeout_seconds':3600}}
        if role == 'treatment':
            arm['gates']={'action':{'mode':'registry'},'performance':{'min_speedup':1},
                'success':{'margin':-.05,'interval':'tango_one_sided95','min_pairs':100}}
        arms['arms'].append(arm)
    if identities:
        _validate_paired_identities(identities)
    if scenes:
        manifest = load_json(Path(scenes).expanduser())
        expected_protocol = ('pi05-libero-frozen-scenes-v1' if adapter_id.startswith('pi05-') else
                             'lingbot-joint-robotwin-scenes-v1' if contract['simulator']=='robotwin' else contract['protocol'])
        if manifest.get('protocol') != expected_protocol:
            raise ConfigurationError('scene manifest protocol differs from the campaign')
        if contract['simulator'] == 'robotwin':
            from .robotwin_driver import RESET_POLICY
            if manifest.get('reset_policy') != RESET_POLICY:
                raise ConfigurationError('RoboTwin reset policy changed; prepare fresh scenes')
        source = manifest.get('sources', {})
        recorded_revision = (source.get('robotwin_revision') or source.get('revision') or
                             source.get('simulator', {}).get('revision'))
        if recorded_revision != simulator_revision:
            raise ConfigurationError('scene simulator revision differs from the local checkout')
    output = Path(output).resolve()
    registry_path = output.with_suffix('.registry.json')
    if output.exists() or registry_path.exists():
        raise ConfigurationError('refusing to overwrite an existing plan or registry')
    _validate(raw)
    registry = Registry(raw,sha256_json(raw),registry_path)
    plan = build_plan(registry,arms,'simulator_screen',[checkpoint['id']])
    if scenes:
        for job in plan['jobs']:
            r = job['request']
            key = (f"{r['suite_id']}/{r['task']}/{r['requested_seed']}" if contract['simulator']=='robotwin'
                   else f"{r['task']}/{r['requested_seed']}")
            scene = manifest.get('scenes', {}).get(key, {})
            if (scene.get('task'), scene.get('suite_id'), scene.get('requested_seed')) != (r['task'],r['suite_id'],r['requested_seed']):
                raise ConfigurationError(f'missing or mismatched frozen scene: {key}')
    write_json_atomic(registry_path,raw)
    write_json_atomic(output,plan)
    return {'plan':str(output),'registry':str(registry_path),'jobs':len(plan['jobs']),
            'plan_id':plan['plan_id'],'screening':True,'stage':'bound' if scenes else 'draft'}


def _validate_paired_identities(identities):
    """Permit execution implementation differences while retaining geometry/precision gates."""
    if len(identities)!=2:raise ConfigurationError("exactly two endpoint identities are required")
    for key in ('checkpoint_sha256','upstream_sha256','packages','pipeline_sha256','runtime_source_sha256','hardware','numeric_environment'):
        if identities[0].get(key) != identities[1].get(key):
            raise ConfigurationError(f'paired endpoints differ in {key}; use an explicit custom numerical campaign')
    left = {k:v for k,v in identities[0]['execution'].items() if k not in {'mode','runtime_explanation','backend','transforms','graph_stats','capture_required','tier_ceiling'}}
    right = {k:v for k,v in identities[1]['execution'].items() if k not in {'mode','runtime_explanation','backend','transforms','graph_stats','capture_required','tier_ceiling'}}
    if left != right:
        raise ConfigurationError('paired endpoints change execution geometry or schedule')
