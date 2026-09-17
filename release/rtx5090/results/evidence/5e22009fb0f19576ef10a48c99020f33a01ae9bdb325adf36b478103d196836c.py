"""Independent CPU saved-byte audit of one pinned NUMERIC supplement and its parent."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE='0a53125c5af6d76260be96e39077cdcc3c01126d'
GPU='GPU-004614f4-173a-187b-4e67-a2e1799e4d42'
TARGET={'name':'rtx5090','capability':[12,0]}
PRIMARY={'vla2':'audit_vla2_v2','edge':'audit_edge_v1','nano':'audit_nano_v1'}

def require(value,message):
    if not value:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def check(raw,value,label):require(isinstance(value,str) and len(value)==64 and sha(raw)==value,'Byte hash differs: '+label)
def read_local(name):return json.loads((HERE/name).read_bytes())
def valid_sha(value):return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')
def delta(left,right):
    diff=left.astype(np.float64)-right.astype(np.float64)
    return {'exact_bytes':left.dtype==right.dtype and left.tobytes()==right.tobytes(),'max_abs':float(np.abs(diff).max()),'rmse':float(np.sqrt(np.mean(diff*diff)))}

def audit(args):
    family=args.family
    require(family in PRIMARY,'Unreviewed family')
    binding=json.loads(Path(args.binding).read_bytes())
    require(binding['schema']=='instinctflash.rtx5090_numeric_archive_binding.v1'
            and binding['family']==family and binding['mode']=='numeric' and binding['source_commit']==SOURCE
            and binding['target']==TARGET and binding['GPU_uuid']==GPU,'Wrong actual supplement binding')
    for key in ('archive_sha256','archive_manifest_sha256','completion_sha256','config_sha256','parent_archive_sha256','parent_audit_sha256','vendor_provenance_sha256'):
        require(valid_sha(binding[key]),'Actual binding remains pending: '+key)
    identity=binding['worker'];require(type(identity['pid']) is int and identity['pid']>0
            and isinstance(identity['start_ticks'],str) and identity['start_ticks'].isdigit() and identity['exited'] is True,'Actual process not admitted')
    primary_dir=HERE.parent/PRIMARY[family];sys.path.insert(0,str(primary_dir))
    spec=importlib.util.spec_from_file_location('numeric_parent_'+family,primary_dir/'audit.py');primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
    from serving_audit import archive_files,serving
    files,manifest,archive_sha=archive_files(args.archive)
    require(archive_sha==binding['archive_sha256'],'Different actual supplement archive')
    check(files['provenance/archive_manifest.json'],binding['archive_manifest_sha256'],'manifest')
    template=read_local(family+'.template.json');run=template['output'];prefix=Path(run).name+'/'
    require(manifest['source_run']==run and manifest['source_commit']==SOURCE
            and manifest['source_completion_sha256']==binding['completion_sha256'],'Wrong supplement run/source/terminal')
    raw=lambda name:files[prefix+name]
    read=lambda name:json.loads(raw(name))
    config,done=read('config.json'),read('completion.json')
    template['parent']=config['parent'];require(config==template,'Actual config differs beyond parent bindings')
    check(raw('config.json'),binding['config_sha256'],'actual config');check(raw('completion.json'),binding['completion_sha256'],'actual terminal')
    source=read_local('source_review.json')['families'][family]
    require(done['schema']=='instinctflash.rtx5090_numeric_supplement.v1' and done['status']=='completed_supplement_only'
            and done['passed'] is True and done['family']==family and done['mode']=='numeric' and done['source_commit']==SOURCE
            and done['target']==TARGET and done['GPU_uuid']==GPU and done['config_sha256']==binding['config_sha256']
            and done['worker_pid']==identity['pid'] and done['worker_start_ticks']==identity['start_ticks']
            and done['worker_sha256']==source['runner_sha256'] and done['shared_runner_sha256']==primary.WORKER_SHA
            and done['automatic_retries']==0 and done['owned_processes_terminal'] is True and done['lease_released'] is True
            and done['primary_baselines_recaptured'] is False
            and (('full_public_run_completed' not in done) if family=='vla2' else done['full_public_run_completed'] is False)
            and done['supplemental_api_cells']==1 and done['supplemental_serve_cells']==1
            and done['captured_cells']==[config['cell_id']],'Actual full supplemental lifecycle differs')
    for name,value in [('supplement_runner.py',source['runner_sha256']),('shared_worker.py',primary.WORKER_SHA),('selected.py',config['helper_sha256']),('expected_plan.json',config['expected_plan_sha256'])]:
        check(files['provenance/'+name],value,'actual producer '+name)
    require(files['provenance/expected_plan.json']==(HERE/(family+'.expected_plan.json')).read_bytes(),'Reviewed plan bytes drift')
    pf,pm,parent_sha=archive_files(args.parent_archive);require(parent_sha==binding['parent_archive_sha256'],'Different actual parent archive')
    audit_raw=Path(args.parent_audit).read_bytes();check(audit_raw,binding['parent_audit_sha256'],'independent parent audit');parent_audit=json.loads(audit_raw)
    require(parent_audit['passed'] is True and parent_audit['schema']=='instinctflash.rtx5090_'+family+'_independent_audit.v1'
            and parent_audit['source_commit']==SOURCE and parent_audit['GPU_uuid']==GPU and parent_audit['archive_sha256']==parent_sha,'Actual qualified parent missing')
    parent_prefix=Path(primary.RUN).name+'/'
    parent_raw=lambda name:pf[parent_prefix+name]
    parent_read=lambda name:json.loads(parent_raw(name))
    parent,model=parent_read('completion.json'),parent_read('config.json')
    require(pm['source_run']==primary.RUN and pm['source_commit']==SOURCE and pm['source_completion_sha256']==parent_audit['completion_sha256'],'Parent manifest/source differs')
    check(parent_raw('completion.json'),parent_audit['completion_sha256'],'audited parent terminal');check(parent_raw('config.json'),parent_audit['config_sha256'],'audited parent config')
    require(raw('parent_completion.json')==parent_raw('completion.json') and raw('parent_config.json')==parent_raw('config.json'),'Parent raw copies differ')
    require(parent['passed'] is True and parent['lease_released'] is True and parent['owned_processes_terminal'] is True
            and parent['worker_pid']==config['parent']['worker_pid'] and parent['worker_start_ticks']==config['parent']['worker_start_ticks'],'Wrong actual completed parent identity')
    require(config['parent']['run']==done['parent_run']==primary.RUN and done['parent_completion_sha256']==config['parent']['completion_sha256'],'Wrong parent linkage')
    for key,name in [('completion_sha256','completion.json'),('config_sha256','config.json'),('capture_run_sha256','capture/run.json'),('installed_source_sha256','installed_source.json')]:
        check(parent_raw(name),config['parent'][key],'actual parent '+key)
    provenance_raw=Path(args.vendor_provenance).read_bytes();check(provenance_raw,binding['vendor_provenance_sha256'],'actual vendor provenance');provenance=json.loads(provenance_raw)
    require(binding['vendor_provenance_sha256']==parent_audit['vendor_cpu_provenance_sha256'],'Different qualified vendor proof')
    installed=parent_read('installed_source.json')
    paths={r['path']:r['sha256'] for d in installed['distributions'].values() for r in d['files'].values()}
    paths.update(provenance['expected_path_sha256'])
    # The passed primary audit also admits original deploy helper modules that
    # are imported directly from the vendor checkout rather than its wheel.
    for identifier,checksum in config['parent']['cell_receipts'].items():
        name='capture/cells/'+identifier+'/receipt.json'
        check(parent_raw(name),checksum,'audited parent source receipt')
        for path,value in parent_read(name)['sources'].items():
            require(path not in paths or paths[path]==value,'Conflicting qualified parent source')
            paths[path]=value
    plan=read('prepared/plan.json');expected=read_local(family+'.expected_plan.json')
    require(plan==expected and plan['execution_mode']=='numeric' and plan['target']==TARGET and plan['model']==family,'Actual public numeric plan differs')
    preparation=read('prepared/preparation.json');require(preparation['status']=='prepared_primary_checkpoint' and preparation['local_files_only'] is True,'Actual offline preparation missing')
    for name,key in [('plan.json','plan_sha256'),('matrix.json','matrix_sha256'),('profiles.json','profiles_sha256'),('inputs/recorded_inputs_v1.npz','fixture_sha256')]:
        check(raw('prepared/'+name),preparation[key],'prepared '+name);require(raw('capture/'+name)==raw('prepared/'+name),'Prepared/captured bytes differ')
    require(read('capture/matrix.json')==plan['matrix'] and raw('capture/preparation.json')==raw('prepared/preparation.json'),'Full prepared matrix differs')
    require(raw('prepared/inputs/recorded_inputs_v1.npz')==parent_raw('capture/inputs/recorded_inputs_v1.npz'),'Different paired fixture bytes')
    admission=read('admission.json');require(admission['status']=='passed' and admission['mode']=='numeric' and admission['cell_id']==config['cell_id'],'Original CPU admission missing')
    check(raw('prepared/plan.json'),admission['plan_sha256'],'admitted plan');check(parent_raw('prepared/plan.json'),admission['parent_plan_sha256'],'admitted parent plan')
    with np.load(io.BytesIO(raw('prepared/inputs/recorded_inputs_v1.npz')),allow_pickle=False) as values:frames=values['frames'].copy()
    cell=next(c for c in plan['matrix']['cells'] if c['id']==config['cell_id'])
    require(cell['arm']=='runtime_selected' and cell['expected_runtime_kwargs']['precision']=='native' and cell['expected_runtime_kwargs']['tier_ceiling']=='numeric','Wrong native NUMERIC selection')
    receipt=read('capture/'+cell['receipt']);primary.hardware(receipt['hardware']);primary.validate_requests(receipt,frames)
    require(receipt['ok'] is True and receipt['family']==family and receipt['cell_id']==cell['id'] and receipt['arm']=='runtime_selected'
            and receipt['target']==TARGET and receipt['interpreter']==model['python'] and receipt['torch']==model['expected_torch_runtime']
            and receipt['precision']=='native' and not receipt.get('e4m3_tensors'),'Wrong actual native cell/stack/precision')
    require(receipt['runtime_kwargs']==cell['expected_runtime_kwargs'] and receipt['optimizer_environment']==cell['expected_optimizer_environment']
            and receipt['effective_schedule']==cell['effective_schedule'],'Actual options/schedule differ')
    check(raw('capture/matrix.json'),receipt['matrix_sha256'],'actual matrix');check(raw('capture/inputs/recorded_inputs_v1.npz'),receipt['input_archive_sha256'],'actual fixture')
    require(receipt['model_id']==plan['checkpoint']['model_id'] and receipt['revision']==plan['checkpoint']['revision'],'Different original checkpoint')
    policy=receipt['execution_policy'];require(policy['precision']=='native' and policy['tier_ceiling']=='numeric' and policy['nfe']==cell['effective_schedule']['nfe']
            and policy['changed_schedule']=={} and not policy['schedule_options'] and policy.get('step_cache') is None,'Native NUMERIC changed sampling policy')
    require(receipt['sources'] and all(paths.get(p)==v for p,v in receipt['sources'].items()),'Actual loaded source outside qualified installed closure')
    check(raw('capture/'+cell['receipt'][:-5]+'.npz'),receipt['actions_sha256'],'complete candidate actions')
    with np.load(io.BytesIO(raw('capture/'+cell['receipt'][:-5]+'.npz')),allow_pickle=False) as values:actions=values['actions'].copy()
    require(list(actions.shape)==[25,*cell['action_shape']] and actions.dtype.kind=='f' and np.isfinite(actions).all(),'Incomplete/nonfinite candidate actions')
    p50=statistics.median(c['ms'] for r,c in zip(receipt['cases'],receipt['calls']) if r['phase']=='measured')
    expected_ids={family+'-'+arm for arm in ('eager_native','runtime_default','runtime_selected')};require(set(config['parent']['cell_receipts'])==expected_ids,'Missing primary raw bindings')
    parent_records={};comparisons=[]
    for identifier,value in config['parent']['cell_receipts'].items():
        name='capture/cells/'+identifier+'/receipt.json';check(parent_raw(name),value,'parent raw receipt');old=parent_read(name);parent_records[identifier]=old
        require(receipt['cases']==old['cases'] and receipt['guidance']==old['guidance'] and receipt['numeric_environment']==old['numeric_environment']
                and receipt['effective_schedule']==old['effective_schedule'] and receipt['default_schedule']==old['default_schedule']
                and receipt['observed_nfe_before']==receipt['observed_nfe_after']==old['observed_nfe_before']==old['observed_nfe_after']
                and receipt['checkpoint_snapshot']==old['checkpoint_snapshot'],'Paired input/RNG/native settings/source checkpoint differ')
        check(parent_raw(name[:-5]+'.npz'),old['actions_sha256'],'complete original parent actions')
        with np.load(io.BytesIO(parent_raw(name[:-5]+'.npz')),allow_pickle=False) as values:old_actions=values['actions'].copy()
        original_p50=statistics.median(c['ms'] for r,c in zip(old['cases'],old['calls']) if r['phase']=='measured')
        require(original_p50==parent_audit['metrics'][identifier]['p50_ms'],'Independent parent timing differs')
        comparisons.append({'baseline':identifier,'candidate':family+'-numeric-supplement/runtime_selected','same_sampling_policy':True,'speedup':original_p50/p50,'actions':delta(old_actions,actions)})
    if family in ('edge','nano'):
        stats=receipt['backend_stats']['stats'];cache=stats['timestep_cache']
        require(receipt['optimizer_environment']=={'IFL_COSMOS3_TIMESTEP_CACHE':'1'} and stats['action_chunk_size']==32 and stats['action_steps']==4 and stats['guidance']==3.0
                and cache['kind']=='native_full_timestep_result' and cache['hits']>0,'Actual full timestep-embedding cache proof missing')
    for name,key in [('validation.json','validation_sha256'),('serve_validation.json','serve_validation_sha256')]:
        check(raw(name),done[key],'producer '+name);validation=read(name)
        require(validation['status']=='passed' and validation['cell_id']==cell['id'] and validation['all_25_finite'] is True
                and validation['parent_receipts']==config['parent']['cell_receipts'],'Original complete validation differs')
        check(raw('capture/'+cell['receipt']),validation['receipt_sha256'],'validated candidate receipt');require(validation['actions_sha256']==receipt['actions_sha256'],'Validated candidate actions differ')
    ws=read('serve/receipt.json');require(ws['seed']==9173 and ws['interpreter']==model['python'],'Wrong numeric serving RNG/interpreter')
    websocket=serving(files,prefix,'serve',validation['serve_receipt_sha256'],cell,plan['checkpoint'],sha(raw('prepared/plan.json')),plan['fixture_sha256'],GPU)
    for i,call in enumerate(ws['calls']):
        observation=primary.observation(frames,i);observation['prompt']=primary.prompt(i//3)
        require(call['request_sha256']==primary.request_hash(observation),'Serving/reset paired inputs differ')
    stages=['prepare_numeric','admit_numeric','capture_numeric','validate_numeric','serve_numeric','validate_serve']
    require([r['name'] for r in done['stages']]==stages,'Missing/repeated stages')
    for row in done['stages']:
        require(row['exit_code']==0 and row['command'][0]==model['python'] and row['elapsed_seconds']>0,'Wrong/failed child stage')
        check(raw(row['name']+'.log'),row['log_sha256'],'actual stage log')
    check(raw('memory_samples.jsonl'),done['memory_samples_sha256'],'observed memory samples')
    require('torch' not in sys.modules,'CPU auditor imported Torch')
    return {'schema':'instinctflash.rtx5090_numeric_independent_audit.v1','family':family,'mode':'numeric','passed':True,'target':TARGET,'GPU_uuid':GPU,'source_commit':SOURCE,'archive_sha256':archive_sha,'completion_sha256':binding['completion_sha256'],'config_sha256':binding['config_sha256'],'parent_archive_sha256':parent_sha,'parent_audit_sha256':binding['parent_audit_sha256'],'vendor_cpu_provenance_sha256':binding['vendor_provenance_sha256'],'worker':identity,'metrics':{'p50_ms':p50,'primary_samples':20,'total_predictions':25,'precision':'native','schedule':cell['effective_schedule']},'comparisons':comparisons,'websockets':[websocket],'actual_numeric_environment':receipt['numeric_environment'],'full_public_run_completed':False,'primary_baselines_recaptured':False,'GPU_used_by_auditor':False,'Torch_imported_by_auditor':False,'task_quality_validated':False,'minimum_RAM_certified':False,'scope':'One additional NUMERIC cell+6callWS; actual parent three-cell results are separate. Runtime comparisons do not isolate FP8 arithmetic.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=tuple(PRIMARY),required=True)
    for n in ('archive','binding','parent-archive','parent-audit','vendor-provenance','output'):p.add_argument('--'+n,required=True)
    args=p.parse_args();prep=read_local('preparation_completion.json')
    for n,r in prep['files'].items():check((HERE/n).read_bytes(),r['sha256'],'frozen auditor '+n)
    d=HERE.parent/PRIMARY[args.family];check((d/'preparation_completion.json').read_bytes(),prep['primary_preparations'][args.family],'frozen parent auditor')
    for n,r in json.loads((d/'preparation_completion.json').read_bytes())['files'].items():check((d/n).read_bytes(),r['sha256'],'frozen parent input '+n)
    out=Path(args.output);require(not out.exists(),'Refusing audit overwrite')
    try:result=audit(args)
    except Exception as error:
        with out.with_suffix(out.suffix+'.failure.json').open('x') as f:json.dump({'passed':False,'GPU_used_by_auditor':False,'error_type':type(error).__name__,'error':str(error)},f,indent=2);f.write('\n')
        raise
    with out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'family':args.family,'metrics':result['metrics'],'comparisons':result['comparisons']}))

if __name__=='__main__':main()
