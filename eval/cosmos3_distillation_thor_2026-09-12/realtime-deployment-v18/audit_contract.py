"""Independent deployment receipt gate; never a task-quality certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def require_contract(report, seed):
    p=report.get('student_provenance',{})
    b=report.get('backend_stats',{})
    prompt=report.get('prompt_contract',{})
    if not (report.get('ok') is True and report.get('trained_student') is True
            and report.get('task_quality_certified') is False and report.get('category')=='SCREEN'
            and p.get('training_seed')==seed and p.get('student_updates')==64
            and p.get('state_key')=='student' and p.get('arm_id')==f'nano_sde1_cfg1_seed{seed}'):
        raise ValueError('Wrong student identity or admission status')
    if (b.get('precision')!='native' or type(b.get('action_steps')) is not int
            or isinstance(b.get('guidance'),bool) or b.get('action_chunk_size')!=32
            or b.get('action_steps')!=1 or b.get('guidance')!=1
            or b.get('action_padding')!='zero'
            or b.get('native_optimizations',{}).get('elided_lm_head_bytes')!=1244659712):
        raise ValueError('Nano action/residency contract differs')
    if (prompt.get('format_prompt_as_json') is not False or prompt.get('history_length')!=1
            or prompt.get('action_chunk_size')!=32 or prompt.get('max_action_dim')!=64
            or prompt.get('metadata_augmentors')!=['viewpoint_augmentor','duration_fps_augmentor','resolution_info_augmentor']):
        raise ValueError('Nano prompt metadata/layout changed')
    if (report.get('first_request_native_branch_clocks')!=[1000.]
            or b.get('sampling',{}).get('sigmas')!=[1.,0.]
            or b.get('sampler_calls')!=36 or b.get('velocity_evaluations')!=36
            or b.get('padding_projection_calls')!=36
            or b.get('padding_projected_velocity_branches')!=36
            or b.get('padding_hooks_restored') is not True):
        raise ValueError('Full one-step clock/conditional-branch/padding evidence differs')


def audit(directory):
    result=dict(status='success',category='SCREEN',task_quality_certified=False,seeds={})
    for seed in (12031,12032):
        folder=directory/f'seed{seed}';arrays={};reports={}
        for arm in ('native','cudnn','combined'):
            report=json.loads((folder/f'{arm}.json').read_text());require_contract(report,seed)
            archive=folder/f'{arm}.npz'
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=report['actions_sha256']:
                raise ValueError('Action archive changed')
            actions=np.load(archive)['actions']
            if actions.shape!=(36,32,8) or not np.isfinite(actions).all():
                raise ValueError('Missing or nonfinite full actions')
            arrays[arm]=actions;reports[arm]=report
        for arm in reports:
            for key in ('student_provenance','checkpoint_manifest_sha256','fixture_sha256','prompt_contract',
                        'benchmark_sha256','cache_helper_sha256','declared_execution','torch','cuda','cudnn'):
                if reports[arm][key]!=reports['native'][key]:raise ValueError(f'Unpaired {key}')
        if arrays['cudnn'].tobytes()!=arrays['combined'].tobytes():
            raise ValueError('Combined optimization changed action bytes')
        result['seeds'][str(seed)]=dict(full_actions_checked=108,
            package_sha256=reports['native']['checkpoint_manifest_sha256'],
            source_reports={arm:hashlib.sha256((folder/f'{arm}.json').read_bytes()).hexdigest() for arm in reports})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh audit output')
    try:result=audit(a.directory)
    except Exception as error:
        a.output.write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2)+'\n')
