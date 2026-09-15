"""Validate full deployed capture identity/execution, without a quality claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate(directory, root=None):
    directory=Path(directory); root=Path(root or Path(__file__).parent)
    plan=json.loads((root/'plan.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert digest(root/'protocol.json')==plan['protocol_sha256']
    report=json.loads((directory/'report.json').read_text())
    assert report['status']=='success' and report['quality_certified'] is False
    assert report['plan_sha256']==digest(root/'plan.json')
    assert report['protocol_sha256']==plan['protocol_sha256']
    configuration=report['configuration_id']; attention=report['attention']
    assert dict(configuration_id=configuration,attention=attention) in plan['variants']
    assert report['checkpoint_manifest_sha256']==plan['packages'][configuration]['manifest_sha256']
    assert report['script_sha256']==plan['capture_sources']['run_capture.py']
    for name in ('capture_request.py','producer_capture.py'):
        matches=[sha for path,sha in report['sources'].items() if Path(path).name==name]
        assert matches==[plan['capture_sources'][name]]
    assets=report['external_runtime_assets']
    assert len(assets)==1 and assets[0]['bytes']==2818839170
    assert assets[0]['sha256']=='20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
    config=next(c for c in protocol['configurations'] if c['id']==configuration)
    branches=1 if config['guidance']==1 else 2
    requests=[r for r in protocol['requests'] if r['configuration_id']==configuration]
    assert len(report['requests'])==len(requests)==128
    for i,(row,request) in enumerate(zip(report['requests'],requests,strict=True)):
        assert row['index']==i and row['json']==f'{i:04d}.json' and row['npz']==f'{i:04d}.npz'
        meta=directory/row['json']; archive=directory/row['npz']
        assert digest(meta)==row['json_sha256'] and digest(archive)==row['npz_sha256']
        record=json.loads(meta.read_text()); trace=record['trace']
        assert record['request']==request and record['arrays_sha256']==row['npz_sha256']
        assert trace['effective_sampler_arguments']['seed']==[request['seed']]
        assert trace['seed_receipt']['actual_generation_seeds']==[[request['seed']]]
        assert trace['seed_receipt']['normal_request_rng_unchanged']
        assert trace['seed_receipt']['normal_seed_config_restored']
        assert trace['callback_timesteps']==[1000.]
        assert (trace['sampler_calls'],trace['preparation_calls'],trace['sampler_denoiser_callbacks'],
                trace['model_velocity_branch_calls'])==(1,1,1,branches)
        assert trace['effective_mask_observed'] and trace['observers_restored']
        assert trace['action_padding_intervention']['hooks_restored']
        with np.load(archive,allow_pickle=False) as data:
            assert set(data.files)=={'action','model_action','measured_action','model_measured_action',
                'endpoint','reference','preserve_mask','initial_noise'}
            for name in data.files:
                assert data[name].dtype==np.float32 and np.isfinite(data[name]).all(), name
            for name in ('action','model_action','measured_action','model_measured_action'):
                assert data[name].shape==(32,8)
            endpoint,reference,mask,noise=(data[k] for k in ('endpoint','reference','preserve_mask','initial_noise'))
            assert endpoint.ndim==2 and endpoint.shape[0]==1
            assert endpoint.shape==reference.shape==mask.shape==noise.shape
            assert np.all((mask==0)|(mask==1))
            offset=trace['action_offset'];assert offset==endpoint.shape[1]-33*64
            action_mask=mask[:,offset:].reshape(1,33,64)
            assert np.all(action_mask[:,0,:8]==1) and np.all(action_mask[:,1:,:8]==0)
            assert np.all(action_mask[:,:,8:]==1)
            assert np.all(endpoint[:,offset:].reshape(1,33,64)[:,:,8:]==0)
            assert np.array_equal(endpoint[mask==1],reference[mask==1])
            assert np.array_equal(data['model_action'],endpoint[:,offset:].reshape(33,64)[1:,:8])
    stats=report['backend_stats']
    assert stats['sampler_calls']==stats['velocity_evaluations']==stats['padding_projection_calls']==128
    assert stats['padding_projected_velocity_branches']==128*branches and stats['padding_hooks_restored']
    assert stats['guidance']==config['guidance'] and stats['sampling']['sigmas']==[1.,0.]
    assert report['execution_policy']['precision']=='native'
    assert report['execution_policy']['category']==('NUMERIC' if attention=='cudnn' else 'BITEXACT')
    if attention=='cudnn':assert stats['numeric_attention']['eligible_python_calls']>0
    return dict(validation_complete=True,quality_certified=False,requests=128,
        configuration_id=configuration,attention=attention,report_sha256=digest(directory/'report.json'),
        scope='Full frozen request/execution/array validation only; no task-quality or realtime certificate')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    output=a.directory/'capture_validation.json'
    output.write_text(json.dumps(dict(validation_complete=False))+'\n')
    result=validate(a.directory)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
