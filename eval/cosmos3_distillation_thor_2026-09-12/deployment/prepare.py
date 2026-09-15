"""Copy audited V14/V15 native exports into separate, unqualified Flash packages."""
import argparse
import json
from pathlib import Path
import shutil

from instinct_compress.artifacts import file_sha256, finalize_artifact, verify_artifact
from instinct_compress.flash.cosmos3_action_padding import padding_sidecar, PADDING_SIDECAR

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('export_roundtrip',type=Path)
p.add_argument('destination',type=Path)
p.add_argument('--model-id',required=True)
a=p.parse_args()
source=(a.export_roundtrip/'checkpoint').resolve()
dest=a.destination.resolve()
if dest.exists() or source==dest or source in dest.parents:
    raise ValueError('Use a fresh destination outside the frozen export')
report=json.loads((a.export_roundtrip/'report.json').read_text())
assert report['status']=='success' and report['cold_reload_bitexact']
assert report['settings']['guidance']==4. and report['settings']['times']==[1.,.75,.5,.25,0.]
inventory=json.loads((a.export_roundtrip/'export_inventory.json').read_text())
assert inventory['status']=='success' and inventory['no_lora_keys'] and inventory['all_shards_independent']
records={}
for row in inventory['inventory']:
    path=Path(row['path']).resolve(); relative=path.relative_to(source)
    assert path.stat().st_size==row['bytes'] and file_sha256(path)==row['sha256'],relative
    records[relative.as_posix()]=row['sha256']
assert set(records)=={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
config=json.loads((source/'config.json').read_text())
assert config['model']['config']['fixed_step_sampler_config']=={
    '_type':'fixed_step_sampler_config','sample_type':'sde','t_list':[1.,.75,.5,.25]}
shutil.copytree(source,dest,symlinks=False)
for name,digest in records.items():
    assert file_sha256(dest/name)==digest,name
(dest/PADDING_SIDECAR).write_text(json.dumps(padding_sidecar(),indent=2)+'\n')
execution=dict(model_id=a.model_id,backbone='cosmos3_policy_action_fixed_step',servable=False,
    nfe={'prefix':1,'action':4},guidance={'action':{'mode':'cfg','scale':4.}},
    sampling={'kind':'rectified_flow_fixed_step','sample_type':'sde',
              'num_train_timesteps':1000.,'sigmas':[1.,.75,.5,.25,0.]},
    action_padding='zero',action_dim=8,action_chunk_size=32,domain_name='droid_lerobot',
    conditioning_fps=15.,format_prompt_as_json=True,image_height=540,image_width=640,shift=1.)
provenance=dict(scope='Copied audited native export; new Flash declaration pending Thor qualification; no task-quality certification',
    original_export=str(a.export_roundtrip.resolve()),
    export_report_sha256=file_sha256(a.export_roundtrip/'report.json'),
    original_inventory_sha256=file_sha256(a.export_roundtrip/'export_inventory.json'))
manifest=finalize_artifact(dest,execution=execution,provenance=provenance,
    required_files=('checkpoint.json',PADDING_SIDECAR))
verify_artifact(dest)
for name,digest in records.items():
    assert file_sha256(source/name)==digest,name
receipt=dict(prepared=True,servable=False,task_quality_certified=False,source=str(source),
    destination=str(dest),original_files_verified=len(records),
    copied_original_files_identical=True,original_postcheck_passed=True,
    manifest_sha256=file_sha256(dest/'instinctcompress_manifest.json'),total_bytes=manifest['total_bytes'])
(dest.parent/(dest.name+'-preparation.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
