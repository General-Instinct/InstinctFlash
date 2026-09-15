"""Copy an audited Edge final-online64 V17 export into a qualification package."""
import argparse
import json
from pathlib import Path
import shutil

from instinct_compress.artifacts import file_sha256, finalize_artifact, verify_artifact
from instinct_compress.flash.cosmos3_action_padding import padding_sidecar, PADDING_SIDECAR
from realtime_adapter import BACKBONE


def checked(record):
    path = Path(record['path'])
    if path.stat().st_size != record['bytes'] or file_sha256(path) != record['sha256']:
        raise ValueError(f'Changed producer evidence: {path}')
    return path


def validate_gate(gate):
    expected = dict(schema_version=17, status='success', student_updates=64, steps=2,
        times=[1., .5, 0.], action_shape=[33, 64], future_action_shape=[32, 8],
        action_padding='zero', state_key='student', live_state_matches_snapshot=True,
        cold_reload_bitexact=True, native_serving_bitexact=True, actual_callbacks=6)
    if any(gate.get(k) != v for k, v in expected.items()):
        raise ValueError('Require successful V17 final-online64 native export and cold-load gate')
    guidance = gate.get('guidance')
    if isinstance(guidance, bool) or guidance != 1.:
        raise ValueError('Require literal CFG1')
    if gate.get('actual_branches') != 6:
        raise ValueError('Mechanical export branch budget differs')
    if gate.get('training_seed') not in (12031, 12032) or gate.get('arm_id') != (
            f"edge_sde2_cfg{int(guidance)}_seed{gate['training_seed']}"):
        raise ValueError('Require one of the two declared Edge training arms')


def execution_declaration(gate, model_id):
    validate_gate(gate)
    return dict(model_id=model_id, backbone=BACKBONE, servable=False,
        nfe={'prefix': 2, 'action': 2}, guidance={'action': {'mode': 'cfg', 'scale': gate['guidance']}},
        sampling=dict(kind='rectified_flow_fixed_step', sample_type='sde',
            num_train_timesteps=1000., sigmas=[1., .5, 0.]),
        action_padding='zero', action_dim=8, action_chunk_size=32, domain_name='droid_lerobot',
        conditioning_fps=15., format_prompt_as_json=True, image_height=540, image_width=640, shift=1.)


def prepare(export, destination, model_id, completion_path):
    export, destination = export.resolve(), destination.resolve()
    source = export / 'checkpoint'
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError('Use a fresh destination outside the producer checkpoint')
    gate_digest = file_sha256(export/'report.json')
    gate = json.loads((export/'report.json').read_text())
    validate_gate(gate)
    completion_digest = file_sha256(completion_path)
    completion = json.loads(completion_path.read_text())
    if completion.get('status') != 'success':
        raise ValueError('Require completed V17 training/audit/export execution')
    bound_gates = [checked(row).resolve() for row in completion['export_gates']]
    if (export/'report.json').resolve() not in bound_gates:
        raise ValueError('Completion does not bind this export gate')
    audit = json.loads(checked(completion['training_audit']).read_text())
    if not (audit.get('schema') == 'twostep_training_v17_cpu_audit_v1'
            and audit.get('status') == 'success' and audit.get('valid') is True
            and audit.get('mode') == 'train' and audit.get('protocol') == gate['protocol']
            and gate['arm_id'] in audit.get('arms', [])):
        raise ValueError('Require matching independent formal-training audit')
    arm = next(row for row in audit['results'] if row['arm_id'] == gate['arm_id'])
    checked(arm['report'])
    if arm['counts']['student_updates'] != 64 or arm['counts']['fake_updates'] != 443:
        raise ValueError('Require final64 with all prescribed fake updates')
    if Path(gate['checkpoint']).resolve() != source:
        raise ValueError('Gate names another native checkpoint')
    for key in ('snapshot', 'protocol', 'plan', 'manifest', 'source_report', 'cold_report', 'export_inventory'):
        checked(gate[key])
    source_report = json.loads(checked(gate['source_report']).read_text())
    cold = json.loads(checked(gate['cold_report']).read_text())
    if source_report['status'] != 'success' or cold['status'] != 'success':
        raise ValueError('Source and cold reports must both succeed')
    if source_report['snapshot'] != gate['snapshot'] or source_report['student_state_sha256'] != gate['student_state_sha256']:
        raise ValueError('Gate/source snapshot or student identity differs')
    if cold['source_report'] != gate['source_report'] or source_report['pid'] == cold['pid']:
        raise ValueError('Require a separate cold process referencing this exact source report')
    for key in ('full_endpoint_bitexact', 'normalized_actions_bitexact', 'decoded_actions_bitexact',
                'condition_and_seed_bitexact', 'fresh_process', 'no_duplicate_lora'):
        if cold.get(key) is not True:
            raise ValueError(f'Cold-load check missing: {key}')
    inventory = json.loads(checked(gate['export_inventory']).read_text())
    if not (inventory['status']=='success' and inventory['no_lora_keys'] and inventory['all_shards_independent']):
        raise ValueError('Require complete merged native inventory')
    records = {}
    for row in inventory['inventory']:
        relative = checked(row).resolve().relative_to(source)
        records[str(relative)] = row['sha256']
    if set(records) != {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}:
        raise ValueError('Inventory does not cover every native file')
    config = json.loads((source/'config.json').read_text())
    if config['model']['config']['fixed_step_sampler_config'] != dict(
            _type='fixed_step_sampler_config', sample_type='sde', t_list=[1., .5]):
        raise ValueError('Native checkpoint must declare the complete SDE2 grid')
    shutil.copytree(source, destination, symlinks=False)
    for name, digest in records.items():
        if file_sha256(destination/name) != digest:
            raise ValueError(f'Copy differs: {name}')
    (destination/PADDING_SIDECAR).write_text(json.dumps(padding_sidecar(), indent=2)+'\n')
    execution = execution_declaration(gate, model_id)
    finalize_artifact(destination, execution=execution,
        provenance=dict(scope='Copied Edge final-online64 V17 export; Thor and quality unqualified',
            original_export=str(export), completion_sha256=completion_digest,
            training_audit_sha256=completion['training_audit']['sha256'], export_gate_sha256=file_sha256(export/'report.json'),
            arm_id=gate['arm_id'], training_seed=gate['training_seed'], state_key='student',
            student_updates=64, student_state_sha256=gate['student_state_sha256']),
        required_files=('checkpoint.json', PADDING_SIDECAR))
    verify_artifact(destination)
    for name, digest in records.items():
        if file_sha256(source/name) != digest:
            raise ValueError(f'Original changed during preparation: {name}')
    for key in ('snapshot', 'protocol', 'plan', 'manifest', 'source_report', 'cold_report', 'export_inventory'):
        checked(gate[key])
    if file_sha256(export/'report.json') != gate_digest:
        raise ValueError('Export gate changed during preparation')
    checked(completion['training_audit'])
    if file_sha256(completion_path) != completion_digest:
        raise ValueError('Completion changed during preparation')
    return dict(prepared=True, servable=False, task_quality_certified=False,
        source=str(source), destination=str(destination), original_files_verified=len(records),
        original_postcheck_passed=True, manifest_sha256=file_sha256(destination/'instinctcompress_manifest.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('export_roundtrip', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--model-id', required=True)
    p.add_argument('--completion', type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.export_roundtrip, a.destination, a.model_id, a.completion)
    (a.destination.parent/(a.destination.name+'-preparation.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result, indent=2))
