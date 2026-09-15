"""Bind final receipts and list work remaining across all registered checkpoints."""
import argparse
from pathlib import Path

from benchmarks.vla.execution_evidence import build_record
from benchmarks.vla.qualification import read_startup, report
from benchmarks.vla.registry import load_registry
from benchmarks.vla.util import load_json, write_json_atomic


def build(root, repository, output):
    if output.exists():
        raise ValueError('refusing to overwrite inventory directory')
    output.mkdir(parents=True)
    bundles = [Path(r['bundle']) for r in load_json(repository / 'eval/simulator_quality_2026-09-06/screening-index.json')]
    previous = load_json(repository / 'eval/simulator_next_steps_2026-09-06/evidence-index.json')
    bundles += [Path(r['path']) for r in previous['simulation_bundles'].values()]
    bundles += [root.parent / 'precision_evidence_20260906_v2' / repeat / 'evidence'
                for repeat in ('repeat1', 'repeat2')]
    starts, records = [], {}
    for folder in (root, root / 'thor-results'):
        for path in sorted(folder.glob('*.receipt.json')):
            if not path.name.startswith(('final.', 'corrected-env.', 'strict.', 'va-final.')):
                continue
            raw = load_json(path)
            if raw.get('startup', {}).get('fault_injection'):
                continue
            read_startup(path)
            starts.append(path)
            timing = path.with_name(path.name.replace('.receipt.json', '.latency.json'))
            record = build_record(path, measurement_path=timing if timing.exists() else None)
            pid = record['profile']['profile_id']
            if pid not in records or timing.exists():
                records[pid] = record
    for pid, record in records.items():
        write_json_atomic(output / f'{pid}.execution.json', record)
    for device in ('H100', 'Thor'):
        value = report(load_registry(), target_device=device, bundles=bundles,
                       records=list(records.values()), startup_receipts=starts, minimum_startups=3)
        write_json_atomic(output / f'{device.lower()}-qualification.json', value)
    print(f'{len(starts)} ordinary startup receipts, {len(records)} distinct profiles, {len(bundles)} historical bundles')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.repository, args.output)
