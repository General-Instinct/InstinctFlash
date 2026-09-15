"""Copy small raw JSON receipts into this reviewable benchmark package."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
base = Path(__file__).resolve().parent
matrix = json.loads((base / 'measured_candidates.json').read_text())
paths = {p.relative_to(args.source) for p in args.source.glob('*.json')}
for row in matrix['rows']:
    for cell in row['frameworks'].values():
        for candidate in cell.get('candidates', []) + cell.get('failed', []):
            paths.add(Path(candidate['receipt']))
inventory = []
for rel in sorted(paths):
    src = args.source / rel
    dst = base / 'receipts' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    inventory.append({'path': str(Path('receipts') / rel), 'bytes': dst.stat().st_size,
                      'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()})
(base / 'receipt_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
print(f'Packaged {len(inventory)} receipts')
