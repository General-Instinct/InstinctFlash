"""CPU fault-injection check for the archived Edge comparison gate."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

base = Path(__file__).resolve().parent
for kind in ('hash', 'protocol', 'latency', 'tail'):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for path in (base / 'receipts').glob('edge-*'):
            shutil.copyfile(path, root / path.name)
        path = root / 'edge-cached.json'
        report = json.loads(path.read_text())
        if kind == 'hash':
            report['actions_sha256'] = '0' * 64
        elif kind == 'protocol':
            report['effective_schedule']['guidance'] = 1.0
        elif kind == 'latency':
            for call in report['calls']:
                call['ms'] *= 2
        else:
            measured = [call for call in report['calls'] if call['phase'] == 'measured']
            measured[-1]['ms'] *= 3
        path.write_text(json.dumps(report))
        # Explicitly seed an old success, including when the archive was copied
        # without its optional comparison receipt.
        (root / 'edge-comparison.json').write_text(json.dumps(dict(passed=True)))
        result = subprocess.run([sys.executable, str(base / 'compare.py'), str(root), 'edge'],
                                capture_output=True)
        assert result.returncode != 0, (kind, result.stdout, result.stderr)
        assert not json.loads((root / 'edge-comparison.json').read_text())['passed'], kind
        print(kind, 'rejected; no stale passing receipt')
