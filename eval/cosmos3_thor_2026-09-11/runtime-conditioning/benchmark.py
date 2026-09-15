"""Public Runtime comparison: no service/model monkeypatches."""
import hashlib
import json
from pathlib import Path
import sys
from cosmos3_iwm import conditioning_cache
from benchmarks.regression import cosmos

output = Path(sys.argv[3])
code = cosmos.main()
report = json.loads(output.read_text())
stats = report.get('backend_stats', {}).get('conditioning_cache')
report['conditioning_cache'] = [stats] if stats is not None else []
report['runtime_cache_sources'] = {'conditioning_cache.py': hashlib.sha256(Path(conditioning_cache.__file__).read_bytes()).hexdigest()}
output.write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(code)
