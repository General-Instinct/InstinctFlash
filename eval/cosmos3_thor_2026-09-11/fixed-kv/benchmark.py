"""Isolate fixed KV storage against the guarded Runtime conditioning path."""
import hashlib
import json
import os
from pathlib import Path
import sys
from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
from cosmos3_iwm import conditioning_cache
from benchmarks.regression import cosmos
import buffers

original_init = RobolabPolicyService.__init__
stats = []
def initialize(self, args):
    original_init(self, args)
    if os.environ.get('IFL_FIXED_KV') != '1':
        return
    original_generate = self.model.generate_samples_from_batch
    # Runtime installs its conditioning wrapper after service construction;
    # the original bound method is invoked through that wrapper, with an active
    # request and completed source admission, on first generation.
    installed = False
    def generate(*a, **kw):
        nonlocal installed
        if not installed:
            assert self._ifl_conditioning_cache_status['admitted']
            stats.append(buffers.install(self))
            installed = True
        return original_generate(*a, **kw)
    self.model.generate_samples_from_batch = generate
RobolabPolicyService.__init__ = initialize
output = Path(sys.argv[3])
code = cosmos.main()
r = json.loads(output.read_text())
r['fixed_kv'] = stats
r['fixed_kv_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for name in ('buffers.py', 'benchmark.py')}
output.write_text(json.dumps(r, indent=2) + '\n')
raise SystemExit(code)
