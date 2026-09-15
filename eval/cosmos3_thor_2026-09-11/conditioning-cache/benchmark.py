"""Native full-action comparison of request-local conditioning reuse."""
import hashlib
import json
import os
from pathlib import Path
import sys

from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
from benchmarks.regression import cosmos
from cache import ConditioningCache

instances = []
original_init = RobolabPolicyService.__init__

def initialize(self, args):
    original_init(self, args)
    if os.environ.get('IFL_CONDITIONING_CACHE') != '1':
        return
    model = self.model
    original_generate = model.generate_samples_from_batch
    cache = None
    def generate(*args, **kwargs):
        nonlocal cache
        if cache is None:
            # Runtime has now installed its default pointwise and layer graphs.
            cache = ConditioningCache(model, verify=os.environ.get('IFL_CACHE_VERIFY') == '1')
            original_velocity = model._get_velocity
            model._get_velocity = lambda **kw: cache.velocity(original_velocity, **kw)
            instances.append(cache)
        return cache.generate(original_generate, *args, **kwargs)
    model.generate_samples_from_batch = generate

RobolabPolicyService.__init__ = initialize
output = Path(sys.argv[3])
code = cosmos.main()
report = json.loads(output.read_text())
report['conditioning_cache'] = [cache.report() for cache in instances]
report['conditioning_cache_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    for name in ('cache.py', 'benchmark.py')}
output.write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(code)
