"""Reserved Thor-only cross-stream ownership/eviction regression."""
import hashlib
import json
from pathlib import Path

import torch
from tensor_cache import TensorResultCache

assert torch.cuda.get_device_capability() == (11, 0)
cache = TensorResultCache(max_entries=1)
producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
checks = []
with torch.no_grad():
    for i in range(20):
        with torch.cuda.stream(producer):
            reference = torch.arange(65536, device='cuda', dtype=torch.float32) + i
            cold = cache.get_or_compute(('a', i), lambda: reference * 2)
        with torch.cuda.stream(consumer):
            hit = cache.get_or_compute(('a', i), lambda: None)
        with torch.cuda.stream(producer):
            cache.get_or_compute(('b', i), lambda: torch.zeros_like(reference))
            churn = [torch.ones_like(reference) for _ in range(8)]
        torch.cuda.synchronize()
        checks.append(torch.equal(hit, reference * 2) and torch.equal(hit, cold))
        hit.zero_()
        assert len(churn) == 8
assert all(checks)
report = dict(ok=True, comparisons=len(checks), device=torch.cuda.get_device_name(),
              torch=torch.__version__, cache=cache.report(),
              source_sha256=hashlib.sha256(Path(__file__).with_name('tensor_cache.py').read_bytes()).hexdigest())
Path(__file__).with_name('cuda_check.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
