"""Render only an independently passed four-arm audit; no model execution."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
audit_path = root/'audit.json'
audit = json.loads(audit_path.read_text())
assert audit['status'] == 'passed' and audit['completed_jobs'] == 4
assert audit['finite_actions_checked'] == 72 and audit['quality_certified'] is False
rows = []
for precision, label in [('native', 'Native (BF16)'), ('fp8', 'FP8')]:
    assert audit['jobs'][precision+'-dynamic']['actual_timing_computed_steps'] == [4] * 9
    receipt = json.loads((root/(precision+'-dynamic.json')).read_text())
    assert receipt['backend_stats']['step_cache']['last_generation']['peak_cache_bytes'] == 453632
    fixed = audit['jobs'][precision+'-checkpoint']['continuation_p50_ms']
    dynamic = audit['jobs'][precision+'-dynamic']['continuation_p50_ms']
    ratio = audit['ratios'][precision+'_checkpoint_over_'+precision+'_dynamic']['ratio_of_continuation_p50']
    rows.append(f'| {label} | {fixed:.2f} | {dynamic:.2f} | {ratio:.2f}× |')
combined = audit['ratios']['native_checkpoint_over_fp8_dynamic']['ratio_of_continuation_p50']
assert {k:v['verified_files'] for k,v in audit['manifest_content_verification'].items()} == {
    'source':392, 'drivers':5, 'external':109}
text = f'''# Thor dynamic step-cache results — 2026-09-14

The shared DreamZero integration passed all four native/FP8 × fixed/dynamic
processes. The independent audit checked **72 finite action arrays** and **24
exact paired action comparisons** against the same native dynamic policy.

| Arithmetic | Fixed 8 DiT calls (ms) | Dynamic cache (ms) | Cache speedup |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

These are synchronized public-predict continuation p50 values, with **four
measured continuation samples per arm**. One full three-cycle episode was
warmed before two measured episodes. Both schedulers retain all sixteen
updates and CFG remains 5. Every measured dynamic request computed four DiT
slots (eight CFG branches), plus two separate observation-KV branch forwards.
The peak owned prediction storage was 453,632 bytes. Raw samples remain in the
individual JSON receipts and [independent audit](audit.json).

Native fixed-mask execution to FP8 plus dynamic reuse measured {combined:.2f}×
in these samples. This combines arithmetic and prediction-reuse changes.
The result is a **bounded latency SCREEN**, not a tail-latency estimate or a
closed-loop task-quality certificate. Dynamic outputs are not claimed equivalent
to the fixed-mask outputs; FP8 and native actions are not claimed equivalent.
The published multi-framework table remains unchanged.

Within each dynamic arithmetic route, matched requests compared the installed
shared controller, the original native dynamic loop and the reinstalled shared
controller. Full actions matched byte for byte. Producer checks also matched
raw video/action endpoints, all 160 KV-array hashes per request, compute masks,
both solver clocks and post-call RNG states. The independent CPU auditor verifies
the stored endpoint/KV receipts; those raw tensors were not archived for an
independent finite-value recheck.

The audit verified the contents of 392 frozen Runtime source/binary entries,
five driver/protocol entries and 109 external source/fixture entries. No GPU
recapture or inference retry was needed. An initial CPU-only checkpoint-path
assumption was rejected because HF snapshot links terminate in content-addressed
blobs; the corrected snapshot-link verification and original rejected check are
both retained. This did not change any inference input or output.

Recheck the copied results without loading a model:

```bash
python eval/dynamic_step_cache_integration_2026-09-14/audit.py \\
  eval/dynamic_step_cache_integration_2026-09-14 \\
  --output /tmp/dynamic-step-cache-audit.json
```

For source-content verification as well, extract `source.tar.gz` into a temporary
directory and supply `--strict-content --source-root <extracted>/source
--external-map <path-map.json>`. The provided `external_path_map.json` records
this workspace's local copies; adjust it for another checkout. The auditor
returns 0 on pass, 1 on failure and 2 for incomplete results.

Audit SHA256: `{hashlib.sha256(audit_path.read_bytes()).hexdigest()}`.
'''
(root/'results.md').write_text(text)
