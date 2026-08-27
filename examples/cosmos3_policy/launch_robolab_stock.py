#!/usr/bin/env python3
"""Launch NVIDIA's stock robolab policy server with guardrails neutered.

nvidia/Cosmos-Guardrail1 is a gated repo. Upstream's own test
(inference/inference_test.py) bypasses guardrails by patching the runner; this launcher does
the same one step earlier — the runners are never constructed — and changes nothing else, so
the measured path is NVIDIA's shipped serving code.
"""
import sys

from cosmos_framework.inference.common import inference as common_inference


class _NoopRunner:
    def run(self, *a, **k):
        return None

    def is_safe(self, **k):
        return True, "guardrails disabled for latency benchmark"


def _no_guardrails(cls, args):
    return cls(text=_NoopRunner(), video=_NoopRunner())


common_inference.GuardrailRunners.create = classmethod(_no_guardrails)

from cosmos_framework.scripts.action_policy_server_robolab import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
