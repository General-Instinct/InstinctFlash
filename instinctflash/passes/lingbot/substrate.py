"""Substrate passes — remove work the model never asked for.

These three are the least glamorous passes in the framework and, so far, the most valuable:
together they were measured at **1.92x, bit-exact** on LingBot-VA (8881 -> 4624 ms per control
cycle; `max|delta action| = 0.000e+00` over 6 paired seeded cycles). See
`eval/lingbot_va_robotwin/RESULTS.md` section 4.

They belong in the framework rather than in a per-model patch file because none of them is
about LingBot-VA. Each fires on a *situation* that any single-GPU batch-1 closed loop can be in,
and the situation is detectable from declarations plus the runtime environment:

  * a model sharded across one GPU
  * an allocator being handed back to the driver between control steps
  * debug telemetry doing a blocking device->host copy on the critical path

A generic serving stack does not look for these because they only hurt at batch 1 with a hot
loop. That is exactly the regime a robot policy lives in, and exactly the regime nobody profiles.
"""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.planners.planner import PassResult, Tier


class FSDPElision:
    """Do not shard a model across one GPU.

    `distributed/util.py:15-19` applies FSDP `fully_shard` whenever `dist.is_initialized()` —
    which is always, because `init_distributed` runs unconditionally even for a single-GPU
    server. `fsdp.py:28-34` wraps 4 units per block (attn1, attn2, ffn, block) across 30 blocks
    plus the root = **121 units**, all with `reshard_after_forward=True`.

    At world_size 1 every all-gather is a no-op collective, but PyTorch still pays the flat-param
    copy and stream sync per unit per forward: ~121 x 79 = **~9,600 shard/unshard round trips per
    control step** to shard a model across one GPU.

    Bit-exact at world_size 1: FSDP shards into exactly one shard, so the all-gather is identity,
    and `MixedPrecisionPolicy(param_dtype=bf16)` on already-bf16 weights is a no-op cast.
    Measured: 8881 -> 5078 ms, `max|delta| = 0`.
    """

    name = "fsdp_elision"

    #: These three passes PATCH THE LINGBOT-VA SERVER OBJECT -- they elide its FSDP wrapping, its
    #: `empty_cache` calls and its per-chunk debug dumps. None of that exists in a model that is not
    #: this server, so without a gate they were reported as APPLY for every checkpoint in the
    #: ecosystem: an external toy GRU got a plan promising "measured 1.75x standalone on LingBot-VA".
    #: The plan is a claim about what will happen to THIS model, so a pass that cannot touch it must
    #: decline. Gating on the backbone is honest rather than lazy here -- these passes live in
    #: `passes/lingbot/` because that is genuinely what they know how to rewrite.
    requires_capabilities = frozenset({"backbone:wan_va"})


    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        if deployment.world_size > 1:
            return PassResult(self.name, False, Tier.BITEXACT,
                              f"world_size={deployment.world_size}: sharding is doing real work")
        return PassResult(
            self.name, True, Tier.BITEXACT,
            "world_size=1, so every FSDP all-gather is identity while still paying a "
            "flat-param copy and stream sync per unit per forward",
            params={"world_size": deployment.world_size},
            expected_win="measured 1.75x standalone on LingBot-VA",
        )


class AllocatorChurnElision:
    """Stop returning the caching allocator to the driver between control steps.

    `torch.cuda.empty_cache()` runs on every chunk (`wan_va_server.py:569`) and every KV update
    (`:603`). In a one-shot generation script that is harmless hygiene. In a closed loop with a
    fixed working set it is pure cost: the allocator releases pooled blocks and the next control
    step re-runs `cudaMalloc` for the same shapes, forever.

    Bit-exact: allocation policy cannot change a tensor value.
    """

    name = "allocator_churn_elision"

    requires_capabilities = frozenset({"backbone:wan_va"})   # see fsdp_elision above


    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        return PassResult(
            self.name, True, Tier.BITEXACT,
            "closed-loop serving has a stable working set, so releasing the pool between "
            "control steps only forces re-allocation next step",
            expected_win="small alone; larger once per-layer KV gathers stop churning the pool",
        )


class DebugDumpElision:
    """Take blocking telemetry off the critical path.

    `save_async` (`utils.py:56-70`) is async only for the *disk write*. The `.cpu()` at :63-64 is
    a **blocking device->host copy** of the full latent and action tensors, executed three times
    per control step, unconditionally, with no upstream flag to disable it.

    Bit-exact: writing a tensor to disk does not change it. Visible mostly in the KV-commit
    phase, measured 453 -> 292 ms.
    """

    name = "debug_dump_elision"

    requires_capabilities = frozenset({"backbone:wan_va"})   # see fsdp_elision above


    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        return PassResult(
            self.name, True, Tier.BITEXACT,
            "per-chunk tensor dumps do a synchronous D2H on the critical path and are not "
            "consumed by the serving client",
            expected_win="measured ~1.10x on top of fsdp_elision",
        )


class ObsDecodeElision:
    """Never build or run the observation-decode tail unless pixels were requested.

    All four WAMs surveyed agree the predicted-video decode is optional at serving time, and all
    four pay for it differently. Cosmos3-Edge is the extreme: it denoises 567 GEN tokens of which
    **550 are future-video tokens that are discarded** at the call site, decoding only under
    `--decode-video`. LingBot-VA returns latents its RoboTwin client drops on the floor.

    This pass only skips the *decode*; it does not touch the denoise loop that produced the video
    tokens. Deleting those is a much larger and genuinely lossy question — the action stream
    attends to the video KV — and belongs to the adaptive-NFE pass, not here.
    """

    name = "obs_decode_elision"

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        if not spec.obs_decode_modules:
            return PassResult(self.name, False, Tier.BITEXACT,
                              "model declares no observation-decode modules")
        if deployment.want_pixels:
            return PassResult(self.name, False, Tier.BITEXACT,
                              "caller requested predicted pixels")
        return PassResult(
            self.name, True, Tier.BITEXACT,
            f"caller wants actions only; {list(spec.obs_decode_modules)} need never be "
            f"resident or executed",
            params={"modules": list(spec.obs_decode_modules)},
            expected_win="no latency win on this model (already skipped); frees resident memory, "
                         "which is what sets episodes-per-GPU",
        )
