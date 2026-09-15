"""InstinctFlash adapter for LeRobot's pi05 VLA family. External plugin: no core changes.

Written to test whether InstinctFlash's abstraction is real, using a model family deliberately unlike
LingBot-VA. Facts below come from lerobot/pi05_base's own config.json, not from guesses.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from instinctflash import (AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec, PurityKey)
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "pi05"

#: The kill-switch for the family's DEFAULT static-KV graph capture. Family-scoped on purpose
#: (the IFL_PI05_* convention): the default it disables is pi05's, and other families' capture
#: policies are their own. Honored by `Pi05Adapter.install`, recorded on the plan, printed.
CAPTURE_KILL_SWITCH = "IFL_PI05_NO_CAPTURE"


class Pi05Adapter:
    """A vision-language-action policy: one observation in, a 50-action chunk out."""

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id="lerobot/pi05_base",
            param_bytes=14_467_165_872,
            # ONE stream, and its lifetime is the interesting part: the prefix K/V is recomputed
            # every control step (n_obs_steps=1, no history), so CHUNK -- not EPISODE like a WM.
            streams=(
                KVStreamSpec(name="prefix", tokens_per_frame=200, lifetime=KVLifetime.CHUNK),
            ),
            # vision+language prefix once, then 10 flow-matching steps over the action chunk.
            phases=(
                PhaseSpec(name="prefix", nfe=1, writes=frozenset({"prefix"})),
                PhaseSpec(name="action", nfe=10, reads=frozenset({"prefix"}),
                          truncatable=True, min_nfe=1, depends_on=("prefix",)),
            ),
            # Flow matching, no classifier-free guidance at all.
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            # The prefix is a pure function of the observation and prompt, constant across all 10
            # action steps -- the same shape of claim LingBot-VA makes at EPISODE scope.
            # The prefix KV is chunk-constant AND upstream already hoists it: `sample_actions` runs
            # the prefix once with use_cache=True and threads the resulting cache through all ten
            # denoise steps. Declaring the purity without `already_hoisted` made the plan promise a
            # conditioning_prefill win that did not exist -- it reported "recomputed on all 11
            # forwards per control step" about an implementation that recomputes it zero times.
            purity=(PurityKey(artifact="prefix_kv", fields=("images", "state", "prompt"),
                              scope=KVLifetime.CHUNK, already_hoisted=True),),
            obs_decode_modules=(),      # a VLA predicts no pixels
            observation=ObservationSpec(
                fields=(ObservationField("observation.images.base_0_rgb", (3, 224, 224)),
                        ObservationField("observation.images.left_wrist_0_rgb", (3, 224, 224)),
                        ObservationField("observation.images.right_wrist_0_rgb", (3, 224, 224)),
                        ObservationField("observation.state", (32,))),
                history=1, conditioning=("prompt",)),
            # "backbone" identifies this spec to backbone-keyed planner checks (the engine
            # geometry gate in passes/generic/engine_offload.py keys on it; notes are the one
            # channel a pass can see). Per-checkpoint action geometry rides on notes too — see
            # spec_for_checkpoint.
            notes={"family": "vla", "backbone": BACKBONE, "chunk_size": "50",
                   "n_obs_steps": "1"},
        )

    def observation_contract(self, checkpoint):
        """What `predict` expects FOR THIS CHECKPOINT, and where that answer came from.

        `spec().observation` is pi05_base's geometry. A fine-tune renames and reshapes the
        cameras — `lerobot/pi05_libero_finetuned_v044` takes two 256x256 cameras, one empty 224
        camera and an 8-dim state, and feeding it the base geometry dies inside its normalizer
        with a shape error AFTER the weights loaded. So geometry is a declaration fact:
        `execution.obs_features` maps each observation key to its per-observation shape, and a
        checkpoint that is not the base release must declare it rather than inherit a different
        robot's cameras.
        """
        import dataclasses

        from instinctflash.adapters.base import ObservationField

        raw = (checkpoint.execution.extra or {}).get("obs_features")
        # "FILL_ME" (or anything that is not a mapping) is a scaffold sentinel, not a value —
        # it falls through to the loud declare-your-obs_features message below.
        feats = dict(raw) if isinstance(raw, dict) else {}
        static = self.spec().observation
        if feats:
            fields = tuple(ObservationField(str(k), tuple(int(x) for x in shape), "float32")
                           for k, shape in feats.items())
            return dataclasses.replace(static, fields=fields), \
                "the checkpoint's declared execution.obs_features"
        if (checkpoint.execution.model_id or "") == "lerobot/pi05_base":
            return static, "the adapter's static declaration (pi05_base geometry)"
        raise RuntimeError(
            f"{checkpoint.execution.model_id or checkpoint.path}: no execution.obs_features "
            f"declared, and pi05 fine-tunes do not share the base checkpoint's cameras "
            f"(v044 takes image/image2/empty_camera_0 + an 8-dim state, the base takes three "
            f"224x224 cameras + a 32-dim state). Declare obs_features in the checkpoint's "
            f"instinctflash.json: a mapping of observation key -> per-observation shape, e.g. "
            f'{{"observation.images.image": [3, 256, 256], "observation.state": [8]}}. '
            f"The values are in the checkpoint's own train_config.json input_features.")

    def spec_for_checkpoint(self, checkpoint) -> AdapterSpec:
        """Bind the family spec to the upstream weights this declaration selects.

        Checkpoint facts ride on the spec, from ONE source each ("action geometry" below joins
        the original two — ``runtime.engine_backend.declared_action_dim`` owns its resolution):

        * ``notes['base_weights']`` — the upstream repo the pointer package selects. Passes that
          carry per-release numeric evidence (``Pi05TF32Numeric``) key on it, because the plan
          header ``model_id`` is rewritten by the facade to the DECLARED package id (the correct
          label for the plan, and the wrong key for evidence measured on the underlying weights).
        * ``observation`` — resolved by ``observation_contract`` above, the single owner of
          per-checkpoint geometry (``execution.obs_features``/KNOWN_DECLARATIONS). This method
          deliberately declares no geometry of its own: an earlier draft hardcoded a 3-field v044
          contract here that contradicted the declared 4-field one (incl.
          ``observation.images.empty_camera_0``), which is exactly the two-hooks failure this
          unification removes. A checkpoint that cannot resolve geometry yet still PLANS with the
          static shape — serving and ``Runtime.observation`` keep failing loud through
          ``observation_contract`` itself.
        """
        base = self.spec()
        notes = dict(base.notes)
        repo = str((checkpoint.execution.extra or {}).get("base_weights") or "")
        if repo:
            notes["base_weights"] = repo
        op = (checkpoint.execution.extra or {}).get("pi05_tf32_numeric")
        if isinstance(op, dict):
            notes["tf32_hardware"] = op.get("hardware")
        # ACTION geometry, for the planner's engine gate (one rule, two surfaces: the runtime's
        # EngineBackend enforces the same fact via the same helper). The engine's pi05 frontend
        # serves LIBERO's 7-dim actions as built and used to silently truncate a 32-dim
        # checkpoint's chunks to (10, 7); the pass can only decline that at plan time if the
        # declared dimensionality is a fact on the spec. Unresolved stays fail-closed at the
        # pass: notes carry the reason instead of a number.
        from instinctflash.runtime.engine_backend import declared_action_dim
        dim, dim_src = declared_action_dim(checkpoint)
        if dim is not None:
            notes["action_dim"] = str(dim)
            notes["action_dim_source"] = dim_src
        else:
            notes["action_dim_unresolved"] = dim_src
        base = dataclasses.replace(base, notes=notes)
        try:
            observation, _source = self.observation_contract(checkpoint)
        except RuntimeError:
            return base            # plan-time tolerance; serve/observation stays fail-loud
        return dataclasses.replace(base, observation=observation)

    #: pi05 needs lerobot and torch. It does NOT need diffusers -- which is what the runtime used to
    #: demand of every model, sending a perfectly hostable VLA to a worker it has no reason to have.
    HOST_REQUIRES = ("torch", "lerobot")

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available
        return imports_available(self.HOST_REQUIRES)

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        """Load the upstream pi05 policy WITH its processor pipeline.

        THE PROCESSOR IS NOT OPTIONAL, and that is the substance of "VLA support". pi05's
        `predict_action_chunk` reads `batch[OBS_LANGUAGE_TOKENS]` and
        `batch[OBS_LANGUAGE_ATTENTION_MASK]` -- already tokenized. Text never reaches the policy. The
        tokenizer, the input normalisation and the action un-normalisation all live in a
        `PolicyProcessorPipeline` published alongside the weights as `policy_preprocessor.json`, so a
        VLA served without it is not slow, it is WRONG: it would be fed unnormalised pixels and would
        return actions in a normalised space nobody can execute.

        This is model semantics, so it belongs here rather than in the runtime. What the runtime sees
        is still one object with `predict` and `reset`.
        """
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        tf32_planned = _validate_tf32_plan(checkpoint, plan)

        repo = (checkpoint.execution.extra or {}).get("base_weights")
        if not repo:
            raise RuntimeError(
                f"{checkpoint.model_id}: no local weights and no execution.base_weights, so there is "
                f"nothing to load. Declare the upstream repo id in base_weights.")
        _require_processor_steps(repo)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from pi05_iwm.precision import Pi05PrecisionLease
        precision = Pi05PrecisionLease(torch, "tf32" if tf32_planned else "fp32")
        try:
            # This must be a CONCRETE config object mutated before PI05Policy constructs any
            # modules.  With LeRobot 0.6.2, PI05Config.from_pretrained(repo, dtype="float32")
            # silently leaves v044's JSON value at bfloat16.  TF32 only accelerates FP32 GEMMs, so
            # accepting that value would publish a NUMERIC plan while running BF16 arithmetic.
            config = PI05Config.from_pretrained(repo)
            if tf32_planned:
                config.dtype = "float32"
                config.compile_model = False
            cap = None
            if str(dev).startswith("cuda") and torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(torch.device(str(dev)))
            # Two independent reasons a published compile_model=true does not stand, each
            # printed with its own cause. Both must run HERE: the flag is consumed by
            # PI05Policy.from_pretrained below, so any decision that needs the plan has to
            # precede construction (the same ordering the T0 plan-gated capture install in
            # lingbot_vla_iwm enforces).
            _neutralize_compile_model_for_planned_capture(config, plan, dev)
            _neutralize_compile_model_on_sm110a(config, cap)
            _apply_compile_permission(config, plan)

            policy = PI05Policy.from_pretrained(repo, config=config)
            policy.eval()
            policy.to(dev)
            if tf32_planned:
                _require_all_floating_parameters_fp32(policy, torch)

            n = dict(nfe or checkpoint.execution.nfe or {})
            if "action" in n and hasattr(policy.config, "num_inference_steps"):
                # the declared flow-matching schedule, applied. Without this a checkpoint declaring
                # nfe {action: 4} would be served at the config's 10 and the plan was priced wrong.
                policy.config.num_inference_steps = int(n["action"])

            # OVERRIDE THE PUBLISHED DEVICE. `lerobot/pi05_base` ships its pipeline with
            # `device_processor: {"device": "cpu"}` -- the publisher's deployment assumption baked
            # into the checkpoint. Left alone it puts language tokens on the CPU while the weights
            # are on cuda:0. Where a model runs is a DEPLOYMENT fact, so the runtime's device wins.
            pre, post = make_pre_post_processors(
                policy.config, pretrained_path=repo,
                preprocessor_overrides={"device_processor": {"device": str(dev)}},
                postprocessor_overrides={"device_processor": {"device": str(dev)}})

            loop = _Pi05Loop(policy, pre, post, dev, precision_lease=precision)
            try:
                from instinctflash.runtime.precision import install_requested_fp8
                install_requested_fp8(policy.model, plan, "pi05")
                Pi05Adapter.install(policy, plan, device=dev)
            except Exception:
                loop.close()
                raise
            return loop
        except Exception:
            precision.close()
            raise

    @staticmethod
    def install(policy, plan, *, device=None) -> list[str]:
        """Act on the plan. Returns the names of what actually got installed.

        The runtime asks the adapter to do this because a plan is a claim, and a claim nobody acts on
        is worse than no claim: pi05 previously ran with a plan reporting an APPLIED pass and nothing
        installed, silently.

        CAPTURE IS THE DEFAULT. When the plan's graph_capture APPLIES and the build device is
        CUDA, the replay-safe static-KV capture (pi05_iwm/static_capture.py) is installed for
        every pi05-class checkpoint — fresh fine-tunes included. It used to key on things fresh
        fine-tunes lack (a published compile_model=true to supersede, or an env opt-in), so a
        checkpoint straight out of lerobot-train served EAGER at ~207 ms/chunk while the same
        weights measured bit-exact at ~73 ms captured. What makes defaulting it safe is not the
        history of measurements on OTHER checkpoints — it is the runtime SELF-CHECK: the first
        capture is compared against upstream eager on staged inputs it was not captured from
        (exact equality, seconds of startup, once per process), and a mismatch releases the
        graphs and falls back to eager loudly while serving continues.

        `IFL_PI05_NO_CAPTURE=1` is the kill-switch (recorded on the plan, printed). The old
        opt-in flags stay recognized as no-ops with a notice.
        """
        import os

        import torch

        from pi05_iwm.surface import Pi05Surface

        wanted = {getattr(r, "name", "") for r in getattr(plan, "results", ())
                  if getattr(r, "applies", False)}
        from pi05_iwm.passes import PASS_NAME
        tf32_numeric = PASS_NAME in wanted
        if "graph_capture" not in wanted:
            if tf32_numeric:
                raise RuntimeError("pi05_tf32_numeric cannot run without graph_capture")
            return []
        if not (device and str(device).startswith("cuda") and torch.cuda.is_available()):
            print("InstinctFlash pi05: graph_capture is planned but needs CUDA; running eager.")
            return []

        capture = next(r for r in plan.results if r.name == "graph_capture" and r.applies)
        if os.environ.get(CAPTURE_KILL_SWITCH) == "1":
            if tf32_numeric:
                # The TF32 operating point is checkpoint-REQUIRED and its manifest declares
                # static_kv_graph: true — serving it eager would be a different execution
                # semantics under the same declaration, the exact substitution this repo refuses.
                raise RuntimeError(
                    f"{CAPTURE_KILL_SWITCH}=1 cannot serve this checkpoint: it declares the "
                    f"TF32 static-KV operating point (execution.pi05_tf32_numeric, "
                    f"static_kv_graph: true), so eager would be a different semantics than the "
                    f"one declared. Unset {CAPTURE_KILL_SWITCH}, or select the FP32 checkpoint.")
            surface = Pi05Surface(policy.model)
            hoisted = surface.hoist_loop_constants()
            note = (f"{CAPTURE_KILL_SWITCH}=1 — the default static-KV capture is disabled by "
                    f"the caller; running eager (upstream's arithmetic exactly)")
            capture.params["decision"] = tuple(capture.params.get("decision", ())) + (note,)
            print(f"InstinctFlash pi05: {note}. {len(hoisted)} bit-exact hoist(s) applied "
                  f"(no measurable win alone).")
            return ["loop_constant_hoist"]

        # The retired opt-ins. Both used to select what is now simply the default, so they
        # change nothing — said out loud rather than silently ignored.
        if os.environ.get(Pi05Surface.STATIC_CAPTURE_OPT_IN) == "1":
            print(f"InstinctFlash pi05: {Pi05Surface.STATIC_CAPTURE_OPT_IN}=1 is a no-op — "
                  f"static-KV capture is the default for pi05-class checkpoints on "
                  f"capture-capable devices now ({CAPTURE_KILL_SWITCH}=1 disables it).")
        if os.environ.get(Pi05Surface.CAPTURE_OPT_IN) == "1":
            print(f"InstinctFlash pi05: {Pi05Surface.CAPTURE_OPT_IN}=1 is a no-op — the "
                  f"DynamicCache capture experiment is retired from install (measured "
                  f"replay-unsafe; the negative result is documented in pi05_iwm/surface.py). "
                  f"The default static-KV capture serves instead.")

        surface = Pi05Surface(policy.model)
        hoisted = surface.hoist_loop_constants()          # BITEXACT, and the prerequisite

        from pi05_iwm.static_capture import install_static_capture

        if tf32_numeric:
            if (torch.get_float32_matmul_precision() != "high"
                    or not torch.backends.cuda.matmul.allow_tf32):
                raise RuntimeError(
                    "pi05_tf32_numeric was planned but its process precision lease is not active"
                )
            install_static_capture(
                policy.model,
                step_tables=True,
                on_self_check=_record_self_check_on_plan(capture),
                full_chunk=False,
                prefix_graph=False,
            )
            for h in hoisted:
                print(f"InstinctFlash pi05: hoisted {h}")
            numeric = next(r for r in plan.results if r.name == PASS_NAME and r.applies)
            delta = numeric.params["max_abs_action_delta"]
            margin = (f"max measured |delta action vs FP32| {float(delta):.3e}"
                      if delta is not None else "no qualified target-device action margin")
            print("InstinctFlash pi05: TF32 + static-KV graph installed; plan tier NUMERIC "
                  f"({margin}). The first capture is "
                  f"gated by the bit-exact self-check (replay vs TF32 eager).")
            return ["loop_constant_hoist", "graph_capture_static_kv", PASS_NAME]

        # The replay-safe path, BY DEFAULT: static max-extent KV buffers, gate numbers in
        # pi05_iwm/static_capture.py and verify_static_capture.py (bitexact on unseen inputs
        # and prompts; 3.55x denoise step, 1.65x chunk on H100/pi05_base; 206.7 -> 72.8 ms
        # on v044). The per-process proof is the runtime self-check wired here.
        compile_superseded = bool(capture.params.get("compile_model_superseded"))
        driver = install_static_capture(
            policy.model, on_self_check=_record_self_check_on_plan(capture),
            **_native_capture_options(torch.cuda.get_device_capability(device)))
        graph_mode = (
            "prefix + full-loop graphs"
            if getattr(driver, "_prefix_graph_enabled", False)
            else "full-loop graph" if getattr(driver, "_full_chunk", False)
            else "per-step graph")
        if (getattr(driver, "_full_chunk", False)
                and getattr(driver, "_step_tables", False)):
            graph_mode += " with baked time/AdaRMS tables"
        for h in hoisted:
            print(f"InstinctFlash pi05: hoisted {h}")
        because = (" — installed in place of the checkpoint's neutralized compile_model"
                   if compile_superseded else
                   " — the pi05-family default on capture-capable devices")
        print(f"InstinctFlash pi05: static-KV {graph_mode} capture installed{because}. The first "
              f"capture is gated by a bit-exact self-check (replay vs eager on staged inputs, "
              f"exact equality); a mismatch releases the graphs and falls back to eager, "
              f"loudly. Kill-switch: {CAPTURE_KILL_SWITCH}=1.")
        return ["loop_constant_hoist", "graph_capture_static_kv"]


class _Pi05Loop:
    """One control cycle over pi05. No commit phase: the prefix is rebuilt every cycle."""

    def __init__(self, policy, pre, post, device, precision_lease=None):
        import torch
        self._torch, self._p, self._pre, self._post, self._dev = torch, policy, pre, post, device
        self._prompt = ""
        self._precision_lease = precision_lease

    @property
    def graph_stats(self):
        driver = getattr(self._p.model, "_ifl_static_denoiser", None)
        return {
            "captured": bool(driver and not driver.rejected and
                             (getattr(driver, "_chunk_graph", None) is not None or
                              getattr(driver, "_graph", None) is not None)),
            "rejected": bool(driver and driver.rejected),
            "full_graph": bool(driver and driver._full_chunk),
            "prefix_graph": bool(driver and driver._prefix_graph_enabled),
            "replays": int(getattr(driver, "replays", 0)),
            "prefix_replays": int(getattr(driver, "prefix_replays", 0)),
            "chunk_replays": int(getattr(driver, "chunk_replays", 0)),
            "self_check": getattr(driver, "self_check", None),
        }

    def reset(self, **conditioning) -> None:
        self._prompt = str(conditioning.get("prompt") or "")
        self._p.reset()                       # drops the buffered 50-step action chunk

    def predict(self, observation):
        # TENSORS, not arrays. pi05's processor does `state.cpu().numpy()` (processor_pi05.py:67), so a
        # numpy observation dies inside the pipeline with `'numpy.ndarray' has no attribute 'cpu'`.
        # ObservationSpec declares shapes and dtypes, not a tensor library, and converting is the
        # adapter's job -- the caller should be able to hand over whatever a camera produced.
        batch = {k: self._as_tensor(v) for k, v in observation.items()
                 if k.startswith("observation.")}
        # LeRobot names the instruction `task`; the declaration calls it `prompt`. Mapping one to the
        # other is exactly the adapter's job -- the runtime must not learn either name.
        batch["task"] = str(observation.get("prompt") or self._prompt)
        with self._torch.no_grad():
            action = self._post(self._p.select_action(self._pre(batch)))
        a = action if self._torch.is_tensor(action) else self._torch.as_tensor(action)
        return {"action": a.squeeze(0).detach().cpu().numpy()}

    def _as_tensor(self, v):
        t = v if self._torch.is_tensor(v) else self._torch.as_tensor(v)
        if t.dtype not in (self._torch.float32, self._torch.uint8):
            t = t.float()
        return t.to(self._dev)

    def close(self) -> None:
        self._p = None
        lease, self._precision_lease = self._precision_lease, None
        if lease is not None:
            lease.close()


def _record_self_check_on_plan(capture):
    """The self-check verdict, put where a reader will look: the plan's graph_capture entry.

    `plan.explain()` / `runtime.explain()` render params['decision'] lines, and the plan object
    is the same one the facade holds — so the line the serve log prints at first capture is the
    line every later explain() shows. The full verdict (per-input deltas, the startup cost)
    rides on params['self_check'] for programmatic readers.
    """
    def on_result(res: dict) -> None:
        if res["bitexact"]:
            refilled = sum(1 for c in res["cases"] if c["prefix"] == "refilled")
            line = (f"self-check bit-exact on {res['n']} inputs (replay == eager exactly, "
                    f"{refilled} on a refilled prefix; {res['seconds']:.1f} s startup cost, "
                    f"once per process)")
        else:
            line = (f"self-check FAILED — replay differs from eager by "
                    f"{res['max_abs_delta']:.3e} on staged inputs it was not captured from; "
                    f"graphs released, running eager (upstream's arithmetic exactly), "
                    f"serve continues")
        capture.params["decision"] = tuple(capture.params.get("decision", ())) + (
            f"graph_capture: {line}",)
        capture.params["self_check"] = res
        # stderr, deliberately: the verdict lands at FIRST CAPTURE, i.e. during serving, and
        # cli_config.execute defers stdout until the command returns — which for a persistent
        # `instinctflash serve` is never. The server's live log stream is stderr.
        import sys
        print(f"InstinctFlash pi05: graph_capture {line}.", file=sys.stderr, flush=True)
    return on_result


def _native_capture_options(capability):
    """Measured native defaults; explicit graph switches remain independent opt-outs."""
    import os
    qualified = tuple(capability) in {(9, 0), (11, 0)}
    full_env = os.environ.get("IFL_PI05_FULL_CHUNK_GRAPH")
    prefix_env = os.environ.get("IFL_PI05_PREFIX_GRAPH")
    prefix = (prefix_env == "1" if prefix_env is not None
              else qualified and full_env != "0")
    full = (full_env == "1" if full_env is not None else qualified or prefix)
    tables = (True if full and qualified and "IFL_PI05_FULL_STEP_TABLES" not in os.environ
              else None)
    return {"full_chunk": full, "prefix_graph": prefix, "step_tables": tables}


def _apply_compile_permission(config, plan):
    """A publisher's compile flag cannot bypass the caller's arithmetic ceiling."""
    if not getattr(config, "compile_model", False):
        return
    from instinctflash.planners.planner import Tier, PassResult
    if getattr(plan, "tier_ceiling", Tier.BITEXACT) < Tier.NUMERIC:
        config.compile_model = False
        print("InstinctFlash pi05: compile_model neutralized by BITEXACT policy; "
              "upstream eager arithmetic retained.")
    else:
        plan.results.append(PassResult(
            "pi05_torch_compile", True, Tier.NUMERIC,
            "Checkpoint compile_model retained with explicit NUMERIC permission"))


def _neutralize_compile_model_on_sm110a(config, device_capability) -> bool:
    """Turn off a checkpoint-published ``compile_model: true`` on sm_110a, saying why.

    ``lerobot/pi05_libero_finetuned_v044`` publishes ``compile_model: true`` in its config, and
    on Thor (sm_110a) ``torch.compile`` dies inside triton with ``ptxas-blackwell: 'sm_110a' is
    not defined`` — triton simply cannot emit for this arch (established; the previous
    workaround was ``TORCH_COMPILE_DISABLE=1`` in the caller's environment, which nobody who
    just loads the checkpoint knows to set). ``compile_model`` is a publisher's deployment
    assumption, not model semantics: the eager module computes the same function, so
    neutralizing the flag changes nothing but the crash. Printed, never silent — the caller
    must be able to see their config key did not take effect and why.

    Returns True when the key was neutralized.
    """
    if not getattr(config, "compile_model", False):
        return False
    if tuple(device_capability or ()) != (11, 0):
        return False
    config.compile_model = False
    print("InstinctFlash pi05: the checkpoint publishes compile_model=true, but torch.compile "
          "is dead on this device (sm_110a: triton fails with \"ptxas-blackwell: 'sm_110a' is "
          "not defined\"). compile_model neutralized — running the eager module, which computes "
          "the same function without the crash.")
    return True


#: Why a planned graph_capture outranks a checkpoint-published ``compile_model: true``. Every
#: number is measured on the same machine and checkpoint (H100, v044, full action chunk): the
#: static-KV captured chunk runs 72.8 ms against torch.compile max-autotune's 173.3 ms, capture
#: replay is BITEXACT against eager on inputs it never saw while inductor's numerics carry no
#: equivalence evidence at all, and capture warms up in seconds where the compiled first start
#: burned 171 s+ of autotune (a 738 s cold `serve --serve.smoke=true`, eight-family UX pass).
COMPILE_SUPERSEDED_REASON = ("superseded by graph_capture: bit-exact, faster "
                             "(72.8 vs 173.3 ms measured), no compile wait")


def _neutralize_compile_model_for_planned_capture(config, plan, device) -> bool:
    """Turn off a checkpoint-published ``compile_model: true`` when this plan captures instead.

    The SECOND, independent neutralization rule. The sm_110a one above is about a device where
    torch.compile crashes; this one is about a plan that already holds something strictly
    better. ``compile_model`` is a publisher's deployment assumption, and when the plan's
    ``graph_capture`` pass APPLIES the runtime's own static-KV capture serves the same denoise
    loop for less on every axis that assumption is about — see COMPILE_SUPERSEDED_REASON for
    the measurements. Honoring the flag on top of capture would buy nothing and cost the
    compile wait, so the flag is neutralized and ``install`` takes the verified static-capture
    path: the promise this print makes is kept by the same plan object carrying
    ``compile_model_superseded`` down to the installer.

    Scope, deliberately narrow:

      * the plan's ``graph_capture`` must APPLY — a device where the planner declines it (CPU,
        or the measured bandwidth-bound-edge class) keeps the publisher's key untouched, so
        the checkpoint author's choice stands everywhere our capture has no case;
      * the build device must be CUDA — an APPLICABILITY-UNCHECKED plan built onto a CPU
        cannot install capture, and neutralizing compile_model there would replace the
        author's choice with nothing.

    Printed, never silent, and recorded as a Decision on the plan's ``graph_capture`` entry so
    ``plan.explain()`` / ``runtime.explain()`` show it. Returns True when the key was
    neutralized.
    """
    if not getattr(config, "compile_model", False):
        return False
    if not str(device or "").startswith("cuda"):
        return False
    capture = next((r for r in getattr(plan, "results", ())
                    if getattr(r, "name", "") == "graph_capture"
                    and getattr(r, "applies", False)), None)
    if capture is None:
        return False
    config.compile_model = False
    capture.params["compile_model_superseded"] = True
    # PassResult is frozen but params is the runtime-facing dict by contract; explain()
    # surfaces this as a 'decision:' line (the conv-layout autotune precedent).
    capture.params["decision"] = tuple(capture.params.get("decision", ())) + (
        f"checkpoint publishes compile_model=true — neutralized: {COMPILE_SUPERSEDED_REASON}",)
    print(f"InstinctFlash pi05: the checkpoint publishes compile_model=true — "
          f"{COMPILE_SUPERSEDED_REASON}. compile_model neutralized; the plan's static-KV "
          f"capture serves the denoise loop instead.")
    return True


def _declares_tf32_operating_point(checkpoint) -> bool:
    """Validate the manifest value, not merely the capability token derived from its key."""
    op = (checkpoint.execution.extra or {}).get("pi05_tf32_numeric")
    if op is None:
        return False
    if not isinstance(op, dict):
        raise RuntimeError("execution.pi05_tf32_numeric must be an object")
    required = {
        "enabled": True,
        "tier": "NUMERIC",
        "math_mode": "tf32",
        "parameter_dtype": "float32",
        "compile_model": False,
        "static_kv_graph": True,
    }
    if op.get("hardware") not in {"sm90", "sm110"}:
        raise RuntimeError("pi05_tf32_numeric hardware must be sm90 or sm110")
    if op["hardware"] == "sm110" and (op.get("qualification") != "unqualified"
                                        or op.get("max_abs_action_delta") is not None):
        raise RuntimeError("Thor TF32 requires explicit unqualified status and no inherited action margin")
    wrong = {k: (op.get(k), want) for k, want in required.items() if op.get(k) != want}
    if wrong:
        raise RuntimeError(f"invalid pi05_tf32_numeric declaration fields: {wrong}")
    return True


def _require_all_floating_parameters_fp32(policy, torch_module) -> None:
    """Hard gate: a declared TF32 operating point may not actually load BF16/FP16 modules."""
    floating = []
    wrong = []
    for name, parameter in policy.named_parameters():
        if not parameter.is_floating_point():
            continue
        floating.append(name)
        if parameter.dtype != torch_module.float32:
            wrong.append((name, str(parameter.dtype)))
    if not floating:
        raise RuntimeError("pi05 TF32 operating point loaded no floating-point parameters")
    if wrong:
        preview = ", ".join(f"{name}={dtype}" for name, dtype in wrong[:5])
        more = f" (+{len(wrong) - 5} more)" if len(wrong) > 5 else ""
        raise RuntimeError(
            "pi05 TF32 operating point requires every floating parameter to be torch.float32; "
            f"found {len(wrong)}/{len(floating)} with another dtype: {preview}{more}"
        )


def _validate_tf32_plan(checkpoint, plan) -> bool:
    """Return TF32 mode, refusing any declaration/plan mismatch before weights load."""
    tf32_declared = _declares_tf32_operating_point(checkpoint)
    applied = {r.name for r in getattr(plan, "results", ()) if r.applies}
    from pi05_iwm.passes import PASS_NAME
    tf32_planned = PASS_NAME in applied
    if tf32_declared != tf32_planned:
        detail = next(
            (r.reason for r in getattr(plan, "results", ()) if r.name == PASS_NAME),
            "pass was not evaluated",
        )
        raise RuntimeError(
            "pi0.5 TF32 operating-point declaration and plan disagree: "
            f"declared={tf32_declared}, applied={tf32_planned}. {detail}. "
            "The runtime refuses to silently substitute FP32 or TF32 semantics."
        )
    if tf32_planned and "graph_capture" not in applied:
        raise RuntimeError(
            "pi05_tf32_numeric requires the measured static-KV graph stack, but "
            "graph_capture is not applied (possibly excluded by the caller)."
        )
    if tf32_planned:
        result = next(r for r in plan.results if r.name == PASS_NAME and r.applies)
        op = checkpoint.execution.extra["pi05_tf32_numeric"]
        for field in ("max_abs_action_delta", "evidence", "hardware"):
            if op.get(field) != result.params.get(field):
                raise RuntimeError(
                    f"pi0.5 TF32 manifest {field}={op.get(field)!r} disagrees with the "
                    f"checkpoint-specific planner evidence {result.params.get(field)!r}"
                )
    return tf32_planned


def _require_processor_steps(repo: str) -> None:
    """Refuse early if this LeRobot cannot build the checkpoint's processor pipeline.

    The real precondition, found by hitting three different walls in order. `lerobot 0.4.4` raises
    `ValueError: An incorrect transformer version is used` from a pi05 assert on
    `transformers.__version__ == "4.53.2"`, which names neither the module nor the fix. Patch that and
    the next wall is the pipeline: `lerobot/pi05_base` declares a `relative_actions_processor` step
    that 0.4.4's registry does not have, because the checkpoint was published by a newer LeRobot.
    `lerobot 0.6.1` has the step AND has dropped the transformers assert, so the version check was
    never the real requirement -- the processor registry is.

    Checking the registry against the checkpoint's own step list says which of those two worlds you are
    in, before 14.5 GB of weights load.
    """
    import json

    from lerobot.processor import ProcessorStepRegistry

    # NOT a bare `except: return`. It was, and it silently swallowed a NameError from a missing
    # `Path` import -- so the whole precondition reported "fine" while checking nothing, and the run
    # still died on the gated tokenizer after loading the weights. A check that cannot run must say so.
    try:
        # `repo` is a local fine-tune directory at least as often as it is a Hub id, and
        # hf_hub_download refuses a path outright (HFValidationError) — which used to demote this
        # whole precondition to "unverified" for exactly the local-serve flow that needs it most:
        # the gated-tokenizer wall would fire only after the full weights had loaded.
        local = Path(repo) / "policy_preprocessor.json"
        if local.is_file():
            cfg = json.loads(local.read_text())
        else:
            from huggingface_hub import hf_hub_download
            cfg = json.loads(Path(hf_hub_download(repo, "policy_preprocessor.json")).read_text())
    except Exception as e:                                        # noqa: BLE001
        print(f"instinctflash: cannot inspect {repo}'s processor pipeline "
              f"({type(e).__name__}: {e}); preconditions unverified, the loader will report any "
              f"failure itself.")
        return
    want = [st.get("registry_name") for st in (cfg.get("steps") or []) if st.get("registry_name")]
    # THE TOKENIZER IS A SEPARATE GATE, and it used to fire after 14.5 GB had loaded. pi05's
    # tokenizer_processor pulls `google/paligemma-3b-pt-224`, which is a GATED repo: without an
    # accepted licence it is a 401, and the pipeline raised only once the policy was already resident.
    # Checking reachability first costs one HTTP request and saves a multi-minute load that cannot
    # succeed.
    for st in (cfg.get("steps") or []):
        name = (st.get("config") or {}).get("tokenizer_name")
        if not name:
            continue
        try:
            # Probe a real FILE FETCH, not `model_info`. A gated repo answers model_info with public
            # metadata and then refuses the download, so metadata reachability was the wrong probe:
            # the check passed and the pipeline still died on a 401 -- after the weights had loaded.
            from huggingface_hub import hf_hub_download
            hf_hub_download(name, "config.json")
        except Exception as e:                                    # noqa: BLE001
            raise RuntimeError(
                f"{repo}'s processor pipeline tokenizes with {name!r}, which is not reachable from "
                f"this machine: {type(e).__name__}.\n\n"
                f"That repo is gated. Accept its licence at https://huggingface.co/{name} with the "
                f"account whose token is configured here, then `hf auth login`. There is no substitute "
                f"-- swapping in a different tokenizer would change how the instruction is encoded and "
                f"silently produce a different policy.\n"
                f"`instinctflash describe` and `instinctflash plan` need neither the tokenizer nor the "
                f"weights.") from None

    try:
        have = set(ProcessorStepRegistry._registry)
    except Exception as e:                                        # noqa: BLE001
        print(f"instinctflash: cannot read LeRobot's processor registry ({type(e).__name__}); "
              f"step availability unverified.")
        return
    missing = [w for w in want if w not in have]
    if missing:
        import lerobot
        raise RuntimeError(
            f"{repo} declares processor steps this LeRobot cannot build: {missing}.\n"
            f"Installed lerobot is {lerobot.__version__}; the checkpoint was published by a newer one. "
            f"pi05 needs a LeRobot whose registry has those steps -- 0.6.1 does, and it also dropped "
            f"the `transformers == 4.53.2` assert that 0.4.4 fails on, so upgrading LeRobot fixes both "
            f"walls at once. Note 0.5+ requires Python >= 3.12.\n"
            f"`instinctflash describe` and `instinctflash plan` need none of this.")
