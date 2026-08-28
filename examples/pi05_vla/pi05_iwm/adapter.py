"""InstinctFlash adapter for LeRobot's pi05 VLA family. External plugin: no core changes.

Written to test whether InstinctFlash's abstraction is real, using a model family deliberately unlike
LingBot-VA. Facts below come from lerobot/pi05_base's own config.json, not from guesses.
"""
from __future__ import annotations

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
            notes={"family": "vla", "chunk_size": "50", "n_obs_steps": "1"},
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

        repo = (checkpoint.execution.extra or {}).get("base_weights")
        if not repo:
            raise RuntimeError(
                f"{checkpoint.model_id}: no local weights and no execution.base_weights, so there is "
                f"nothing to load. Declare the upstream repo id in base_weights.")
        _require_processor_steps(repo)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # A CONCRETE config object, mutated before PI05Policy constructs any modules: the
        # compile_model flag is consumed inside the model __init__ (lerobot 0.6.1
        # modeling_pi05.py wraps sample_actions/forward in torch.compile there — and also flips
        # torch.set_float32_matmul_precision("high") process-wide before the wrappers), so any
        # decision that needs the plan has to precede construction.
        config = PI05Config.from_pretrained(repo)
        _neutralize_compile_model_for_planned_capture(config, plan, dev)
        policy = PI05Policy.from_pretrained(repo, config=config)
        policy.eval()
        policy.to(dev)

        n = dict(nfe or checkpoint.execution.nfe or {})
        if "action" in n and hasattr(policy.config, "num_inference_steps"):
            # the declared flow-matching schedule, applied. Without this a checkpoint declaring
            # nfe {action: 4} would be served at the config's 10 and the plan would be priced wrong.
            policy.config.num_inference_steps = int(n["action"])

        # OVERRIDE THE PUBLISHED DEVICE. `lerobot/pi05_base` ships its pipeline with
        # `device_processor: {"device": "cpu"}` -- the publisher's deployment assumption baked into the
        # checkpoint. Left alone it puts language tokens on the CPU while the weights are on cuda:0,
        # and the run dies inside a Gemma embedding with "Expected all tensors to be on the same
        # device". Where a model runs is a DEPLOYMENT fact, so the runtime's device wins over anything
        # a checkpoint asserts about it.
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=repo,
            preprocessor_overrides={"device_processor": {"device": str(dev)}},
            postprocessor_overrides={"device_processor": {"device": str(dev)}})

        loop = _Pi05Loop(policy, pre, post, dev)
        Pi05Adapter.install(policy, plan, device=dev)
        return loop

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
        if "graph_capture" not in wanted:
            return []
        if not (device and str(device).startswith("cuda") and torch.cuda.is_available()):
            print("InstinctFlash pi05: graph_capture is planned but needs CUDA; running eager.")
            return []

        capture = next(r for r in plan.results if r.name == "graph_capture" and r.applies)
        if os.environ.get(CAPTURE_KILL_SWITCH) == "1":
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

        # The replay-safe path, BY DEFAULT: static max-extent KV buffers, gate numbers in
        # pi05_iwm/static_capture.py and verify_static_capture.py (bitexact on unseen inputs
        # and prompts; 3.55x denoise step, 1.65x chunk on H100/pi05_base; 206.7 -> 72.8 ms
        # on v044). The per-process proof is the runtime self-check wired here.
        from pi05_iwm.static_capture import install_static_capture
        compile_superseded = bool(capture.params.get("compile_model_superseded"))
        install_static_capture(policy.model,
                               on_self_check=_record_self_check_on_plan(capture))
        for h in hoisted:
            print(f"InstinctFlash pi05: hoisted {h}")
        because = (" — installed in place of the checkpoint's neutralized compile_model"
                   if compile_superseded else
                   " — the pi05-family default on capture-capable devices")
        print(f"InstinctFlash pi05: static-KV graph capture installed{because}. The first "
              f"capture is gated by a bit-exact self-check (replay vs eager on staged inputs, "
              f"exact equality); a mismatch releases the graphs and falls back to eager, "
              f"loudly. Kill-switch: {CAPTURE_KILL_SWITCH}=1.")
        return ["loop_constant_hoist", "graph_capture_static_kv"]


class _Pi05Loop:
    """One control cycle over pi05. No commit phase: the prefix is rebuilt every cycle."""

    def __init__(self, policy, pre, post, device):
        import torch
        self._torch, self._p, self._pre, self._post, self._dev = torch, policy, pre, post, device
        self._prompt = ""

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

    ``compile_model`` is a publisher's deployment assumption, and when the plan's
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
    # surfaces this as a 'decision:' line.
    capture.params["decision"] = tuple(capture.params.get("decision", ())) + (
        f"checkpoint publishes compile_model=true — neutralized: {COMPILE_SUPERSEDED_REASON}",)
    print(f"InstinctFlash pi05: the checkpoint publishes compile_model=true — "
          f"{COMPILE_SUPERSEDED_REASON}. compile_model neutralized; the plan's static-KV "
          f"capture serves the denoise loop instead.")
    return True


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
