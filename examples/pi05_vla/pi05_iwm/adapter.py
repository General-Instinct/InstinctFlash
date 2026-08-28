"""InstinctFlash adapter for LeRobot's pi05 VLA family. External plugin: no core changes.

Written to test whether InstinctFlash's abstraction is real, using a model family deliberately unlike
LingBot-VA. Facts below come from lerobot/pi05_base's own config.json, not from guesses.
"""
from __future__ import annotations

from pathlib import Path

from instinctflash import (AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec, PurityKey)
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "pi05"


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
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        repo = (checkpoint.execution.extra or {}).get("base_weights")
        if not repo:
            raise RuntimeError(
                f"{checkpoint.model_id}: no local weights and no execution.base_weights, so there is "
                f"nothing to load. Declare the upstream repo id in base_weights.")
        _require_processor_steps(repo)
        policy = PI05Policy.from_pretrained(repo)
        policy.eval()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        installed, silently. The generic `GraphCapture` pass does the work -- see `surface.py` for why
        pi05 can be captured when LingBot-VA cannot, and for the one constant that blocked it.
        """
        import torch

        from instinctflash.passes.graph_capture import GraphCapture
        from instinctflash.passes.interface import run_pass

        from pi05_iwm.surface import Pi05Surface

        wanted = {getattr(r, "name", "") for r in getattr(plan, "results", ())
                  if getattr(r, "applies", False)}
        if "graph_capture" not in wanted:
            return []
        if not (device and str(device).startswith("cuda") and torch.cuda.is_available()):
            print("InstinctFlash pi05: graph_capture is planned but needs CUDA; running eager.")
            return []

        surface = Pi05Surface(policy.model)
        hoisted = surface.hoist_loop_constants()          # BITEXACT, and the prerequisite

        import os
        if os.environ.get(Pi05Surface.STATIC_CAPTURE_OPT_IN) == "1":
            # The replay-safe path: static max-extent KV buffers, gate numbers in
            # pi05_iwm/static_capture.py and verify_static_capture.py (bitexact on unseen
            # inputs and prompts; 3.55x denoise step, 1.65x chunk on H100/pi05_base).
            from pi05_iwm.static_capture import install_static_capture
            install_static_capture(policy.model)
            for h in hoisted:
                print(f"InstinctFlash pi05: hoisted {h}")
            print("InstinctFlash pi05: static-KV graph capture installed (bitexact-verified path).")
            return ["loop_constant_hoist", "graph_capture_static_kv"]

        result = run_pass(GraphCapture(), surface, device=torch.device(str(device)))
        if getattr(result, "skipped_reason", None) or not surface.install():
            # Declining is the EXPECTED outcome here, not a failure. pi05's denoise region is not
            # replay-safe -- measured, see surface.py -- so the site declares capturable=False and the
            # generic pass refuses it. The hoists are bit-exact and stay; they buy nothing on their
            # own, and saying so beats implying the run was optimized.
            print(f"InstinctFlash pi05: graph_capture declined "
                  f"({getattr(result, 'skipped_reason', 'no rewrite applied')}). Running eager, which "
                  f"is upstream's arithmetic exactly. {len(hoisted)} bit-exact hoist(s) applied "
                  f"(no measurable win alone).")
            return ["loop_constant_hoist"]
        for h in hoisted:
            print(f"InstinctFlash pi05: hoisted {h}")
        print("InstinctFlash pi05: graph_capture installed on the denoise step. NOTE: this path is "
              "opt-in and not equivalence-verified -- see pi05_iwm/surface.py.")
        return ["loop_constant_hoist", "graph_capture"]


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
