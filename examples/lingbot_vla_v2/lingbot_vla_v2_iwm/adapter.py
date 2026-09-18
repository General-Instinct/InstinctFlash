"""Runtime adapter for ``robbyant/lingbot-vla-v2-6b-robotwin``.

The adapter preserves the upstream deploy server's preprocessing and action un-normalisation
contract.  Its default GPU preprocessing path keeps the upstream resize byte-for-byte and moves
the unchanged Qwen normalization/patchification work to CUDA.  Model execution uses dedicated
LingBot kernels and the replay-safe denoise executor installed by :mod:`static_capture`.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path

from instinctflash import AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec, PurityKey
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "lingbot_vla_v2"
MODEL_ID = "robbyant/lingbot-vla-v2-6b-robotwin"
#: The kill-switch for the family's DEFAULT capture arm (denoise static-KV graph AND the
#: vision/prefill graphs). Family-scoped, the IFL_PI05_NO_CAPTURE convention. Honored by
#: `install`, recorded on the plan, printed.
CAPTURE_KILL_SWITCH = "IFL_VLA2_NO_CAPTURE"
#: Where the upstream checkout is looked for when LINGBOT_VLA_V2_ROOT is unset. There is no
#: universal default — the env var is the contract; these are the documented conventions.
SOURCE_ROOT_CANDIDATES = (
    Path.home() / "lingbot-vla-v2-repo",
    Path.home() / "lingbot-vla-v2",
)
_CWD_LOCK = threading.RLock()


class LingBotVLAV2Adapter:
    """Three-camera RobotWin VLA: one Qwen3-VL prefill and ten action-flow steps."""

    HOST_REQUIRES = (
        "torch", "torchvision", "transformers", "safetensors", "yaml", "lerobot",
    )
    # The Qwen3 serving path does not import qwen_vl_utils, and the qualified
    # Thor checkout uses SDPA. A blanket flash_attn requirement incorrectly sends
    # that working native stack to a worker. Upstream still validates whichever
    # attention implementation its actual model constructor selects.

    def spec(self) -> AdapterSpec:
        # At the published 256x256 processor size each image contributes 64 visual tokens plus
        # Qwen's two vision-boundary tokens: 3*66.  The checkpoint pads language to 72 and appends
        # two 8-token task-query groups, giving a fixed prefix extent of 286.
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=25_503_630_044,
            streams=(KVStreamSpec("prefix", tokens_per_frame=286, lifetime=KVLifetime.CHUNK),),
            phases=(
                PhaseSpec("prefix", nfe=1, writes=frozenset({"prefix"})),
                PhaseSpec("action", nfe=10, reads=frozenset({"prefix"}), truncatable=True,
                          min_nfe=1, depends_on=("prefix",)),
            ),
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            purity=(PurityKey("prefix_kv", ("images", "state", "prompt"), KVLifetime.CHUNK,
                              already_hoisted=True),),
            observation=ObservationSpec(
                fields=(
                    ObservationField("observation.images.cam_high", (480, 640, 3), "uint8"),
                    ObservationField("observation.images.cam_left_wrist", (480, 640, 3), "uint8"),
                    ObservationField("observation.images.cam_right_wrist", (480, 640, 3), "uint8"),
                    ObservationField("observation.state", (14,), "float32"),
                ),
                history=1,
                batched=False,
                conditioning=("prompt",),
            ),
            notes={
                "family": "vla",
                # "backbone" identifies this spec to backbone-keyed planner checks (the engine
                # operating-point gate in passes/generic/engine_offload.py keys on it: the
                # vla2_thor frontend bakes a 10-step schedule, and a plan at any other nfe
                # must decline the engine rather than print one schedule and run another).
                "backbone": BACKBONE,
                "chunk_size": "50",
                "numeric_tier": "NUMERIC (upstream fused-MoE is nondeterministic)",
                # The capture gate accepts nonzero deltas; the planner must enforce that tier.
                "capture_tier": "NUMERIC",
            },
        )

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        ok, reason = imports_available(self.HOST_REQUIRES)
        if not ok:
            return ok, reason
        root = _source_root(required=False)
        if root is None or not (root / "deploy" / "lingbot_vla_v2_policy.py").exists():
            return False, (f"LingBot-VLA-V2 source not found "
                           f"({'at ' + str(root) if root else 'no candidate exists'}); "
                           f"set LINGBOT_VLA_V2_ROOT to the upstream checkout")
        return True, f"the model stack imports and LingBot-VLA-V2 source is at {root}"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-VLA-V2 inference requires CUDA")
        dev = str(device or "cuda")
        if not dev.startswith("cuda"):
            raise RuntimeError(f"LingBot-VLA-V2 only supports a CUDA device, got {dev!r}")
        if ":" in dev:
            torch.cuda.set_device(int(dev.rsplit(":", 1)[1]))
        toolchain = _configure_thor_ptxas(torch.cuda.get_device_capability())

        root = _source_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        extra = dict(checkpoint.execution.extra or {})
        qwen = os.environ.get("QWEN3VL_PATH") or str(
            extra.get("tokenizer_repo") or "Qwen/Qwen3-VL-4B-Instruct"
        )
        os.environ["QWEN3VL_PATH"] = qwen
        model_path = _resolve_model_path(checkpoint)

        mode = os.environ.get("IFL_VLA2_BACKEND", "static").strip().lower()
        if mode not in {"static", "compile", "eager"}:
            raise RuntimeError("IFL_VLA2_BACKEND must be one of: static, compile, eager")

        if mode == "compile":
            from instinctflash.runtime.precision import require_transform_permission
            from instinctflash.planners.planner import Tier, PassResult
            require_transform_permission(plan, Tier.NUMERIC, "IFL_VLA2_BACKEND=compile")
            if any(r.name == "graph_capture" and getattr(r, "excluded", False)
                   for r in plan.results):
                raise ValueError("compile conflicts with excluded graph_capture")
            plan.results.append(PassResult("vla2_torch_compile", True, Tier.NUMERIC,
                                           "Explicit torch.compile backend; numerical changes permitted"))

        # transformers 4.57.3 _patch_mistral_regex probes the Hub API even in offline mode when
        # the Qwen3-VL tokenizer loads; the tokenizer is not a mistral model, skip it. Same
        # workaround as the GR00T adapter and the N1.7 verify scripts — the adapter must load
        # from a warm cache with HF_HUB_OFFLINE=1.
        try:
            import transformers.tokenization_utils_base as tub

            def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
                return tokenizer

            tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)
        except Exception:  # noqa: BLE001 - a transformers without the probe needs no patch
            pass

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        server = LingbotVLAv2Server(
            str(model_path), use_length=50, chunk_ret=True, use_bf16=True, use_fp32=False,
            use_compile=(mode == "compile"),
        )
        server._instinctflash_ptxas = toolchain
        schedule = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        server.vla.model.config.num_steps = int(schedule.get("action", 10))

        from .sm120_native import NativeSDPA, selected
        if selected(plan, torch.cuda.get_device_capability(), mode,
                    enabled=_env_flag("IFL_VLA2_SM120_SDPA", default=False)):
            if os.environ.get(CAPTURE_KILL_SWITCH) == "1":
                raise ValueError("SM120 native SDPA conflicts with disabled graph capture")
            server._instinctflash_native_sdpa = NativeSDPA(server.vla.model)

        from instinctflash.runtime.precision import install_requested_fp8
        install_requested_fp8(server.vla.model, plan, "lingbot_vla_v2")
        driver = self.install(server, plan, mode=mode, device=dev)

        robot = str(extra.get("robot") or "robotwin")
        return _LingBotVLAV2Loop(server, root, robot=robot, driver=driver)

    def install(self, server, plan, *, mode="static", device=None):
        """Install the executor selected for this load, honoring the compiled plan.

        This public hook also tells ``InProcessBackend`` that the adapter acts on applicable plan
        results.  The plan is READ, not decorative: the CUDA-graph executors install only when the
        plan applies ``graph_capture`` (mirroring the GR00T adapter) — a plan whose capture pass
        declined, or was excluded by the caller, must not be optimized around anyway.
        """
        if mode == "static":
            wanted = {
                getattr(result, "name", "")
                for result in getattr(plan, "results", ())
                if getattr(result, "applies", False)
            }
            capture_planned = "graph_capture" in wanted
            from instinctflash.planners.planner import Tier
            if capture_planned:
                capture = next(r for r in plan.results if r.name == "graph_capture" and r.applies)
                if capture.tier < Tier.NUMERIC:
                    raise ValueError("LingBot-VLA-V2 capture requires a NUMERIC plan: its self-check accepts nonzero deltas")
            extras = (_env_flag("IFL_VLA2_CUDA_KERNELS", default=False)
                      or _env_flag("IFL_VLA2_MOE_KERNEL", default=False)
                      or _env_flag("IFL_VLA2_RMSNORM_KERNEL", default=False))
            if extras and (not capture_planned or os.environ.get(CAPTURE_KILL_SWITCH) == "1"):
                raise ValueError("experimental numeric kernels require an active NUMERIC capture plan")
            if not capture_planned:
                print(
                    "InstinctFlash LingBot-VLA-V2: the plan does not apply graph_capture, so the "
                    "static-KV CUDA Graph backend is not installed; running the upstream path."
                )
            # The Triton kernels DEFAULT OFF, now with the H100 6-case gate on record
            # (verify_moe_kernel.py / moe_kernel_results.json): the MoE kernel PASSES the
            # null-control envelope (3.84e-2 vs 5.08e-2) with self-consistency 0.0 — the
            # deterministic reduction verified — but measured ~2% slower than vendor robby_moe
            # on the eager path and is ungated under capture; the fused RMSNorm FAILED the
            # envelope (6.10e-2) and is NOT RECOMMENDED until it passes.
            all_kernels = _env_flag("IFL_VLA2_CUDA_KERNELS", default=False)
            if _env_flag("IFL_VLA2_MOE_KERNEL", default=all_kernels) \
                    and _triton_kernels_allowed("IFL_VLA2_MOE_KERNEL"):
                from .moe_kernel import install_lingbot_moe_kernel

                report = install_lingbot_moe_kernel(server.vla.model)
                server._instinctflash_moe_kernel = report
                print(
                    "InstinctFlash LingBot-VLA-V2: sparse-MoE CUDA kernel installed "
                    f"({report.layers} layers active, {report.converted_layers} converted)."
                )
            if _env_flag("IFL_VLA2_RMSNORM_KERNEL", default=all_kernels) \
                    and _triton_kernels_allowed("IFL_VLA2_RMSNORM_KERNEL"):
                from .rmsnorm_kernel import install_lingbot_rmsnorm_kernel

                report = install_lingbot_rmsnorm_kernel(server.vla.model)
                server._instinctflash_rmsnorm_kernel = report
                print(
                    "InstinctFlash LingBot-VLA-V2: fused RMSNorm CUDA kernel installed "
                    f"({report.modules} modules, hidden={report.hidden_size})."
                )
            driver = None
            if capture_planned and os.environ.get(CAPTURE_KILL_SWITCH) == "1":
                capture = next(r for r in plan.results
                               if r.name == "graph_capture" and r.applies)
                note = (f"{CAPTURE_KILL_SWITCH}=1 — the default capture arm (static-KV denoise "
                        f"graph + vision/prefill graphs) is disabled by the caller; running "
                        f"eager (upstream's arithmetic exactly)")
                capture.params["decision"] = tuple(capture.params.get("decision", ())) + (note,)
                print(f"InstinctFlash LingBot-VLA-V2: {note}.")
                capture_planned = False
            if capture_planned:
                if _env_flag("IFL_VLA2_GPU_PREPROCESS", default=True):
                    # FeatureTransform is created by the first server.reset(), which the Runtime loop
                    # performs after installing compute backends. Defer this one transform-dependent
                    # installer until that reset has completed.
                    server._instinctflash_gpu_preprocess_pending = (
                        str(device or "cuda"),
                        os.environ.get("IFL_VLA2_GPU_PREPROCESS_MODE", "processor")
                        .strip()
                        .lower(),
                    )
                from instinctflash.runtime.capture_self_check import record_self_check_on_plan

                from .static_capture import NULL_ENVELOPE, install_static_capture

                capture = next(r for r in plan.results
                               if r.name == "graph_capture" and r.applies)
                from instinctflash.planners.planner import Tier
                if capture.tier < Tier.NUMERIC:
                    raise ValueError("LingBot-VLA-V2 capture requires a NUMERIC plan: its self-check accepts nonzero deltas")
                driver = install_static_capture(
                    server.vla.model,
                    on_self_check=_release_prefix_graphs_on_fail(
                        record_self_check_on_plan(capture, "LingBot-VLA-V2"), server))
                if _env_flag("IFL_VLA2_PREFIX_GRAPH", default=True):
                    from .prefix_capture import install_prefix_capture

                    prefix = install_prefix_capture(server.vla.model)
                    server._instinctflash_prefix_capture = prefix
                    print(
                        "InstinctFlash LingBot-VLA-V2: static vision/prefill CUDA Graph "
                        "backend installed."
                    )
                print("InstinctFlash LingBot-VLA-V2: static-KV CUDA Graph backend installed — "
                      "the NUMERIC arm on eligible devices. The first capture is "
                      "gated by a startup self-check (replay vs upstream eager on staged "
                      "inputs it was not captured from, three comparisons per input) against a legacy "
                      f"cross-domain implementation guard ({NULL_ENVELOPE:.3e} — not a calibrated velocity envelope; the fused-MoE "
                      "kernel is nondeterministic even against itself, so its capture tier is "
                      "NUMERIC, not BITEXACT); a miss releases every graph and falls back to "
                      f"eager, loudly. Kill-switch: {CAPTURE_KILL_SWITCH}=1.")
            return driver
        if mode == "compile":
            print("InstinctFlash LingBot-VLA-V2: using upstream torch.compile backend.")
            return None
        print("InstinctFlash LingBot-VLA-V2: using upstream eager backend.")
        return None


class _LingBotVLAV2Loop:
    def __init__(self, server, source_root: Path, *, robot: str, driver=None):
        self._server = server
        self._root = source_root
        self._robot = robot
        self._driver = driver
        self._prompt = ""
        self.reset(robot=robot)

    def reset(self, **conditioning) -> None:
        self._prompt = str(conditioning.get("prompt") or "")
        self._robot = str(conditioning.get("robot") or self._robot)
        # Upstream resolves robot config and norm stats relative to its project root.  Scope the
        # process-wide cwd change tightly and serialize it; inference itself uses no relative paths.
        with _project_cwd(self._root):
            self._server.reset(self._robot)
        pending = getattr(
            self._server, "_instinctflash_gpu_preprocess_pending", None
        )
        if pending is not None:
            from .image_preprocess import install_lingbot_gpu_image_preprocess

            pending_device, pending_mode = pending
            report = install_lingbot_gpu_image_preprocess(
                self._server, device=pending_device, mode=pending_mode
            )
            self._server._instinctflash_gpu_preprocess = report
            delattr(self._server, "_instinctflash_gpu_preprocess_pending")
            print(
                "InstinctFlash LingBot-VLA-V2: GPU image preprocessing installed "
                f"({report.mode}, {report.camera_count} cameras, "
                f"{report.input_hw}->{report.output_hw})."
            )

    def predict(self, observation):
        obs = dict(observation)
        prompt = str(obs.get("prompt") or obs.get("task") or self._prompt)
        if not prompt:
            raise ValueError("LingBot-VLA-V2 requires a prompt (in reset() or predict())")
        obs["prompt"] = obs["task"] = prompt
        if getattr(self._server, "_instinctflash_fallback_refused", False):
            raise RuntimeError("capture rejected with experimental kernels; reload a native eager Runtime")
        # A rejection can happen midway through a policy call, after its preprocessing and
        # prefix already ran. Discard that mixed call and replay the original observation
        # with its RNG/counter state restored after all capture-side transforms are removed.
        pending = self._driver is not None and not self._driver.rejected and self._driver.graph is None
        if pending:
            import random
            import numpy as np
            import torch
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
            state = {key:getattr(self._server,key) for key in
                     ("global_step","last_action_chunk","last_normalized_action_chunk")
                     if hasattr(self._server,key)}
        result = self._server.infer(obs)
        if pending and self._driver.rejected:
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            if rng[3] is not None:torch.cuda.set_rng_state(rng[3])
            for key,value in state.items():setattr(self._server,key,value)
            result = self._server.infer(dict(observation,prompt=prompt,task=prompt))
        if "action" not in result:
            raise RuntimeError(f"upstream LingBot-VLA-V2 returned no 'action': {result.keys()}")
        return {"action": result["action"]}

    @property
    def graph_stats(self) -> dict:
        d = self._driver
        moe = getattr(self._server, "_instinctflash_moe_kernel", None)
        norm = getattr(self._server, "_instinctflash_rmsnorm_kernel", None)
        preprocess = getattr(self._server, "_instinctflash_gpu_preprocess", None)
        prefix = getattr(self._server, "_instinctflash_prefix_capture", None)
        return {
            "captured": bool(d and d.graph is not None),
            "replays": int(d.replays if d else 0),
            "cuda_kernels": bool(moe or norm),
            "moe_layers": int(moe.layers if moe else 0),
            "rmsnorm_modules": int(norm.modules if norm else 0),
            "gpu_image_preprocess": bool(preprocess),
            "vision_graph": bool(prefix and prefix.vision.graph is not None),
            "vision_replays": int(prefix.vision.replays if prefix else 0),
            "prefill_graph": bool(prefix and prefix.prefill.graph is not None),
            "prefill_replays": int(prefix.prefill.replays if prefix else 0),
            "ptxas": getattr(self._server, "_instinctflash_ptxas", {}),
            "native_sm120": (self._server._instinctflash_native_sdpa.report()
                             if getattr(self._server, "_instinctflash_native_sdpa", None) else None),
        }

    def close(self) -> None:
        prefix = getattr(self._server, "_instinctflash_prefix_capture", None)
        if prefix is not None:
            prefix.close()
        if self._driver is not None:
            self._driver.close()
        moe = getattr(self._server, "_instinctflash_moe_kernel", None)
        if moe is not None:
            moe.close()
        preprocess = getattr(self._server, "_instinctflash_gpu_preprocess", None)
        if preprocess is not None:
            preprocess.close()
        native_sdpa = getattr(self._server, "_instinctflash_native_sdpa", None)
        if native_sdpa is not None:
            native_sdpa.close()
        self._server = None


def _release_prefix_graphs_on_fail(recorder, server):
    """Wrap the plan recorder so a FAILED denoise self-check also closes the prefix graphs.

    The vision/prefill graphs are the same replay bet as the denoise graph; evidence that
    replay disagrees with eager in this process is evidence against all of them, so the
    loop discards the in-flight mixed call and reruns upstream after restoring its RNG/counters.
    Experimental custom kernels cannot claim this fallback and instead refuse further serving.
    """
    def on_verdict(res: dict) -> None:
        recorder(res)
        if res.get("passed"):
            return
        preprocess = getattr(server, "_instinctflash_gpu_preprocess", None)
        if preprocess is not None:
            preprocess.close()
            server._instinctflash_gpu_preprocess = None
        if hasattr(server, "_instinctflash_gpu_preprocess_pending"):
            delattr(server, "_instinctflash_gpu_preprocess_pending")
        if any(getattr(server,key,None) is not None for key in
               ("_instinctflash_moe_kernel","_instinctflash_rmsnorm_kernel")):
            server._instinctflash_fallback_refused = True
            raise RuntimeError("capture rejected with experimental kernels; automatic native fallback is unavailable")
        prefix = getattr(server, "_instinctflash_prefix_capture", None)
        if prefix is not None:
            prefix.close()
            server._instinctflash_prefix_capture = None
            import sys
            print("InstinctFlash LingBot-VLA-V2: vision/prefill graphs released with the "
                  "rejected denoise graph — the whole capture arm falls back together.",
                  file=sys.stderr, flush=True)
        native_sdpa = getattr(server, "_instinctflash_native_sdpa", None)
        if native_sdpa is not None:
            native_sdpa.close()
            server._instinctflash_native_sdpa = None
    return on_verdict


def _source_root(*, required: bool = True) -> Path | None:
    env = os.environ.get("LINGBOT_VLA_V2_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in SOURCE_ROOT_CANDIDATES:
        if (candidate / "deploy" / "lingbot_vla_v2_policy.py").exists():
            return candidate.resolve()
    if required:
        raise RuntimeError(
            "LingBot-VLA-V2 upstream source not found. Set LINGBOT_VLA_V2_ROOT to the checkout "
            f"(searched: {[str(c) for c in SOURCE_ROOT_CANDIDATES]}).")
    return None


def _configure_thor_ptxas(capability):
    """Use a Thor-capable system assembler for upstream MoE compilation.

    Some Triton wheels bundle a ptxas which rejects sm_110a. Preserve explicit
    user choices; fill absent overrides only when the toolkit advertises Thor.
    This selects a compiler, not a different MoE implementation or precision.
    """
    import shutil
    import subprocess
    if tuple(capability) != (11, 0):
        return {}
    names = ('TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH')
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        cuda_root = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        candidates = ([str(Path(cuda_root)/'bin/ptxas')] if cuda_root else [])
        candidates += ['/usr/local/cuda/bin/ptxas', shutil.which('ptxas')]
        for candidate in dict.fromkeys(candidates):
            if not candidate or not Path(candidate).is_file():
                continue
            try:
                probe = subprocess.run([candidate, '--help'], capture_output=True,
                                       text=True, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if probe.returncode == 0 and 'sm_110a' in probe.stdout + probe.stderr:
                for name in missing:
                    os.environ[name] = candidate
                print(f'InstinctFlash LingBot-VLA-V2: Thor MoE compiler {candidate} '
                      f"({', '.join(missing)}).")
                break
    return {name: os.environ.get(name) for name in names}


def _triton_kernels_allowed(flag_name: str) -> bool:
    """Refuse the Triton kernels on Thor (SM110) — measured dead, and worse than dead.

    Triton codegen fails on sm_110a with a PTXAS internal error (thor_column/vla2.json), and the
    vendor's MoE except-handler references an undefined ``logger``, so the failure surfaces as a
    NameError instead of a fallback; the RMSNorm patch has no try/except at all and dies at the
    first 768-wide norm. Refusing here is the only honest behaviour on that device.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0):
        raise RuntimeError(
            f"{flag_name} requests a Triton kernel on SM110 (Thor), where Triton codegen is "
            f"measured-dead (PTXAS internal error) and the vendor fallback path crashes. Use the "
            f"Thor engine arm (config 'lingbot_vla_v2', arch 'thor') instead.")
    return True


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean flag, got {value!r}")


def _resolve_model_path(checkpoint) -> Path:
    """Resolve flat, declared-view, and pointer-only package layouts."""
    root = Path(checkpoint.path)
    extra = dict(checkpoint.execution.extra or {})
    subdir = str(extra.get("checkpoint_subdir") or "")
    candidates = [root / subdir] if subdir else []
    candidates.append(root)
    for candidate in candidates:
        if ((candidate / "model.safetensors.index.json").exists()
                or next(candidate.glob("*.safetensors"), None) is not None):
            return candidate

    pointer = extra.get("base_weights")
    if pointer and Path(str(pointer)).exists():
        base = Path(str(pointer))
    elif pointer:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    candidate = base / subdir if subdir else base
    if not ((candidate / "model.safetensors.index.json").exists()
            or next(candidate.glob("*.safetensors"), None) is not None):
        raise RuntimeError(f"LingBot-VLA-V2 weights not found under {candidate}")
    return candidate


@contextlib.contextmanager
def _project_cwd(root: Path):
    with _CWD_LOCK:
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(previous)
