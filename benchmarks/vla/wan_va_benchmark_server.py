"""Opt-in identity and episode seeding for serve_variant's real single-GPU VA server.

Installed only for benchmark serving. The checkpoint must be a pinned Hub snapshot;
the receipt is generated from the loaded config, source and weight bytes, never from
the client's claimed identity. The ordinary serving entrypoint is unchanged.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import re
import sys
import types

from .util import ConfigurationError, sha256_file, sha256_json, write_json_atomic


def snapshot_identity(path):
    path = Path(path).absolute()
    if path.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", path.name):
        raise ConfigurationError("benchmark serving requires LINGBOT_CKPT=/.../models--ORG--REPO/snapshots/<revision>")
    repo = path.parent.parent.name
    if not repo.startswith("models--") or "--" not in repo[len("models--"):]:
        raise ConfigurationError("cannot derive model id from checkpoint snapshot path")
    files = {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not any(name.startswith("transformer/") and name.endswith(".safetensors") for name in files):
        raise ConfigurationError("snapshot has no transformer safetensors")
    return {"model_id": repo[len("models--"):].replace("--", "/", 1),
            "model_revision": path.name, "checkpoint_sha256": sha256_json(files)}


def execution_identity(config, applied):
    return {"nfe": {"video": config.num_inference_steps, "action": config.action_num_inference_steps},
            "guidance": {"video": float(config.guidance_scale), "action": float(config.action_guidance_scale)},
            "grid_shifts": {"video": float(config.snr_shift), "action": float(config.action_snr_shift)},
            "geometry": {name: getattr(config, name) for name in (
                "height", "width", "frame_chunk_size", "action_per_frame", "env_type", "obs_cam_keys")},
            "applied": list(applied), "dtype": str(config.param_dtype),
            "action_semantics_sha256": sha256_json({k: getattr(config, k) for k in
                ("attn_window", "action_dim", "used_action_channel_ids", "action_norm_method", "norm_stat", "video_exec_step")})}


class SeededPolicy:
    def __init__(self, model, identity, seed_fn):
        self.model, self.identity, self.seed_fn = model, identity, seed_fn
        self.seed = None
        self.original_infer = model._infer
        owner = self
        def seeded_infer(this, obs, frame_st_id=0):
            if owner.seed is None:
                raise ConfigurationError("benchmark episode must be reset with a seed")
            owner.seed_fn(owner.seed + int(frame_st_id))
            return owner.original_infer(obs, frame_st_id=frame_st_id)
        model._infer = types.MethodType(seeded_infer, model)

    def infer(self, observation):
        observation = dict(observation)
        if observation.get("reset"):
            seed = observation.pop("benchmark_seed", None)
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ConfigurationError("reset requires a non-negative benchmark_seed")
            expected = sha256_json(self.identity)
            if observation.pop("benchmark_identity_sha256", None) != expected:
                raise ConfigurationError("reset identity mismatch")
            self.seed = seed
            self.seed_fn(seed)
            result = self.model.infer(observation)
            return {**result, "benchmark_seed": seed, "benchmark_identity_sha256": expected}
        if self.seed is None:
            raise ConfigurationError("inference before seeded reset")
        return self.model.infer(observation)


def install_server(server_module, receipt, applied, protocol=None):
    """Call after all pass installers, before S.run constructs the model."""
    from .robotwin_driver import PROTOCOL
    from .plan import pipeline_digest

    def serve(model, local_rank, host, port):
        import torch
        from instinctflash.serving.msgpack_numpy import Packer, unpackb
        import asyncio
        from websockets.asyncio.server import serve as ws_serve
        from websockets.exceptions import ConnectionClosed

        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or local_rank != 0:
            raise ConfigurationError("benchmark server requires exactly one GPU/process")
        identity = {"schema_version": 1, "protocol": protocol or PROTOCOL,
                    **snapshot_identity(model.job_config.wan22_pretrained_model_name_or_path),
                    "execution": execution_identity(model.job_config, applied),
                    "seed_mode": "episode_plus_frame", "synthetic": False,
                    "server_code_sha256": sha256_file(Path(__file__)),
                    "pipeline_sha256": pipeline_digest(),
                    "upstream_sha256": sha256_json({
                        str(p.relative_to(Path(server_module.__file__).resolve().parent.parent)): sha256_file(p)
                        for p in sorted(Path(server_module.__file__).resolve().parent.parent.rglob("*.py"))
                        if ".venv" not in p.parts}),
                    "optimization_source_sha256": sha256_json({
                        str(p.relative_to(Path(__file__).resolve().parents[2])): sha256_file(p)
                        for folder in ("instinctflash", "eval/lingbot_va_robotwin")
                        for p in sorted((Path(__file__).resolve().parents[2] / folder).rglob("*.py"))}),
                    "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "packages_sha256": sha256_json(sorted(
                        f"{d.metadata.get('Name')}=={d.version}" for d in importlib.metadata.distributions()))}
        def seed_all(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        policy = SeededPolicy(model, identity, seed_all)

        async def run():
            busy = False
            async def handle(connection):
                nonlocal busy
                if busy:
                    await connection.close(code=1013, reason="benchmark episode already owns server")
                    return
                busy = True
                policy.seed = None
                packer = Packer()
                try:
                    await connection.send(packer.pack({"benchmark_identity": identity}))
                    async for message in connection:
                        obs = unpackb(message)
                        # Keep the event loop serving handshakes while one inference runs.
                        # Awaiting completion before releasing busy prevents episode overlap.
                        result = await asyncio.to_thread(policy.infer, obs)
                        await connection.send(packer.pack(result))
                except ConnectionClosed:
                    pass
                except Exception as error:
                    await connection.send(f"{type(error).__name__}: {error}")
                    await connection.close(code=1011)
                finally:
                    policy.seed = None
                    busy = False
            async with ws_serve(handle, host, port, compression=None, max_size=None, ping_interval=None):
                # Receipt means the socket is bound and the actual model has finished loading.
                write_json_atomic(Path(receipt), identity)
                print(f"benchmark server ready; receipt: {receipt}", flush=True)
                await asyncio.Future()
        asyncio.run(run())

    server_module.run_async_server_mode = serve
