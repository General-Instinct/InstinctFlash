"""RemoteEnginePolicy — lerobot's policy interface locally, the Thor engine remotely.

Presents exactly the surface lerobot_eval's rollout touches (nn.Module, .config, .reset(),
.select_action(batch), .predict_action_chunk(batch)) and forwards each chunk request over a
websocket to thor_server.py, which wraps Runtime.from_pretrained on the Jetson.

Observation mapping is a deliberate mirror of PI05Policy._preprocess_images: the batch
arrives as [1,3,256,256] float32 in [0,1], we resize_with_pad to the config resolution,
map to [-1,1], and ship fp16 HWC — the engine frontend passes negative-valued float16
through untouched, so the pixels the engine sees are the same pixels the local bf16 arm
computes on.

T3 REAL-certificate revision — STATE CROSSES THE WIRE. The pilot stripped the
", State: ..." suffix (set_prompt cost seconds per prompt change); the server now holds
one captured graph set per prompt token length, so this proxy ships the EXACT PaliGemma
token ids the local arm consumes — batch[observation.language.tokens] under its attention
mask, produced by the same TokenizerProcessorStep in both arms. Tokenizer parity is
therefore by construction, and the discretized state reaches the remote model exactly as
it reaches the local one.

The server returns RAW normalized actions (the model's _g_noise[:, :7]) — NOT engine-
unnormalized — because lerobot_eval applies the checkpoint's own unnormalizer
postprocessor after select_action in both arms. Every infer reply carries
set_prompt_calls; this proxy asserts it never moves after the first chunk (a serve-time
recapture would be a protocol violation, not a slow chunk).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import deque

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msgpack_numpy  # noqa: E402  vendored, openpi wire format


def _sha(arr) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def to_engine_frame(img: torch.Tensor, resolution=(224, 224)):
    """One image, lerobot batch form -> engine wire form.

    In:  [1,3,H,W] float32 in [0,1] (what the eval preprocessor hands the policy).
    Out: (h,w,3) float16 in [-1,1] — the same resize_with_pad + [-1,1] mapping
    PI05Policy._preprocess_images applies, so both arms compute on the same pixels.
    """
    import numpy as np
    from lerobot.policies.common.vla_utils import resize_with_pad_torch
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.shape[0] != 1:
        raise ValueError(f"batch-1 only, got batch {img.shape[0]}")
    img = img.to(torch.float32)
    if img.shape[1] == 3:                                         # BCHW -> BHWC
        img = img.permute(0, 2, 3, 1)
    if img.shape[1:3] != tuple(resolution):
        img = resize_with_pad_torch(img, *resolution)
    img = img * 2.0 - 1.0
    return np.ascontiguousarray(img[0].to(torch.float16).cpu().numpy())


class EngineClient:
    """Blocking one-request-one-reply client; RoboTwin's proven connect settings."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 300.0):
        import websockets.sync.client
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        deadline = time.time() + connect_timeout_s
        last = None
        while True:
            try:
                # ping_interval=None: long engine pauses (set_prompt capture) must not
                # kill the connection. max_size=None: fp16 frames exceed the 1MB default.
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    ping_interval=None, close_timeout=10)
                break
            except Exception as e:                                # noqa: BLE001
                last = e
                if time.time() > deadline:
                    raise RuntimeError(f"no engine server at {self._uri}: {last!r}") from e
                time.sleep(2)
        self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=60))

    def call(self, msg: dict, timeout_s: float = 900.0) -> dict:
        self._ws.send(self._packer.pack(msg))
        rep = self._ws.recv(timeout=timeout_s)
        if isinstance(rep, str):                # server sent a traceback
            raise RuntimeError(f"engine server error:\n{rep}")
        rep = msgpack_numpy.unpackb(rep)
        if not rep.get("ok", False):
            raise RuntimeError(f"engine server refused: {rep}")
        return rep

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:                                          # noqa: BLE001
            pass


from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402


class RemoteEnginePolicy(PreTrainedPolicy):
    """Arm-B proxy: pops actions from a local queue, refills it from the Thor engine.

    Subclasses PreTrainedPolicy because lerobot_eval's eval_policy type-checks the policy
    (ValueError otherwise); the weights-bearing parts of the base class are never touched.
    """

    config_class = PI05Config
    name = "pi05_remote_engine"

    def __init__(self, config, host: str, port: int, *,
                 log_path: str | None = None, engine_seed: int = 7):
        super().__init__(config)
        # An anchor buffer so .to(device) / device probing on the "policy" behave.
        self.register_buffer("_anchor", torch.zeros(1))
        self._host, self._port = host, port
        self._client: EngineClient | None = None
        self._queue: deque = deque(maxlen=config.n_action_steps)
        self._engine_seed = int(engine_seed)
        self._ep = -1
        self._chunk = 0
        self._set_prompt_calls: int | None = None   # frozen after first chunk; recapture = bug
        self._log = open(log_path, "a", buffering=1) if log_path else None

    # -- lerobot policy surface ----------------------------------------------------------
    def reset(self) -> None:
        self._ensure_client()
        self._queue.clear()
        self._ep += 1
        self._chunk = 0
        # Per-episode seed: reproducible given the episode index, distinct across episodes
        # within a job. Two invocations of the same job replay the same seed sequence —
        # that is the property the null control certifies.
        seed = self._engine_seed + self._ep
        rep = self._client.call({"op": "reset", "seed": seed})
        self._log_event({"event": "reset", "ep": self._ep, "engine_seed": seed,
                         "server_counters": {k: rep.get(k) for k in ("n_infer", "n_reset")}})

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        if len(self._queue) == 0:
            actions = self.predict_action_chunk(batch)            # (1, n, dim)
            self._queue.extend(actions.transpose(0, 1))           # rows (1, dim)
        return self._queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **_kw) -> torch.Tensor:
        import numpy as np
        self._ensure_client()
        prompt = self._full_prompt(batch)                         # state INCLUDED, for the log
        prompt_ids = self._prompt_ids(batch)                      # exact local-arm token ids
        imgs = self._images_for_engine(batch)                     # list of (224,224,3) f16
        payload = {"op": "infer", "prompt_ids": prompt_ids, "image": imgs[0]}
        payload["wrist_image"] = imgs[1] if len(imgs) > 1 else imgs[0]
        t0 = time.perf_counter()
        rep = self._client.call(payload)
        rtt_ms = (time.perf_counter() - t0) * 1e3
        spc = rep.get("set_prompt_calls")
        if self._set_prompt_calls is None:
            self._set_prompt_calls = spc
        elif spc != self._set_prompt_calls:
            raise RuntimeError(
                f"server recaptured mid-serving: set_prompt_calls moved "
                f"{self._set_prompt_calls} -> {spc}; arms are no longer the same candidate")
        actions = np.asarray(rep["actions"], dtype=np.float32)    # (chunk, dim) RAW normalized
        n = min(self.config.n_action_steps, actions.shape[0])
        actions = actions[:n]
        self._log_event({
            "event": "chunk", "ep": self._ep, "chunk": self._chunk, "prompt": prompt,
            "prompt_len": int(len(prompt_ids)),
            "obs_sha": _sha(np.stack(imgs)), "actions_sha": _sha(actions),
            "actions": actions.tolist(), "set_prompt_calls": spc,
            "rtt_ms": round(rtt_ms, 2), "server_ms": round(rep.get("server_ms", 0.0), 2)})
        self._chunk += 1
        return torch.from_numpy(actions).unsqueeze(0)             # (1, n, dim)

    # -- PreTrainedPolicy abstract surface not used at eval time --------------------------
    def get_optim_params(self) -> dict:
        raise NotImplementedError("RemoteEnginePolicy is an eval-time proxy; it does not train")

    def forward(self, batch: dict, **_kw):
        raise NotImplementedError("RemoteEnginePolicy is an eval-time proxy; it does not train")

    # -- plumbing -------------------------------------------------------------------------
    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = EngineClient(self._host, self._port)
            self._log_event({"event": "connect", "uri": f"ws://{self._host}:{self._port}",
                             "metadata": self._client.metadata})

    def _full_prompt(self, batch: dict) -> str:
        """The full rewritten prompt ('Task: <text>, State: <bins>;\\nAction: ') — log evidence
        that state crossed the wire, never stripped."""
        task = batch.get("task")
        if not task:
            return ""
        t = task[0] if isinstance(task, (list, tuple)) else task
        return str(t)

    def _prompt_ids(self, batch: dict):
        """The EXACT token ids the local arm consumes: TokenizerProcessorStep output,
        unpadded via its attention mask. No re-tokenization on either side of the wire."""
        import numpy as np
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        mask = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        if tokens is None or mask is None:
            raise ValueError(
                f"batch lacks {OBS_LANGUAGE_TOKENS}/{OBS_LANGUAGE_ATTENTION_MASK} — the "
                f"tokenizer processor step did not run; refusing to re-tokenize remotely "
                f"(got keys {sorted(batch)[:10]})")
        ids = tokens[0][mask[0].to(torch.bool)]
        return np.ascontiguousarray(ids.detach().cpu().numpy().astype(np.int64))

    def _images_for_engine(self, batch: dict) -> list:
        """Mirror PI05Policy._preprocess_images up to the point pixels leave the CPU."""
        res = tuple(getattr(self.config, "image_resolution", (224, 224)))
        out = [to_engine_frame(batch[key], res)
               for key in self.config.image_features if key in batch]  # config order = camera order
        if not out:
            raise ValueError(f"no image features {list(self.config.image_features)} "
                             f"in batch keys {sorted(batch)[:8]}")
        return out

    def _log_event(self, rec: dict) -> None:
        if self._log is not None:
            rec["ts"] = round(time.time(), 3)
            self._log.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
