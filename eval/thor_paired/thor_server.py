"""Websocket serving loop around the InstinctFlash engine backend — runs ON THOR.

    ~/frt_env/bin/python thor_server.py --view ~/ifl/pi05_v044_view \
        --assets ~/ifl/t3_assets/thor_assets.json --calib ~/ifl/t3_assets/calib_obs.npz \
        --min-len 41 --max-len 62 --port 29930

T3 REAL-certificate revision: STATE CROSSES THE WIRE. The pilot stripped the
", State: ..." prompt suffix because set_prompt pays seconds of capture per call; this
server instead adopts the thor-2 production pattern (tools/pi05_thor_server.py): capture
one CUDA-graph set per prompt TOKEN LENGTH at startup (--min-len..--max-len, proven to
close libero_spatial's 8-bin state prompts at [41, 62]), then per infer copy the new
prompt's embeddings into the matching length's live _lang_emb buffer — the captured graph
reads it by pointer, so a per-chunk prompt change costs one embedding gather + device
copy, never a recapture. set_prompt() is counter-wrapped and the counter is reported on
every reply so clients can PROVE no serve-time recapture happened.

Wire protocol (msgpack-numpy, one message in -> one out, openpi-style):

    {"op": "ping"}                                  -> {"ok": true, counters, "lengths": [...]}
    {"op": "reset", "seed": S}                      -> {"ok": true, counters}
    {"op": "infer", "prompt_ids": int64[n],
     "image": (224,224,3) u8/f16,
     "wrist_image": (224,224,3) u8/f16}             -> {"ok": true, "actions": (10,7) f32 RAW,
                                                        "server_ms", "prompt_len", "se",
                                                        "set_prompt_calls"}

RAW means the model's normalized action output (_g_noise[:, :7]), NOT engine-unnormalized:
the lerobot eval applies the checkpoint's own unnormalizer postprocessor after
select_action in BOTH arms, so the engine returning already-unnormalized actions would
double-unnormalize arm B (invisible on pi05_base's identity stats, fatal on v044's).

Tokenization happens on the CLIENT: the eval batch already carries the exact PaliGemma
ids the local arm consumes (TokenizerProcessorStep), so shipping ids is tokenizer parity
by construction. A prompt whose length has no captured bucket is a HARD ERROR (a ~6 s
recapture stall inside a control loop is worse than a refused request).

fp8 calibration: per length, on the REAL frames in --calib, at startup, once per server
lifetime (calibrate() N>=2 recaptures and sets _real_data_calibrated, so the lazy
first-frame recalibration can never fire mid-serving). One server lifetime per
certificate, as before.

Determinism contract unchanged: flow noise is numpy's global RNG, one randn per infer;
"reset" reseeds it. Same seed + same obs bytes + same ids -> bit-identical actions.

On error the server sends a plain-text traceback (str, not msgpack) — the client raises
it verbatim, openpi-style.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msgpack_numpy  # noqa: E402  vendored, openpi wire format

RAW_ACTION_DIM = 7  # LIBERO action dim; the local arm slices predict_action_chunk the same way


class BucketEngine:
    """Per-prompt-length CUDA-graph sets over one Pi05TorchFrontendThor instance.

    Mirror of the validated thor-2 production mechanism (tools/pi05_thor_server.py),
    including its two measured landmines: (1) buckets are keyed by EXACT token count, not
    Se — adjacent lengths share an even-padded Se but their _lang_emb row counts differ;
    (2) _dec_rope is ONE shared buffer that set_prompt overwrites in place with
    length-dependent decode positions, so each bucket snapshots a clone and restores it
    before replay.
    """

    def __init__(self, frontend, synth_ids: dict, calib_obs: list, lo: int, hi: int,
                 percentile: float = 99.9):
        import torch
        self.torch = torch
        self.fe = frontend
        self.set_prompt_calls = 0
        orig_set_prompt = frontend.set_prompt

        def counted_set_prompt(p):
            self.set_prompt_calls += 1
            return orig_set_prompt(p)

        frontend.set_prompt = counted_set_prompt

        self.graphs: dict[int, dict] = {}
        for L in range(lo, hi + 1):
            ids = synth_ids.get(str(L))
            if ids is None:
                raise SystemExit(f"assets json has no synth prompt for length {L}")
            t0 = time.time()
            frontend.set_prompt(np.asarray(ids, dtype=np.int64))
            # Real-frame fp8 calibration per length (N>=2 -> multi-frame path, which
            # recaptures the enc+ae graph and sets _real_data_calibrated). Snapshot AFTER.
            frontend.calibrate(calib_obs, percentile=percentile)
            self.graphs[L] = {
                "len": L, "se": int(frontend.Se),
                "siglip": frontend._siglip_graph,          # noqa: SLF001
                "enc_ae": frontend._enc_ae_graph,          # noqa: SLF001
                "lang_emb": frontend._lang_emb,            # noqa: SLF001
                "dec_rope": frontend._dec_rope.clone(),    # noqa: SLF001
                "keep": (frontend._enc_calib_scales,       # noqa: SLF001
                         frontend._ae_calib_scales,        # noqa: SLF001
                         frontend._sa_all, frontend._sf_all, frontend._fs_all,  # noqa: SLF001
                         getattr(frontend, "_attn", None)),
            }
            print(f"[thor_server] captured len {L} -> Se {frontend.Se} "
                  f"in {time.time() - t0:.1f}s", flush=True)
        if not getattr(frontend, "_real_data_calibrated", False):
            raise SystemExit("ABORT: frontend not real-data calibrated after capture loop — "
                             "the lazy recalibration would recapture mid-serving")
        self.capture_baseline = self.set_prompt_calls
        torch.cuda.synchronize()
        print(f"[thor_server] {len(self.graphs)} buckets for lengths {sorted(self.graphs)} "
              f"({self.set_prompt_calls} set_prompt calls at startup)", flush=True)

    def infer(self, prompt_ids: np.ndarray, obs: dict) -> tuple[np.ndarray, dict]:
        torch = self.torch
        fe = self.fe
        ids = np.asarray(prompt_ids, dtype=np.int64).ravel()
        g = self.graphs.get(len(ids))
        if g is None:
            raise ValueError(
                f"prompt has {len(ids)} tokens; no bucket captured for that length. "
                f"Captured: {sorted(self.graphs)}. Widen --min-len/--max-len and restart — "
                f"never recapture inside an episode.")
        # Point the frontend at this length's graphs/buffers; the graphs read them by pointer.
        fe._siglip_graph = g["siglip"]                     # noqa: SLF001
        fe._enc_ae_graph = g["enc_ae"]                     # noqa: SLF001
        fe._lang_emb = g["lang_emb"]                       # noqa: SLF001
        fe._dec_rope.copy_(g["dec_rope"])                  # noqa: SLF001
        fe.Se = g["se"]
        fe.total_keys = g["se"] + fe.Sa
        emb = fe.embedding_weight[torch.as_tensor(ids, device="cuda")]
        emb = emb * float(emb.shape[-1] ** 0.5)            # Gemma convention (set_prompt parity)
        n_tok, n_buf = emb.shape[0], g["lang_emb"].shape[0]
        if n_buf == n_tok:
            g["lang_emb"].copy_(emb.to(g["lang_emb"].dtype))
        elif n_buf == n_tok + 1:
            # Se is padded even by DUPLICATING the last embedding (set_prompt does the
            # same); reproduce it so the extra row is never stale from capture time.
            g["lang_emb"][:n_tok].copy_(emb.to(g["lang_emb"].dtype))
            g["lang_emb"][n_tok].copy_(emb[-1].to(g["lang_emb"].dtype))
        else:
            raise RuntimeError(f"prompt has {n_tok} embeddings but bucket len {g['len']} "
                               f"holds {n_buf}; expected equal or one more")
        fe.infer(obs)                                       # advances the numpy noise RNG
        # RAW normalized actions — the local lerobot postprocessor unnormalizes both arms.
        raw = np.ascontiguousarray(
            fe._g_noise.float().cpu().numpy()[:, :RAW_ACTION_DIM].astype(np.float32))  # noqa: SLF001
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("non-finite raw actions")
        return raw, {"prompt_len": int(len(ids)), "se": g["se"],
                     "set_prompt_calls": self.set_prompt_calls}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default=os.path.expanduser("~/ifl/pi05_v044_view"),
                    help="declared checkpoint view (instinctflash.json + safetensors)")
    ap.add_argument("--assets", default=os.path.expanduser("~/ifl/t3_assets/thor_assets.json"))
    ap.add_argument("--calib", default=os.path.expanduser("~/ifl/t3_assets/calib_obs.npz"))
    ap.add_argument("--min-len", type=int, default=41)
    ap.add_argument("--max-len", type=int, default=62)
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29930)
    a = ap.parse_args()

    from instinctflash import Runtime
    t0 = time.time()
    rt = Runtime.from_pretrained(a.view, placement="engine")
    print(f"[thor_server] runtime up in {time.time() - t0:.1f}s "
          f"(model_id={rt.model_id})", flush=True)
    frontend = rt._backend._frontend                        # noqa: SLF001  bucket surgery

    assets = json.load(open(a.assets))
    z = np.load(a.calib)
    calib_obs = [{"image": z["image"][i], "wrist_image": z["wrist_image"][i]}
                 for i in range(z["image"].shape[0])]
    print(f"[thor_server] {len(calib_obs)} calibration frames from {a.calib}", flush=True)

    t0 = time.time()
    eng = BucketEngine(frontend, assets["synth_ids"], calib_obs, a.min_len, a.max_len,
                       percentile=a.percentile)
    print(f"[thor_server] bucket capture done in {time.time() - t0:.1f}s", flush=True)

    packer = msgpack_numpy.Packer()
    lock = threading.Lock()          # engine is single-stream; serialize everything
    counters = {"n_infer": 0, "n_reset": 0}

    from websockets.sync.server import serve

    def handler(ws):
        peer = getattr(ws, "remote_address", "?")
        print(f"[thor_server] client {peer} connected", flush=True)
        ws.send(packer.pack({"server": "instinctflash-thor-t3-cert",
                             "view": a.view, "model_id": rt.model_id,
                             "lengths": sorted(eng.graphs),
                             "raw_actions": True, "state_in_prompt": True}))
        for raw_msg in ws:
            try:
                msg = msgpack_numpy.unpackb(raw_msg)
                op = msg.get("op")
                with lock:
                    if op == "ping":
                        rep = {"ok": True, **counters,
                               "lengths": sorted(eng.graphs),
                               "set_prompt_calls": eng.set_prompt_calls}
                    elif op == "reset":
                        # Reseed BEFORE anything else: the episode's noise stream must be
                        # a pure function of the seed, not of what ran earlier.
                        np.random.seed(int(msg.get("seed", 0)))
                        counters["n_reset"] += 1
                        rep = {"ok": True, **counters,
                               "set_prompt_calls": eng.set_prompt_calls}
                    elif op == "infer":
                        if "prompt_ids" not in msg:
                            raise ValueError(
                                "T3-cert protocol requires 'prompt_ids' (exact client-side "
                                "PaliGemma ids, state included); the pilot's string-prompt "
                                "path is retired because it was state-blind")
                        t1 = time.perf_counter()
                        actions, meta = eng.infer(
                            msg["prompt_ids"],
                            {"image": msg["image"], "wrist_image": msg["wrist_image"]})
                        counters["n_infer"] += 1
                        rep = {"ok": True, "actions": actions,
                               "server_ms": (time.perf_counter() - t1) * 1e3, **meta}
                    else:
                        rep = {"ok": False, "error": f"unknown op {op!r}"}
                ws.send(packer.pack(rep))
            except Exception:                                     # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[thor_server] ERROR:\n{tb}", flush=True)
                try:
                    ws.send(tb)                                    # str -> client raises
                except Exception:                                  # noqa: BLE001
                    break
        print(f"[thor_server] client {peer} disconnected", flush=True)

    # ping_interval=None: bucket capture already happened, but a slow relay must not
    # sever an episode mid-chunk.
    with serve(handler, a.host, a.port, compression=None, max_size=None,
               ping_interval=None) as srv:
        print(f"[thor_server] listening on ws://{a.host}:{a.port}", flush=True)
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
