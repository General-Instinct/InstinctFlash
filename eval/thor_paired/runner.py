"""T3 REAL-certificate runner — paired closed-loop LIBERO gate, weights = v044 both arms.

    /home/ubuntu/tools/pi05env313/bin/python runner.py {smoke|smokeB|null|declare|paired|report|all}

Arms (SAME weights by construction — lerobot/pi05_libero_finetuned_v044, declared on Thor
as ~/ifl/pi05_v044_view):
    A  local bf16 PI05Policy on H100 (GPUs 2+3, two parallel lanes), stock lerobot path
    B  RemoteEnginePolicy -> ws://thor:29930 -> Runtime(engine backend, fp8 W+A),
       STATE IN PROMPT via per-length capture buckets, RAW actions over the wire,
       sequential on GPU 2, ONE server lifetime for everything it ever answers.

Protocol (pre-declared, pi05_compress discipline):
    suite libero_spatial, ALL 10 tasks x 3 episodes = 30 paired episodes, seed 1000,
    batch_size 1, chunk_size = n_action_steps = 10 (the engine's captured geometry, both
    arms), non-inferiority margin 5.0 pp declared in certificate.json BEFORE any paired
    episode runs, exact pairing + McNemar via the pi05_compress comparator.

Order: smoke -> null (same-obs replay + state-liveness gates, live v044 server)
       -> declare (write the pre-declaration) -> paired (arm B first, alone on the box;
       then arm A on two GPU lanes) -> report (certificate.json).

Differences from the pilot, each closing a measured confound:
    - v044 (LIBERO fine-tune) instead of pi05_base: real successes, so the comparator's
      verdict is no longer degenerate.
    - state-blindness closed: exact token ids (state included) cross the wire per chunk.
    - raw actions over the wire: the checkpoint's unnormalizer runs locally in BOTH arms
      (engine-side unnorm + local unnorm would double-unnormalize v044).
    - no rename_map: v044 declares LIBERO-native camera keys.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = "/home/ubuntu/tools/pi05env313/bin/python"
HERE = Path(__file__).parent
INJ = str(HERE / "inj")
OUT = Path("/home/ubuntu/iwm_distill/thor_t3/cert")
CERT = Path("/home/ubuntu/iwm_distill/thor_t3/certificate.json")
ASSETS = OUT / "assets" / "thor_assets.json"
CALIB = OUT / "assets" / "calib_obs.npz"
ARM_A_GPUS = ["2", "3"]
ARM_B_GPU = "2"
SEED = 1000
SUITE = "libero_spatial"
TASKS = list(range(10))
# T3_EXT=1 runs the POWER EXTENSION: same margin, same seeds discipline, same server
# lifetime, more episodes. The pre-declared n=30 primary measured diff 0.00pp with 6/30
# discordant — honest but underpowered for a 5pp NI margin (one-sided p=0.270); at the
# observed ~20% discordance, ~500 pairs give >=80% power at true diff 0. The margin
# NEVER moves; the extension supersedes the primary as the powered analysis and the
# primary is preserved verbatim in the certificate.
EXT = os.environ.get("T3_EXT") == "1"
N_EPS = 50 if EXT else 3
PAIRED_SUB = "ext_paired" if EXT else "paired"
MARGIN_PP = 5.0
THOR_HOST = os.environ.get("IFL_THOR_HOST", "100.68.159.80")
THOR_PORT = int(os.environ.get("IFL_THOR_PORT", "29930"))
ENGINE_SEED = 7
JOB_TIMEOUT = 7200 if EXT else 3600
CKPT_GLOB = ("/home/ubuntu/.cache/huggingface/hub/"
             "models--lerobot--pi05_libero_finetuned_v044/snapshots/*/")


def ckpt_dir() -> str:
    hits = glob.glob(CKPT_GLOB)
    if not hits:
        sys.exit("no local lerobot/pi05_libero_finetuned_v044 snapshot")
    return hits[0].rstrip("/")


def job_cmd(task: int, n_eps: int, outdir: Path) -> list[str]:
    return [PY, "-m", "lerobot.scripts.lerobot_eval",
            f"--policy.path={ckpt_dir()}",
            "--policy.compile_model=false",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            # chunk 10 = the engine's captured geometry; the local arm samples the same
            # 10-step horizon so the two arms share chunk geometry, not just weights.
            # (NOT comparable to pi05_compress chunk-50 numbers — pilot wall, still true.)
            "--policy.n_action_steps=10",
            "--policy.chunk_size=10",
            "--env.type=libero", f"--env.task={SUITE}", f"--env.task_ids=[{task}]",
            f"--eval.n_episodes={n_eps}", "--eval.batch_size=1",
            "--eval.use_async_envs=false",
            f"--seed={SEED}", f"--output_dir={outdir}"]


def job_env(remote: bool, log_path: Path, gpu: str) -> dict:
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": gpu, "MUJOCO_GL": "egl",
           "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONPATH": INJ,
           "IFL_ACTION_LOG": str(log_path)}
    if remote:
        env.update({"IFL_THOR_REMOTE": "1", "IFL_THOR_HOST": THOR_HOST,
                    "IFL_THOR_PORT": str(THOR_PORT), "IFL_ENGINE_SEED": str(ENGINE_SEED)})
    return env


def launch_job(task: int, n_eps: int, outdir: Path, remote: bool, gpu: str):
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "actions.jsonl"
    if log_path.exists():
        log_path.rename(outdir / "actions.jsonl.prev")
    joblog = outdir / "job.log"
    lf = open(joblog, "w")
    p = subprocess.Popen(job_cmd(task, n_eps, outdir), env=job_env(remote, log_path, gpu),
                         stdout=lf, stderr=subprocess.STDOUT)
    return {"proc": p, "task": task, "out": outdir, "log": joblog,
            "t0": time.time(), "gpu": gpu, "remote": remote}


def finish_job(j) -> dict:
    try:
        rc = j["proc"].wait(timeout=max(1, JOB_TIMEOUT - (time.time() - j["t0"])))
    except subprocess.TimeoutExpired:
        j["proc"].kill()
        rc = -9
    outdir = j["out"]
    ok = rc == 0 and (outdir / "eval_info.json").exists()
    wall = time.time() - j["t0"]
    arm = "armB" if j["remote"] else "armA"
    print(f"  [{arm}] {SUITE} t{j['task']} gpu{j['gpu']} "
          f"{'ok' if ok else f'FAIL rc={rc}'} ({wall:.0f}s) -> {outdir}", flush=True)
    r = {"task": j["task"], "ok": ok, "rc": rc, "wall_s": round(wall, 1),
         "gpu": j["gpu"], "out": str(outdir), "log": str(j["log"])}
    if ok:
        r["successes"] = read_successes(outdir)
    return r


def run_job(task: int, n_eps: int, outdir: Path, remote: bool, gpu: str) -> dict:
    return finish_job(launch_job(task, n_eps, outdir, remote, gpu))


def done_successes(outdir: Path, n_eps: int):
    """Successes of an ALREADY-COMPLETE job (resume support: the harness supervising a
    multi-hour phase can be killed without invalidating finished task jobs — eval_info.json
    is written atomically at job end, partial jobs never have one). Returns None unless the
    job finished with the full episode count."""
    try:
        succ = read_successes(outdir)
    except Exception:                                              # noqa: BLE001
        return None
    return succ if len(succ) == n_eps else None


def read_successes(outdir: Path) -> list[bool]:
    info = json.load(open(outdir / "eval_info.json"))
    succ = []
    for entry in info["per_task"]:
        succ += [bool(x) for x in entry["metrics"]["successes"]]
    return succ


def read_chunks(log_path: Path) -> list[dict]:
    recs = []
    for line in open(log_path):
        r = json.loads(line)
        if r.get("event") == "chunk":
            recs.append(r)
    return recs


def progress(msg: str) -> None:
    with open("/tmp/t3_cert_progress.md", "a") as f:
        f.write(f"- {time.strftime('%H:%M:%S')} {msg}\n")
    print(f"[progress] {msg}", flush=True)


# ------------------------------------------------------------------ phases
def phase_smoke() -> dict:
    """Ping the Thor server; verify it serves the RIGHT view (v044, state-in-prompt,
    raw actions) with every needed length bucket. No infer — nothing to warm up anymore
    (fp8 calibration happened per length at server startup), this is a contract check."""
    sys.path.insert(0, str(HERE))
    from remote_policy import EngineClient
    c = EngineClient(THOR_HOST, THOR_PORT, connect_timeout_s=20)
    rep = c.call({"op": "ping"})
    c.close()
    meta = c.metadata
    assert "pi05_libero_finetuned_v044" in str(meta.get("model_id", "")), \
        f"server serves {meta.get('model_id')!r}, not v044 — wrong view"
    assert meta.get("raw_actions") and meta.get("state_in_prompt"), \
        f"server metadata lacks the T3-cert contract flags: {meta}"
    need = json.load(open(ASSETS))
    lo, hi = need["min_len"], need["max_len"]
    have = set(rep.get("lengths", []))
    missing = [L for L in range(lo, hi + 1) if L not in have]
    assert not missing, f"server lacks buckets for lengths {missing}"
    out = {"server": meta, "ping": rep}
    print(f"smoke: model_id={meta['model_id']} lengths={rep['lengths'][:3]}..."
          f"{rep['lengths'][-1]} set_prompt_calls={rep['set_prompt_calls']}", flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=1))
    return out


def phase_null() -> dict:
    """Both null gates through the live v044 server (see null_replay.py docstring)."""
    d = OUT / "null"
    d.mkdir(parents=True, exist_ok=True)
    out_json = d / "null_controls.json"
    cp = subprocess.run([PY, str(HERE / "null_replay.py"), str(ASSETS), str(CALIB),
                         str(out_json)],
                        env={**os.environ, "IFL_THOR_HOST": THOR_HOST,
                             "IFL_THOR_PORT": str(THOR_PORT)})
    result = json.loads(out_json.read_text())
    if cp.returncode != 0 or not result.get("passed"):
        raise SystemExit(f"NULL CONTROLS FAILED — do not run the paired gate: {out_json}")
    progress(f"null controls PASSED (replay 2x28 bit-identical, state liveness, "
             f"no recapture, server_ms_max={result['state_liveness']['server_ms_max']})")
    return result


def phase_smokeB() -> dict:
    """One arm-B episode end-to-end (proxy -> ids -> buckets -> raw actions) before
    committing 30 episodes to the protocol. Output kept OUTSIDE the paired tree."""
    r = run_job(TASKS[0], 1, OUT / "smokeB" / f"{SUITE}_t{TASKS[0]}", remote=True,
                gpu=ARM_B_GPU)
    if not r["ok"]:
        raise SystemExit(f"arm-B smoke failed: {r['log']}")
    chunks = read_chunks(Path(r["out"]) / "actions.jsonl")
    lens = sorted({c["prompt_len"] for c in chunks})
    spc = {c["set_prompt_calls"] for c in chunks}
    assert len(spc) == 1, f"recapture during smokeB: set_prompt_calls {spc}"
    progress(f"arm-B smoke episode ok: {len(chunks)} chunks, prompt lens {lens}, "
             f"success={r['successes']}, no recapture")
    return r


def phase_declare() -> dict:
    """Pre-declare the gate BEFORE any paired episode: margin, protocol, pairing rule."""
    if EXT:
        return phase_declare_ext()
    if CERT.exists():
        cert = json.loads(CERT.read_text())
        assert cert["pre_declared_margin"] == MARGIN_PP, \
            "certificate.json exists with a different margin — refusing to move the goalposts"
        return cert
    cert = {
        "what": "T3 REAL certificate: paired closed-loop LIBERO gate, arm A local H100 "
                "bf16 lerobot policy, arm B Thor engine (fp8 W+A) serving the SAME "
                "weights over the network, state-in-prompt via length-bucket graphs",
        "declared_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "weights": "lerobot/pi05_libero_finetuned_v044 (snapshot 8e174154) BOTH arms",
        "pre_declared_margin": MARGIN_PP,
        "decision_rule": "exact pairing on (suite, task, episode), McNemar on discordant "
                         "pairs, one-sided non-inferiority at the pre-declared margin via "
                         "the pi05_compress comparator; verdict taken from the comparator "
                         "output, not recomputed",
        "protocol": {"suite": SUITE, "tasks": TASKS, "n_eps": N_EPS, "seed": SEED,
                     "batch_size": 1, "n_action_steps": 10, "chunk_size": 10,
                     "engine_seed_base": ENGINE_SEED,
                     "server": f"ws://{THOR_HOST}:{THOR_PORT}",
                     "one_server_lifetime": True,
                     "armA": "local bf16, H100 GPUs 2+3 (two parallel task lanes)",
                     "armB": "remote fp8 engine via ws proxy, sequential, sim on GPU 2"},
        "status": "DECLARED — paired runs not started",
    }
    CERT.write_text(json.dumps(cert, indent=1))
    progress(f"pre-declared margin {MARGIN_PP}pp in {CERT} before any paired episode")
    return cert


def phase_declare_ext() -> dict:
    """Declare the power extension BEFORE running it: same margin, more episodes.

    Rationale recorded verbatim: the pre-declared n=30 primary measured diff 0.00pp
    (McNemar p=1.0, 3/3 discordant) but a 5pp NI margin needs ~500 pairs at the observed
    ~20% discordance for >=80% power at true diff 0. This is a POWER fix, not a goalpost
    move: margin unchanged, all primary data preserved and reported.
    """
    cert = json.loads(CERT.read_text())
    assert cert["pre_declared_margin"] == MARGIN_PP
    assert cert.get("status") == "MEASURED", "primary gate must be measured first"
    if "power_extension_declared" in cert:
        return cert
    sys.path.insert(0, str(HERE))
    from remote_policy import EngineClient
    c = EngineClient(THOR_HOST, THOR_PORT, connect_timeout_s=20)
    ping = c.call({"op": "ping"})
    c.close()
    assert ping["set_prompt_calls"] == 22, \
        f"server lifetime changed (set_prompt_calls={ping['set_prompt_calls']}); a new " \
        f"lifetime means a new fp8 calibration — extension would not be the same candidate"
    cert["power_extension_declared"] = {
        "declared_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "n_eps_per_task": N_EPS, "n_pairs": N_EPS * len(TASKS),
        "margin_pp": MARGIN_PP,
        "rationale": "primary n=30 underpowered: diff 0.00pp, 6/30 discordant, NI "
                     "one-sided p=0.270 at 5pp; ~500 pairs give >=80% power at true "
                     "diff 0 with ~20% discordance (project precedent: 925 pairs for "
                     "the QAD-KD certificate)",
        "same_server_lifetime": {"ping_at_declaration": ping},
        "status": "DECLARED — extension not started",
    }
    CERT.write_text(json.dumps(cert, indent=1))
    progress(f"power extension declared: {N_EPS} eps/task = {N_EPS * len(TASKS)} pairs, "
             f"margin unchanged {MARGIN_PP}pp, same server lifetime "
             f"(set_prompt_calls={ping['set_prompt_calls']})")
    return cert


def phase_paired() -> dict:
    cert = json.loads(CERT.read_text())
    assert cert["pre_declared_margin"] == MARGIN_PP, "margin not pre-declared"
    if EXT:
        assert "power_extension_declared" in cert, "extension not declared"
    res = {"armA": [], "armB": []}

    # Arm B first, alone on the box (concurrency blinded evals before; keep it clean).
    for t in TASKS:
        outdir = OUT / PAIRED_SUB / "armB" / f"{SUITE}_t{t}"
        prior = done_successes(outdir, N_EPS)
        if prior is not None:
            r = {"task": t, "ok": True, "rc": 0, "wall_s": None, "gpu": ARM_B_GPU,
                 "out": str(outdir), "log": str(outdir / "job.log"),
                 "successes": prior, "resumed": True}
            print(f"  [armB] {SUITE} t{t} already complete ({sum(prior)}/{N_EPS}) — reused",
                  flush=True)
        else:
            r = run_job(t, N_EPS, outdir, remote=True, gpu=ARM_B_GPU)
        res["armB"].append(r)
        (OUT / f"{PAIRED_SUB}_runs.json").write_text(json.dumps(res, indent=1))
        if not r["ok"]:
            raise SystemExit(f"armB task {t} failed — see {r['log']}")
    progress(f"arm B complete: {sum(sum(r['successes']) for r in res['armB'])}/"
             f"{N_EPS * len(TASKS)} successes")

    # Arm A: two parallel lanes over GPUs 2 and 3.
    pending = []
    for t in TASKS:
        outdir = OUT / PAIRED_SUB / "armA" / f"{SUITE}_t{t}"
        prior = done_successes(outdir, N_EPS)
        if prior is not None:
            res["armA"].append({"task": t, "ok": True, "rc": 0, "wall_s": None,
                                "gpu": None, "out": str(outdir),
                                "log": str(outdir / "job.log"),
                                "successes": prior, "resumed": True})
            print(f"  [armA] {SUITE} t{t} already complete ({sum(prior)}/{N_EPS}) — reused",
                  flush=True)
        else:
            pending.append(t)
    running: list = []
    while pending or running:
        while pending and len(running) < len(ARM_A_GPUS):
            used = {j["gpu"] for j in running}
            gpu = next(g for g in ARM_A_GPUS if g not in used)
            t = pending.pop(0)
            running.append(launch_job(t, N_EPS, OUT / PAIRED_SUB / "armA" / f"{SUITE}_t{t}",
                                      remote=False, gpu=gpu))
        for j in running:
            if time.time() - j["t0"] > JOB_TIMEOUT and j["proc"].poll() is None:
                j["proc"].kill()
        done = [j for j in running if j["proc"].poll() is not None]
        if not done:
            time.sleep(5)
            continue
        for j in done:
            running.remove(j)
            r = finish_job(j)
            res["armA"].append(r)
            (OUT / f"{PAIRED_SUB}_runs.json").write_text(json.dumps(res, indent=1))
            if not r["ok"]:
                for k in running:
                    k["proc"].kill()
                raise SystemExit(f"armA task {r['task']} failed — see {r['log']}")
    res["armA"].sort(key=lambda r: r["task"])
    (OUT / f"{PAIRED_SUB}_runs.json").write_text(json.dumps(res, indent=1))
    progress(f"arm A complete: {sum(sum(r['successes']) for r in res['armA'])}/"
             f"{N_EPS * len(TASKS)} successes")
    return res


def merged_json(arm: str, label: str) -> Path:
    suites = {SUITE: {}}
    total_s = total_n = 0
    for t in TASKS:
        d = OUT / PAIRED_SUB / arm / f"{SUITE}_t{t}"
        succ = read_successes(d)
        suites[SUITE][str(t)] = succ
        total_s += sum(succ)
        total_n += len(succ)
    merged = {"ckpt": label, "nfe": None, "seed": SEED, "n_eps": N_EPS,
              "n_action_steps": 10, "suites": suites,
              "overall": {"successes": total_s, "episodes": total_n,
                          "pc_success": round(100 * total_s / total_n, 2) if total_n else None}}
    p = OUT / PAIRED_SUB / arm / "merged.json"
    p.write_text(json.dumps(merged, indent=1))
    return p


def phase_report() -> dict:
    a = merged_json("armA", "lerobot/pi05_libero_finetuned_v044 [local bf16, H100]")
    b = merged_json("armB", "lerobot/pi05_libero_finetuned_v044 [thor engine fp8, remote ws, "
                            "state-in-prompt]")
    pairing_path = OUT / ("ext_pairing.json" if EXT else "pairing.json")
    cp = subprocess.run(
        [PY, "-m", "pi05_compress", "compare", str(a), str(b),
         "--margin", str(MARGIN_PP), "--out", str(pairing_path)],
        env={**os.environ, "PYTHONPATH": "/home/ubuntu/InstinctCompress"},
        capture_output=True, text=True)
    print(cp.stdout)
    if cp.returncode != 0:
        print(cp.stderr, file=sys.stderr)
        raise SystemExit("compare_evals failed")
    pairing = json.loads(pairing_path.read_text())

    null = json.loads((OUT / "null" / "null_controls.json").read_text())
    paired = json.loads((OUT / f"{PAIRED_SUB}_runs.json").read_text())
    smoke = json.loads((OUT / "smoke.json").read_text())

    per_task, rtts, server_ms, spc_seen, prompt_lens = [], [], [], set(), set()
    for t in TASKS:
        entry = {"task": t}
        for arm in ("armA", "armB"):
            d = OUT / PAIRED_SUB / arm / f"{SUITE}_t{t}"
            succ = read_successes(d)
            entry[arm] = succ
            chunks = read_chunks(d / "actions.jsonl")
            if arm == "armB":
                for c in chunks:
                    if "rtt_ms" in c:
                        rtts.append(c["rtt_ms"])
                    if "server_ms" in c:
                        server_ms.append(c["server_ms"])
                    if "set_prompt_calls" in c:
                        spc_seen.add(c["set_prompt_calls"])
                    if "prompt_len" in c:
                        prompt_lens.add(c["prompt_len"])
        entry["pairs"] = [
            {"ep": i, "armA": entry["armA"][i], "armB": entry["armB"][i],
             "discordant": entry["armA"][i] != entry["armB"][i]}
            for i in range(min(len(entry["armA"]), len(entry["armB"])))]
        per_task.append(entry)

    def pct(v, q):
        return v[min(len(v) - 1, int(len(v) * q))]

    rtts.sort()
    server_ms.sort()
    lat = {
        "n_chunks": len(rtts),
        "rtt_ms": {"p50": pct(rtts, 0.5), "p95": pct(rtts, 0.95), "max": rtts[-1]},
        "server_ms": {"p50": pct(server_ms, 0.5), "p95": pct(server_ms, 0.95),
                      "max": server_ms[-1]},
        "note": "rtt includes tailscale relay; server_ms is engine time on Thor",
    } if rtts else None

    assert len(spc_seen) <= 1, f"set_prompt_calls moved during arm B: {sorted(spc_seen)}"

    cert = json.loads(CERT.read_text())
    assert cert["pre_declared_margin"] == MARGIN_PP
    overall = pairing["overall"]

    if EXT:
        # End-of-run lifetime check: the whole extension must have been served by the
        # SAME process that served the primary and the null controls.
        sys.path.insert(0, str(HERE))
        from remote_policy import EngineClient
        c = EngineClient(THOR_HOST, THOR_PORT, connect_timeout_s=20)
        ping_end = c.call({"op": "ping"})
        c.close()
        assert ping_end["set_prompt_calls"] == 22, "server lifetime changed mid-extension"
        # Preserve the underpowered primary verbatim before the powered fields replace it.
        if "primary_gate_n30" not in cert:
            cert["primary_gate_n30"] = {
                k: cert[k] for k in (
                    "per_task_paired_results", "pairing_report", "armA_pc_success",
                    "armB_pc_success", "diff_pp", "n_pairs", "discordant", "mcnemar_p",
                    "ni_one_sided_p", "ni_verdict", "latency_stats", "armB_no_recapture",
                    "completed_utc") if k in cert}
            cert["primary_gate_n30"]["note"] = (
                "pre-declared primary (10 tasks x 3 eps): diff 0.00pp, honest but "
                "underpowered for the 5pp margin — superseded as the powered analysis "
                "by the declared extension below, margin unchanged")
        cert["power_extension_declared"]["status"] = "MEASURED"
        cert["power_extension_declared"]["ping_at_completion"] = ping_end
        cert["verdict_source"] = (f"power extension: {N_EPS} eps/task = "
                                  f"{overall['n_pairs']} pairs, margin unchanged")

    cert.update({
        "status": "MEASURED",
        "completed_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "server": smoke["server"],
        "null_controls": null,
        "per_task_paired_results": per_task,
        "pairing_report": pairing,
        "armA_pc_success": overall["ref_pc"],
        "armB_pc_success": overall["cand_pc"],
        "diff_pp": overall["diff_pp"],
        "n_pairs": overall["n_pairs"],
        "discordant": {"cand_wins_n01": overall["n01_cand_wins"],
                       "cand_loses_n10": overall["n10_cand_loses"]},
        "mcnemar_p": overall["mcnemar_p"],
        "ni_one_sided_p": overall["ni_one_sided_p"],
        "ni_verdict": "NON-INFERIOR" if overall["non_inferior"] else "NOT NON-INFERIOR",
        "latency_stats": lat,
        "armB_no_recapture": {"set_prompt_calls_constant": len(spc_seen) <= 1,
                              "prompt_lens_served": sorted(prompt_lens)},
        "confounds_closed_vs_pilot": [
            "state-blindness: exact PaliGemma ids (state included) cross the wire per "
            "chunk into per-length capture buckets; state-liveness null gate proves the "
            "state suffix changes the served action",
            "double-unnormalization: server returns RAW normalized actions; the "
            "checkpoint's unnormalizer postprocessor runs locally in BOTH arms",
            "degenerate verdict: v044 produces real successes, so the comparator has "
            "variance to test against",
        ],
        "walls_carried_forward": [
            "renderer nondeterminism (MuJoCo EGL): closed-loop bit-gates undefined; "
            "pairing is statistical over (suite, task, episode) — measured in the pilot",
            "fp8 calibration is per server lifetime (per-length, real frames, at server "
            "startup); ONE lifetime served every arm-B episode and null control",
            "chunk geometry 10 both arms — NOT comparable to pi05_compress chunk-50 runs",
            "third camera: local arm masks a padded empty camera, engine is num_views=2 — "
            "structurally different inputs, same information content",
            "engine pads odd prompt lengths by duplicating the last token embedding "
            "(Se even for fp8 GEMM); the local arm has no such duplicate — known, "
            "T1-accepted structural difference",
        ],
    })
    CERT.write_text(json.dumps(cert, indent=1))
    print(f"certificate -> {CERT}")
    print(f"  armA {overall['ref_pc']}% armB {overall['cand_pc']}% "
          f"diff {overall['diff_pp']}pp mcnemar_p {overall['mcnemar_p']} "
          f"-> {cert['ni_verdict']} at margin {MARGIN_PP}pp")
    progress(f"certificate written: armA {overall['ref_pc']}% armB {overall['cand_pc']}% "
             f"{cert['ni_verdict']} (mcnemar_p={overall['mcnemar_p']})")
    return cert


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("smoke",):
        phase_smoke()
    if phase in ("null", "all"):
        phase_smoke()
        phase_null()
    if phase in ("smokeB",):
        phase_smokeB()
    if phase in ("declare",):
        phase_declare()
    if phase in ("paired", "all"):
        phase_declare()
        phase_paired()
    if phase in ("report", "all"):
        phase_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
