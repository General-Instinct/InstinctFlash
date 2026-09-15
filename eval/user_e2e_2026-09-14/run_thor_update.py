"""Finite serial public API sweep. Failures stop this queue; no automatic retry."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
EXTRA = {
    "pi05": {}, "vla4": {"LINGBOT_VLA_ROOT": "/home/guanming/lingbot-vla-repo"},
    "vla2": {"LINGBOT_VLA_V2_ROOT": "/home/guanming/lingbot-vla-v2-repo"},
    "groot": {"GR00T_ROOT": "/home/guanming/thorcol/Isaac-GR00T"},
    "edge": {}, "nano": {}, "va": {"LINGBOT_ROOT": "/home/guanming/lingbot-va"},
    "dreamzero": {"DREAMZERO_ROOT": "/home/guanming/thorcol/dreamzero"},
}
FIXTURE = "/home/guanming/ifl_eval/four_frameworks_20260913/source/eval/native_total_2026-09-10/fixtures/va_eval_obs.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="vla4-prompt-reuse")
    parser.add_argument("--cells", nargs="*")
    parser.add_argument("--matrix", default="matrix_update.json")
    args = parser.parse_args()
    status_path = ROOT / f"status-{args.batch}.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    matrix = json.loads((ROOT / args.matrix).read_text())
    installation = json.loads((ROOT / "installation_update.json").read_text())
    assert installation["ok"]
    cells = [c for c in matrix["cells"] if args.cells is None or c["id"] in args.cells]
    assert cells and (args.cells is None or {c["id"] for c in cells} == set(args.cells))
    status = dict(schema=1, status="queued", pid=os.getpid(), jobs=[], started=time.time(),
                  matrix_sha256=sha(ROOT / args.matrix), installation_sha256=sha(ROOT / "installation_update.json"))

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(status_path)

    def gpu_pids():
        return subprocess.check_output(["/usr/sbin/nvidia-smi", "--query-compute-apps=pid",
                                        "--format=csv,noheader,nounits"], text=True).strip()

    def verify():
        assert sha(ROOT / args.matrix) == status["matrix_sha256"]
        for path, expected in json.loads((ROOT / "installed_files_update.json").read_text()).items():
            assert sha(path) == expected, path

    save()
    child = None
    try:
        with open("/tmp/thor_gpu.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status["status"] = "running"
            save()
            for cell in cells:
                verify()
                assert not gpu_pids(), "Thor has another active GPU process"
                output = ROOT / cell["receipt"]
                assert not output.exists(), f"Refusing to overwrite {output}"
                output.parent.mkdir(parents=True, exist_ok=True)
                python = installation["environments"][cell["family"]]["python"]
                env = {k: v for k, v in os.environ.items() if not k.startswith("IFL_") and k not in
                       ("PYTHONPATH", "LINGBOT_CKPT", "DREAMZERO_ROOT", "LINGBOT_ROOT", "LINGBOT_VLA_ROOT", "LINGBOT_VLA_V2_ROOT", "GR00T_ROOT")}
                env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1",
                    PYTHONUNBUFFERED="1", PYTHONHASHSEED="0", MASTER_PORT="29824",
                    NO_ALBUMENTATIONS_UPDATE="1", DYNAMIC_CACHE_SCHEDULE="false", NUM_DIT_STEPS="8",
                    ENABLE_TENSORRT="false", TORCHDYNAMO_DISABLE="1" if cell["arm"] == "eager_native" else "0",
                    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
                    TRITON_PTXAS_BLACKWELL_PATH="/usr/local/cuda/bin/ptxas",
                    UV_OFFLINE="1",
                    PATH=str(Path(python).parent)+":"+installation["environments"][cell["family"]]["vendor_stack"]+"/bin:/home/guanming/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin",
                    **EXTRA[cell["family"]], **cell["expected_optimizer_environment"])
                command = [python, "-I", "-m", "benchmarks.regression.user_e2e", "--matrix", str(ROOT / args.matrix),
                           "--cell", cell["id"], "--output-root", str(ROOT), "--fixture", FIXTURE]
                job = dict(cell=cell["id"], status="running", command=command, started=time.time(),
                           environment={k: v for k, v in env.items() if k.startswith("IFL_") or k in
                            ("DYNAMIC_CACHE_SCHEDULE", "NUM_DIT_STEPS", "TORCHDYNAMO_DISABLE", "ENABLE_TENSORRT")})
                status["jobs"].append(job)
                save()
                with output.with_suffix(".log").open("x") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    job["pid"] = child.pid
                    save()
                    child.wait(timeout=3600)
                job.update(exit_code=child.returncode, ended=time.time())
                receipt = json.loads(output.read_text()) if output.exists() else {}
                job["status"] = "passed" if child.returncode == 0 and receipt.get("ok") else "failed"
                job["error"] = receipt.get("error")
                save()
                assert job["status"] == "passed", f"Stopped at {cell['id']}"
            verify()
            status.update(status="passed", ended=time.time())
            save()
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status.update(status="failed", error=traceback.format_exc(), ended=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
