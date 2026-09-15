"""Owned operational queue of installed public Thor commands, no implicit retries."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    base = Path(config["base"])
    root = base / config["output"]
    root.mkdir(exist_ok=False)
    record = {"status": "running", "started_unix": time.time(), "jobs": []}
    try:
        for job in config["jobs"]:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"], text=True).strip()
            if apps:
                raise RuntimeError("unexpected GPU compute process before next job")
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            for name in job.get("activation", []):
                for line in (base / name).read_text().splitlines():
                    parts = shlex.split(line)
                    if not parts or parts[0].startswith("#"):
                        continue
                    if parts[0] == "unset":
                        for key in parts[1:]:
                            env.pop(key, None)
                    elif parts[0] == "export" and len(parts) == 2:
                        key, value = parts[1].split("=", 1)
                        if key == "PATH":
                            value = value.replace("${PATH}", env.get("PATH", "")).replace("$PATH", env.get("PATH", ""))
                        if "$" in value or "`" in value:
                            raise RuntimeError("unsupported activation expansion")
                        env[key] = value
                    else:
                        raise RuntimeError("unsupported activation syntax")
            env.update(job.get("environment", {}))
            command = [job["python"], "-I", "-m", job["module"], *job["arguments"]]
            row = {"id": job["id"], "started_unix": time.time(), "command": command, "activation": job.get("activation", [])}
            keys = ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_COMPILE_THREADS")
            row["applied_compiler_environment"] = {k: env[k] for k in keys if k in env}
            row["assembler_sha256"] = {
                k: hashlib.sha256(Path(env[k]).read_bytes()).hexdigest()
                for k in keys[:2] if k in env
            }
            record["jobs"].append(row)
            write(root / "progress.json", record)
            with (root / (job["id"] + ".log")).open("x") as log:
                done = subprocess.run(command, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=job.get("timeout", 5400))
            row.update(returncode=done.returncode, completed_unix=time.time())
            receipt = json.loads((base / job["receipt"]).read_text())
            row["receipt_status"] = receipt["status"]
            if done.returncode or receipt["status"] != "passed":
                raise RuntimeError("public command failed; original evidence retained: " + job["id"])
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=repr(error))
    record["completed_unix"] = time.time()
    write(root / "completion.json", record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
