"""Run one frozen VLA4 update cell after the existing queue passes; never retry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent


def main():
    receipt = ROOT / "followthrough_update_status.json"
    assert not receipt.exists(), receipt
    status = dict(status="waiting", pid=os.getpid(), started=time.time(),
                  predecessor="status-path-recovery.json", predecessor_pid=232622)

    def save():
        temp = receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2)+"\n")
        temp.replace(receipt)

    save()
    try:
        while True:
            predecessor = json.loads((ROOT / status["predecessor"]).read_text())
            assert predecessor["pid"] == status["predecessor_pid"]
            assert predecessor["status"] != "failed", "Predecessor failed; no continuation"
            if predecessor["status"] == "passed":
                break
            assert time.time() - status["started"] < 10800, "Wait bound exceeded"
            time.sleep(15)
        # Terminal status is written before the predecessor releases its lock.
        time.sleep(3)
        status.update(status="running", predecessor_sha256=hashlib.sha256(
            (ROOT / status["predecessor"]).read_bytes()).hexdigest())
        save()
        command = ["/home/guanming/frt_env/bin/python", str(ROOT / "run_thor_update.py"),
                   "--cells", "vla4-runtime_update"]
        subprocess.run(command, cwd=ROOT, check=True, timeout=3700)
        status.update(status="passed", ended=time.time())
    except BaseException as error:
        status.update(status="failed", ended=time.time(), error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
