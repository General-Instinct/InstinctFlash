"""Read-only GPU admission snapshot for experimental timing; never stop other jobs.

Call before AND after a timing arm and retain both snapshots. This is a point-in-
time check, not proof of exclusivity over the entire measurement interval.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import subprocess
import time


def gpu_snapshot(gpu="0", *, run=subprocess.run):
    def query(options):
        result = run(["nvidia-smi", *options, "--format=csv,noheader,nounits"],
                     check=True, capture_output=True, text=True, timeout=15)
        return list(csv.reader(io.StringIO(result.stdout)))
    rows = query(["-i", str(gpu), "--query-gpu=uuid,name,utilization.gpu,memory.used,clocks.sm,temperature.gpu"])
    if len(rows) != 1 or len(rows[0]) != 6:
        raise RuntimeError("Expected exactly one GPU with a complete nvidia-smi snapshot")
    uuid, name, usage, memory, clocks, temperature = (cell.strip() for cell in rows[0])
    processes = query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory"])
    matching = []
    for row in processes:
        if len(row) != 3:
            raise RuntimeError("Malformed GPU process snapshot")
        if row[0].strip() == uuid:
            matching.append({"pid": int(row[1].strip()), "memory_MiB": row[2].strip()})
    return {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_uuid": uuid, "name": name, "utilization_percent": int(usage),
            "memory_MiB": int(memory), "sm_clock_MHz": int(clocks),
            "temperature_C": int(temperature), "processes": matching}


def require_uncontended(snapshot, *, own_pids=(), max_idle_utilization=5):
    own = set(own_pids)
    other = [row["pid"] for row in snapshot["processes"] if row["pid"] not in own]
    if other:
        raise RuntimeError(f"GPU timing refused: other compute processes are present: {other}")
    if snapshot["utilization_percent"] > max_idle_utilization:
        raise RuntimeError("GPU timing refused: device is not idle; let outstanding work finish first")


def idle_admission(gpu="0", *, snapshot_fn=gpu_snapshot, sleep=time.sleep):
    """Allow a completed arm's utilization sample to age out, not competitors.

    No CUDA work is submitted while waiting. A live compute process always
    refuses immediately. Retain all admission samples in the test report.
    """
    samples = [snapshot_fn(gpu)]
    for _ in range(3):
        if samples[-1]["processes"] or samples[-1]["utilization_percent"] <= 5:
            break
        sleep(1.0)
        samples.append(snapshot_fn(gpu))
    require_uncontended(samples[-1])
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="Physical nvidia-smi index or GPU UUID, not a remapped Torch ordinal")
    args = parser.parse_args()
    snapshot = gpu_snapshot(args.gpu)
    print(json.dumps(snapshot, indent=2), flush=True)
    try:
        require_uncontended(snapshot, own_pids=(os.getpid(),))
    except RuntimeError as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
