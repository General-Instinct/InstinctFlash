"""Freeze an explicit native LIBERO observation for paired GR00T startup."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.vla import groot_libero_driver as driver
from benchmarks.vla.groot_policy_server import startup_observation
from benchmarks.vla.util import ConfigurationError, sha256_file


def prepare(groot_root, libero_root, output, *, task=0, seed=9173):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_identity = driver.source_identity(libero_root, groot_root)
    suite, factory = driver.simulator(groot_root, 'libero_10')
    prompt = suite.get_task(task).language
    snapshots, digests = [], []
    for _ in range(2):
        env = driver.new_env(suite, factory, task, seed)
        try:
            obs = driver.reset(env, seed)
            digests.append(driver.observation_digest(obs))
            snapshots.append({k: np.ascontiguousarray(v) for k, v in obs.items()
                              if k.startswith(('video.', 'state.'))})
        finally:
            env.close()
    if digests[0] != digests[1]:
        raise ConfigurationError('native startup reset is not repeatable')
    path = output/'observation.npz'
    np.savez(path, **snapshots[0])
    loaded = startup_observation(path)
    for key, value in loaded.items():
        if value.dtype != snapshots[1][key].dtype or value.tobytes() != snapshots[1][key].tobytes():
            raise ConfigurationError('startup observation bytes changed on serialization')
    report = dict(purpose='calibration/startup only; exclude this task/seed from evaluation',
                  suite='libero_10', task=task, seed=seed, prompt=prompt,
                  observation_sha256=sha256_file(path), native_observation_digest=digests[0],
                  repeated_native_reset_byte_equal=True,
                  sources=source_identity,
                  preparation_source_sha256=sha256_file(Path(__file__)),
                  native_protocol='Gr00t LiberoEnv.reset(seed), no added settling or init-state replacement',
                  arrays={k:dict(shape=list(v.shape), dtype=v.dtype.str) for k,v in loaded.items()})
    (output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groot-root', required=True, type=Path)
    parser.add_argument('--libero-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--seed', type=int, default=9173)
    args = parser.parse_args()
    print(json.dumps(prepare(args.groot_root, args.libero_root, args.output,
                             task=args.task, seed=args.seed), indent=2))
