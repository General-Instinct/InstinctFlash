# Recorded simulator input

`va_eval_obs.npz` is the exact recorded RoboTwin camera/action archive used in this campaign. The benchmark consumes `frame0_0`, the first 12 rows of `jpeg_0`, and recorded `actions_0`; other recordings are retained to preserve the original artifact hash. States and prompts are constructed by the benchmark, so this is not a closed-loop rollout.

SHA-256: `d6f08f968287b78eadd0ef3001e90f0a46397b17294dbd2c46c854feb5b93172`. JPEG payloads are stored in NumPy object arrays, as in the original recording.
