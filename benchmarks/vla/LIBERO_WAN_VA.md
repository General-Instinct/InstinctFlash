# LingBot-VA × LIBERO-Long

`wan_va_libero_driver.py` implements the pinned official LIBERO-Long checkpoint:
`robbyant/lingbot-va-posttrain-libero-long@0e89d1e753019988aba484e8da2dc0810e264d9f`.
It is separate from the pi05 LIBERO driver and the RoboTwin VA bridge.

The first profile covers the complete `libero_10` suite (10 tasks), 2 initial states per
task, two arms, 40 episodes. It is screening only. The inherited -0.05 margin is schema
plumbing, NOT a product acceptance budget; fewer than 100 pairs cannot certify and the
profile is always non-reportable. Quality reports do not certify byte equivalence.

The protocol follows upstream `evaluation/libero/client.py`: 128×128 vertically flipped
cameras, five zero-action settling steps, action shape (7,4,4), skip the first frame on the
first chunk, commit 12 observed frames initially and 16 thereafter. A successful terminal
chunk is not committed. The outer loop checks timestep <800; as upstream, a final chunk
can cross that limit. No clean/randomized duplication is added to LIBERO.

Prepare scenes once before either arm: pin simulator source/assets by bytes and revision,
choose explicit init-state indices, freeze prompt, state and post-settling RGB digest. The
simulator seed is frozen too. A mismatched initial observation aborts instead of scoring
another scene. All jobs run in separate processes with identity-checked remote model servers.
Malformed actions or infrastructure failures leave incomplete results; they are never dropped
from a denominator or interpreted as a success.

The current campaign deliberately preserves original 20V/50A, guidance5/1, grid shifts5/0.05
and bf16. It refuses unknown passes and numeric/behavioral variants. The allowed candidates
are FSDP/allocator/debug elision, conditioning prefill and ring KV. These are exact-intended,
not automatically proven exact on this new checkpoint. The control must have no passes.
Both endpoints must attest identical checkpoint bytes and all execution settings other than
that pass list. Use `--legacy-passes` on both endpoints to disable default generic rewrites.
Do not enable NDHWC, FP8, reduced NFE or guidance changes in this campaign.

Use `serve_variant.py --config-name libero --benchmark-receipt <receipt>` with the explicit
`LINGBOT_CKPT` snapshot path. The benchmark entrypoint supplies that path to upstream's LIBERO
configuration (which otherwise contains a placeholder). The model's existing loader selects
Torch attention; the downloaded snapshot is not edited. A second endpoint uses the same
command plus the selected candidate flags. Receipts identify weight bytes, source, packages,
protocol, execution settings and per-episode seeding support.

Construct arms following `ROBOTWIN.md`, changing the driver command to this module, passing
`LIBERO_ROOT`, `LIBERO_CONFIG_PATH`, `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, `LP_NUM_THREADS=1`, `PYTHONHASHSEED=0`, and the intended
simulator interpreter. `LIBERO_CONFIG_PATH/config.yaml` must point at the pinned checkout's
assets, init_files and bddl_files, not a user's unrelated default ~/.libero tree.
Use `make_plan(arms, output, simulator_root)` to write a draft and registry snapshot;
`--prepare-plan draft.json --output scenes.json` runs scene preparation. Then call
`make_plan(..., scenes='scenes.json')` using real endpoint receipts, and use standard
`benchmarks.vla --registry plan.registry.json run --plan plan.json --output run` and `report`.
Existing plans refuse replacement. Code changes require a new plan.

A result includes success, cycles, executed steps, action-byte digest, scene provenance and
transport timings. Comparing the paired action digests certifies ONLY observed executed
controller actions on those scenes, not all hidden state, arbitrary inputs or other devices.
Persistent KV/latent equivalence and full strict-policy enforcement in the general runtime
remain separate work from this explicitly constrained benchmark path.

Local rendering validation found nondeterministic pixels on H100 EGL even for consecutive
renders with identical physical/model/texture state. Single-thread OSMesa reproduced bytes
across fresh environments, so this campaign explicitly uses OSMesa. This is a separately
pinned rendering configuration, not a claim of identical pixels to EGL or a published score.
Simulator package versions and rendering environment are bound into the scene manifest.
