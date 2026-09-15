# Model / simulator adapter contracts

List supported explicit contracts without importing a model or simulator:

```sh
python -m benchmarks.vla adapters
```

The catalog is `config/adapters.json`. It contains LingBot-VA LIBERO/RoboTwin,
pi05 LIBERO, LingBot-VLA 4B/V2 RoboTwin and the GR00T N1.7 LIBERO fine-tune.
See [Simulator evaluation](SIMULATOR_EVALUATION.md) for lifecycle and coverage. Sharing a backbone does not make cameras, action
geometry or checkpoints interchangeable. Contract integration is not evidence of
model quality or numerical equivalence.

For a suite declaring `protocol.bridge`, plan construction resolves that bridge,
checks the checkpoint ID AND revision, backbone, suite, evaluation mode and driver
entrypoint, then embeds the complete contract and its SHA-256 under
`request.adapter`. This is covered by the request hash. The migrated drivers
check the binding before evaluation (VA also checks before scene preparation). Changed contracts require a
new plan, even if the checkpoint ID remains unchanged. Driver revisions also hash
the contract catalog and validation code.

Supported entrypoints are `python -m <declared module>` or `python
/absolute/path/to/the/local/driver.py`. Opaque wrappers are refused because the
planner cannot establish which adapter they execute. Existing suites with no
bridge continue to use their existing driver checks. This is incremental migration,
not a claim that the catalog covers every legacy benchmark path.

Each contract declares:

- Exact checkpoint allowlist and backbone; simulator and compatible suite IDs.
- Camera keys, image preprocessing and instruction source.
- Wire action shape, controller dimension, conversion semantics and digest encoding.
- Observed-history commit sizes and the evaluation timing mode.

A contract is not an executable conversion function or an accuracy certificate.
The implementation must enforce its semantics; live endpoint identity, frozen
scenes, normalization/config hashes and paired results remain required evidence.
No padding, camera substitution, action projection or checkpoint substitution is
performed automatically.

## Adding a checkpoint or model

1. Establish its simulator training compatibility, camera preprocessing, state
   requirements, action space/normalization, history and reset behavior. A checkpoint
   trained for another robot does not become compatible by changing its label.
2. Implement or extend the appropriate driver. A shared transport does not require a
   shared robot action decoder. Use a new protocol version for changed semantics.
3. Add its pinned checkpoint to the model registry and explicit adapter catalog;
   define the suite's bridge. Do not add an untested checkpoint to an existing
   allowlist merely because its tensor dimensions match.
4. Add refusal and protocol tests, freeze scenes, then run original versus candidate
   with matched inputs and noise. Start with a smoke profile before broader screening.
5. Declare the digest convention to the action reporter. Keep action equivalence,
   quality success, and latency/realtime evidence separate.

Currently the LIBERO VA driver also pins the official checkpoint in its own
validation. Supporting another checkpoint needs an implementation review as well
as a catalog entry. This prevents an allowlist edit from falsely advertising
plug-and-play compatibility. Arbitrary checkpoints cannot yet be dropped into the
benchmark without an adapter.

Old frozen runs remain readable for analysis. Do not retrofit new adapter fields
into an old request or overwrite its plan: that would change its evidence identity.

## pi05 LIBERO schedule protocol

The shipped pi05 sweep specs declare `driver.adapter_id=pi05-libero-schedule-v1`.
For closed-loop jobs the planner binds the corresponding bridge and evaluation
mode, refusing conflicts with suite declarations. Latency jobs stay on their existing
path. The driver rejects missing/changed bindings before loading a checkpoint.

This protocol intentionally preserves the previous evaluation setup: 50-action
policy chunks, `n_action_steps=10`, TF32 enabled, cuDNN benchmark enabled and
compilation disabled. These settings are explicit in the contract. They are not a
claim of original-checkpoint execution semantics (the checkpoint defaults to
`n_action_steps=50`) or of bit-exact execution. NFE remains the arm's declared
schedule; this migration does not initiate a reduced-NFE experiment.

The pinned config declares two 256x256 RGB inputs, an 8D state and 7D controller
actions, plus the policy's configured empty camera. LeRobot's checkpoint processors
remain responsible for normalization and image/action conversion.

pi05 preserves `executed_steps` and now requires a frozen scene manifest for live
closed-loop runs. Prepare it with `python -m benchmarks.vla.pi05_scenes
--prepare-plan <draft> --output <scenes.json>`, then bind its path and SHA-256 in
both arms' `operating_point.scene_manifest` and build a new plan. The helper pins
LIBERO source/assets, LeRobot source, package versions, renderer environment,
initial state/index, instruction and the complete raw reset observation digest.
It checks rollout's actual reset without performing an extra reset.

The standard reporter recognizes this protocol's canonical FP64 action digest
only when results carry verified scene provenance. This does not certify the
policy's original tensor dtype or internal state. Test the original against itself
before evaluating optimizations; TF32/cuDNN settings can affect repeatability.
The existing committed-revision requirement for live pi05 runs remains in force;
use an isolated committed execution snapshot when developing in a dirty workspace.

For pi05, model loading is additionally checked tensor by tensor against the local
checkpoint before evaluation. The older upstream loader can print a loading error
and still return a policy. Missing keys are accepted only when the model tensor
shares the exact storage, shape, stride and dtype of another verified checkpoint
key. Canonical controller action values are retained for first-divergence analysis.
