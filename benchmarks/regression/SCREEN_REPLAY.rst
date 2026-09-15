Recorded Cosmos SCREEN input replay
==================================

``python -m benchmarks.regression.screen_replay`` replays the recorded RGB,
joint/gripper state, prompts, request seeds and six episode resets for one
historical Cosmos arm. It uses the original UniPC4/CFG3 native BF16 recipe and
the same cold-inclusive ``seeded_predict`` host timer. New runs retain their
own source, checkpoint, action and timing identities.

This is latency and raw-action replay. It cannot recreate the simulator's task
success: each arm follows its previously recorded observation trajectory.
The two historical tasks and six pairs per family remain a separate SCREEN,
without task-quality certification.

The helper source ships in the core wheel. Rendered observations are separate
inputs and are not included in that wheel. The original scenes use mixed
CC BY-NC-SA, CC BY-SA and additional asset terms; their public or commercial
redistribution has not been cleared. See the exact pinned RoboLab
`third-party notices <https://github.com/NVlabs/RoboLab/blob/9db0aaf09d9fe5d4f37b168320788258c7012463/THIRD_PARTY_NOTICES.md>`_
and `local scene-rights review <../../eval/public_release_2026-09-15/robolab_rendered_fixture_license_review_v1.json>`_.
These restrictions concern the RoboLab renders, not the distinct MIT-declared
RoboTwin recorded-input fixture used by the uniform reproduction runner.

For an authorized local copy of the original study, losslessly export all
316 requests into four NumPy archives. No pickle, simulator or model is loaded::

   python -m benchmarks.regression.screen_replay export \
     --study /path/to/timed_screen_v1 \
     --latency /path/to/timed_screen_latency_v1.json \
     --output /path/to/new-local-screen-fixture

The export validates the frozen timing aggregate, original wire/receipt hashes,
decoded observation hashes and complete reference actions. It prints the new
fixture manifest hash. Keep that exact hash with the fixture.

Activate the separately bootstrapped Cosmos environment and prepare its pinned
auxiliary assets using ``scripts/prepare_auxiliary_assets.py``. Prepare one arm
with its explicit fixture hash, then run it in a fresh installed process::

   python -m benchmarks.regression.screen_replay prepare \
     --fixture /path/to/local-screen-fixture --fixture-sha256 FIXTURE_MANIFEST_SHA256 \
     --cell edge-runtime_selected --bf16-library /path/to/libifl_bf16.so \
     --output /path/to/new-preparation
   python -m benchmarks.regression.screen_replay run \
     --prepared /path/to/new-preparation --output /path/to/new-replay

Other cells are ``edge-eager_native``, ``nano-eager_native`` and
``nano-runtime_selected``; omit ``--bf16-library`` for those cells. ``prepare``
downloads the exact primary checkpoint; ``run`` is offline and bounded, retains
errors without retry, and publishes a report only after all requests and the
native ledger close. ``report`` independently checks the saved completion/file
bindings and can display the same completed result again.

Generating new simulator episodes additionally requires RoboLab commit
``9db0aaf09d9fe5d4f37b168320788258c7012463``, its licensed assets, Isaac Sim,
the asset/renderer manifests, a separately admitted policy identity and the
original protocol. ``python -m benchmarks.vla.robolab_driver --help`` lists the
explicit inputs. The historical author-host launch files are retained evidence;
they are not a portable simulator installer. New simulation produces new
trajectories and must retain its own evidence and outcome identities.
