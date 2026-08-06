"""Training oracles: teacher/student/grid factories over a live backbone. LAYER 1 ONLY.

These are NOT adapters. A `BackendAdapter` states facts about a model and applies a plan to it; an
oracle wraps a built server so a training recipe can pull velocities out of it. `lingbot_velocity.py`
implements zero methods of the adapter protocol -- no `spec()`, no `sites()`, no `apply()` -- and lived
in `adapters/` only because that is where it was first written.

That mislabelling had a cost: it made `runtime/block_heads.py` importing a PDD training oracle look
like a runtime module importing an adapter, which is unremarkable, rather than the serving path
importing a training library, which is the one thing the project's organising principle forbids. See
AUDIT.md finding F1. The import still exists and is now visibly `from instinctwm.train...` inside
`runtime/`, which is the point -- make it ugly, then remove it (AUDIT.md Stage 1).

Nothing under `runtime/`, `planners/`, `executors/`, or `passes/` should import from here.
"""
