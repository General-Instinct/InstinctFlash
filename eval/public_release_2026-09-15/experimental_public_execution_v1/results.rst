Experimental public SDE1 recipes
===============================

Three fresh public recipe executions passed on Jetson Thor on September 15, 2026.
These are BF16 NUMERIC latency SCREENs, separate from the full UniPC4/CFG3 headline
results. They do not certify task quality or robot control frequency.

.. csv-table:: Prediction p50 after warm-up
   :header-rows: 1

   "Recipe","p50 (ms)","Warm-up / measured requests"
   "Edge distilled seed12031, SDE1/CFG1","270.07","6 / 10"
   "Edge distilled seed12032, SDE1/CFG1","270.66","6 / 10"
   "Nano original weights, SDE1/CFG1","710.96","12 / 24"

All three use the complete SDE [1, 0] schedule, guidance scale 1, zero action
padding and one native action clock at 1000. Nano uses the original untrained
weights with persistent cache and Triton SwiGLU. These are distinct operating
points from each original full-schedule policy; the distilled Edge checkpoints
did not recover the historical UniPC4 quality profile.

The `completion <native_evidence_v1/completion.json>`_ binds all seven successful
preparation and execution subprocesses. Individual receipts retain timing samples
and full action arrays: `Edge12031 <native_evidence_v1/archives/edge-seed12031/run/receipt.json>`_,
`Edge12032 <native_evidence_v1/archives/edge-seed12032/run/receipt.json>`_,
`Nano <native_evidence_v1/archives/nano-original/run/receipt.json>`_.

An `independent CPU audit <independent_numerical_lifecycle_audit_v1.json>`_
recomputed all three medians and checked 68 complete float32 action arrays.
Its process/archive review also used local operational records; the public subset
retains the 65 numerical, configuration and archive records. Private administrative
paths in those records are provenance, not required public reproduction inputs.

Use the `public SDE1 guide <../../../examples/cosmos3_sde1/README.rst>`_ to prepare
the exact released overlays and original Nano recipe in a fresh environment.
