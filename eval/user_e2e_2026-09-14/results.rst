Public API results on Jetson Thor
================================

P50 latency in milliseconds; ratios are eager p50 divided by the measured p50.

.. csv-table:: Same declared sampling policy
   :header: "Family", "Eager ms", "Default route", "Default ms", "Eager/default", "Fastest Runtime route (cell)", "Runtime ms", "Eager/Runtime"

   "pi05","408.4","native; ceiling BITEXACT","306.2","1.33x","FP8; ceiling NUMERIC (pi05-runtime_selected)","51.7","7.91x"
   "LingBot-VLA 4B","655.5","native; ceiling BITEXACT","249.2","2.63x","FP8; ceiling NUMERIC (vla4-runtime_update)","217.8","3.01x"
   "LingBot-VLA V2 6B","730.1","native; ceiling BITEXACT","737.5","0.99x","FP8; ceiling NUMERIC (vla2-runtime_update)","388.7","1.88x"
   "GR00T N1.7","138.1","native; ceiling BITEXACT","111.1","1.24x","native; ceiling BITEXACT (groot-runtime_selected)","110.9","1.24x"
   "Cosmos3 Edge","3387.9","native; ceiling BITEXACT","2556.1","1.33x","native; ceiling NUMERIC (edge-runtime_selected)","1041.5","3.25x"
   "Cosmos3 Nano","10259.5","native; ceiling BITEXACT","8585.3","1.20x","native; ceiling NUMERIC (nano-runtime_selected)","4800.4","2.14x"
   "LingBot-VA","15105.7","native; ceiling BITEXACT","5632.1","2.68x","FP8; ceiling NUMERIC (va-runtime_selected)","2889.5","5.23x"
   "DreamZero","23599.0","native; ceiling BITEXACT","23560.3","1.00x","FP8; ceiling NUMERIC (dreamzero-runtime_selected)","21035.7","1.12x"

.. csv-table:: Changed operating points (separate)
   :header: "Cell", "Route", "Actual schedule", "P50 ms", "Eager/p50"

   "va-2v4a-native","native; ceiling BEHAVIORAL","2 video / 4 action steps","772.4","19.56x"
   "va-2v4a-fp8","FP8; ceiling BEHAVIORAL","2 video / 4 action steps","457.3","33.03x"
   "dreamzero-dynamic-native","native; ceiling BEHAVIORAL","16 solver updates; dynamic reuse","13281.9","1.78x"
   "dreamzero-dynamic-fp8","FP8; ceiling BEHAVIORAL","16 solver updates; dynamic reuse","11843.4","1.99x"

Ratios above 1 mean lower latency than eager; a default need not be faster.
Fastest Runtime is the minimum measured eligible Runtime p50; eager stays a separate reference.
This is neither task-quality certification nor a recommendation.

Ceilings limit permitted Runtime transforms; they do not assert action equality against eager.
Eager disables TorchDynamo; Runtime retains upstream compile behavior. See report.json for action differences.
One Thor, same checkpoint revisions and sampling policy within each main row; precision may differ.
Stateless: five warmups then 20 measured calls with resets and alternating prompts.
History: one warmup episode, then six three-cycle episodes; p50 uses 12 continuation calls.
Recorded cameras and synthetic states are prepared; public preprocessing/predict/commit is timed.
Setup, first calls and pi05's separate 51-call queue remain in the full report.
DreamZero eager setup includes verification of 2,146 loaded tensors.
Changed operating-point ratios do not isolate infrastructure gains or establish task quality.
Prompt updates enter the main minimum only after their corresponding audited gate passes.
The full 31-cell report retains all original controls, including slower FP8 routes.
Updates excluded from the minimum: none.

Report SHA-256: 2a0a5e8f85b7d5b1362d1a98b4c2ef910678e80582b43ac6768ee0ec2ad8046c
Prompt gate SHA-256: c68135317f69564725c05f4b676d4b59d1b7116b8034f9014c4f2f3bfba30aa1
