"""Inclusive CUDA-event spans for a diagnostic eager run, not a speed benchmark."""
from collections import defaultdict
import time
import torch

class CudaSpans:
    def __init__(self):
        self.enabled = False
        self.request = None
        self.patches = []
        self.pending = []
        self.rows = []

    def patch(self, obj, name, group, label):
        original = getattr(obj, name)
        owned = name in vars(obj)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tensor = next((x for x in args if isinstance(x, torch.Tensor)), None)
            row = dict(request=self.request, group=group, label=label,
                       input_shape=None if tensor is None else list(tensor.shape))
            begin.record()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                row['host_dispatch_ms'] = 1000*(time.perf_counter()-start)
                self.pending.append((row, begin, end))
        setattr(obj, name, wrapped)
        self.patches.append((obj, name, original, owned, wrapped))

    def install(self, model):
        self.patch(model, '_prepare_inference_data', 'prepare', 'prepare')
        self.patch(model, '_get_velocity', 'velocity', 'velocity')
        layers = model.net.language_model.model.layers
        assert len(layers) == 36
        for i, layer in enumerate(layers):
            self.patch(layer, 'forward', 'layers', str(i))
            self.patch(layer.self_attn, 'forward', 'attention', str(i))
            self.patch(layer.mlp, 'forward', 'text_mlp', str(i))
            self.patch(layer.mlp_moe_gen, 'forward', 'generation_mlp', str(i))

    def collect(self):
        # Caller must synchronize after the complete request.
        for row, begin, end in self.pending:
            row['cuda_span_ms'] = begin.elapsed_time(end)
            self.rows.append(row)
        self.pending.clear()

    def report(self):
        grouped = defaultdict(lambda: defaultdict(float))
        for row in self.rows:
            grouped[row['group']][row['request']] += row['cuda_span_ms']
        return dict(inclusive=True, cuda_spans_include_stream_idle_time=True,
                    nested_groups_must_not_be_added=True, layer_graphs=False,
                    rows=self.rows, mean_ms_per_request={k:sum(v.values())/len(v) for k,v in grouped.items()},
                    requests_per_group={k:len(v) for k,v in grouped.items()})

    def close(self):
        for obj, name, original, owned, wrapper in reversed(self.patches):
            if getattr(obj, name) is not wrapper:
                raise RuntimeError(f'Profiler wrapper replaced: {name}')
            if owned:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        self.patches.clear()
