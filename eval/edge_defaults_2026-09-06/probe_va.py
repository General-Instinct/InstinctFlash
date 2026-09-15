"""Native-schedule Thor stress replay; phase instrumentation is excluded from latency samples.

Inputs are decoded real observation frames replayed cyclically past ring saturation.
This is a fixed-input systems/parity experiment, not simulator success-rate evaluation.
Run one arm per torch.distributed process, with the existing LingBot-VA environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from instinctflash.runtime.lingbot_install import (
    import_lingbot_server, install_conditioning_prefill,
    install_debug_dump_elision, install_fsdp_elision,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--terminal-elision', action='store_true')
    p.add_argument('--action-graphs', action='store_true')
    p.add_argument('--cycles', type=int, default=48)
    p.add_argument('--runs', type=int, default=2)
    a = p.parse_args()
    if a.cycles < 40 or a.runs < 2:
        p.error('at least 40 cycles and two reset episodes are required')
    if a.output.exists():
        p.error('refusing to overwrite evidence')
    a.output.mkdir(parents=True)
    inputs = np.load(a.input, allow_pickle=False)
    prompt = str(inputs['prompt'])
    S = import_lingbot_server()
    cfg = S.VA_CONFIGS['robotwin'].copy()
    # Preserve EasyDict attribute access.
    from easydict import EasyDict
    cfg = EasyDict(cfg)
    cfg.save_root = str(a.output / 'visualization')
    Path(cfg.save_root).mkdir()
    S.init_distributed(1, int(os.getenv('LOCAL_RANK', '0')), 0)
    cfg.rank, cfg.local_rank, cfg.world_size = 0, 0, 1
    assert (cfg.num_inference_steps, cfg.action_num_inference_steps) == (25, 50)
    install_fsdp_elision(S)
    torch.cuda.empty_cache = lambda: None
    server = S.VA_Server(cfg)
    from instinctflash.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(S, type(server))
    list(install_conditioning_prefill(S, type(server)))
    list(install_debug_dump_elision(S))
    # Preserve native convolutions: P007 layout selection is NUMERIC and outside this arm.
    graph = None
    if a.action_graphs:
        # Experimental selection around the existing host-bookkeeping-safe graph executor.
        # Only action denoise repeats amortize capture here; commits remain eager.
        import types
        from instinctflash.passes.lingbot.graph_capture import GraphBlockStack
        graph = GraphBlockStack(max_graphs=2)
        graph.install(S, type(server))
        graphed = server.transformer._iwm_stack
        action_tokens = cfg.frame_chunk_size * cfg.action_per_frame
        def selective(model, hidden, text, tproj, rot, update_cache, cache_name):
            if hidden.shape[1] == action_tokens and update_cache == 0:
                return graphed(hidden, text, tproj, rot, update_cache, cache_name)
            out = graph._stack_fn(model, hidden, text, tproj, rot, update_cache, cache_name)
            graph._commit_all(model, hidden, update_cache, cache_name)
            return out
        server.transformer._iwm_stack = types.MethodType(selective, server.transformer)
    if a.terminal_elision:
        from instinctflash.passes.lingbot.action_terminal_elision import ActionTerminalForwardElision
        ActionTerminalForwardElision().install(S, type(server))
    events, phase = [], {'enabled': False, 'request': ''}
    orig = server.transformer.forward
    def forward(*args, **kwargs):
        if not phase['enabled']:
            return orig(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        t = time.perf_counter()
        result = orig(*args, **kwargs)
        host_ms = (time.perf_counter() - t) * 1000
        end.record()
        events.append((phase['request'], bool(kwargs.get('action_mode', False)),
                       int(kwargs.get('update_cache', 0)), start, end, host_ms))
        return result
    server.transformer.forward = forward
    cams = list(cfg.obs_cam_keys)
    obs = {k: inputs['initial'][i] for i, k in enumerate(cams)}
    frames = inputs['frames']
    records, actions = [], []
    for run in range(a.runs):
        torch.manual_seed(20260906)
        server.infer(dict(reset=True, prompt=prompt, save_visualization=False))
        offset = 0
        for cycle in range(a.cycles):
            torch.manual_seed(50100 + cycle)
            # Instrument only selected cycles of the last episode. Never pool these with timings.
            phase['enabled'] = run == a.runs - 1 and cycle in (2, 8, 35, 36, 40, 47)
            events.clear()
            kfs = [{k: frames[(offset + f) % len(frames), i] for i, k in enumerate(cams)}
                   for f in range(4 if cycle == 0 else 8)]
            offset += len(kfs)
            torch.cuda.synchronize()
            t = time.perf_counter()
            phase['request'] = 'infer'
            action = server.infer(dict(obs=[obs], prompt=prompt, save_visualization=False))['action']
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            phase['request'] = 'commit'
            server.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                              save_visualization=False, state=action))
            torch.cuda.synchronize()
            commit_ms = (time.perf_counter() - t) * 1000
            phases = {}
            for req, action_mode, update, start, end, host_ms in events:
                key = f'{req}/{"action" if action_mode else "video"}/update{update}'
                bucket = phases.setdefault(key, {'calls': 0, 'stream_ms': 0., 'host_submit_ms': 0.})
                bucket['calls'] += 1
                bucket['stream_ms'] += start.elapsed_time(end)
                bucket['host_submit_ms'] += host_ms
            sig = server.transformer.blocks[0].attn1._iwm_ring_signature(server.cache_name)
            arr = np.asarray(action).copy()
            if not np.isfinite(arr).all():
                raise RuntimeError("nonfinite action in native stress replay")
            actions.append(arr)
            row = dict(run=run, cycle=cycle, instrumented=phase['enabled'],
                       infer_ms=infer_ms, commit_ms=commit_ms, total_ms=infer_ms+commit_ms,
                       phases=phases, ring_signature=sig,
                       action_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                       graph_captures=graph.n_captures if graph else 0,
                       graph_replays=graph.n_replays if graph else 0,
                       graph_failure=graph.failed if graph else None)
            records.append(row)
            with (a.output/'cycles.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
    np.savez_compressed(a.output/'actions.npz', actions=np.stack(actions))
    result = dict(complete=True, schedule={'video':25,'action':50}, guidance='native',
                  terminal_elision=a.terminal_elision, action_graphs=a.action_graphs,
                  cycles=a.cycles, runs=a.runs,
                  input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                  numeric_environment=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  scope='Repeated real frames; fixed-input saturation stress, not closed-loop accuracy. CUDA event intervals include stream idle time; they are not kernel busy time.')
    if a.terminal_elision:
        from instinctflash.passes.lingbot.action_terminal_elision import controller_of
        ctl = controller_of(server.transformer)
        result['elision_controller'] = {k:v for k,v in vars(ctl).items() if isinstance(v,(str,int,float,bool,type(None)))}
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
