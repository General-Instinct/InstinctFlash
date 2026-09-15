"""48-cycle public Runtime stress test using explicitly repeated recorded windows.

This is synthetic history stress, not a physically continuous robot episode.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from instinctflash import Runtime
from instinctflash.runtime.lingbot_install import import_lingbot_server

root = Path('/home/guanming/ifl_eval/thor_precision_completion_20260909')
family, precision = sys.argv[1:3]
assert family in ('libero', 'robotwin')
assert precision in ('native', 'fp8')
dest = root / f'va-saturated-stress-{family}-{precision}'
assert not dest.with_suffix('.json').exists(), 'Preserve prior receipts'
report = dict(precision=precision, family=family, scope='Paired 48-cycle synthetic-history stress; cyclic recorded windows, no closed-loop quality or sustained-tail certificate')
loop = None
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ['LINGBOT_ROOT'] = '/home/guanming/lingbot-va'
    cfg = import_lingbot_server(os.environ['LINGBOT_ROOT']).VA_CONFIGS[family]
    cameras = tuple(cfg.obs_cam_keys)
    if family == 'libero':
        data = np.load(root / 'va-libero-observations.npz')
        frames = [{k: data[k][i].copy() for k in cameras} for i in range(29)]
        prompt = str(data['prompt'])
        first = np.concatenate((np.zeros((4, 7), dtype=np.float32), data['executed_actions'][:12]))
        executed = [first.reshape(4, 4, 7).transpose(2, 0, 1).copy(),
                    data['executed_actions'][12:28].reshape(4, 4, 7).transpose(2, 0, 1).copy(), None]
        observations = [[frames[0]], frames[1:13], frames[13:29]]
        shape, video_steps = (7, 4, 4), 20
    else:
        import io
        from PIL import Image
        data = np.load('/home/guanming/thor_va_engine/va_eval_obs.npz', allow_pickle=True)
        def decode(values):
            return {k: np.asarray(Image.open(io.BytesIO(bytes(v))).convert('RGB')) for k, v in zip(cameras, values)}
        frames = [decode(data['frame0_0'])] + [decode(v) for v in data['jpeg_0'][:12]]
        prompt = 'Use the right arm for red block, then the left arm for green block, and the right arm for blue block, arranging them left to right.'
        executed = [data['actions_0'][0].copy(), data['actions_0'][1].copy(), None]
        observations = [[frames[0]], frames[1:5], frames[5:13]]
        shape, video_steps = (16, 2, 16), 25
    # Keep the real first-history geometry, then cycle the full recorded window.
    # Both precision arms consume exactly the same observations/executed actions.
    observations = observations[:2] + [observations[2]] * 46
    executed = [executed[0]] + [executed[1]] * 47
    outputs, calls = {}, []
    start = time.perf_counter()
    loop = Runtime.from_pretrained(root / f'runtime-{family}-checkpoint', precision=precision)
    loop.reset(prompt=prompt)
    report['setup_seconds'] = time.perf_counter() - start
    if precision == 'native':
        native = loop._backend._ensure()._server
    else:
        bridge = loop._backend._loop._loop._server
        native = bridge.native
        report['fp8_declaration'] = bridge.frontend.declaration()
    assert native.job_config.num_inference_steps == video_steps
    assert native.job_config.action_num_inference_steps == 50
    report['nfe'] = dict(video=video_steps, action=50)
    report['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    report['torch'] = torch.__version__
    from instinctflash.adapters import lingbot_va
    from instinctflash.runtime import wan_va_engine
    report['source_sha256'] = {m.__name__: hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
                              for m in (lingbot_va, wan_va_engine)}

    def cache_state():
        if precision == 'fp8':
            e = bridge.frontend
            return dict(frame=e.frame_st_id, capacity=e.pool_slots,
                        live=e.slab.tail-e.slab.head, head=e.slab.head, tail=e.slab.tail)
        result = {}
        for name, cache in native.transformer.blocks[0].attn1.attn_caches.items():
            if cache is not None and cache.get('mask') is not None:
                result[name] = dict(capacity=cache['mask'].numel(),
                                    live=int(cache['mask'].sum().item()))
        return result

    def predict(label, obs, cycle, executed):
        torch.manual_seed(1800 + cycle)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = loop.predict({'obs': obs}, executed_action=executed)['action']
        torch.cuda.synchronize()
        assert out.shape == shape and np.isfinite(out).all()
        elapsed_ms = (time.perf_counter()-start)*1000
        outputs[label] = out.copy()
        calls.append(dict(label=label, ms=elapsed_ms, cache=cache_state(),
                          frame=native.frame_st_id))
        np.savez_compressed(dest.with_suffix('.npz'), **outputs)
        print(json.dumps(calls[-1]), flush=True)
        return out

    resets = []
    for episode in range(2):
        start = time.perf_counter()
        loop.reset(prompt=prompt)
        torch.cuda.synchronize()
        resets.append((time.perf_counter()-start)*1000)
        for cycle, obs in enumerate(observations):
            predict(f'episode_{episode}_cycle_{cycle}', obs, cycle, executed[cycle])
            calls[-1].update(episode=episode, cycle=cycle,
                             phase='warmup' if episode == 0 else 'measured')
    for episode in range(1, 2):
        for cycle in range(48):
            assert outputs[f'episode_0_cycle_{cycle}'].tobytes() == outputs[f'episode_{episode}_cycle_{cycle}'].tobytes()
    np.savez_compressed(dest.with_suffix('.npz'), **outputs)
    measured = [c['ms'] for c in calls if c['phase'] == 'measured']
    report.update(ok=True, family=family, calls=calls, reset_ms=resets,
                  scope='Paired checkpoint-native Runtime with 48-cycle synthetic history sequences, original schedule; first episode warmup, second measured. Initial recorded history then repeated full recorded observation/action window. This is saturation stress, not a continuous real robot episode, sustained-tail or closed-loop quality certificate.',
                  repeated_episode_byte_equal=True, p50_ms=float(np.percentile(measured, 50)),
                  p99_ms=float(np.percentile(measured, 99)),
                  per_cycle_p50_ms={str(i):float(np.median([c['ms'] for c in calls if c['phase']=='measured' and c['cycle']==i])) for i in range(48)},
                  actions_sha256=hashlib.sha256(dest.with_suffix('.npz').read_bytes()).hexdigest())
except Exception as error:
    report.update(ok=False, error=repr(error), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    if loop is not None:
        loop.close()
    dest.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
if not report.get('ok'):
    raise SystemExit(1)
