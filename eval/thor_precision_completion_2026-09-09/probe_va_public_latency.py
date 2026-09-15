"""Independent public-API input controls, preserving LIBERO's native schedule."""
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
dest = root / f'va-public-latency-{family}-{precision}'
assert not dest.with_suffix('.json').exists(), 'Preserve prior receipts'
report = dict(precision=precision, family=family, scope='Paired early-history native/FP8 public Runtime latency; no closed-loop quality or sustained latency certificate')
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

    def predict(label, obs, cycle, executed):
        torch.manual_seed(1800 + cycle)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = loop.predict({'obs': obs}, executed_action=executed)['action']
        torch.cuda.synchronize()
        assert out.shape == shape and np.isfinite(out).all()
        outputs[label] = out.copy()
        calls.append(dict(label=label, ms=(time.perf_counter()-start)*1000,
                          frame=native.frame_st_id))
        print(json.dumps(calls[-1]), flush=True)
        return out

    resets = []
    for episode in range(4):
        start = time.perf_counter()
        loop.reset(prompt=prompt)
        torch.cuda.synchronize()
        resets.append((time.perf_counter()-start)*1000)
        for cycle, obs in enumerate(observations):
            predict(f'episode_{episode}_cycle_{cycle}', obs, cycle, executed[cycle])
            calls[-1].update(episode=episode, cycle=cycle,
                             phase='warmup' if episode == 0 else 'measured')
    for episode in range(1, 4):
        for cycle in range(3):
            assert outputs[f'episode_0_cycle_{cycle}'].tobytes() == outputs[f'episode_{episode}_cycle_{cycle}'].tobytes()
    np.savez_compressed(dest.with_suffix('.npz'), **outputs)
    measured = [c['ms'] for c in calls if c['phase'] == 'measured']
    report.update(ok=True, family=family, calls=calls, reset_ms=resets,
                  scope='Paired checkpoint-native public Runtime, fixed recorded observations/executed history, original schedule; one warmup episode and three measured three-cycle episodes. Early history only, not saturated ring latency, sustained tail or closed-loop task quality.',
                  repeated_episode_byte_equal=True, p50_ms=float(np.percentile(measured, 50)),
                  p99_ms=float(np.percentile(measured, 99)),
                  per_cycle_p50_ms={str(i):float(np.median([c['ms'] for c in calls if c['phase']=='measured' and c['cycle']==i])) for i in range(3)},
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
