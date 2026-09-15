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
precision = sys.argv[1]
assert precision in ('native', 'fp8')
dest = root / f'va-libero-public-inputs-{precision}'
assert not dest.with_suffix('.json').exists(), 'Preserve prior receipts'
report = dict(precision=precision, scope='Independent input/reset controls, recorded LIBERO frames and executed actions; no closed-loop quality or sustained latency certificate')
loop = None
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    os.environ['LINGBOT_ROOT'] = '/home/guanming/lingbot-va'
    cfg = import_lingbot_server(os.environ['LINGBOT_ROOT']).VA_CONFIGS['libero']
    cameras = tuple(cfg.obs_cam_keys)
    data = np.load(root / 'va-libero-observations.npz')
    frames = [{k: data[k][i].copy() for k in cameras} for i in range(29)]
    prompt = str(data['prompt'])
    first = np.concatenate((np.zeros((4, 7), dtype=np.float32), data['executed_actions'][:12]))
    first = first.reshape(4, 4, 7).transpose(2, 0, 1).copy()
    changed_execution = first.copy()
    changed_execution[0, :, 1:] += 0.05
    outputs, calls = {}, []
    start = time.perf_counter()
    loop = Runtime.from_pretrained(root / 'runtime-libero-checkpoint', precision=precision)
    loop.reset(prompt=prompt)
    report['setup_seconds'] = time.perf_counter() - start
    if precision == 'native':
        native = loop._backend._ensure()._server
    else:
        bridge = loop._backend._loop._loop._server
        native = bridge.native
        report['fp8_declaration'] = bridge.frontend.declaration()
    assert native.job_config.num_inference_steps == 20
    assert native.job_config.action_num_inference_steps == 50
    report['nfe'] = dict(video=20, action=50)
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
        assert out.shape == (7, 4, 4) and np.isfinite(out).all()
        outputs[label] = out.copy()
        calls.append(dict(label=label, ms=(time.perf_counter()-start)*1000,
                          frame=native.frame_st_id))
        print(json.dumps(calls[-1]), flush=True)
        return out

    def equal(a, b):
        return outputs[a].dtype == outputs[b].dtype and outputs[a].tobytes() == outputs[b].tobytes()

    predict('baseline_initial', [frames[0]], 0, first)
    predict('baseline_history', frames[1:13], 1, None)
    for camera in cameras:
        initial = {k: v.copy() for k, v in frames[0].items()}
        initial[camera] = frames[-1][camera].copy()
        assert not np.array_equal(initial[camera], frames[0][camera]), 'Recorded camera control must differ'
        loop.reset(prompt=prompt)
        predict('camera_' + camera, [initial], 0, first)
        assert not equal('baseline_initial', 'camera_' + camera), camera + ' has no measured influence'
    changed_prompt = 'Open the drawer and place the object inside.'
    assert prompt != changed_prompt
    loop.reset(prompt=changed_prompt)
    predict('changed_prompt', [frames[0]], 0, first)
    assert not equal('baseline_initial', 'changed_prompt'), 'Prompt has no measured influence'
    loop.reset(prompt=prompt)
    predict('execution_initial', [frames[0]], 0, changed_execution)
    assert equal('baseline_initial', 'execution_initial'), 'Future executed action changed current prediction'
    predict('changed_execution', frames[1:13], 1, None)
    assert not equal('baseline_history', 'changed_execution'), 'Executed history has no measured influence'
    loop.reset(prompt=prompt)
    predict('restored_initial', [frames[0]], 0, first)
    predict('restored_history', frames[1:13], 1, None)
    assert equal('baseline_initial', 'restored_initial') and equal('baseline_history', 'restored_history')
    np.savez_compressed(dest.with_suffix('.npz'), **outputs)
    report.update(ok=True, calls=calls, camera_controls=list(cameras), prompt_changes_actions=True,
                  executed_history_changes_actions=True, reset_restores_action_bytes=True,
                  future_feedback_does_not_change_current_action=True,
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
