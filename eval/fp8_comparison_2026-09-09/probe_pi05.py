"""Matched Thor computation probe; preprocessed images/tokens, not robot API latency.

Each invocation is one fresh process. Never overwrites results. Fixed 10 denoise
steps; 10-action paired comparison and separate 50-action native reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['stock10', 'capture10', 'stock50', 'capture50', 'engine16', 'engine8'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--frames', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repeat', type=int, required=True)
    p.add_argument('--iterations', type=int, default=128)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise RuntimeError('refusing to overwrite result')
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1701 + a.repeat)
    np.random.seed(1701 + a.repeat)
    frames = np.load(a.frames)
    obs = [dict(image=frames['image'][i], wrist_image=frames['wrist_image'][i]) for i in range(5)]
    ids = json.loads(Path(a.assets).read_text())['synth_ids']['48']
    assert len(ids) == 48
    chunk = 50 if a.arm.endswith('50') else 10
    engine = a.arm.startswith('engine')
    info = dict(arm=a.arm, repeat=a.repeat, chunk=chunk, steps=10, views=2,
                prompt_tokens=48, calibration_indices=[0, 1, 2], evaluation_indices=[3, 4],
                scope='CPU preprocessed fp16 images and token IDs to CPU normalized 7D action chunk; no tokenizer, robot preprocessing, action unnormalization, transport or simulator',
                input_sha256=digest(a.frames), assets_sha256=digest(a.assets),
                script_sha256=digest(__file__), checkpoint=str(Path(a.checkpoint).resolve()),
                torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), warmup=8, iterations=a.iterations)
    start = time.perf_counter()
    if engine:
        import flash_rt
        from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
        fe = Pi05TorchFrontendThor(a.checkpoint, num_views=2, use_fp8=a.arm == 'engine8', autotune=3)
        fe.set_prompt(ids)
        if a.arm == 'engine8':
            fe.calibrate(obs[:3], percentile=99.9)
        info['frontend_source'] = str(Path(flash_rt.__file__).parent)
        info['frontend_sha256'] = digest(Path(flash_rt.__file__).parent / 'frontends/torch/pi05_thor.py')
        def predict(index, noise):
            # Explicit actual-noise injection; same fp16 values fed to upstream.
            original = np.random.randn
            def fixed(*shape):
                if shape != (10, 32):
                    raise RuntimeError(f'unexpected noise request {shape}')
                return noise.astype(np.float64)
            np.random.randn = fixed
            try:
                fe.infer(obs[index])
                return fe._g_noise.float().cpu().numpy()[:, :7].copy()
            finally:
                np.random.randn = original
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        cfg = PI05Config.from_pretrained(a.checkpoint)
        cfg.compile_model = False
        cfg.chunk_size = chunk
        cfg.n_action_steps = chunk
        policy = PI05Policy.from_pretrained(a.checkpoint, config=cfg).to('cuda').eval()
        model = policy.model
        info['parameter_dtype'] = str(next(model.parameters()).dtype)
        if a.arm.startswith('capture'):
            from pi05_iwm.surface import Pi05Surface
            from pi05_iwm.static_capture import install_static_capture
            Pi05Surface(model).hoist_loop_constants()
            den = install_static_capture(model)
        def predict(index, noise):
            images = [torch.from_numpy(obs[index][k]).permute(2, 0, 1)[None].to('cuda', dtype=torch.float32) for k in ('image', 'wrist_image')]
            masks = [torch.ones(1, device='cuda', dtype=torch.bool) for _ in images]
            tokens = torch.tensor([ids], device='cuda', dtype=torch.long)
            tm = torch.ones_like(tokens, dtype=torch.bool)
            nz = torch.from_numpy(noise.astype(np.float32))[None].to('cuda')
            with torch.no_grad():
                return model.sample_actions(images, masks, tokens, tm, noise=nz, num_steps=10)[0, :, :7].float().cpu().numpy().copy()
    torch.cuda.synchronize()
    info['load_calibrate_s'] = time.perf_counter() - start
    # Noise generation is outside timing for every arm. Same inputs across repeats.
    rng = np.random.default_rng(2917)
    noises = rng.standard_normal((max(128, a.iterations) + 8, chunk, 32)).astype(np.float16)
    t = time.perf_counter()
    for i in range(8):
        predict(3 + i % 2, noises[i])
    torch.cuda.synchronize()
    info['warmup_s'] = time.perf_counter() - t
    torch.cuda.reset_peak_memory_stats()
    samples, outputs = [], []
    for i in range(a.iterations):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = predict(3 + i % 2, noises[8 + i])
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1000)
        if not np.isfinite(result).all():
            raise RuntimeError('nonfinite action')
        outputs.append(result)
    # Separate repeatability probe, excluded from latency distribution.
    null = [predict(3, noises[8]) for _ in range(3)]
    info['latency_ms'] = {f'p{q}': float(np.percentile(samples, q)) for q in (50, 95, 99)}
    info['samples_ms'] = samples
    info['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    info['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    info['memory_caveat'] = 'PyTorch allocator only; engine external CUDA allocations excluded'
    info['device_free_total_bytes'] = list(torch.cuda.mem_get_info())
    info['graph_captured'] = bool(fe._enc_ae_graph) if engine else bool(a.arm.startswith('capture') and den._graph is not None)
    info['status'] = 'measured'
    info['numeric_environment'] = dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark)
    np.savez(out.with_suffix('.npz'), actions=np.stack(outputs), null=np.stack(null), noises=noises[8:8+a.iterations])
    info['actions_sha256'] = digest(out.with_suffix('.npz'))
    out.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({k: info[k] for k in ('arm', 'repeat', 'latency_ms', 'graph_captured')}), flush=True)


if __name__ == '__main__':
    main()
