<div align="center">

<img src="assets/instinctwm_2.png" width="360"/>

# InstinctWM

### Load, optimize, and deploy world-action models.

[Documentation](docs/) •
[Examples](examples/) •
[Benchmarks](eval/) •
[Models](#supported-models)

</div>

---

InstinctWM is an optimization framework for world-action models (WAMs), covering the full stack from model optimization to hardware deployment.

---

# What's New

- Layered optimization framework for world-action models
- Generic optimization pass system
- Bit-exact runtime optimization for LingBot-VA
- Cosmos3-Edge support
- Evaluation and certification framework

---

# Optimization Stack

InstinctWM organizes optimizations into six complementary layers.

| **MODEL** | **GRAPH** | **CACHE** | **ATTENTION** | **KERNEL** | **HARDWARE** |
|:--|:--|:--|:--|:--|:--|
| **Step Reduction** | **Prefill Extraction** | **TeaCache** | **FlashAttention** | **Triton** | **TensorRT** |
| Parallel Decoding Distillation | Execution Graph Rewrite | XCache | FlashInfer | CUDA | FP8 |
| rCM | CUDA Graph Capture | SeaCache | Sana-Video Hybrid Attention | CUDA Kernels | INT8 |
| sCM | Stream Overlap | KV Reuse | LongSana | Operator Fusion | INT4 |
| DMD2 | Persistent State Analysis | Cross-attention Cache | Linear Attention | Fused AdaLN | Jetson |
| DreamZero-Flash | Static Memory Planning | Episode Cache | Mamba / DeltaNet | Fused CFG | Thor |
| **Latent Compression** | Prefill Cache | Window Cache | | Fused Scheduler | Snapdragon |
| DCAE / DCVE | | Energy-based Cache | | Fused VAE | |
| Other Latent Tokenizers | | | | Paged KV Kernels | |
| **Architecture** | | | | | |
| Hybrid AR + Diffusion | | | | | |

---

# Quick Start

Install InstinctWM

```bash
git clone https://github.com/general-instinct/InstinctWM
cd InstinctWM

pip install -e .
```

Optimize a model

```python
from instinctwm import load
from instinctwm import Optimizer

model = load("lingbot-va")

optimizer = Optimizer()

engine = optimizer.compile(model)

engine.serve()
```

---

# Supported Models

| Model | Status |
|:--|:--|
| LingBot-VA | Full runtime support |
| Cosmos3-Edge | Engine support |

Additional world-action models will be added over time.

---

# Documentation

## Layer 1 — Model Optimization

Reduce model computation while preserving downstream performance.

Topics include

- Step Reduction
- Latent Compression
- Architecture Optimization
- Model Certification

→ `docs/layer1.md`

---

## Layer 2 — Graph Optimization

Optimize execution graphs without changing model behavior.

Topics include

- Graph Rewrite
- CUDA Graph
- Prefill Extraction
- Stream Overlap
- Static Memory Planning

→ `docs/layer2.md`

---

## Layer 3 — Cache Optimization

Reuse computation across timesteps and episodes.

Topics include

- KV Reuse
- Cross-attention Cache
- Episode Cache
- TeaCache
- XCache
- SeaCache

→ `docs/layer3.md`

---

## Layer 4 — Attention Optimization

Optimize attention implementations.

Topics include

- FlashAttention
- FlashInfer
- Hybrid Attention
- Linear Attention
- Mamba / DeltaNet

→ `docs/layer4.md`

---

## Layer 5 — Kernel Optimization

Optimize low-level kernels.

Topics include

- Triton
- CUDA
- Operator Fusion
- Fused AdaLN
- Fused CFG
- Paged KV

→ `docs/layer5.md`

---

## Layer 6 — Hardware Optimization

Deploy efficiently across different hardware platforms.

Topics include

- TensorRT
- FP8 / INT8 / INT4
- Jetson
- Thor
- Snapdragon

→ `docs/layer6.md`

---

# Evaluation

InstinctWM includes evaluation and certification tooling for world-action models.

Features include

- RoboTwin evaluation
- Paired evaluation
- Certification
- Latency benchmarking
- Bit-exact verification

See

```
eval/
```

---

# Examples

```text
examples/
├── optimize_lingbot.py
├── optimize_cosmos3.py
├── runtime_server.py
└── evaluation.py
```

---

# Roadmap

Current focus areas include

- Layer 1 model optimization
- Layer 3 cache optimization
- Layer 5 kernel optimization
- Layer 6 hardware optimization

---

# License

AGPL-3.0

See `LICENSE`.
