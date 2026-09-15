# LingBot-VA latency regimes — do the published cells survive KV-pool saturation? (2026-09-04)

**Question.** Stage-2 M3 (`thor_va_engine/stage2_report.md` §4) found LingBot-VA cycle latency has two regimes: *early*
(ring-KV pool still filling, cycles < 37 of an episode) and *saturated* (pool full, cycles ≥ 37; the knee is the ring wrap
at 0-based cycle 36). Our published cells were measured over the first 8 (Thor) / 12 (H100) cycles of an episode, i.e. in
the early regime. The stock vendor server also fills a pool, so the question is whether the **ratios** survive saturation,
and by how much the **absolute** cells understate a long episode.

**Answer in one line.** Every published RATIO stands and is conservative — the stock server's cycle grows more with pool size
than ours does (same-computation: Thor 3.21× published → **3.82×** saturated; H100 3.27× published → **4.10×** saturated on
the re-measurement box; 2V/4A vs stock default: Thor 20.2× → **22.8×**, H100 23.4× → **27.7×**). The Thor ABSOLUTE cells are
early-regime only: at saturation stock 18027 → **31533 ms**, ours 5611 → **8250 ms**, ours @2V/4A 892.9 → **1382 ms**
(the "35.8 Hz effective" becomes **23.1 Hz**; 24.6 Hz with P010). On H100 our default-NFE chain is regime-flat
(sat/early 1.00); our 2V/4A cell grows +6 %; the stock cell grows +20 %.

## 0. Provenance (takeover)

* Measurement designed and launched by the agent that died with its host restart on 2026-09-02 ~22:30Z. Both chains had been
  started with `setsid nohup` and **ran to completion unattended**; nothing was re-run. Tools: `va_regime_tools/`
  (`h100_chain.sh`, `h100_va_arm.sh`, `thor_regime_chain.sh`, `thor_regime_arm.sh`, `stock_nfe_server.py`,
  `regime_summarize.py`, `regime_table.py`; `regime_report.py` added by the takeover). Notebook: `va_regime_tools/PROGRESS.md`.
* **H100:** the 4xH100 box (`4xh100`, host 68-209-74-33), GPU 0 (0 MiB at start; GPUs 1–3 stayed at 0 MiB for the whole
  chain), 2026-09-02 22:18:42Z → 23:00:03Z. InstinctFlash tree = internal main **c68dc53** (P010 in `shipped_configuration`),
  vendor tree `/home/ubuntu/lingbot-va`, checkpoint `robbyant/lingbot-va-posttrain-robotwin`. **Not the box the published H100
  cells came from** (those are the local box's H100s, 2026-08-24 — off-limits: GPUs 0–3 robot servers, 4–7 head-to-head).
* **Thor:** Jetson Thor, MAXN, `~/InstinctFlash_thor` (serve_variant md5 0717405c == c68dc53), 2026-09-02 15:17:57 → 17:21:57 PDT
  (22:17Z → 00:21Z), each arm under `flock /tmp/thor_gpu.lock`. Same device and stack as the published Thor cells.
  Concurrency check: the Stage-2 M3 campaigns' last probe log on Thor is 15:05:11 PDT, before this chain started → no overlap.
* Raw artifacts: `va_regime_results/{h100,thor}/{chain.log,logs/probe_*_run{0,1,2}.log,logs/server_*.log,logs/telemetry_*}`;
  per-device summaries with every per-cycle trace `va_regime_results/{h100,thor}_regime_summary.json`; combined table/ratio/verdict
  arithmetic `va_latency_regimes.json` (this directory). Copies in `InstinctFlash/eval/results_2026-08_thor_h100/`.

## 1. Protocol (Stage-2 M3 verbatim)

`probe_latency.py` real message order (reset → infer → compute_kv_cache → …), random-pixel frames, keyframes 4-then-8,
`--cycles 48 --repeats 1`, **3 runs per arm, run 0 discarded**, one server per arm, nothing else on the GPU, `nvidia-smi` /
thermal-zone telemetry every 10 s plus before/after each arm. Cycle = 32 control steps. Windows (1-based cycle numbers as the
task states them; the probe prints 0-based `cycle 0..47`):

| window | cycles | statistic |
|---|---|---|
| **early** | 5–30 | p50 (p99 also reported) |
| **saturated** | 37–48 | p50 and p99 |
| steady | 2–48 | p50 (Stage-2 M3's "cycle p50 (1-47)") |
| published-protocol window, H100 | 2–12 | mean (the published 12-cycle × 3 protocol's kept-run window) |
| published-protocol window, Thor | 2–8 | p50 (the published 8-cycle × 3, run-0-discarded protocol's kept-run window) |

Arms (9 per device): stock vendor server `wan_va/wan_va_server.py` at its default 25V/50A@w5; stock at 2V/4A@w5 and 2V/2A@w1 via
`stock_nfe_server.py` (byte-for-byte vendor code path; only `num_inference_steps` / `action_num_inference_steps` /
`guidance_scale` of `VA_CONFIGS['robotwin']` overridden in-process — the same fields `serve_variant --degrade-nfe/--guidance`
write; w1 ⇒ `use_cfg=False`, batch-1); our shipped chain `serve_variant.py --no-fsdp --no-empty-cache --no-debug-dump
--conditioning-prefill --ring-kv --conv-layout` at the three points, each **with and without `--action-terminal-elision` (P010)**,
which is in `shipped_configuration()` since d7e6103 (the published cells predate it).

Stage-2 M3 defined "early" as cycles 2–13, which gave torch 2V/4A 903 ms on Thor; this protocol's early window (5–30) gives 1013.
Both are early-regime numbers — the difference is the window; the saturated windows are identical and agree within 1.5 %
(M3 1362 vs 1382 here).

**Integrity.** All 54 probe runs rc=0 with 48 cycles; kept-run spreads ≤ 1.3 % (steady p50) on every arm. H100 GPU 0 held
1980 MHz on every loaded sample, throttle mask 0x0, ≤ 455 W (700 W cap), 27–50 °C. Thor GPC held 1575 MHz on every loaded
sample except the first sample of each arm (server loading, 315–495 MHz) and one teardown sample; gpu-thermal peaked at
81.2 °C on the default-NFE torch arms with no clock change. The Thor torch arms reproduce Stage-2 M3 to ≤ 2 % (2V/4A 1086 vs
1080 steady p50; P010 1012 vs 1030; 2V/2A 703 / 655 vs 701 / 652). H100 run 0 shows the known first-episode transient
(~900 ms/cycle p50 for both stock and ours at the few-step points, e.g. 2V/4A run 0 p50 988 vs 267 in runs 1–2) — discarded by
protocol; Thor shows no such transient.

## 2. Table — arm × point × device × regime (ms per 32-step cycle; kept runs 1–2 averaged)

| arm | point | device | early p50 (cyc 5–30) | early p99 | **saturated p50 (cyc 37–48)** | saturated p99 | sat/early | steady p50 (cyc 2–48) | published-protocol window | kept-run spread (steady / sat) | clocks / thermal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock vendor server | default 25V/50A@w5 | H100 (4xH100 box) | 6902.6 | 7461.7 | **7768.7** | 7791.2 | 1.13× | 7252.1 | 6468.6 (mean cyc 2–12) | 1.1% / 1.3% | SM 1980 MHz on all 106 loaded samples, 28–44 °C, ≤328 W, no throttle |
| stock vendor server | default 25V/50A@w5 | Jetson Thor | 23420.7 | 29478.9 | **31532.8** | 31827.2 | 1.35× | 27140.2 | 17851.0 (p50 cyc 2–8) | 0.0% / 0.4% | GPC 1575 MHz on 372/373 loaded samples, gpu-thermal 37–65 °C |
| shipped chain, no P010 (the published arm) | default 25V/50A@w5 | H100 (4xH100 box) | 1897.9 | 1912.2 | **1896.2** | 1916.6 | 1.00× | 1897.7 | 1896.4 (mean cyc 2–12) | 0.1% / 0.1% | SM 1980 MHz on all 31 loaded samples, 34–50 °C, ≤452 W |
| shipped chain, no P010 (the published arm) | default 25V/50A@w5 | Jetson Thor | 6619.4 | 7577.4 | **8250.1** | 8271.3 | 1.25× | 7213.1 | 5632.5 (p50 cyc 2–8) | 0.2% / 0.1% | GPC 1575 MHz on 102/103, gpu-thermal 54–81 °C |
| shipped chain + P010 (shipped since d7e6103) | default 25V/50A@w5 | H100 (4xH100 box) | 1848.0 | 1869.5 | **1851.7** | 1871.5 | 1.00× | 1847.9 | 1845.7 (mean cyc 2–12) | 0.5% / 0.0% | SM 1980 MHz on all 30, 34–49 °C, ≤455 W |
| shipped chain + P010 (shipped since d7e6103) | default 25V/50A@w5 | Jetson Thor | 6560.3 | 7486.4 | **8182.3** | 8217.7 | 1.25× | 7126.4 | 5577.2 (p50 cyc 2–8) | 0.4% / 0.0% | GPC 1575 MHz on 101/102, gpu-thermal 60–81 °C |
| stock vendor server at the point (config override) | 2V/4A@w5 | H100 (4xH100 box) | 1010.5 | 1057.1 | **1097.8** | 1109.6 | 1.09× | 1040.7 | 981.1 (mean cyc 2–12) | 1.2% / 0.4% | SM 1980 MHz on all 18, 32–40 °C, ≤274 W |
| stock vendor server at the point (config override) | 2V/4A@w5 | Jetson Thor | 3225.6 | 3946.2 | **4272.1** | 4313.0 | 1.32× | 3665.6 | 2519.7 (p50 cyc 2–8) | 0.6% / 1.1% | GPC 1575 MHz on 51/52, gpu-thermal 60–68 °C |
| shipped chain, no P010 (the published arm) | 2V/4A@w5 | H100 (4xH100 box) | 265.2 | 277.4 | **280.9** | 286.1 | 1.06× | 266.4 | 266.1 (mean cyc 2–12) | 0.8% / 0.1% | SM 1980 MHz on all 7, 34–43 °C, ≤372 W |
| shipped chain, no P010 (the published arm) | 2V/4A@w5 | Jetson Thor | 1013.1 | 1145.8 | **1382.3** | 1403.4 | 1.36× | 1086.0 | 882.2 (p50 cyc 2–8) | 0.8% / 0.4% | GPC 1575 MHz on 16/17, gpu-thermal 55–76 °C |
| shipped chain + P010 (shipped since d7e6103) | 2V/4A@w5 | H100 (4xH100 box) | 246.5 | 257.4 | **262.0** | 265.5 | 1.06× | 247.2 | 248.3 (mean cyc 2–12) | 0.2% / 0.5% | SM 1980 MHz on all 7, 34–44 °C, ≤379 W |
| shipped chain + P010 (shipped since d7e6103) | 2V/4A@w5 | Jetson Thor | 949.9 | 1065.5 | **1300.2** | 1319.6 | 1.37× | 1012.0 | 834.6 (p50 cyc 2–8) | 0.1% / 0.9% | GPC 1575 MHz on 15/16, gpu-thermal 57–75 °C |
| stock vendor server at the point (config override) | 2V/2A@w1 | H100 (4xH100 box) | 814.1 | 828.0 | **843.6** | 863.0 | 1.04× | 819.2 | 814.7 (mean cyc 2–12) | 0.4% / 0.9% | SM 1980 MHz on all 15, 33–37 °C, ≤213 W |
| stock vendor server at the point (config override) | 2V/2A@w1 | Jetson Thor | 1919.8 | 2040.3 | **2159.7** | 2183.3 | 1.12× | 1997.9 | 1787.1 (p50 cyc 2–8) | 0.2% / 0.8% | GPC 1575 MHz on 28/30 (1 loading + 1 teardown sample), gpu-thermal 56–64 °C |
| shipped chain, no P010 | 2V/2A@w1 | H100 (4xH100 box) | 226.6 | 229.8 | **226.6** | 233.3 | 1.00× | 226.8 | 226.8 (mean cyc 2–12) | 0.7% / 0.5% | SM 1980 MHz on all 7, 34–41 °C, ≤305 W |
| shipped chain, no P010 | 2V/2A@w1 | Jetson Thor | 681.2 | 725.1 | **782.4** | 791.0 | 1.15× | 702.6 | 625.5 (p50 cyc 2–8) | 0.3% / 0.7% | GPC 1575 MHz on 10/11, gpu-thermal 58–69 °C |
| shipped chain + P010 (shipped since d7e6103) | 2V/2A@w1 | H100 (4xH100 box) | 203.6 | 215.4 | **204.5** | 207.0 | 1.00× | 204.1 | 205.0 (mean cyc 2–12) | 0.2% / 0.4% | SM 1980 MHz on all 6, 32–38 °C, ≤262 W |
| shipped chain + P010 (shipped since d7e6103) | 2V/2A@w1 | Jetson Thor | 624.6 | 672.4 | **729.1** | 738.8 | 1.17× | 654.7 | 579.6 (p50 cyc 2–8) | 1.1% / 0.0% | GPC 1575 MHz on 9/10, gpu-thermal 56–70 °C |

Every arm reached saturation within the 48-cycle budget on both devices (the trace plateaus from cycle 37 on every arm that
grows at all; see `trace_total_ms` in the summary JSONs). Regime declared per cell above; no value is extrapolated.

### 2a. Ratios per regime

"vs stock DEFAULT" is the published headline convention (the operating-point cells are quoted against the vendor's default 25V/50A
server); "vs stock AT THE POINT" is the same-computation ratio at that operating point (new here).

| arm @ point | device | vs stock DEFAULT: early | **saturated p50** | saturated p99 | published-window | published cell | vs stock AT THE POINT: early | **saturated p50** |
|---|---|---|---|---|---|---|---|---|
| shipped chain, no P010 (the published arm) @ default | H100 (4xH100 box) | 3.64× | **4.10×** | 4.07× | 3.41× | 3.27× | 3.64× | **4.10×** |
| shipped chain + P010 @ default | H100 (4xH100 box) | 3.74× | **4.20×** | 4.16× | 3.50× | — | 3.74× | **4.20×** |
| stock at the point @ 2V/4A@w5 | H100 (4xH100 box) | 6.83× | **7.08×** | 7.02× | 6.59× | — | 1.00× | **1.00×** |
| shipped chain, no P010 (the published arm) @ 2V/4A@w5 | H100 (4xH100 box) | 26.02× | **27.66×** | 27.24× | 24.31× | 23.43× | 3.81× | **3.91×** |
| shipped chain + P010 @ 2V/4A@w5 | H100 (4xH100 box) | 28.00× | **29.65×** | 29.34× | 26.05× | — | 4.10× | **4.19×** |
| stock at the point @ 2V/2A@w1 | H100 (4xH100 box) | 8.48× | **9.21×** | 9.03× | 7.94× | — | 1.00× | **1.00×** |
| shipped chain, no P010 @ 2V/2A@w1 | H100 (4xH100 box) | 30.47× | **34.28×** | 33.39× | 28.52× | — | 3.59× | **3.72×** |
| shipped chain + P010 @ 2V/2A@w1 | H100 (4xH100 box) | 33.91× | **37.99×** | 37.64× | 31.56× | — | 4.00× | **4.13×** |
| shipped chain, no P010 (the published arm) @ default | Jetson Thor | 3.54× | **3.82×** | 3.85× | 3.17× | 3.21× | 3.54× | **3.82×** |
| shipped chain + P010 @ default | Jetson Thor | 3.57× | **3.85×** | 3.87× | 3.20× | — | 3.57× | **3.85×** |
| stock at the point @ 2V/4A@w5 | Jetson Thor | 7.26× | **7.38×** | 7.38× | 7.08× | — | 1.00× | **1.00×** |
| shipped chain, no P010 (the published arm) @ 2V/4A@w5 | Jetson Thor | 23.12× | **22.81×** | 22.68× | 20.23× | 20.19× | 3.18× | **3.09×** |
| shipped chain + P010 @ 2V/4A@w5 | Jetson Thor | 24.66× | **24.25×** | 24.12× | 21.39× | — | 3.40× | **3.29×** |
| stock at the point @ 2V/2A@w1 | Jetson Thor | 12.20× | **14.60×** | 14.58× | 9.99× | — | 1.00× | **1.00×** |
| shipped chain, no P010 @ 2V/2A@w1 | Jetson Thor | 34.38× | **40.30×** | 40.24× | 28.54× | — | 2.82× | **2.76×** |
| shipped chain + P010 @ 2V/2A@w1 | Jetson Thor | 37.50× | **43.25×** | 43.08× | 30.80× | — | 3.07× | **2.96×** |

### 2b. P010 (`--action-terminal-elision`, BITEXACT, in `shipped_configuration` since d7e6103) at saturation

| point | device | no P010 sat p50 | + P010 sat p50 | Δ ms | Δ % | no P010 sat p99 | + P010 sat p99 |
|---|---|---|---|---|---|---|---|
| default 25V/50A@w5 | H100 (4xH100 box) | 1896.2 | 1851.7 | −44.6 | −2.4% | 1916.6 | 1871.5 |
| default 25V/50A@w5 | Jetson Thor | 8250.1 | 8182.3 | −67.9 | −0.8% | 8271.3 | 8217.7 |
| 2V/4A@w5 | H100 (4xH100 box) | 280.9 | 262.0 | −18.8 | −6.7% | 286.1 | 265.5 |
| 2V/4A@w5 | Jetson Thor | 1382.3 | 1300.2 | −82.1 | −5.9% | 1403.4 | 1319.6 |
| 2V/2A@w1 | H100 (4xH100 box) | 226.6 | 204.5 | −22.1 | −9.8% | 233.3 | 207.0 |
| 2V/2A@w1 | Jetson Thor | 782.4 | 729.1 | −53.3 | −6.8% | 791.0 | 738.8 |

(The Thor −68 ms/cycle at default NFE is exactly the P010 report's projection; at 2V/4A saturation the saved forward is worth
−82 ms because the elided forward runs over the full window.)

## 3. Verdict per published cell

Published cells and where they live: `InstinctFlash/README.md` lines 37–38 (+ "twenty-three times" in the What's-new bullet, line
21); `iwm_distill/thor_column/final_table.md` and `InstinctFlash/eval/results_2026-08_thor_h100/final_table.md` (VA row and the
2V/4A declared-operating-point row; sources `va.json`, `bench_h100_full/table.json`). `InstinctWM/README.md` carries no absolute
VA latency cell (its "2.88× bit-exact, plus 1.405×" decomposition is not a regime claim; on H100 the default-NFE chain is
regime-flat, so nothing there needs a regime qualifier).

| # | published cell | verdict | numbers |
|---|---|---|---|
| 1 | **H100 same-computation** "8448 → 2583 ms, 3.27×" | **Ratio STANDS (conservative).** OUR absolute cell stands (regime-flat on H100: 1896 in every window, sat/early 1.00). The STOCK absolute cell is **early-regime only**: on the re-measurement box the stock server's saturated cycle is 1.20× its published-window value (7768.7 vs 6468.6). The published box cannot be re-measured (off-limits), so no corrected absolute is stated — the correction is the factor. | same-box published-window ratio 3.41× → **saturated 4.10×** (p99 4.07×); with P010 4.20×. Same-box values are 23–27 % below the published local-box cells on both arms (the known faster box); ratios are within 4 %. |
| 2 | **Thor same-computation** "18027 → 5611 ms, 3.21×" | **Early-regime only (both absolute cells)** — the published-protocol window reproduces them today (17851 / 5633, −1.0 % / +0.4 %), and the same episode saturates at **31533 → 8250 ms** (1.77× / 1.46× the published ms). **Ratio STANDS (conservative): 3.82× saturated** (p99 3.85×), 3.85× with P010. Effective rate 1.8 → 1.0 Hz stock, 5.7 → 3.9 Hz ours. | saturated p50 31532.8 → 8250.1 (nop010) / 8182.3 (P010); sat p99 31827.2 → 8271.3 / 8217.7 |
| 3 | **H100 @ 2V/4A** "8448 → 360.5 ms, 23.4×" (README "360 ms, 23×", "twenty-three times") | **Ratio STANDS (conservative).** Our absolute cell is **early-regime only by +6 %** on the same box (266.1 → 280.9); the stock denominator grows +20 %, so the ratio rises. With P010 (now shipped) the saturated same-box value (262.0) is below the no-P010 published-window value (266.1). | same-box published-window 24.3× → **saturated 27.7×** (p99 27.2×); with P010 29.7×. Same-computation at the point (stock 2V/4A vs ours 2V/4A): 3.81× early → 3.91× saturated (4.19× with P010). Effective 120 → 114 Hz (122 Hz with P010). |
| 4 | **Thor @ 2V/4A** "18027 → 892.9 ms, 20.2× (35.8 Hz effective)" (README "893 ms, 20×") | **Early-regime only (absolute ms and the Hz claim)** — the window reproduces (882.2, −1.2 %); the same episode saturates at **1382.3 ms p50 / 1403.4 p99 → 23.1 Hz** (1300.2 / 1319.6 → **24.6 Hz** with P010). **Ratio STANDS: 22.8× saturated** vs stock default (24.3× with P010) ≥ the published 20.2×. Consequence already recorded in Stage-2: this arm misses the 960 ms real-time tier at saturation; the 2V/2A@w1 + P010 arm (729.1 p50 / 738.8 p99, 43.9 Hz; certified NON-INFERIOR robust 2026-09-03, n=642) meets it. | saturated 31532.8 → 1382.3 (nop010) / 1300.2 (P010); same-computation at the point: 3.18× early → 3.09× saturated (3.40× → 3.29× with P010) |

**Why the ratios rise.** On H100 our infer path is flat early→saturated (default 1817 → 1799 ms p50; 2V/4A 187 → 185) and only the
commit (kv) message grows (81 → 98; 79 → 96), while the stock server's infer path grows (6546 → 7410; 661 → 744): the growth sits in
the stock KV-cache addressing's per-forward, pool-size-dependent bookkeeping, which the ring-KV pass replaces (the P010 A/B logged the
stock allocator's per-layer `nonzero`+gather). On Thor both arms' infer paths grow — attention over the full window is bandwidth-bound
there — stock +35 % (22589 → 30496) vs ours +23 % (6321 → 7753), and our commit message +67 % (298 → 499: VAE encode + the commit
forwards over the full window). At 2V/4A on Thor the two arms grow alike (1.32× vs 1.36×), so that ratio is flat across regimes
(23.1× early, 22.8× saturated) and above the published 20.2× only because the published window (cycles 2–8) sits even earlier
than this protocol's early window.

## 4. Proposed README wording (NOT applied — README untouched)

Minimal (keeps every existing number, adds the regime declaration; H100 same-computation cells keep the local-box values per the
final-table policy, since ratios not absolute ms are the stable cross-box quantity):

```
| **LingBot-VA** (5B WAM) | 8448 → 2583 ms, **3.27×** ‡ | 18027 → 5611 ms, **3.21×** (early) ‡ / 31533 → 8250 ms, **3.82×** (saturated) | NUMERIC |
| **LingBot-VA @ 2V/4A** (5B WAM) | 8448 → 360 ms, **23×** ‡ | 18027 → 893 ms, **20×** (early) ‡ / 31533 → 1382 ms, **23×** (saturated; 23 Hz, 24.6 Hz with P010) | OPERATING-POINT (certified) |

‡ LingBot-VA latency has two regimes within an episode: early (the ring-KV pool still filling, cycles < 37) and saturated
(pool full, cycles ≥ 37). The cells marked "early" were measured over cycles 2–8 (Thor) / 2–12 (H100) of an episode; the
saturated cells over cycles 37–48 (48-cycle episodes × 3 runs, run 0 discarded, real message order). The vendor server's cycle
grows more with pool size than ours does, so every ratio holds or rises at saturation (same computation: H100 4.1×, Thor 3.8×;
2V/4A vs the vendor default: H100 27.7×, Thor 22.8×), while the Thor absolute ms above understate a long episode by 1.5–1.8×
in the early cells. On H100 our default-NFE chain is regime-flat (1896 ms in both regimes on the re-measurement box); the
vendor server is +20 % at saturation. Table, protocol, per-cycle traces: eval/results_2026-08_thor_h100/va_latency_regimes.md.
```

What's-new bullet (line 21): "runs at twenty-three times its upstream serving cost" — stands as written (saturated 22.8× on Thor,
27.7× on H100); optionally "…twenty-three times its upstream serving cost in both the early and the saturated KV-pool regime".

Full variant (if the row is re-based rather than footnoted): quote saturated p50 as the cell and the early value in parentheses —
Thor "31533 → 8250 ms, 3.82× (early 18027 → 5611, 3.21×)", "31533 → 1382 ms, 22.8×, 23 Hz (early 893 ms, 36 Hz)"; H100 keeps the
local-box cells with the footnote until the local box can be re-measured over 48 cycles.

## 5. Notes and limits

* **Not the published H100 box.** The 4xH100 box is 23–27 % faster than the local box on both arms at the published window
  (stock 6469 vs 8448, ours 1896 vs 2583, 2V/4A 266 vs 360.5), consistent with the 2026-08-28 re-sweep; the same-box ratio at the
  published window (3.41×, 24.3×) is within 4 % of the published (3.27×, 23.4×). H100 conclusions here are ratios and same-box
  sat/early factors; no cross-box absolute is projected.
* **Saturation reached everywhere** — the 48-cycle budget covers 12 saturated cycles per run on every arm, including the 26–32 s/cycle
  Thor stock default (3 × 21 min).
* **Stage-2 M3 "early" (cycles 2–13) ≠ this protocol's early (cycles 5–30).** Both are early-regime; the saturated windows coincide.
* **The published Thor protocol (cycles 2–8) sits before this protocol's early window**, so the early column here is already above the
  published cells (e.g. 2V/4A 1013 vs 882/893).
* **H100 first-episode transient** (~900 ms/cycle for most of run 0 on the few-step arms, both stock and ours; run 0 is discarded by
  protocol and the kept runs agree ≤ 1.3 %) — the "discard one full warm-up episode" rule is mandatory on that box.
* **P010** is a served, BITEXACT pass (d7e6103); both columns are reported. The published cells predate it.
* The 2V/2A@w1 point: certified NON-INFERIOR (robust) 2026-09-03 (`fewstep/cert_2v2a_w1_certificate.*`, n=642, most-conservative
  LB −0.0022 vs −0.05); saturated Thor 729.1 p50 / 738.8 p99 with P010 (43.9 Hz) is the only torch arm inside the 960 ms tier at
  saturation; H100 204.5 / 207.0 (156 Hz).
