#!/usr/bin/env python3
"""Export the real LingBot-VA training output as a publishable InstinctWM package.

    THE 23 GB DIRECTORY IS A TRAINING OUTPUT, NOT A PUBLISHED PACKAGE.

`/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin` is diffusers-style multi-folder --
transformer/ vae/ text_encoder/ tokenizer/ -- and it fails `validate_package` because there is no
root declaration. The fix is not to widen the validator. It is to export the servable subset, which
is what LeRobot did with this exact checkpoint: `lerobot/lingbot_va_robotwin` is FLAT, one 10.2 GB
weight file, with `"wan_pretrained_path": "robbyant/lingbot-va-posttrain-robotwin"` in its config --
the frozen stack referenced by repo id rather than vendored.

So this script produces:

    lingbot-va/
      instinctwm.json                              the declaration (execution + provenance)
      config.json                                  the transformer's own config, copied
      diffusion_pytorch_model-0000N-of-0000M.safetensors + .index.json    the TRAINABLE weights only
      README.md                                    generated model card
      LICENSE                                      copied if present upstream
      instinctwm_certificate.json                  optional, from verify/released.py
      instinctwm_benchmark.json                    optional, from verify/released.py

and nothing else. The VAE, text encoder and tokenizer are NOT copied: they are frozen, they are the
same bytes for every fine-tune of this backbone, and duplicating 13 GB of them into every published
checkpoint is the cost the pointer avoids.

    python tools/export_lingbot_hub_package.py --out /home/ubuntu/hub/lingbot-va
    python -m instinctwm.descriptors.package /home/ubuntu/hub/lingbot-va
    hf upload general-instinct/lingbot-va /home/ubuntu/hub/lingbot-va
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC_DEFAULT = "/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin"
#: Where the frozen stack lives, referenced rather than copied. This is the upstream publication of
#: the same training output; a fork should point at its own.
BASE_WEIGHTS_DEFAULT = "robbyant/lingbot-va-posttrain-robotwin"


def _copy(src: Path, dst: Path, *, label: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    n = dst.stat().st_size
    print(f"  {label:52} {n:>15,} bytes")
    return n


def build_declaration(*, model_id: str, base_weights: str, param_bytes: int,
                      nfe_video: int, nfe_action: int, provenance: bool) -> dict:
    execution = {
        "model_id": model_id,
        "backbone": "wan_va",
        "servable": True,
        "guidance": {"video": "cfg", "action": "positive_only"},
        "nfe": {"video": nfe_video, "action": nfe_action},
        # The frozen stack -- VAE, text encoder, tokenizer -- by reference. An EXECUTION fact: the
        # runtime cannot load the model without it, and it says nothing about training.
        "base_weights": base_weights,
        "param_bytes": param_bytes,
    }
    doc = {"instinctwm_schema": 1, "execution": execution}
    if provenance:
        doc["provenance"] = {
            "note": "FOR HUMANS. The runtime never reads this block; publishability() proves the "
                    "package still serves with it removed.",
            "training_output": SRC_DEFAULT,
            "upstream_release": base_weights,
        }
    return doc


def collect_evidence() -> tuple[dict | None, dict | None]:
    """Certificate and benchmark, from the release registry. Both optional by design."""
    try:
        from instinctwm.verify.released import RELEASED, disposition_of, SERVED
    except Exception:                                          # noqa: BLE001
        return None, None
    cert = None
    for r in RELEASED:
        if disposition_of(r.pid).status is SERVED and r.tier.name != "BITEXACT" and r.certificate:
            cert = {"pass_id": r.pid, "name": r.name, "tier": r.tier.name,
                    "evidence_kind": r.evidence_kind(), "certificate": r.certificate}
            break
    bench = {
        "benchmark": "RoboTwin 2.0",
        "protocol": "paired, identical pinned seeds, margin declared before the run",
        "note": "Success rates are from the P007 certification run; see certificate for the exact "
                "protocol. Latency is measured on our own hardware and does not transfer.",
    }
    return cert, bench


def generate_card(*, model_id: str, base_weights: str, decl: dict,
                  cert: dict | None, bench: dict | None) -> str:
    """The model card, GENERATED from the declaration rather than hand-written.

    LeRobot renders its cards from a template at publish time so they cannot drift from the
    artifact, and ships the Evaluation section as an explicit hole when there is nothing to report.
    Both are adopted here.
    """
    ex = decl["execution"]
    caps_note = "" if cert else ("\n_No verification certificate was included in this package._\n")
    eval_block = ("\n_No evaluation results have been provided for this checkpoint._\n"
                  if not bench else f"""
| benchmark | protocol |
|:--|:--|
| {bench['benchmark']} | {bench['protocol']} |

{bench['note']}
""")
    cert_block = caps_note if not cert else f"""
**{cert['pass_id']} `{cert['name']}` — tier {cert['tier']}**, evidence: {cert['evidence_kind']}.

```
{cert['certificate'][:700]}
```

The served chain is therefore **{cert['tier']}**, not bit-exact end to end. That is what the
certificate is for.
"""
    return f"""---
library_name: instinctwm
pipeline_tag: robotics
license: apache-2.0
tags:
  - world-action-model
  - robotics
  - instinctwm
inference: false
base_model: {base_weights}
base_model_relation: finetune
---

# {model_id.split('/')[-1]}

A LingBot-VA world-action model packaged for the [InstinctWM](https://github.com/General-Instinct/InstinctWM)
runtime. **This package contains the trainable transformer only.** The frozen stack — VAE, text
encoder, tokenizer — is referenced by repo id and resolved at load:

```
execution.base_weights = "{base_weights}"
```

## Quick start

```bash
pip install instinctwm
```

```python
from instinctwm import Runtime

runtime = Runtime.from_pretrained("{model_id}")
runtime.reset(prompt="put the bottle in the dustbin")
action = runtime.predict(observation)
```

`observation` is a mapping in the backbone's own schema — for this checkpoint, one frame per camera
under the LeRobot key convention:

```python
observation = {{"obs": [{{"observation.images.cam_high": frame, ...}}],
               "prompt": "put the bottle in the dustbin"}}
```

**One prediction per `reset()` today.** A wan_va control cycle is two phases — the action
prediction, then a KV-commit that advances the ring — and the public API currently expresses only
the first, so a closed `while True: runtime.predict(...)` loop is not yet supported. Start a new
episode for each prediction until multi-phase cycles land.

Inspect it first, without downloading {ex.get('param_bytes', 0) / 1e9:.1f} GB of weights:

```python
from instinctwm import describe
describe("{model_id}")
```

## What this checkpoint declares

| field | value |
|:--|:--|
| `backbone` | `{ex['backbone']}` — must resolve to a registered adapter |
| `servable` | `{ex['servable']}` |
| `guidance` | video `{ex['guidance']['video']}`, action `{ex['guidance']['action']}` |
| `nfe` | video {ex['nfe']['video']}, action {ex['nfe']['action']} forwards per control step |
| `base_weights` | `{ex['base_weights']}` |

The runtime reads that block and nothing else. How these weights were trained lives under
`provenance`, which the loader parses only to discard — delete it entirely and the checkpoint still
serves.

## Operating points

An operating point is a **descriptor delta, not a mode flag**. This package declares
{ex['nfe']['video']}V/{ex['nfe']['action']}A. To serve a different schedule, either publish a second
checkpoint whose `execution.nfe` says so, or override the declared field explicitly:

```python
Runtime.from_pretrained("{model_id}", nfe={{"video": 25, "action": 50}})
```

There is no `Fast`/`Quality` preset, and there will not be one: a preset table inside the runtime
would be per-checkpoint tuning living in the wrong place.

## Verification
{cert_block}
## Evaluation
{eval_block}
## Limitations

- Requires an adapter registered for backbone `{ex['backbone']}`. InstinctWM ships one.
- The frozen stack is fetched from `{base_weights}`; that repo must remain reachable.
- Latency figures in the InstinctWM repository are measured on specific hardware and do not transfer.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT, help="the training output directory")
    ap.add_argument("--out", required=True, help="where to write the publishable package")
    ap.add_argument("--model-id", default="general-instinct/lingbot-va")
    ap.add_argument("--base-weights", default=BASE_WEIGHTS_DEFAULT)
    ap.add_argument("--nfe-video", type=int, default=2)
    ap.add_argument("--nfe-action", type=int, default=4)
    ap.add_argument("--no-provenance", action="store_true",
                    help="omit the provenance block entirely -- the package still serves")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    tf = src / "transformer"
    if not tf.is_dir():
        print(f"{src}: no transformer/ subdirectory -- this does not look like a LingBot-VA "
              f"training output")
        return 2

    weights = sorted(tf.glob("*.safetensors"))
    index = tf / "diffusion_pytorch_model.safetensors.index.json"
    cfg = tf / "config.json"
    for p in (cfg,):
        if not p.exists():
            print(f"{p}: missing")
            return 2

    total = sum(w.stat().st_size for w in weights)
    print(f"source : {src}")
    print(f"  transformer: {len(weights)} shard(s), {total:,} bytes"
          f"{' + index' if index.exists() else ''}")
    frozen = sum(f.stat().st_size for d in ("vae", "text_encoder", "tokenizer")
                 for f in (src / d).rglob("*") if f.is_file())
    print(f"  frozen stack NOT copied: {frozen:,} bytes -> referenced as {a.base_weights!r}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {out}")
    written = 0
    written += _copy(cfg, out / "config.json", label="config.json")
    for w in weights:
        written += _copy(w, out / w.name, label=w.name)
    if index.exists():
        written += _copy(index, out / index.name, label=index.name)
    for extra in ("LICENSE", "LICENSE.txt"):
        if (src / extra).exists():
            written += _copy(src / extra, out / "LICENSE", label=extra)

    decl = build_declaration(model_id=a.model_id, base_weights=a.base_weights, param_bytes=total,
                             nfe_video=a.nfe_video, nfe_action=a.nfe_action,
                             provenance=not a.no_provenance)
    (out / "instinctwm.json").write_text(json.dumps(decl, indent=2) + "\n")
    print(f"  {'instinctwm.json':52} {(out / 'instinctwm.json').stat().st_size:>15,} bytes")

    cert, bench = collect_evidence()
    if cert:
        (out / "instinctwm_certificate.json").write_text(json.dumps(cert, indent=2) + "\n")
        print(f"  {'instinctwm_certificate.json':52} "
              f"{(out / 'instinctwm_certificate.json').stat().st_size:>15,} bytes")
    if bench:
        (out / "instinctwm_benchmark.json").write_text(json.dumps(bench, indent=2) + "\n")
        print(f"  {'instinctwm_benchmark.json':52} "
              f"{(out / 'instinctwm_benchmark.json').stat().st_size:>15,} bytes")

    card = generate_card(model_id=a.model_id, base_weights=a.base_weights, decl=decl,
                         cert=cert, bench=bench)
    (out / "README.md").write_text(card)
    print(f"  {'README.md (generated)':52} {(out / 'README.md').stat().st_size:>15,} bytes")

    print(f"\npackage total: {written:,} bytes "
          f"({written / 1e9:.1f} GB, vs {(total + frozen) / 1e9:.1f} GB for the training output)")

    # -- the gates, run here so a bad package is never uploaded ---------------------------------
    from instinctwm.descriptors.package import publishability, validate_package
    print("\n" + "=" * 78)
    rep = validate_package(out)
    print(rep.explain())
    ok, findings = publishability(out)
    print(f"  publishable without training internals: {'YES' if ok else 'NO'}")
    for f in findings:
        print(f"    - {f}")
    if not (rep.ok and ok):
        print("\nREFUSING to declare success: the package does not pass its own gates.")
        return 1
    print(f"\nReady. Publish with:\n    hf upload {a.model_id} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
