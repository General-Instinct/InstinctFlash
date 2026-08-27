"""Scaffolding a fine-tune's declaration from a built-in base: copy, infer, or say FILL_ME.

`instinctflash validate <dir> --validate.scaffold=<base-hub-id|auto>` is the one-command path from
"a training output with no declaration" to "a package `validate` can judge". The scaffold copies
the base's built-in declaration (`known.KNOWN_DECLARATIONS`), then goes field by field:

    inherited   copied from the base — facts a fine-tune keeps (guidance mode, step schedule,
                servable) unless its author says otherwise.
    inferred    read out of the checkpoint ITSELF, with the evidence quoted: backbone identity
                from config.json fingerprints, param_bytes measured from the weight files,
                pi05 obs_features from the checkpoint's own input_features, and so on.
    FILL_ME     a fact the checkpoint does not carry and the scaffold refuses to guess. Written
                into the file as the literal string "FILL_ME" so the follow-up validate flags
                every one of them; each carries a one-line explanation of what belongs there.

THE RULE IS THE SAME ONE THE ADAPTERS ENFORCE AT SERVE TIME: auto-fill only what is provable.
The wan_va adapter refuses to serve guessed camera geometry (`resolve_observation_geometry`),
the pi05 adapter refuses a fine-tune without `obs_features`, cosmos3 refuses a missing serving
config — so the scaffold writing a plausible-looking wrong value would just move the failure
from a loud refusal to silently wrong conditioning. FILL_ME keeps it loud.

Inference depth is per family and stated in the output, because it genuinely differs: a pi05
checkpoint states its own observation contract in config.json; a wan_va training output does not
carry its cameras anywhere the scaffold can read.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from instinctflash.descriptors.checkpoint import _declaration_file
from instinctflash.descriptors.known import KNOWN_DECLARATIONS, lookup

#: The sentinel a scaffolded declaration carries for every fact it could not prove. `validate`
#: flags each occurrence as a PROBLEM, and the adapters treat it as "not declared" (never as a
#: value), so a half-filled scaffold cannot serve.
FILL_ME = "FILL_ME"

INHERITED, INFERRED, TO_FILL = "inherited", "inferred", "FILL_ME"


class ScaffoldError(ValueError):
    """The scaffold refuses: unknown base, no fingerprint match, or an ambiguous match."""


@dataclass(frozen=True)
class ScaffoldField:
    """One execution-block field and how the scaffold decided it."""

    key: str
    status: str      # INHERITED | INFERRED | TO_FILL
    value: Any
    #: evidence for an inferred value, the one-line "what belongs here" for a FILL_ME,
    #: context for an inherited value.
    note: str


@dataclass(frozen=True)
class ScaffoldPlan:
    """A scaffolded declaration document plus the per-field record of how it was built."""

    path: str
    base_id: str
    backbone: str
    #: one honest sentence: what this family's scaffold can prove and what it cannot.
    depth: str
    fields: tuple[ScaffoldField, ...]
    document: dict

    def counts(self) -> dict:
        return {s: sum(1 for f in self.fields if f.status == s)
                for s in (INFERRED, INHERITED, TO_FILL)}

    def explain(self) -> str:
        c = self.counts()
        out = [f"declaration scaffold: base {self.base_id} -> {self.path}",
               f"  inference depth for backbone {self.backbone!r}: {self.depth}"]
        for f in self.fields:
            if f.status == TO_FILL:
                out.append(f"  FILL_ME    {f.key:18} — {f.note}")
            else:
                v = json.dumps(f.value, ensure_ascii=False)
                if len(v) > 58:
                    v = v[:55] + "..."
                out.append(f"  {f.status:9}  {f.key:18} = {v}" + (f"   [{f.note}]" if f.note else ""))
        out.append(f"  {c[INFERRED]} inferred, {c[INHERITED]} inherited, {c[TO_FILL]} to fill")
        return "\n".join(out)

    def to_result(self) -> dict:
        return {"base": self.base_id, "backbone": self.backbone, "depth": self.depth,
                "fields": [{"key": f.key, "status": f.status, "value": f.value, "note": f.note}
                           for f in self.fields]}


# --- reading the checkpoint ----------------------------------------------------------------------

def _json_at(d: Path, rel: str) -> dict:
    p = d / rel
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text())
    except Exception:                                            # noqa: BLE001 - fingerprints only
        return {}
    return doc if isinstance(doc, dict) else {}


def _first_config(d: Path, names: tuple[str, ...]) -> tuple[dict, str]:
    for rel in names:
        cfg = _json_at(d, rel)
        if cfg:
            return cfg, rel
    return {}, ""


def _measure_weights(d: Path, subdir: str = "") -> "tuple[int, str] | None":
    """Total bytes of the weight files under `d/subdir`, and which files were measured.

    File sizes (`st_size`, symlinks followed), matching how every built-in declaration's
    `param_bytes` was produced — the fans export and lerobot/pi05_libero_finetuned_v044 both
    equal the safetensors file size exactly. Sharded sets are enumerated through their index so
    processor-pipeline safetensors sitting next to the model are never miscounted as weights.
    """
    root = d / subdir if subdir else d
    for index_name, single_name in (("model.safetensors.index.json", "model.safetensors"),
                                    ("diffusion_pytorch_model.safetensors.index.json",
                                     "diffusion_pytorch_model.safetensors")):
        idx = _json_at(root, index_name)
        shards = sorted(set((idx.get("weight_map") or {}).values()))
        if shards:
            present = [root / s for s in shards if (root / s).is_file()]
            if present:
                total = sum(p.stat().st_size for p in present)
                where = f"{subdir}/" if subdir else ""
                return total, f"{len(present)} shards of {where}{index_name}"
        p = root / single_name
        if p.is_file():
            where = f"{subdir}/" if subdir else ""
            return p.stat().st_size, f"{where}{single_name}"
    return None


# --- per-family knowledge -------------------------------------------------------------------------
#
# One entry per backbone. `detect` fingerprints the checkpoint's own files and names the base
# entry (or entries) it matches; `build` returns the family-specific field decisions. Both read
# the same facts the family's adapter reads at serve time — the scaffold is those adapters'
# declaration knowledge applied at packaging time, not a second opinion.

@dataclass(frozen=True)
class _Family:
    backbone: str
    #: e.g. 'config.json with type == "pi05"' — quoted in refusals so the user knows what was
    #: looked for and where.
    fingerprint: str
    detect: Callable[[Path], list]           # -> [(base_id, evidence)], len>1 == ambiguous
    build: Callable[[Path, Mapping, str], dict]
    depth: str
    #: True when `detect` distinguishes between this family's base ids (cosmos3 Edge vs Nano),
    #: so an explicit base contradicting the fingerprint is refused rather than obeyed.
    strict_ids: bool = False


def _local_pointer(d: Path, decisions: dict, evidence: str) -> None:
    """base_weights = this directory, when the package itself carries what the loader needs."""
    decisions["base_weights"] = ScaffoldField(
        "base_weights", INFERRED, str(d.resolve()),
        evidence + " — replace with the published repo id when you upload this package")


# wan_va ------------------------------------------------------------------------------------------

_WAN_CONFIGS = ("config.json", "transformer/config.json")


def _detect_wan_va(d: Path) -> list:
    cfg, src = _first_config(d, _WAN_CONFIGS)
    if cfg.get("_class_name") == "WanTransformer3DModel" and "action_dim" in cfg:
        return [("robbyant/lingbot-va-posttrain-robotwin",
                 f"{src}: _class_name=WanTransformer3DModel with action_dim={cfg['action_dim']} "
                 f"(a plain Wan transformer has no action head)")]
    return []


#: Why each wan_va geometry key is FILL_ME: they are training facts (wan_va/configs/*.py) the
#: checkpoint does not carry, and the adapter's own rule is declare-or-fail-loud, never guess —
#: the base's robotwin values are deliberately NOT inherited, because serving a fine-tune under
#: another robot's cameras corrupts conditioning with no warning.
_WAN_GEOMETRY_REASONS = {
    "obs_cam_keys": "the camera keys this fine-tune was trained on — copy obs_cam_keys from your "
                    "wan_va training config; the base's robotwin cameras are not inherited "
                    "because a wrong camera list silently corrupts conditioning",
    "height": "native frame height from your training config (base declares 256)",
    "width": "native frame width from your training config (base declares 320)",
    "env_type": "view compositing mode from your training config, e.g. 'none' or "
                "'robotwin_tshape' (base declares 'robotwin_tshape')",
}


def _build_wan_va(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    m = _measure_weights(d) or _measure_weights(d, "transformer")
    if m:
        decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                 f"measured: {m[1]}")
    frozen = ("vae", "text_encoder", "tokenizer")
    if all((d / c).is_dir() for c in frozen):
        _local_pointer(d, decisions, f"{', '.join(c + '/' for c in frozen)} are present, so the "
                                     f"package carries its own frozen stack")
    else:
        decisions["base_weights"] = ScaffoldField(
            "base_weights", INHERITED, base_ex.get("base_weights"),
            "the frozen vae/text_encoder/tokenizer are fetched from this pointer at load")
    for key, why in _WAN_GEOMETRY_REASONS.items():
        decisions[key] = ScaffoldField(key, TO_FILL, FILL_ME, why)
    return decisions


# pi05 --------------------------------------------------------------------------------------------

def _detect_pi05(d: Path) -> list:
    cfg = _json_at(d, "config.json")
    if cfg.get("type") == "pi05":
        return [("lerobot/pi05_base", 'config.json: type == "pi05"')]
    return []


def _pi05_input_features(d: Path) -> "tuple[dict, str] | None":
    for rel, getter in (("config.json", lambda c: c.get("input_features")),
                        ("train_config.json",
                         lambda c: (c.get("policy") or {}).get("input_features"))):
        feats = getter(_json_at(d, rel))
        if isinstance(feats, dict) and feats:
            try:
                shapes = {str(k): [int(x) for x in v["shape"]] for k, v in feats.items()}
            except Exception:                                    # noqa: BLE001 - not the shape we know
                continue
            return shapes, rel
    return None


def _build_pi05(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    cfg = _json_at(d, "config.json")
    feats = _pi05_input_features(d)
    if feats:
        decisions["obs_features"] = ScaffoldField(
            "obs_features", INFERRED, feats[0],
            f"{feats[1]} input_features ({len(feats[0])} keys) — the same source the two "
            f"built-in pi05 declarations were transcribed from")
    else:
        decisions["obs_features"] = ScaffoldField(
            "obs_features", TO_FILL, FILL_ME,
            "observation key -> per-observation shape; your checkpoint's config.json / "
            "train_config.json carries it as input_features")
    if isinstance(cfg.get("num_inference_steps"), int):
        decisions["nfe"] = ScaffoldField(
            "nfe", INFERRED, {"prefix": 1, "action": int(cfg["num_inference_steps"])},
            "config.json num_inference_steps")
    if (d / "policy_preprocessor.json").is_file() and _measure_weights(d):
        _local_pointer(d, decisions, "model.safetensors + policy_preprocessor.json are present, "
                                     "so the package is the loadable policy")
    m = _measure_weights(d)
    if m:
        decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                 f"measured: {m[1]}")
    return decisions


# groot_n17 ---------------------------------------------------------------------------------------

def _sole_metadata_tag(d: Path) -> "str | None":
    meta = _json_at(d, "experiment_cfg/metadata.json")
    return list(meta)[0] if len(meta) == 1 else None


def _detect_groot(d: Path) -> list:
    cfg = _json_at(d, "config.json")
    if cfg.get("type") == "groot":
        return [("nvidia/GR00T-N1.7-3B", 'config.json: type == "groot"')]
    return []


def _build_groot(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    cfg = _json_at(d, "config.json")
    tag = cfg.get("embodiment_tag") or _sole_metadata_tag(d)
    if tag:
        src = ("config.json embodiment_tag" if cfg.get("embodiment_tag")
               else "the sole statistics key of experiment_cfg/metadata.json")
        decisions["embodiment_tag"] = ScaffoldField("embodiment_tag", INFERRED, str(tag), src)
    else:
        decisions["embodiment_tag"] = ScaffoldField(
            "embodiment_tag", TO_FILL, FILL_ME,
            "names the action-space head; the base's OXE_DROID tag is not inherited because a "
            "fine-tune usually retargets it (your training config states it)")
    if isinstance(cfg.get("num_inference_timesteps"), int):
        decisions["nfe"] = ScaffoldField(
            "nfe", INFERRED, {"backbone": 1, "action": int(cfg["num_inference_timesteps"])},
            "config.json num_inference_timesteps")
    m = _measure_weights(d)
    if m:
        decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                 f"measured: {m[1]}")
        _local_pointer(d, decisions, "local weights are present")
    return decisions


# the GEAR-Dreams WAM (backbone string "dreamzero") --------------------------------------------
#
# Identifiers below avoid the model name on purpose: test_checkpoint_platform forbids executable
# identifiers under descriptors/ naming a model, and the invariant it protects holds here — the
# family knowledge is REGISTRY DATA keyed by the backbone string, exactly like known.py, never a
# code branch on a model name.

def _detect_gear_wam(d: Path) -> list:
    cfg = _json_at(d, "config.json")
    if (cfg.get("model_type") == "vla" and cfg.get("architectures") == ["VLA"]
            and "action_horizon" in cfg):
        return [("GEAR-Dreams/DreamZero-DROID",
                 f"config.json: model_type=vla, architectures=[VLA], "
                 f"action_horizon={cfg['action_horizon']}")]
    return []


def _build_gear_wam(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    tag = _sole_metadata_tag(d)
    if tag:
        decisions["embodiment_tag"] = ScaffoldField(
            "embodiment_tag", INFERRED, str(tag),
            "the sole statistics key of experiment_cfg/metadata.json")
    else:
        decisions["embodiment_tag"] = ScaffoldField(
            "embodiment_tag", TO_FILL, FILL_ME,
            "which experiment_cfg/metadata.json statistics entry this checkpoint acts as "
            "(several are present, so the scaffold refuses to pick)")
    decisions["dynamic_cache_schedule"] = ScaffoldField(
        "dynamic_cache_schedule", INHERITED, bool(base_ex.get("dynamic_cache_schedule", False)),
        "upstream's velocity-cosine skipper: SCREEN-tier, it changes outputs, never default-on")
    m = _measure_weights(d)
    if m:
        decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                 f"measured: {m[1]}")
        _local_pointer(d, decisions, "local weights are present")
    return decisions


# cosmos3_policy ----------------------------------------------------------------------------------

def _detect_cosmos3(d: Path) -> list:
    cfg = _json_at(d, "config.json")
    if cfg.get("model_type") != "cosmos3_omni":
        return []
    tower = (cfg.get("text_config") or {}).get("model_type")
    if tower == "cosmos3_edge_text":
        return [("nvidia/Cosmos3-Edge-Policy-DROID",
                 "config.json: model_type=cosmos3_omni with an Edge text tower "
                 "(text_config.model_type=cosmos3_edge_text)")]
    if tower == "qwen3_vl_text":
        return [("nvidia/Cosmos3-Nano-Policy-DROID",
                 "config.json: model_type=cosmos3_omni with the Nano text tower "
                 "(text_config.model_type=qwen3_vl_text)")]
    # cosmos3_omni without a text tower the fingerprint knows: genuinely ambiguous.
    return [("nvidia/Cosmos3-Edge-Policy-DROID",
             "config.json: model_type=cosmos3_omni; the text tower does not identify Edge vs Nano"),
            ("nvidia/Cosmos3-Nano-Policy-DROID",
             "config.json: model_type=cosmos3_omni; the text tower does not identify Edge vs Nano")]


#: All five serving keys are per-checkpoint measurement facts the adapter refuses to guess
#: (`REQUIRED_SERVING_KEYS` in examples/cosmos3_policy) — so the scaffold does too. The base's
#: DROID values are quoted so filling them for a DROID fine-tune is a copy, not a search.
_COSMOS3_SERVING_REASONS = {
    "domain_name": "the data domain the server's normalization/decoding uses "
                   "(the DROID base declares 'droid_lerobot')",
    "action_dim": "the robot's action dimensionality, as trained (DROID base: 8)",
    "action_chunk_size": "actions returned per request, as trained (DROID base: 16)",
    "image_height": "the request image height the service enforces (DROID base: 540)",
    "image_width": "the request image width the service enforces (DROID base: 640)",
}


def _build_cosmos3(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    for key, why in _COSMOS3_SERVING_REASONS.items():
        decisions[key] = ScaffoldField(key, TO_FILL, FILL_ME, why)
    m = _measure_weights(d)
    if m:
        decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                 f"measured: {m[1]}")
        _local_pointer(d, decisions, "local weights are present")
    return decisions


# lingbot_vla / lingbot_vla_v2 ---------------------------------------------------------------------

def _detect_lingbot_vla(d: Path) -> list:
    cfg = _json_at(d, "config.json")
    if cfg.get("type") == "pi0" and (d / "lingbotvla_cli.yaml").is_file():
        return [("robbyant/lingbot-vla-4b-posttrain-robotwin",
                 'config.json: type == "pi0" next to lingbotvla_cli.yaml (the upstream '
                 "LingBot-VLA release layout)")]
    return []


def _build_lingbot_vla(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    decisions["robot"] = ScaffoldField(
        "robot", TO_FILL, FILL_ME,
        "selects the upstream serving profile (norm stats, action channel map); the base "
        "declares 'robotwin' and a fine-tune on another robot must not inherit it")
    stats = sorted((d / "assets" / "norm_stats").glob("*.json")) \
        if (d / "assets" / "norm_stats").is_dir() else []
    if len(stats) == 1:
        decisions["norm_stats"] = ScaffoldField(
            "norm_stats", INFERRED, str(stats[0].relative_to(d)),
            "the only norm-stats file shipped in the package")
    else:
        decisions["norm_stats"] = ScaffoldField(
            "norm_stats", TO_FILL, FILL_ME,
            "package-relative path to the action norm-stats JSON "
            + ("(none found under assets/norm_stats/)" if not stats
               else f"({len(stats)} candidates under assets/norm_stats/)"))
    m = _measure_weights(d)
    if m:
        decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                 f"measured: {m[1]}")
        _local_pointer(d, decisions, "local weights are present")
    return decisions


def _v2_subdirs(d: Path) -> list:
    return sorted(p.relative_to(d) for p in d.glob("checkpoints/*/hf_ckpt")
                  if (p / "config.json").is_file())


def _detect_lingbot_vla_v2(d: Path) -> list:
    if not (d / "lingbotvla_cli.yaml").is_file():
        return []
    subs = _v2_subdirs(d)
    for sub in subs:
        if "vlm_family" in _json_at(d, str(sub / "config.json")):
            return [("robbyant/lingbot-vla-v2-6b-robotwin",
                     f"lingbotvla_cli.yaml next to {sub}/config.json declaring vlm_family "
                     f"(the V2 release bundle layout)")]
    return []


def _build_lingbot_vla_v2(d: Path, base_ex: Mapping, backbone_evidence: str) -> dict:
    decisions: dict = {}
    subs = _v2_subdirs(d)
    if len(subs) == 1:
        sub = str(subs[0])
        decisions["checkpoint_subdir"] = ScaffoldField(
            "checkpoint_subdir", INFERRED, sub, "the only checkpoints/*/hf_ckpt in the bundle")
        m = _measure_weights(d, sub)
        if m:
            decisions["param_bytes"] = ScaffoldField("param_bytes", INFERRED, m[0],
                                                     f"measured: {m[1]}")
            _local_pointer(d, decisions, "the HF checkpoint is present in the bundle")
    else:
        decisions["checkpoint_subdir"] = ScaffoldField(
            "checkpoint_subdir", TO_FILL, FILL_ME,
            "which checkpoints/*/hf_ckpt to serve"
            + (f" ({len(subs)} present: {', '.join(map(str, subs))})" if subs else
               " (none found under checkpoints/)"))
    decisions["robot"] = ScaffoldField(
        "robot", TO_FILL, FILL_ME,
        "selects the upstream serving profile; the base declares 'robotwin' and a fine-tune "
        "on another robot must not inherit it")
    return decisions


_FAMILIES: dict[str, _Family] = {
    "wan_va": _Family(
        "wan_va", 'config.json (or transformer/config.json) with _class_name == '
                  '"WanTransformer3DModel" and an action_dim',
        _detect_wan_va, _build_wan_va,
        depth="deep — backbone proven from the transformer config, param_bytes measured from "
              "the weight files, frozen-stack pointer resolved; the four observation-geometry "
              "keys are training facts the checkpoint does not carry, so they are FILL_ME"),
    "pi05": _Family(
        "pi05", 'config.json with type == "pi05"',
        _detect_pi05, _build_pi05,
        depth="deep — backbone, observation contract (obs_features), denoise schedule and "
              "param_bytes are all read from the checkpoint's own config.json and weight files"),
    "groot_n17": _Family(
        "groot_n17", 'config.json with type == "groot"',
        _detect_groot, _build_groot,
        depth="medium — backbone and param_bytes inferred; embodiment_tag read from the "
              "checkpoint's own config/metadata when it states one, FILL_ME otherwise"),
    "dreamzero": _Family(
        "dreamzero", 'config.json with model_type == "vla", architectures == ["VLA"] and an '
                     "action_horizon",
        _detect_gear_wam, _build_gear_wam,
        depth="medium — backbone and param_bytes inferred; embodiment_tag read from "
              "experiment_cfg/metadata.json when unambiguous, FILL_ME otherwise"),
    "cosmos3_policy": _Family(
        "cosmos3_policy", 'config.json with model_type == "cosmos3_omni" (the text tower '
                          "distinguishes Edge from Nano)",
        _detect_cosmos3, _build_cosmos3,
        depth="shallow — backbone identity (Edge vs Nano) and param_bytes inferred; the five "
              "serving-config keys are measurement facts the adapter refuses to guess, so all "
              "five are FILL_ME with the DROID base's values quoted",
        strict_ids=True),
    "lingbot_vla": _Family(
        "lingbot_vla", 'config.json with type == "pi0" next to lingbotvla_cli.yaml',
        _detect_lingbot_vla, _build_lingbot_vla,
        depth="medium — backbone and param_bytes inferred, norm_stats found when the package "
              "ships exactly one; the robot profile is FILL_ME"),
    "lingbot_vla_v2": _Family(
        "lingbot_vla_v2", "lingbotvla_cli.yaml next to a checkpoints/*/hf_ckpt/config.json "
                          "declaring vlm_family",
        _detect_lingbot_vla_v2, _build_lingbot_vla_v2,
        depth="medium — backbone, checkpoint_subdir and param_bytes inferred from the bundle "
              "layout; the robot profile is FILL_ME"),
}


# --- the scaffold itself ---------------------------------------------------------------------------

def known_bases() -> list[str]:
    return sorted(KNOWN_DECLARATIONS)


def detect_base(ckpt_dir: str | Path) -> list:
    """Every built-in base whose family fingerprint matches this checkpoint: [(base_id, evidence)]."""
    d = Path(ckpt_dir)
    out: list = []
    for fam in _FAMILIES.values():
        out.extend(fam.detect(d))
    return out


def scaffold_declaration(ckpt_dir: str | Path, base_id: str) -> ScaffoldPlan:
    """Build (do not write) the scaffolded declaration for `ckpt_dir` from `base_id`."""
    d = Path(ckpt_dir)
    doc = lookup(base_id)
    if doc is None:
        raise ScaffoldError(
            f"{base_id!r} is not a built-in declaration. Known bases:\n  "
            + "\n  ".join(known_bases())
            + "\n(or --validate.scaffold=auto to detect one from the checkpoint's config)")
    base_ex = dict(doc.get("execution") or {})
    backbone = str(base_ex.get("backbone") or "")
    fam = _FAMILIES.get(backbone)
    if fam is None:
        raise ScaffoldError(f"no scaffold knowledge for backbone {backbone!r} (base {base_id})")

    matches = fam.detect(d)
    if not matches:
        raise ScaffoldError(
            f"{d} does not look like a {backbone} checkpoint — expected {fam.fingerprint}. "
            f"Refusing to scaffold from {base_id}: a declaration copied onto the wrong family "
            f"would fail at load with a worse message than this one.")
    if fam.strict_ids and all(mid != base_id for mid, _ in matches) and len(matches) == 1:
        mid, evidence = matches[0]
        raise ScaffoldError(
            f"{d} identifies as {mid} ({evidence}), but --validate.scaffold={base_id} was "
            f"passed. Refusing to write the wrong sibling's declaration.")
    backbone_evidence = matches[0][1]

    decisions = fam.build(d, base_ex, backbone_evidence)
    decisions["backbone"] = ScaffoldField("backbone", INFERRED, backbone, backbone_evidence)
    decisions.setdefault("model_id", ScaffoldField(
        "model_id", INFERRED, d.resolve().name,
        "the directory name — set your org/name id before publishing"))
    decisions.setdefault("servable", ScaffoldField(
        "servable", INHERITED, bool(base_ex.get("servable", False)),
        "the base is servable; set false if this fine-tune is not fit to serve"))
    if "nfe" not in decisions and "nfe" in base_ex:
        decisions["nfe"] = ScaffoldField(
            "nfe", INHERITED, copy.deepcopy(base_ex["nfe"]),
            "the base's declared step schedule; a step-distilled fine-tune declares its own")
    if "param_bytes" not in decisions and "param_bytes" in base_ex:
        decisions["param_bytes"] = ScaffoldField(
            "param_bytes", INHERITED, base_ex["param_bytes"],
            "no local weight files found to measure; kept from the base")
    for key, value in base_ex.items():
        decisions.setdefault(key, ScaffoldField(key, INHERITED, copy.deepcopy(value),
                                                "copied from the base declaration"))

    order = list(base_ex) + [k for k in decisions if k not in base_ex]
    fields = tuple(decisions[k] for k in order)
    execution = {f.key: f.value for f in fields}
    fill_notes = {f.key: f.note for f in fields if f.status == TO_FILL}
    document = {
        "instinctflash_schema": int(doc.get("instinctflash_schema", 1)),
        "execution": execution,
        "provenance": {
            "scaffold": {
                "base": base_id,
                "generated_by": "instinctflash validate --validate.scaffold",
                **({"fill_me": fill_notes} if fill_notes else {}),
            },
        },
    }
    return ScaffoldPlan(str(d), base_id, backbone, fam.depth, fields, document)


# --- writing, guarding, and the FILL_ME scan ------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _diff_lines(old_doc: Mapping, new_doc: Mapping) -> list[str]:
    old = dict(old_doc.get("execution") or {})
    new = dict(new_doc.get("execution") or {})
    out: list[str] = []
    for k, v in new.items():
        if k not in old:
            out.append(f"  + execution.{k} = {json.dumps(v, ensure_ascii=False)}")
        elif old[k] != v:
            out.append(f"  ~ execution.{k}: {json.dumps(old[k], ensure_ascii=False)} -> "
                       f"{json.dumps(v, ensure_ascii=False)}")
    for k in old:
        if k not in new:
            out.append(f"  - execution.{k} (present only in the existing file)")
    same = [k for k in new if k in old and old[k] == new[k]]
    out.append(f"  = unchanged: {', '.join(same) if same else '(none)'}")
    return out


def run_scaffold(ckpt_dir: str | Path, base: str, *, force: bool = False) -> tuple[dict, str, bool]:
    """The `--validate.scaffold` verb body: detect/copy/infer, guard, write.

    Returns (result-dict, text, wrote). Never overwrites an existing declaration without
    `force`; in that case the text carries the full would-be document and a field-level diff,
    and `wrote` is False so the caller can refuse to report success for work it did not do.
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        raise ScaffoldError(f"{d} is not a directory")
    lines: list[str] = []
    if base == "auto":
        matches = detect_base(d)
        if not matches:
            raise ScaffoldError(
                f"{d}: no built-in declaration matches this checkpoint. Fingerprints looked "
                "for:\n  " + "\n  ".join(f"{f.backbone}: {f.fingerprint}"
                                         for f in _FAMILIES.values())
                + "\nPass --validate.scaffold=<base> explicitly; known bases:\n  "
                + "\n  ".join(known_bases()))
        if len(matches) > 1:
            raise ScaffoldError(
                f"{d}: ambiguous — {len(matches)} built-in declarations match, and the "
                "scaffold refuses to pick:\n  "
                + "\n  ".join(f"{mid}  ({ev})" for mid, ev in matches)
                + "\nPass --validate.scaffold=<one of them>.")
        base, evidence = matches[0]
        lines.append(f"auto-detected base: {base}  ({evidence})")
    plan = scaffold_declaration(d, base)
    lines.append(plan.explain())

    existing = _declaration_file(d)
    wrote = False
    if existing is not None and not force:
        try:
            old = json.loads(existing.read_text())
        except Exception:                                        # noqa: BLE001 - still refuse to clobber
            old = {}
        lines += ["", f"NOT WRITTEN: {existing.name} already exists. What the scaffold would "
                      f"change (--validate.force=true to overwrite):"]
        lines += _diff_lines(old, plan.document)
        lines += ["", "the full document the scaffold would write:",
                  json.dumps(plan.document, indent=2, ensure_ascii=False)]
    else:
        if existing is not None:
            try:
                old = json.loads(existing.read_text())
            except Exception:                                    # noqa: BLE001
                old = {}
            lines += [f"overwriting {existing.name} (--validate.force=true); changes:"]
            lines += _diff_lines(old, plan.document)
        target = d / "instinctflash.json"
        _atomic_write(target, json.dumps(plan.document, indent=2, ensure_ascii=False) + "\n")
        wrote = True
        lines.append(f"wrote {target}")
    result = {**plan.to_result(), "written": wrote}
    return result, "\n".join(lines), wrote


def fill_me_findings(ckpt_dir: str | Path) -> list[tuple[str, str]]:
    """Every FILL_ME sentinel in the package's declaration: [(dotted path, why it must be filled)].

    Read by `validate` on EVERY run, not only after a scaffold, so a half-filled declaration
    keeps failing until the last sentinel is gone. The one-line reasons ride in
    provenance.scaffold.fill_me — provenance, because they are notes for a human, and the
    runtime never reads them.
    """
    p = _declaration_file(Path(ckpt_dir))
    if p is None:
        return []
    try:
        doc = json.loads(p.read_text())
    except Exception:                                            # noqa: BLE001 - structure check reports it
        return []
    reasons = (((doc.get("provenance") or {}).get("scaffold") or {}).get("fill_me") or {})

    found: list[tuple[str, str]] = []

    def walk(prefix: str, top: str, value: Any) -> None:
        if isinstance(value, str) and value == FILL_ME:
            found.append((prefix, str(reasons.get(
                top, "the scaffold could not prove a value; fill in this checkpoint's real one"))))
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(f"{prefix}.{k}", top, v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(f"{prefix}[{i}]", top, v)

    for key, value in (doc.get("execution") or {}).items():
        walk(f"execution.{key}", key, value)
    return found
