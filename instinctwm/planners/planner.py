"""Optimizer/Compiler — the layer that turns declarations into optimizations.

A pass never asks "did the user enable me?". It asks "do this model's declarations make me
legal, and profitable?" and answers from `AdapterSpec` alone. That is what makes the framework
model-aware rather than model-specific: the same `CFGBranchElision` pass fires on any adapter
that declares a `POSITIVE_ONLY` stream, whether that is LingBot-VA today or a model nobody has
written yet.

Accuracy tiers do NOT compose upward. A plan containing one BEHAVIORAL pass is BEHAVIORAL, no
matter how many BITEXACT passes sit beside it. This is enforced in `Plan.tier()` rather than
left to discipline, because the failure mode — quoting a bit-exactness claim for a plan that
contains a lossy pass — is exactly the kind of thing that survives review and then invalidates
a benchmark.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from instinctwm.adapters.base import AdapterSpec
from instinctwm.descriptors.deployment import DeploymentSpec


class Tier(enum.IntEnum):
    """Ordered weakest-claim-last so `max()` gives the tier of a composed plan."""

    BITEXACT = 0    # torch.equal on per-step latents and committed K/V, production kernels
    NUMERIC = 1     # bounded ||delta|| justified by a NAMED structural invariant
    BEHAVIORAL = 2  # changes outputs; needs a paired non-inferiority run vs a measured floor


@dataclass(frozen=True)
class PassResult:
    """What a pass decided, including why it declined."""

    name: str
    applies: bool
    tier: Tier
    reason: str
    #: free-form knobs the runtime layer consumes when installing this pass
    params: dict = field(default_factory=dict)
    expected_win: str = "unknown"
    # Populated only for registry/YAML passes. Keeping these fields on the result means the exact
    # resolved decision can cross a worker boundary without loading or trusting the original YAML.
    config_id: str | None = None
    config_version: str | None = None
    config_layer: str | None = None
    config_mode: str | None = None
    install_phase: str | None = None
    required: bool = False
    config_params: dict = field(default_factory=dict)


class OptimizationPass(Protocol):
    name: str

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        """Decide whether this pass is legal and profitable.

        Both arguments are facts, never requests: `spec` is what the model declared about
        itself, `deployment` is how this particular server is running it. A pass reads them
        and decides; it never asks whether the user enabled it.
        """
        ...


@dataclass
class Plan:
    """An ordered set of passes the optimizer chose for one model."""

    model_id: str
    results: list[PassResult]
    optimization_name: str | None = None
    optimization_fingerprint: str | None = None
    optimization_source: str | None = None
    optimization_skipped: tuple[dict[str, str], ...] = ()
    operating_point: dict[str, int] = field(default_factory=dict)

    @property
    def applied(self) -> list[PassResult]:
        return [r for r in self.results if r.applies]

    def tier(self) -> Tier:
        """The weakest claim in the plan. A plan is only BITEXACT if EVERY applied pass is."""
        if not self.applied:
            return Tier.BITEXACT
        return max(r.tier for r in self.applied)

    def bitexact_subset(self) -> "Plan":
        """The largest sub-plan that can still be claimed bit-exact.

        Useful in practice: it is the configuration you can ship without buying a paired
        non-inferiority run, which costs roughly 10x the GPU time of measuring the speedup.
        """
        required_lossy = [r.name for r in self.applied if r.required and r.tier > Tier.BITEXACT]
        if required_lossy:
            raise ValueError(
                f"cannot construct a bit-exact subset without required pass(es): {required_lossy}")
        return Plan(self.model_id, [r for r in self.results if r.tier == Tier.BITEXACT],
                    self.optimization_name, self.optimization_fingerprint,
                    self.optimization_source, self.optimization_skipped,
                    dict(self.operating_point))

    def without(self, *names: str) -> "Plan":
        """The same plan with the named passes demoted to skipped.

        The named passes stay in `results` with `applies=False` and a reason, so `explain()`
        still shows that they were legal and were dropped by hand. Silently deleting them
        would make the plan indistinguishable from one where they never fired.
        """
        unknown = set(names) - {r.name for r in self.results}
        if unknown:
            raise KeyError(f"no such pass in this plan: {sorted(unknown)}")
        required = [r.name for r in self.results if r.name in names and r.required]
        if required:
            raise ValueError(f"cannot drop required optimization pass(es): {required}")
        return Plan(self.model_id, [
            PassResult(name=r.name, applies=False, tier=r.tier,
                       reason=f"dropped by caller via Plan.without(): {r.reason}",
                       params=r.params, expected_win=r.expected_win,
                       config_id=r.config_id, config_version=r.config_version,
                       config_layer=r.config_layer, config_mode=r.config_mode,
                       install_phase=r.install_phase, required=r.required,
                       config_params=r.config_params)
            if r.name in names else r
            for r in self.results
        ], self.optimization_name, self.optimization_fingerprint,
           self.optimization_source, self.optimization_skipped, dict(self.operating_point))

    def _artifact_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_id": self.model_id,
            "operating_point": dict(self.operating_point),
            "optimization": {
                "name": self.optimization_name,
                "fingerprint": self.optimization_fingerprint,
                "source": self.optimization_source,
                "skipped": list(self.optimization_skipped),
            },
            "results": [
                {
                    "name": r.name, "applies": r.applies, "tier": r.tier.name,
                    "reason": r.reason, "params": dict(r.params),
                    "expected_win": r.expected_win, "config_id": r.config_id,
                    "config_version": r.config_version, "config_layer": r.config_layer,
                    "config_mode": r.config_mode, "install_phase": r.install_phase,
                    "required": r.required, "config_params": dict(r.config_params),
                }
                for r in self.results
            ],
        }

    def execution_fingerprint(self) -> str:
        """Fingerprint the evaluated decisions, including manual ``without()`` changes."""
        blob = json.dumps(self._artifact_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """A versioned, checksummed, JSON-safe execution artifact for managed workers."""
        payload = self._artifact_payload()
        payload["execution_fingerprint"] = self.execution_fingerprint()
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Plan":
        if not isinstance(raw, Mapping):
            raise ValueError("optimization plan must be a mapping")
        if raw.get("format_version") != 1:
            raise ValueError(f"unsupported optimization plan format {raw.get('format_version')!r}")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("optimization plan model_id must be a non-empty string")
        rows = raw.get("results")
        if not isinstance(rows, list):
            raise ValueError("optimization plan results must be a list")
        results = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("optimization plan result must be a mapping")
            try:
                tier = Tier[str(row["tier"])]
            except (KeyError, TypeError) as e:
                raise ValueError(f"invalid optimization plan tier {row.get('tier')!r}") from e
            for field_name in ("name", "reason"):
                if not isinstance(row.get(field_name), str):
                    raise ValueError(f"optimization plan result {field_name} must be a string")
            if not isinstance(row.get("applies"), bool):
                raise ValueError("optimization plan result applies must be a boolean")
            if "required" in row and not isinstance(row["required"], bool):
                raise ValueError("optimization plan result required must be a boolean")
            params = row.get("params") or {}
            if not isinstance(params, Mapping):
                raise ValueError("optimization plan result params must be a mapping")
            config_params = row.get("config_params") or {}
            if not isinstance(config_params, Mapping):
                raise ValueError("optimization plan result config_params must be a mapping")
            for field_name in ("config_id", "config_version", "config_layer", "config_mode",
                               "install_phase"):
                if row.get(field_name) is not None and not isinstance(row[field_name], str):
                    raise ValueError(
                        f"optimization plan result {field_name} must be a string or null")
            results.append(PassResult(
                name=row["name"], applies=row["applies"], tier=tier,
                reason=row["reason"], params=dict(params),
                expected_win=str(row.get("expected_win", "unknown")),
                config_id=row.get("config_id"), config_version=row.get("config_version"),
                config_layer=row.get("config_layer"), config_mode=row.get("config_mode"),
                install_phase=row.get("install_phase"), required=bool(row.get("required", False)),
                config_params=dict(config_params),
            ))
        optimization = raw.get("optimization") or {}
        if not isinstance(optimization, Mapping):
            raise ValueError("optimization plan optimization metadata must be a mapping")
        skipped = optimization.get("skipped") or []
        if not isinstance(skipped, list) or not all(isinstance(x, Mapping) for x in skipped):
            raise ValueError("optimization plan skipped metadata must be a list of mappings")
        for field_name in ("name", "fingerprint", "source"):
            if optimization.get(field_name) is not None and \
                    not isinstance(optimization[field_name], str):
                raise ValueError(f"optimization plan {field_name} must be a string or null")
        if not all(isinstance(x.get("id"), str) and isinstance(x.get("reason"), str)
                   for x in skipped):
            raise ValueError("optimization plan skipped entries need string id and reason")
        operating_point = raw.get("operating_point") or {}
        if not isinstance(operating_point, Mapping) or not all(
                isinstance(name, str) and isinstance(nfe, int) and not isinstance(nfe, bool)
                and nfe >= 0 for name, nfe in operating_point.items()):
            raise ValueError("optimization plan operating_point must map phase names to integers")
        plan = cls(model_id, results, optimization.get("name"),
                   optimization.get("fingerprint"), optimization.get("source"),
                   tuple(dict(x) for x in skipped), dict(operating_point))
        expected = raw.get("execution_fingerprint")
        if not isinstance(expected, str) or expected != plan.execution_fingerprint():
            raise ValueError("optimization plan execution fingerprint mismatch")
        return plan

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
        return path

    @classmethod
    def read(cls, path: str | Path) -> "Plan":
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"optimization plan is unexpectedly large: {path}")
        return cls.from_dict(json.loads(path.read_text()))

    def serve(self, model, port: int, **kwargs):
        """Install this plan on `model` and start serving it.

        Deliberately thin: the plan knows which passes fired, the backend knows how to apply
        them to its own server, and neither needs to know the other's internals. A backend
        that cannot install an applied pass raises rather than serving a plan it did not
        actually apply — the alternative is a server whose `explain()` output is a lie.
        """
        return model.serve(self, port=port, **kwargs)

    def explain(self) -> str:
        out = [f"InstinctWM plan for {self.model_id}", f"  plan tier: {self.tier().name}", ""]
        if self.optimization_name:
            out[1:1] = [
                f"  configuration: {self.optimization_name} "
                f"[{(self.optimization_fingerprint or 'unfingerprinted')[:12]}]",
                f"  source: {self.optimization_source}",
            ]
        for r in self.results:
            mark = "APPLY " if r.applies else "skip  "
            out.append(f"  {mark} {r.name:26s} [{r.tier.name:10s}] {r.reason}")
            if r.applies and r.expected_win != "unknown":
                out.append(f"         expected: {r.expected_win}")
        for skipped in self.optimization_skipped:
            out.append(f"  skip   {skipped['id']:26s} [CONFIG    ] {skipped['reason']}")
        if self.tier() > Tier.BITEXACT:
            lossy = [r.name for r in self.applied if r.tier > Tier.BITEXACT]
            out += [
                "",
                f"  NOTE: plan is {self.tier().name} because of {lossy}.",
                "        Any accuracy-neutrality claim for this plan requires a paired",
                "        non-inferiority run. `plan.bitexact_subset()` is the largest",
                "        configuration that does not.",
            ]
        return "\n".join(out)


class Optimizer:
    """Runs every registered pass against a model's declarations and produces a Plan."""

    def __init__(
        self,
        passes: Sequence[OptimizationPass] | None = None,
        tier_ceiling: Tier | None = None,
        optimization_config=None,
    ):
        #: passes are evaluated in registration order; ordering matters where one pass is a
        #: precondition for another (sync elimination gates graph capture, for instance).
        self._resolved_pipeline = None
        if optimization_config is not None:
            if passes is not None:
                raise ValueError("passes= and optimization_config= are mutually exclusive")
            from instinctwm.config import resolve_pipeline

            self._resolved_pipeline = resolve_pipeline(optimization_config)
            passes = [_ConfiguredPass(item) for item in self._resolved_pipeline.ordered]
            if tier_ceiling is None:
                tier_ceiling = {
                    "bitexact": Tier.BITEXACT,
                    "numeric": Tier.NUMERIC,
                    "behavioral": Tier.BEHAVIORAL,
                }[self._resolved_pipeline.policy.tier_ceiling]
        elif passes is None:
            # Imported lazily: the pass modules import this one, so a module-scope import
            # here would be circular.
            from instinctwm.passes.lingbot import default_passes

            passes = default_passes()
        self._passes = list(passes)
        self._ceiling = Tier.BITEXACT if tier_ceiling is None else tier_ceiling

    def compile(self, spec: AdapterSpec, deployment: DeploymentSpec | None = None,
                capabilities: frozenset[str] | None = None) -> Plan:
        """Evaluate every pass against one model's declarations and one server's situation.

        `deployment` defaults to `DeploymentSpec()` — single GPU, actions only — because that
        is the regime this framework targets. Pass one explicitly when it is not true; the
        passes that care will decline on their own.

        `capabilities` is `Checkpoint.capabilities()` — tokens derived from the checkpoint's
        EXECUTION block and nothing else. A pass that declares `requires_capabilities` is skipped
        unless every token it needs is present. Passing `None` means "do not filter", which is the
        behaviour every existing pass has always had: an empty requirement composes with every
        checkpoint, and that is the default on purpose.

        THERE IS NO ARGUMENT HERE THAT CARRIES A TRAINING METHOD, and there is no way to add one
        without changing this signature. `capabilities` cannot smuggle one either: it is built by
        `Checkpoint.capabilities()` from the execution block, which `load_declaration` populates
        without ever parsing provenance. tests/test_checkpoint_platform.py asserts the resulting
        plan is invariant to provenance.
        """
        deployment = deployment if deployment is not None else DeploymentSpec()
        results: list[PassResult] = []
        configured_outcomes: dict[str, bool] = {}
        for p in self._passes:
            if isinstance(p, _ConfiguredPass):
                missing_dependencies = [
                    dependency for dependency in p.definition.requires
                    if not configured_outcomes.get(dependency, False)
                ]
                if missing_dependencies:
                    r = p.decorate(PassResult(
                        name=p.name, applies=False, tier=Tier.BITEXACT,
                        reason=f"required optimization dependency did not apply: "
                               f"{missing_dependencies}",
                    ))
                    configured_outcomes[p.name] = False
                    if p.required:
                        from instinctwm.config import ConfigurationError
                        raise ConfigurationError(f"required pass {r.name!r} was refused: {r.reason}")
                    results.append(r)
                    continue
            need = frozenset(getattr(p, "requires_capabilities", ()) or ())
            capability_unknown = isinstance(p, _ConfiguredPass) and capabilities is None
            if need and (capability_unknown or
                         (capabilities is not None and not need <= capabilities)):
                missing = need if capabilities is None else need - capabilities
                why = (f"checkpoint capabilities were not supplied; pass requires {sorted(need)}"
                       if capabilities is None else
                       f"checkpoint does not declare {sorted(missing)}")
                r = PassResult(
                    name=getattr(p, "name", type(p).__name__), applies=False, tier=Tier.BITEXACT,
                    reason=f"{why}; the pass is not "
                           f"applicable to it. This is a CAPABILITY decision, not a recipe one.",
                )
                if isinstance(p, _ConfiguredPass):
                    r = p.decorate(r)
                if getattr(p, "required", False):
                    from instinctwm.config import ConfigurationError
                    raise ConfigurationError(f"required pass {r.name!r} was refused: {r.reason}")
                if isinstance(p, _ConfiguredPass):
                    configured_outcomes[p.name] = False
                results.append(r)
                continue
            r = p.evaluate(spec, deployment)
            if r.applies and r.tier > self._ceiling:
                r = replace(
                    r,
                    name=r.name, applies=False, tier=r.tier,
                    reason=f"legal but tier {r.tier.name} exceeds ceiling "
                           f"{self._ceiling.name}: {r.reason}",
                )
            if getattr(p, "required", False) and not r.applies:
                from instinctwm.config import ConfigurationError
                raise ConfigurationError(f"required pass {r.name!r} was refused: {r.reason}")
            if isinstance(p, _ConfiguredPass):
                configured_outcomes[p.name] = r.applies
            results.append(r)
        resolved = self._resolved_pipeline
        if resolved is not None:
            # A dependency pulled in solely for a pass that later fails applicability/profitability
            # is not an independent user request. Keep explicit/unlisted roots, then their applied
            # transitive closure; demote any orphan instead of silently installing extra work.
            by_id = {r.config_id: r for r in results if r.config_id}
            definitions = {item.id: item.definition for item in resolved.ordered}
            needed = {
                item.id for item in resolved.ordered
                if item.enabled_by in {"user", "policy.unlisted"}
                and by_id.get(item.id) is not None and by_id[item.id].applies
            }
            pending = list(needed)
            while pending:
                current = pending.pop()
                for dependency in definitions[current].requires:
                    outcome = by_id.get(dependency)
                    if outcome is not None and outcome.applies and dependency not in needed:
                        needed.add(dependency)
                        pending.append(dependency)
            results = [
                replace(r, applies=False,
                        reason="dependency was enabled transitively, but no applied root pass needs it")
                if r.config_id and r.applies and r.config_id not in needed else r
                for r in results
            ]
        return Plan(
            spec.model_id, results,
            optimization_name=(resolved.name if resolved else None),
            optimization_fingerprint=(resolved.fingerprint if resolved else None),
            optimization_source=(resolved.source if resolved else None),
            optimization_skipped=(tuple({"id": x.id, "layer": x.layer.value,
                                         "mode": x.mode.value, "reason": x.reason}
                                        for x in resolved.skipped) if resolved else ()),
            operating_point={phase.name: phase.nfe for phase in spec.phases},
        )


class _ConfiguredPass:
    """Adapter from a resolved registry selection to the original planner protocol."""

    def __init__(self, selection):
        self.selection = selection
        self.definition = selection.definition
        self.name = self.definition.id
        self.required = selection.mode.value == "required"
        self.requires_capabilities = self.definition.requires_capabilities
        self._pass = None

    def decorate(self, result: PassResult) -> PassResult:
        return replace(
            result, name=self.definition.id,
            config_id=self.definition.id, config_version=self.definition.version,
            config_layer=self.definition.layer.value, config_mode=self.selection.mode.value,
            install_phase=self.definition.install_phase.value, required=self.required,
            config_params=dict(self.selection.params),
        )

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        if self._pass is None:
            self._pass = self.definition.factory(self.selection.params)
        return self.decorate(self._pass.evaluate(spec, deployment))
