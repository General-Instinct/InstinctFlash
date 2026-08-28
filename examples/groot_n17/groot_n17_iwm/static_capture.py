"""Replay-safe CUDA Graph capture for GR00T N1.7's DiT action head.

The four flow-matching calls use the same tensor shapes for a given prompt
shape. Each signature owns static input buffers and one captured DiT forward;
values are copied into those buffers before replay and outputs are cloned out
of the graph pool before returning to upstream code.

DEFAULT, GATED BY A SELF-CHECK. ``GR00TN17Adapter.install`` routes every N1.7-class checkpoint
here when the plan applies graph_capture on a CUDA build — fresh fine-tunes included
(IFL_GROOT_NO_CAPTURE=1 is the kill-switch; the old IFL_GROOT_STATIC_CAPTURE opt-in is
superseded). A capture is not trusted by construction: immediately after EACH signature's graph
is taken, ``_self_check`` replays it against the UPSTREAM eager DiT forward (bound at install
time) on staged inputs the capture never saw. Unlike the KV-cache families, this region has no
out-of-graph prefix — the backbone's vision-language features enter the graph as explicit tensor
inputs — so the staging that kills value-baking is to REDRAW every floating-point input
(perturbed from the captured values by a dedicated generator; integer/bool inputs keep their
captured values, since eager would branch on them while a graph cannot). Exact equality is
required (this family's capture tier is BITEXACT: the standalone gate merged in ``3142eee``
measured 0.0 across unseen observations and prompt shapes). PASS → replay serves; FAIL → the
graphs are released and every later call runs the upstream eager forward, announced loudly —
serving continues.

IFL_GROOT_SELFCHECK_FAULT=1 is the drill switch: it rebinds one input buffer between capture
and check — the stale-address bug class the check exists for — so the loud-fallback path stays
demonstrable on demand.
"""

from __future__ import annotations

import torch

from instinctflash.runtime.capture_self_check import run_capture_self_check

FAMILY = "GR00T N1.7"
#: the drill switch — see the module docstring.
SELF_CHECK_FAULT_ENV = "IFL_GROOT_SELFCHECK_FAULT"


class StaticDiT:
    """Capture one DiT graph per tensor-shape/non-tensor-argument signature."""

    #: staged inputs per captured signature: every case redraws every floating-point input.
    SELF_CHECK_INPUTS = 6

    def __init__(self, dit_forward, on_self_check=None, self_check: bool = True,
                 self_check_inputs: "int | None" = None):
        self._fwd = dit_forward
        self._graphs: dict = {}
        self.replays = 0
        self.captures = 0
        self._on_self_check = on_self_check
        self._self_check_enabled = bool(self_check)
        self._self_check_n = (self.SELF_CHECK_INPUTS if self_check_inputs is None
                              else int(self_check_inputs))
        #: the LAST verdict — see instinctflash.runtime.capture_self_check
        self.self_check: "dict | None" = None
        #: True once a self-check failed. Permanent for the process: every later call runs
        #: the upstream eager forward.
        self.rejected = False

    @property
    def captured(self) -> bool:
        return bool(self._graphs)

    def _signature(self, tensors: dict, other: dict):
        return (
            tuple((key, tuple(value.shape), value.dtype)
                  for key, value in sorted(tensors.items())),
            tuple(sorted((key, repr(value)) for key, value in other.items())),
        )

    def __call__(self, *args, **kwargs):
        if args:
            # Upstream's action loop calls with kwargs only. Preserve eager
            # behavior for any other caller rather than capturing a new API.
            return self._fwd(*args, **kwargs)
        if self.rejected:
            # a rejected capture is permanent: upstream's arithmetic, exactly
            return self._fwd(**kwargs)
        tensors = {key: value for key, value in kwargs.items() if torch.is_tensor(value)}
        other = {key: value for key, value in kwargs.items() if not torch.is_tensor(value)}
        signature = self._signature(tensors, other)
        entry = self._graphs.get(signature)
        if entry is None:
            buffers = {key: value.clone() for key, value in tensors.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._fwd(**buffers, **other)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self._fwd(**buffers, **other)
            entry = (graph, buffers, output)
            self._graphs[signature] = entry
            self.captures += 1

            # FAULT DRILL, between capture and check on purpose: rebinding an input buffer
            # makes every later copy_ land at an address the graph does not read — the
            # stale-address bug class the self-check exists for.
            import os
            import sys
            if os.environ.get(SELF_CHECK_FAULT_ENV) == "1":
                key = next(k for k, v in sorted(buffers.items())
                           if v.is_floating_point())
                print(f"[{FAMILY} static_capture] FAULT INJECTED ({SELF_CHECK_FAULT_ENV}=1): "
                      f"input buffer {key!r} rebound between capture and self-check — the "
                      f"check must now fail.", file=sys.stderr, flush=True)
                buffers[key] = buffers[key].clone()

            # THE GATE. Capturing successfully proves nothing about replaying; the graph
            # serves only if replay equals upstream eager EXACTLY on staged inputs it was
            # not captured from (this family's capture tier is BITEXACT).
            if self._self_check_enabled and self._self_check_n > 0:
                if not self._self_check(entry, other):
                    return self._fwd(**kwargs)
        graph, buffers, output = entry
        for key, value in tensors.items():
            buffers[key].copy_(value)
        graph.replay()
        self.replays += 1
        return output.clone() if torch.is_tensor(output) else output

    # -- the post-capture gate ---------------------------------------------------------------
    def _self_check(self, entry, other) -> bool:
        """Replay vs upstream eager on staged inputs the capture never saw. Exact equality.

        Startup-only, once per captured signature; the model's own RNG stream never moves
        (staged draws come from a dedicated generator, the eager arm is deterministic, replay
        consumes no randomness). Buffers need no restoring: the caller's values are copied in
        by the normal replay path right after the verdict.
        """
        graph, buffers, output = entry
        anchor = next((v for v in buffers.values() if v.is_floating_point()), None)
        if anchor is None or not torch.is_tensor(output):
            # nothing this check can stage or compare — an UNVERIFIABLE capture must not
            # serve by default (measured DiT calls carry float inputs and a tensor output,
            # so this arm exists for API drift, not for the shipped model)
            import sys
            print(f"[{FAMILY} static_capture] the captured region is not self-checkable "
                  f"(float inputs: {anchor is not None}, tensor output: "
                  f"{torch.is_tensor(output)}); refusing to serve an unverified graph.",
                  file=sys.stderr, flush=True)
            self._release_and_fall_back()
            return False
        gen = torch.Generator(device=anchor.device)
        gen.manual_seed(0x51F)
        signature_others = {key: value for key, value in buffers.items()
                            if not value.is_floating_point()}

        def one_case(i):
            staged = dict(signature_others)
            staged.update(other)
            for key, value in buffers.items():
                if not value.is_floating_point():
                    continue
                # perturbed from the captured values, never zero-information noise: the DiT's
                # conditioning inputs (backbone features, adaln state) must stay on-manifold
                # enough that eager takes no degenerate path, while the VALUES are ones the
                # capture never saw — so a graph that baked any input cannot pass
                noise = torch.empty(value.shape, device=value.device, dtype=torch.float32)
                noise.normal_(generator=gen)
                scale = value.detach().float().std().clamp_min(1e-3) * 0.02
                staged[key] = (value.detach().float() + noise * scale).to(value.dtype)

            def run_eager():
                return self._fwd(**staged)

            def run_replay():
                for key, value in staged.items():
                    if torch.is_tensor(value) and value.is_floating_point():
                        buffers[key].copy_(value)
                graph.replay()
                return output
            return "fresh-inputs (all float tensors redrawn)", run_eager, run_replay

        verdict = run_capture_self_check(
            family=FAMILY, cases=(one_case(i) for i in range(self._self_check_n)),
            tolerance=0.0)
        self.self_check = verdict
        if not verdict["passed"]:
            self._release_and_fall_back()
        if self._on_self_check is not None:
            self._on_self_check(dict(verdict))
        return verdict["passed"]

    def _release_and_fall_back(self) -> None:
        """The FAIL arm: graphs released, upstream serves, said out loud. Serving continues."""
        import sys
        self.rejected = True
        self._graphs.clear()
        print(f"[{FAMILY} static_capture] Graphs released; every call now runs the upstream "
              f"eager DiT forward — serving continues on eager arithmetic (upstream's, "
              f"exactly).", file=sys.stderr, flush=True)

    def close(self) -> None:
        self._graphs.clear()


def install_static_capture(model, on_self_check=None, self_check: bool = True) -> StaticDiT:
    """Replace the action-head DiT forward and return its capture counters.

    ``self_check`` (default on) gates each captured signature on the bit-exact
    replay-vs-eager check — see the module docstring. ``on_self_check`` receives each verdict
    dict so an installer can put it on the plan.
    """
    dit = model.action_head.model
    if isinstance(dit.forward, StaticDiT):
        return dit.forward
    handle = StaticDiT(dit.forward, on_self_check=on_self_check, self_check=self_check)
    dit.forward = handle
    return handle


__all__ = ["StaticDiT", "SELF_CHECK_FAULT_ENV", "install_static_capture"]
