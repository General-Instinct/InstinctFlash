"""Existing BF16 DiT graph, optionally combined with shared text-block fusions.

graph_only = DiT graph; fused_graph = DiT graph plus text-block fused graphs.
The benchmark's historical norm_graphs field now records both region types.
"""
from groot_n17_iwm.static_capture import install_static_capture


class DiTReceipt:
    name = 'action_head.dit'

    def __init__(self, driver):
        self.driver = driver

    @property
    def stats(self):
        return dict(captured=self.driver.captured, replays=self.replays,
                    captures=self.driver.captures, self_check=self.verdict, disabled=self.disabled)

    @property
    def replays(self):
        return self.driver.replays

    @property
    def verdict(self):
        return self.driver.self_check

    @property
    def disabled(self):
        return self.driver.rejected


def install(model, *, fused):
    graphs = [DiTReceipt(install_static_capture(model))]
    if fused:
        from block_candidate import install as install_blocks
        graphs.extend(install_blocks(model, fused=True))
    return graphs
