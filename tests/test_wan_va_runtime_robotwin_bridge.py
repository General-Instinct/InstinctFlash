from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.vla.robotwin_driver import WanVaWireBridge
from benchmarks.vla.wan_va_runtime_robotwin_bridge import WanVaRuntimeWireBridge
from benchmarks.vla.util import ConfigurationError
from instinctflash.runtime.wan_va_engine import WanVaEngineLoop


class Server:
    def __init__(self):
        self.native = SimpleNamespace(job_config=SimpleNamespace(obs_cam_keys=['camera'], frame_chunk_size=2))
        self.events = []
        self.cycle = 0

    def infer(self, obs):
        if obs.get('reset'):
            self.cycle = 0
            self.events.append(('reset',))
            return {}
        if obs.get('compute_kv_cache'):
            self.events.append(('commit', [x['camera'] for x in obs['obs']], obs['state'].copy()))
            return {}
        self.events.append(('predict',))
        value = np.arange(512, dtype=np.float32).reshape(16, 2, 16) + self.cycle
        self.cycle += 1
        return {'action': value}


class Remote:
    def __init__(self, runtime):
        self.server = Server()
        self.loop = WanVaEngineLoop(self.server) if runtime else None
        self.requests = []

    def reset_episode(self, prompt, seed):
        if self.loop:
            self.loop.reset(prompt=prompt)
        else:
            self.server.infer({'reset': True})
        return {}

    def infer(self, obs):
        self.requests.append(obs)
        return self.loop.predict(obs) if self.loop else self.server.infer(obs)


def test_client_actions_and_nonterminal_commits_match_actual_runtime_loop():
    scene = {'prompt': 'test', 'resolved_seed': 7}
    old_remote, new_remote = Remote(False), Remote(True)
    old = WanVaWireBridge(old_remote, scene)
    new = WanVaRuntimeWireBridge(new_remote, scene)
    for bridge in (old, new):
        bridge.infer({'reset': True, 'prompt': 'test'})
    for cycle in range(3):
        request = {'obs': {'camera': 0}, 'prompt': 'test'}
        a, b = old.infer(request)['action'], new.infer(request)['action']
        assert a.tobytes() == b.tobytes()
        frames = [{'camera': cycle * 10 + x} for x in range(4 if cycle == 0 else 8)]
        for bridge, action in ((old, a), (new, b)):
            bridge.infer({'obs': frames, 'compute_kv_cache': True, 'state': action})
    assert old.cycles == new.cycles == 3 and old.phase == new.phase == 'infer'
    assert [len(x['obs']) for x in new_remote.requests] == [1, 4, 8]
    old_commits = [e for e in old_remote.server.events if e[0] == 'commit']
    new_commits = [e for e in new_remote.server.events if e[0] == 'commit']
    assert len(old_commits) == 3 and len(new_commits) == 2
    for a, b in zip(old_commits, new_commits):
        assert a[1] == b[1] and a[2].tobytes() == b[2].tobytes()
    # Terminal observations are staged locally; no extra prediction is requested.
    assert len(new.frames) == 8
    # Each rollout gets a new bridge, while its Runtime endpoint is reused.
    # Reset must discard the previous pending action without committing it.
    next_episode = WanVaRuntimeWireBridge(new_remote, scene)
    next_episode.infer({'reset': True, 'prompt': 'test'})
    action = next_episode.infer({'obs': {'camera': 99}})['action']
    assert [e[0] for e in new_remote.server.events[-2:]] == ['reset', 'predict']
    np.testing.assert_array_equal(action, np.arange(512, dtype=np.float32).reshape(16, 2, 16))
    assert len([e for e in new_remote.server.events if e[0] == 'commit']) == 2


def test_changed_action_history_or_short_history_is_refused():
    remote = Remote(True)
    bridge = WanVaRuntimeWireBridge(remote, {'prompt': 'test', 'resolved_seed': 7})
    bridge.infer({'reset': True, 'prompt': 'test'})
    action = bridge.infer({'obs': {'camera': 0}})['action']
    with pytest.raises(ConfigurationError, match='4 observed frames'):
        bridge.infer({'compute_kv_cache': True, 'obs': [], 'state': action})
    with pytest.raises(ConfigurationError, match='changed the predicted action'):
        bridge.infer({'compute_kv_cache': True, 'obs': [{'camera': 0}] * 4, 'state': action + 1})
    assert bridge.phase == 'commit' and bridge.cycles == 0
    assert len(remote.requests) == 1
