"""Check actual deferred VA loop against the simulator's explicit-commit protocol."""
from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.vla import wan_va_libero_driver as driver
from benchmarks.vla.util import ConfigurationError
from instinctflash.runtime.wan_va_engine import WanVaEngineLoop
from benchmarks.vla.wan_va_runtime_policy import RuntimePolicy
from benchmarks.vla.util import sha256_json


class Environment:
    def __init__(self, terminal=34):
        self.env = self
        self.timestep = 0
        self.terminal = terminal
        self.actions = []

    def seed(self, seed):
        pass

    def reset(self):
        self.timestep = 0
        self.actions = []

    def set_init_state(self, state):
        pass

    def step(self, action):
        self.timestep += 1
        self.actions.append(np.array(action, copy=True))
        # Each real execution step has a different observed frame.
        obs = {key: np.full((3, 3, 3), self.timestep % 200 + 1, dtype=np.uint8)
               for key in ('agentview_image', 'robot0_eye_in_hand_image')}
        return obs, 0, self.timestep >= self.terminal, {}


class Server:
    def __init__(self, bad=False):
        self.native = SimpleNamespace(job_config=SimpleNamespace(
            obs_cam_keys=('observation.images.agentview_rgb',
                          'observation.images.eye_in_hand_rgb'), frame_chunk_size=4))
        self.events = []
        self.cycle = 0
        self.bad = bad

    def infer(self, payload):
        if payload.get('reset'):
            self.cycle = 0
            self.events.append(('reset',))
            return {}
        if payload.get('compute_kv_cache'):
            frames = payload['obs']
            values = [int(f['observation.images.agentview_rgb'][0, 0, 0]) for f in frames]
            self.events.append(('commit', values, np.array(payload['state'], copy=True)))
            return {}
        self.events.append(('predict',))
        action = np.arange(112, dtype=np.float32).reshape(7, 4, 4) + self.cycle * 200
        self.cycle += 1
        if self.bad:
            action[0, 0, 0] = np.nan
        return {'action': action}


class Remote:
    def __init__(self, runtime=False, bad=False):
        self.server = Server(bad)
        self.loop = WanVaEngineLoop(self.server) if runtime else None
        self.windows = []

    def reset_episode(self, prompt, seed):
        if self.loop:
            self.loop.reset(prompt=prompt)
        else:
            self.server.infer({'reset': True, 'prompt': prompt})

    def infer(self, payload):
        if self.loop:
            assert 'compute_kv_cache' not in payload
            self.windows.append(len(payload['obs']))
            return self.loop.predict(payload)
        return self.server.infer(payload)


def scene():
    obs = driver.initialize(Environment(), [0], 7)
    return {'init_state': [0], 'resolved_seed': 7, 'prompt': 'test',
            'initial_observation_sha256': driver.observation_digest(obs)}


def test_observed_windows_and_actions_match_explicit_commit_protocol():
    legacy_env, runtime_env = Environment(), Environment()
    legacy, runtime = Remote(), Remote(runtime=True)
    old_metrics, old_trace = driver.rollout(legacy_env, legacy, scene())
    metrics, trace = driver.rollout_runtime(runtime_env, runtime, scene())
    assert metrics == old_metrics and trace == old_trace
    assert metrics['executed_steps'] == 29
    assert runtime.windows == [1, 12, 16]
    np.testing.assert_array_equal(runtime_env.actions, legacy_env.actions)
    commits = [e for e in runtime.server.events if e[0] == 'commit']
    assert [e[1] for e in commits] == [list(range(7, 19)), list(range(19, 35))]
    old_commits = [e for e in legacy.server.events if e[0] == 'commit']
    for new, old in zip(commits, old_commits):
        assert new[1] == old[1]
        np.testing.assert_array_equal(new[2], old[2])
    assert [e[0] for e in runtime.server.events] == [
        'reset', 'predict', 'commit', 'predict', 'commit', 'predict']
    # A new episode discards the terminal chunk rather than committing it.
    driver.rollout_runtime(Environment(terminal=6), runtime, scene())
    assert [e[0] for e in runtime.server.events[-2:]] == ['reset', 'predict']


def test_runtime_nan_refused_before_any_predicted_action_executes():
    env = Environment()
    with pytest.raises(ConfigurationError, match='finite'):
        driver.rollout_runtime(env, Remote(runtime=True, bad=True), scene())
    assert env.timestep == 5  # Only the native settling steps executed.


def test_runtime_initial_scene_drift_refused_before_reset():
    frozen = scene()
    frozen['initial_observation_sha256'] = 'bad'
    remote = Remote(runtime=True)
    with pytest.raises(ConfigurationError, match='initial observation'):
        driver.rollout_runtime(Environment(), remote, frozen)
    assert remote.server.events == []


class PositionedServer(Server):
    def __init__(self):
        super().__init__()
        self.native.frame_st_id = 0

    def infer(self, payload):
        result = super().infer(payload)
        if payload.get('reset'):
            self.native.frame_st_id = 0
        elif payload.get('compute_kv_cache'):
            self.native.frame_st_id += 4
        return result


def test_seed_uses_position_after_real_deferred_commit_and_restores_hook():
    server = PositionedServer()
    seeds = []
    policy = RuntimePolicy(WanVaEngineLoop(server), server, {'test': 1}, seeds.append)
    reset = {'reset': True, 'prompt': 'test', 'benchmark_seed': 7,
             'benchmark_identity_sha256': sha256_json(policy.identity)}

    class Wire:
        def reset_episode(self, prompt, seed):
            return policy.infer(reset)

        def infer(self, payload):
            return policy.infer(payload)

    driver.rollout_runtime(Environment(), Wire(), scene())
    assert seeds == [7, 7, 11, 15]
    driver.rollout_runtime(Environment(terminal=6), Wire(), scene())
    assert seeds[-2:] == [7, 7]
    original = policy.original_infer
    policy.close()
    policy.close()
    assert server.infer == original
    with pytest.raises(ConfigurationError, match='closed'):
        policy.infer(reset)


@pytest.mark.parametrize('invalid', [True, -1, 2**63, None])
def test_invalid_reset_revokes_previous_seed(invalid):
    server = PositionedServer()
    policy = RuntimePolicy(WanVaEngineLoop(server), server, {}, lambda seed: None)
    reset = {'reset': True, 'prompt': 'test', 'benchmark_seed': 7,
             'benchmark_identity_sha256': sha256_json({})}
    policy.infer(reset)
    with pytest.raises(ConfigurationError, match='benchmark_seed'):
        policy.infer({**reset, 'benchmark_seed': invalid})
    with pytest.raises(ConfigurationError, match='before seeded reset'):
        policy.infer({'obs': []})
    policy.close()


def test_identity_mismatch_revokes_previous_seed():
    server = PositionedServer()
    policy = RuntimePolicy(WanVaEngineLoop(server), server, {}, lambda seed: None)
    reset = {'reset': True, 'prompt': 'test', 'benchmark_seed': 7,
             'benchmark_identity_sha256': sha256_json({})}
    policy.infer(reset)
    with pytest.raises(ConfigurationError, match='identity mismatch'):
        policy.infer({**reset, 'benchmark_identity_sha256': 'wrong'})
    assert policy.seed is None
    policy.close()
