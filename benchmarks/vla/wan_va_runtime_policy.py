"""Seeded benchmark transport for an owned, already-built VA Runtime.

Identity admission and Runtime construction belong to the endpoint. Construct
without a fixed Runtime seed: this wrapper owns per-episode noise seeding.
The supplied server must be the exact instance used by Runtime's control loop.
"""
from __future__ import annotations

from .util import ConfigurationError, sha256_json


class RuntimePolicy:
    def __init__(self, runtime, server, identity, seed_fn):
        self.runtime = runtime
        self.server = server
        self.identity = identity
        self.seed_fn = seed_fn
        self.seed = None
        self.original_infer = server.infer
        self.closed = False
        # FP8 owns a native conditioning server; native uses that server directly.
        self.position_owner = getattr(server, 'native', server)
        if not hasattr(self.position_owner, 'frame_st_id'):
            raise ConfigurationError('VA server must expose its actual history position')
        self.hook = self._seeded_infer
        server.infer = self.hook

    def _seeded_infer(self, observation):
        if self.seed is None:
            raise ConfigurationError('inference before seeded reset')
        if not observation.get('reset') and not observation.get('compute_kv_cache'):
            # Deferred commits have now advanced the actual native history.
            self.seed_fn(self.seed + int(self.position_owner.frame_st_id))
        return self.original_infer(observation)

    def infer(self, observation):
        if self.closed:
            raise ConfigurationError('VA benchmark policy is closed')
        observation = dict(observation)
        if observation.get('reset'):
            # A rejected/failed reset also revokes the previous episode.
            self.seed = None
            seed = observation.get('benchmark_seed')
            expected = sha256_json(self.identity)
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ConfigurationError('reset requires a non-negative benchmark_seed')
            if observation.get('benchmark_identity_sha256') != expected:
                raise ConfigurationError('reset identity mismatch')
            self.seed = seed
            try:
                self.seed_fn(seed)
                self.runtime.reset(prompt=observation.get('prompt'))
            except BaseException:
                self.seed = None
                raise
            return {'benchmark_seed': seed, 'benchmark_identity_sha256': expected}
        if self.seed is None:
            raise ConfigurationError('inference before seeded reset')
        if observation.get('compute_kv_cache') or 'executed_action' in observation:
            raise ConfigurationError('benchmark requires Runtime deferred history and unchanged actions')
        try:
            return self.runtime.predict(observation)
        except BaseException:
            self.seed = None
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.seed = None
            if self.server.infer is self.hook:
                self.server.infer = self.original_infer
