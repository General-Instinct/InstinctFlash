"""Explicit per-request native seed for an isolated offline quality worker.

Runtime's normal base seed initializes a stream; this experiment instead uses
the producer's frozen request seed. Never use concurrently with other requests.
"""
import copy
import dataclasses
import functools
from numbers import Integral


def predict_with_request_seed(runtime, observation, seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError('Require an explicit uint32 request seed')
    seed = int(seed)
    loop = runtime._backend._impl
    service = loop._native_loop._service
    if getattr(loop, '_padding', None) is None or getattr(loop, '_sampler', None) is None:
        raise ValueError('Require the installed zero-padding fixed-step runtime')
    config, rng = service.cfg, service._rng
    rng_before = copy.deepcopy(rng.bit_generator.state)
    model = service.model
    owned = 'generate_samples_from_batch' in vars(model)
    original = model.generate_samples_from_batch
    calls = []

    @functools.wraps(original)
    def capture(*args, **kwargs):
        if calls or kwargs.get('seed') != [seed]:
            raise RuntimeError('Require exactly one generation call with the declared request seed')
        calls.append(list(kwargs['seed']))
        return original(*args, **kwargs)

    replacement = dataclasses.replace(config, seed=seed, deterministic_seed=True)
    service.cfg = replacement
    model.generate_samples_from_batch = capture
    try:
        result = runtime.predict(observation)
        if calls != [[seed]]:
            raise RuntimeError('No native generation call observed')
    finally:
        intact = model.generate_samples_from_batch is capture and service.cfg is replacement
        if owned:
            model.generate_samples_from_batch = original
        else:
            del model.generate_samples_from_batch
        service.cfg = config
        # Native deterministic mode must not consume the ordinary request stream.
        rng_unchanged = service._rng is rng and rng.bit_generator.state == rng_before
        service._rng = rng
        rng.bit_generator.state = rng_before
        if not intact or not rng_unchanged:
            raise RuntimeError('Offline seed override encountered concurrent mutation or consumed the request RNG')
    return result, dict(request_seed=seed, actual_generation_seeds=calls,
        seed_mode='explicit offline deterministic request; not Runtime base-stream seed',
        normal_seed_config_restored=True, normal_request_rng_unchanged=True)
