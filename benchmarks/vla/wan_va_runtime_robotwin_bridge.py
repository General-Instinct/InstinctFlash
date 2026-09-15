"""Adapt the original RoboTwin client to Runtime's deferred observed history.

The client still owns pose composition, quaternion normalization and execution.
Its commit message stages real observations for the next Runtime prediction.
The final staged history is not sent when an episode ends without another
prediction; this follows Runtime's API lifecycle, not the legacy server's final
cache-write work. Endpoint identity admission remains the caller's responsibility.
"""
from __future__ import annotations

import numpy as np

from .util import ConfigurationError


class WanVaRuntimeWireBridge:
    def __init__(self, remote, scene):
        self.remote, self.scene = remote, scene
        self.phase, self.cycles = 'reset', 0
        self.frames = None
        self.action = None

    def infer(self, observation):
        if observation.get('reset'):
            if self.phase != 'reset' or observation.get('prompt') != self.scene['prompt']:
                raise ConfigurationError('unexpected episode reset or prompt')
            reply = self.remote.reset_episode(self.scene['prompt'], self.scene['resolved_seed'])
            self.phase = 'infer'
            return reply
        if observation.get('compute_kv_cache'):
            if self.phase != 'commit':
                raise ConfigurationError('commit arrived without an inference')
            frames = observation.get('obs')
            expected = 4 if self.cycles == 0 else 8
            if not isinstance(frames, list) or len(frames) != expected:
                raise ConfigurationError(f'Runtime commit requires {expected} observed frames')
            state = np.asarray(observation.get('state'))
            if (state.dtype != self.action.dtype or state.shape != self.action.shape
                    or state.tobytes() != self.action.tobytes()):
                raise ConfigurationError('client changed the predicted action history')
            self.frames = frames
            self.phase = 'infer'
            self.cycles += 1
            return {}
        if self.phase != 'infer':
            raise ConfigurationError('infer arrived before reset/commit')
        request = {k: v for k, v in observation.items()
                   if k not in {'video_guidance_scale', 'action_guidance_scale'}}
        if self.cycles:
            request['obs'] = self.frames
        else:
            first = request.get('obs')
            if not isinstance(first, dict):
                raise ConfigurationError('first prediction requires the native camera mapping')
            request['obs'] = [first]
        request['save_visualization'] = False
        reply = self.remote.infer(request)
        action = np.asarray(reply.get('action'))
        if action.shape != (16, 2, 16) or not np.isfinite(action).all():
            raise ConfigurationError('Runtime requires finite action shape (16, 2, 16)')
        self.action = action.copy()
        self.frames = None
        self.phase = 'commit'
        return reply
