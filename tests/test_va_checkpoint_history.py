import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from instinctflash.adapters.lingbot_va import _ControlLoop, LingBotVA
from instinctflash.descriptors.known import lookup
from instinctflash.runtime.execution import WorkerBackend


class Server:
    def __init__(self):
        self.messages = []

    def infer(self, observation):
        self.messages.append(observation)
        return {'action': 'native-action'}


class CheckpointHistory(unittest.TestCase):
    def test_native_history_and_current_frame_are_distinct(self):
        for latents, first, later in [(2, 4, 8), (4, 12, 16)]:
            server = Server()
            loop = _ControlLoop(server, ('camera',), frame_chunk_size=latents)
            loop.reset(prompt='task')
            loop.predict({'obs': [{'camera': 0}]})
            loop.commit({}, 'executed-action')
            frames = [{'camera': i} for i in range(first)]
            loop.predict({'obs': frames})
            self.assertEqual(server.messages[-2]['obs'], frames)
            self.assertEqual(server.messages[-2]['state'], 'executed-action')
            self.assertEqual(server.messages[-1]['obs'], frames[-1:])
            loop.commit({}, 'next-action')
            frames = [{'camera': i} for i in range(later)]
            loop.predict({'obs': frames})
            self.assertEqual(server.messages[-2]['obs'], frames)

    def test_libero_refuses_robotwin_history_without_advancing_state(self):
        server = Server()
        loop = _ControlLoop(server, ('camera',), frame_chunk_size=4)
        loop.reset(prompt='task')
        loop.predict({'obs': [{'camera': 0}]})
        loop.commit({}, 'action')
        before = len(server.messages)
        with self.assertRaisesRegex(ValueError, 'at least 12'):
            loop.predict({'obs': [{'camera': i} for i in range(4)]})
        self.assertEqual(len(server.messages), before)
        self.assertEqual(loop._pending_action, 'action')

    def test_invalid_geometry_refused(self):
        for size in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                _ControlLoop(Server(), ('camera',), frame_chunk_size=size)

    def test_close_destroys_only_the_process_group_owned_by_runtime(self):
        with patch("torch.distributed.is_initialized", return_value=True), patch(
                "torch.distributed.destroy_process_group") as destroy:
            owned = _ControlLoop(
                Server(), ('camera',), owns_process_group=True)
            owned.close()
            owned.close()
            destroy.assert_called_once_with()

            external = _ControlLoop(
                Server(), ('camera',), owns_process_group=False)
            external.close()
            destroy.assert_called_once_with()

    def test_public_observation_contract_uses_libero_history(self):
        execution = SimpleNamespace(extra={'obs_cam_keys': ['camera'], 'height': 128,
                                           'width': 128, 'env_type': 'none'})
        with patch.dict('os.environ', {'IFL_CFG': 'libero'}), patch(
                'instinctflash.adapters.lingbot_va._upstream_va_configs',
                return_value={'libero': SimpleNamespace(frame_chunk_size=4)}):
            observation, _ = LingBotVA().observation_contract(SimpleNamespace(execution=execution))
        self.assertEqual(observation.history, 16)

    def test_published_defaults_preserve_native_schedule(self):
        for suffix, nfe, config, chunk in [('robotwin', {'video': 25, 'action': 50}, 'robotwin', 2),
                                         ('libero-long', {'video': 20, 'action': 50}, 'libero', 4)]:
            ex = lookup('robbyant/lingbot-va-posttrain-' + suffix)['execution']
            self.assertEqual(ex['nfe'], nfe)
            self.assertEqual((ex['va_config'], ex['frame_chunk_size']), (config, chunk))

    def test_worker_commits_the_executed_action_with_native_history(self):
        ex = lookup('robbyant/lingbot-va-posttrain-libero-long')['execution']
        checkpoint = SimpleNamespace(execution=SimpleNamespace(extra=ex))
        backend = WorkerBackend(LingBotVA(), checkpoint, None, port=29990)
        server = Server()
        with patch.object(backend, '_spawn'), patch(
                'instinctflash.runtime.execution._connect_client', return_value=server):
            backend.reset(prompt='task')
            backend.predict({'obs': [{'camera': 0}]}, executed_action='actually-executed')
            frames = [{'camera': i} for i in range(12)]
            backend.predict({'obs': frames})
        self.assertTrue(server.messages[-2]['compute_kv_cache'])
        self.assertEqual(server.messages[-2]['state'], 'actually-executed')
        self.assertEqual(server.messages[-2]['obs'], frames)
        self.assertEqual(server.messages[-1]['obs'], frames[-1:])

    def test_worker_launch_preserves_checkpoint_config_and_schedule(self):
        ex = lookup('robbyant/lingbot-va-posttrain-libero-long')['execution']
        checkpoint = SimpleNamespace(execution=SimpleNamespace(extra=ex, nfe=ex['nfe'], guidance=ex['guidance']))
        adapter = LingBotVA()
        with patch.object(adapter, 'materialize', return_value='/pinned/checkpoint'):
            args, _ = adapter.worker_command(checkpoint, None, port=29990, python='python')
        self.assertEqual(args[args.index('--config-name') + 1], 'libero')
        self.assertEqual(args[args.index('--degrade-nfe') + 1], '20,50')
        self.assertEqual(args[args.index('--expected-frame-chunk-size') + 1], '4')
        checkpoint.execution.nfe = {'action': 50}
        with patch.object(adapter, 'materialize', return_value='/pinned/checkpoint'):
            with self.assertRaisesRegex(ValueError, 'both video and action'):
                adapter.worker_command(checkpoint, None, port=29990, python='python')

    def test_native_frozen_stack_uses_the_pinned_package_without_refs_main(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); package = root / 'pinned'; package.mkdir()
            for name in ('transformer', *LingBotVA.FROZEN_COMPONENTS):
                (package / name).mkdir()
            checkpoint = SimpleNamespace(path=package, execution=SimpleNamespace(
                model_id='org/native', extra={'base_weights': 'org/native'}))
            with patch.dict('os.environ', {'LINGBOT_CKPT': '', 'IFL_CACHE': str(root / 'cache')}):
                composed = Path(LingBotVA.materialize(checkpoint))
                for name in LingBotVA.FROZEN_COMPONENTS:
                    self.assertEqual((composed / name).resolve(), package / name)
                (package / 'tokenizer').rmdir()
                with self.assertRaisesRegex(RuntimeError, 'pinned native checkpoint'):
                    LingBotVA.materialize(checkpoint)

    def test_worker_replaces_native_placeholder_checkpoint_path(self):
        from instinctflash.runtime.lingbot_worker import bind_checkpoint_config
        cfg = SimpleNamespace(wan22_pretrained_model_name_or_path='/path/to/pretrained/model')
        bind_checkpoint_config(cfg, '/pinned/composed')
        self.assertEqual(cfg.wan22_pretrained_model_name_or_path, '/pinned/composed')


if __name__ == '__main__':
    unittest.main()
