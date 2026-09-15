"""Compatibility must be explicit before starting a model or simulator."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.vla.adapters import bind_adapter, load_adapters, validate_bound_adapter
from benchmarks.vla.plan import build_plan, load_arms
from benchmarks.vla.registry import Registry, load_registry
from benchmarks.vla.util import ConfigurationError, sha256_json


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_adapters()[0]
        self.request = {'model': {'backbone': self.contract['backbone'],
            'checkpoint': copy.deepcopy(self.contract['checkpoints'][0])},
            'suite': {'id': self.contract['suite_ids'][0], 'kind': 'closed_loop',
                      'protocol': {'bridge': self.contract['protocol'], 'evaluation_mode': 'paused_simulation'}}}
        self.driver = {'command': [sys.executable, '-m', self.contract['driver_module']]}

    def test_contract_bound_and_roundtrips(self):
        bind_adapter(self.request, self.driver)
        self.assertEqual(validate_bound_adapter(self.request), self.contract)
        self.assertEqual(self.request['adapter']['sha256'], sha256_json(self.contract))

    def test_checkpoint_from_same_backbone_is_not_compatible(self):
        self.request['model']['checkpoint'] = load_adapters()[1]['checkpoints'][0]
        with self.assertRaisesRegex(ConfigurationError, 'checkpoint'):
            bind_adapter(self.request, self.driver)

    def test_revision_override_cannot_claim_original_contract(self):
        self.request['model']['checkpoint']['revision'] = 'a'*40
        with self.assertRaisesRegex(ConfigurationError, 'checkpoint'):
            bind_adapter(self.request, self.driver)

    def test_unknown_bridge_suite_mode_and_driver_refused(self):
        original = copy.deepcopy(self.request)
        for key, value in [('bridge', 'unknown'), ('evaluation_mode', 'realtime')]:
            self.request = copy.deepcopy(original)
            self.request['suite']['protocol'][key] = value
            with self.assertRaises(ConfigurationError): bind_adapter(self.request, self.driver)
        self.request = copy.deepcopy(original)
        self.request['suite']['id'] = 'another-simulator'
        with self.assertRaises(ConfigurationError): bind_adapter(self.request, self.driver)
        self.request = original
        with self.assertRaisesRegex(ConfigurationError, 'driver module'):
            bind_adapter(self.request, {'command': ['python', '-c', 'pass', '-m', self.contract['driver_module']]})

    def test_missing_and_modified_binding_fail(self):
        with self.assertRaisesRegex(ConfigurationError, 'contract'):
            validate_bound_adapter(self.request)
        bind_adapter(self.request, self.driver)
        self.request['adapter']['contract']['actions']['wire_shape'][0] = 100
        self.request['adapter']['sha256'] = sha256_json(self.request['adapter']['contract'])
        with self.assertRaisesRegex(ConfigurationError, 'contract'):
            validate_bound_adapter(self.request)

    def test_legacy_unbridged_suite_still_uses_existing_driver(self):
        self.request['suite']['protocol'] = {}
        bind_adapter(self.request, {'command': ['legacy-driver']})
        self.assertNotIn('adapter', self.request)

    def test_explicit_driver_adapter_refuses_unknown_id_and_protocol_conflict(self):
        descriptor = dict(self.driver, adapter_id='does-not-exist')
        with self.assertRaisesRegex(ConfigurationError, 'unknown driver adapter'):
            bind_adapter(self.request, descriptor)
        descriptor['adapter_id'] = 'pi05-libero-schedule-v1'
        with self.assertRaisesRegex(ConfigurationError, 'conflicts'):
            bind_adapter(self.request, descriptor)

    def test_build_plan_binds_every_real_robotwin_job(self):
        reg = load_registry(); raw = copy.deepcopy(reg.raw)
        for suite in raw['suites']:
            if suite['id'] == 'robotwin50_easy':
                suite['protocol'].update(bridge='wan-va-robotwin-paused-v1', evaluation_mode='paused_simulation')
        raw['profiles']['adapter_test'] = {'suites': ['robotwin50_easy'],
            'limits': {'tasks': 1, 'seeds_per_task': {'closed_loop': 1}},
            'arm_repeats': 1, 'latency': {'warmup': 0, 'iterations': 1}}
        registry = Registry(raw, sha256_json(raw), reg.path)
        arms = load_arms(Path(__file__).resolve().parents[1]/'benchmarks/vla/config/arms.ci.json')
        for arm in arms['arms']:
            arm['driver']['command'] = [sys.executable, '-m', 'benchmarks.vla.robotwin_driver']
        plan = build_plan(registry, arms, 'adapter_test', ['robbyant/lingbot-va-posttrain-robotwin'])
        self.assertTrue(plan['jobs'])
        for job in plan['jobs']:
            validate_bound_adapter(job['request'])
            self.assertEqual(job['request_sha256'], sha256_json(job['request']))


if __name__ == '__main__': unittest.main()
