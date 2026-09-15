import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.vla.campaign import build_campaign
from benchmarks.vla.doctor import inspect_plan
from benchmarks.vla.registry import load_registry
from benchmarks.vla.util import load_json

class CampaignDoctorTests(unittest.TestCase):
    def test_simulator_checkout_is_checked_instead_of_hub_dataset_cache(self):
        spec={'schema_version':1,'adapter_id':'groot-libero-v1','driver_python':sys.executable,
              'environment':{'LIBERO_ROOT':'/tmp/libero'},'tasks':1,'seeds':1,'candidate_tier':'BITEXACT'}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'plan.json'
            with patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n'):
                build_campaign(spec,output)
            plan=load_json(output);registry=load_registry(output.with_suffix('.registry.json'))
            with patch('benchmarks.vla.doctor.subprocess.check_output',return_value='a'*40+'\n'):
                result=inspect_plan(plan,registry)
            self.assertFalse(any('dataset not cached' in w for w in result['warnings']))
            with patch('benchmarks.vla.doctor.subprocess.check_output',return_value='b'*40+'\n'):
                result=inspect_plan(plan,registry)
            self.assertTrue(any('expected '+ 'a'*40 in e for e in result['errors']))

    def test_capture_preflight_requires_plugin_in_selected_driver_interpreter(self):
        spec={'schema_version':1,'adapter_id':'pi05-libero-schedule-v1','driver_python':sys.executable,
              'environment':{'LIBERO_ROOT':'/tmp/libero'},'tasks':1,'seeds':1,'candidate_tier':'BITEXACT'}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'plan.json'
            with patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n'):
                build_campaign(spec,output)
            plan=load_json(output);registry=load_registry(output.with_suffix('.registry.json'))
            for installed in (False, True):
                def probe(command, **kwargs):
                    return str(installed)+'\n' if command[0] == sys.executable else 'a'*40+'\n'
                with patch('benchmarks.vla.doctor.subprocess.check_output',side_effect=probe):
                    result=inspect_plan(plan,registry)
                missing=any('adapter entry point is missing' in e for e in result['errors'])
                self.assertEqual(missing,not installed)

if __name__=='__main__':unittest.main()
