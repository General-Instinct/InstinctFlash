import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.vla.campaign import build_campaign
from benchmarks.vla.util import ConfigurationError

class CampaignTests(unittest.TestCase):
    def spec(self):
        return {'schema_version':1,'adapter_id':'groot-libero-v1','driver_python':sys.executable,
                'environment':{'LIBERO_ROOT':'/tmp/libero','GR00T_ROOT':'/tmp/groot'},
                'tasks':10,'seeds':2,'candidate_tier':'BITEXACT'}
    @patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n')
    def test_draft_has_complete_frozen_pairs_and_no_reportable_claim(self,_):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'draft.json';result=build_campaign(self.spec(),output)
            self.assertEqual(result['jobs'],40)
            plan=json.loads(output.read_text())
            self.assertTrue(all(j['request']['suite']['protocol']['screening'] for j in plan['jobs']))
            self.assertTrue(all(j['request']['adapter']['contract']['id']=='groot-libero-v1' for j in plan['jobs']))
            with self.assertRaises(ConfigurationError):build_campaign(self.spec(),output)
    @patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n')
    def test_refuses_silent_task_truncation_and_missing_remote_receipts(self,_):
        with tempfile.TemporaryDirectory() as d:
            spec=self.spec();spec['tasks']=11
            with self.assertRaises(ConfigurationError):build_campaign(spec,Path(d)/'a.json')
            scene=Path(d)/'scenes.json';scene.write_text('{}')
            with self.assertRaises(ConfigurationError):build_campaign(self.spec(),Path(d)/'b.json',scene)
    def test_refuses_unknown_checkpoint_and_numeric_mislabel(self):
        with tempfile.TemporaryDirectory() as d:
            spec=self.spec();spec['checkpoint']={'id':'some/untrained','revision':'a'*40}
            with self.assertRaises(ConfigurationError):build_campaign(spec,Path(d)/'a.json')
            spec=self.spec();spec['adapter_id']='lingbot_vla_v2-robotwin-v1'
            with self.assertRaises(ConfigurationError):build_campaign(spec,Path(d)/'b.json')
    @patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n')
    def test_repeat_comparison_tiers_and_explicit_task_selection(self,_):
        with tempfile.TemporaryDirectory() as d:
            for comparison,tier in [('aa','BITEXACT'),('bb','NUMERIC'),('ab','NUMERIC')]:
                spec=dict(self.spec(),adapter_id='lingbot_vla_v2-robotwin-v1',
                          environment={'ROBOTWIN_ROOT':'/tmp/robotwin','LINGBOT_ROOT':'/tmp/va'},
                          tasks=1,seeds=1,task_names=['beat_block_hammer'],
                          comparison=comparison,candidate_tier=tier,record_inputs=True)
                output=Path(d)/(comparison+'.json');build_campaign(spec,output)
                jobs=json.loads(output.read_text())['jobs'];self.assertEqual(len(jobs),4)
                for job in jobs:
                    request=job['request'];point=request['arm']['operating_point']
                    self.assertEqual(request['task'],'beat_block_hammer')
                    self.assertTrue(point['record_inputs'])
                    expected=tier if comparison=='bb' or request['arm']['role']=='treatment' else 'BITEXACT'
                    self.assertEqual(point['tier'],expected)
    def test_pi05_repeat_alias_and_nonboolean_recording_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            for changes in [dict(adapter_id='pi05-libero-schedule-v1',comparison='aa'),dict(record_inputs='false')]:
                with self.assertRaises(ConfigurationError):
                    build_campaign(dict(self.spec(),**changes),Path(d)/'plan.json')
if __name__=='__main__':unittest.main()
