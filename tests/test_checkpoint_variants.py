import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.vla.campaign import build_campaign
from benchmarks.vla.util import ConfigurationError
class VariantTests(unittest.TestCase):
 def spec(self):return {'schema_version':1,'adapter_id':'groot-libero-v1','driver_python':sys.executable,'environment':{'GR00T_ROOT':'/tmp/groot','LIBERO_ROOT':'/tmp/libero'},'tasks':10,'seeds':2,'candidate_tier':'BITEXACT','checkpoint_subdir':'libero_spatial'}
 @patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n')
 def test_same_contract_freezes_distinct_checkpoint_directory(self,_):
  with tempfile.TemporaryDirectory() as d:
   output=Path(d)/'plan.json';build_campaign(self.spec(),output);plan=json.loads(output.read_text())
   self.assertEqual(len(plan['jobs']),40)
   for j in plan['jobs']:
    self.assertEqual(j['request']['model']['checkpoint']['subdir'],'libero_spatial')
    self.assertEqual(j['request']['suite_id'],'groot_libero_spatial')
    self.assertTrue(j['request']['task'].startswith('libero_spatial/'))
 @patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n')
 def test_wrong_suite_and_unreviewed_subdirectory_refuse(self,_):
  with tempfile.TemporaryDirectory() as d:
   for change in [{'suites':['groot_libero_10']},{'checkpoint_subdir':'untrained'}]:
    with self.assertRaises(ConfigurationError):build_campaign(dict(self.spec(),**change),Path(d)/'p.json')
if __name__=='__main__':unittest.main()
