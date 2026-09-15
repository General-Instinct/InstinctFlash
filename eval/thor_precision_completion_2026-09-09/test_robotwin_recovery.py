"""Safety controls for admitting pre-policy retries; no GPU/model execution."""
import json,tempfile,unittest
from pathlib import Path
from recover_robotwin_fp8 import retryable

class RetryAdmission(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.path=Path(self.tmp.name)/'trial.json'
  self.trace=self.path.with_suffix('.episode')/'trace/trace.json';self.trace.parent.mkdir(parents=True)
  self.identity={'precision':'fp8','checkpoint':'pinned'}
  self.data={'complete':True,'calls':[],'identity':self.identity}
  self.log=self.path.with_suffix('.log')
  self.log.write_text('expert gate rejected pinned seed 110101 (accepted previously)\nRuntimeError: [InstinctWM] pinned seed list exhausted after 0/1 accepted episodes')
 def admitted(self):
  self.trace.write_text(json.dumps(self.data))
  return retryable(self.path,self.identity,110101)
 def test_only_explicit_empty_completed_trace_admitted(self):self.assertTrue(self.admitted())
 def test_reset_call_alone_prevents_retry(self):
  self.data['calls']=[{'reset':True}];self.assertFalse(self.admitted())
 def test_incomplete_trace_prevents_retry(self):
  self.data['complete']=False;self.assertFalse(self.admitted())
 def test_identity_change_prevents_retry(self):
  self.data['identity']={'precision':'native'};self.assertFalse(self.admitted())
 def test_existing_task_result_prevents_retry(self):
  self.path.write_text('{"success": false}');self.assertFalse(self.admitted())
 def test_unexpected_failure_prevents_retry(self):
  self.log.write_text('CUDA out of memory');self.assertFalse(self.admitted())
 def test_seed_substitution_prevents_retry(self):
  self.log.write_text(self.log.read_text().replace('110101','110102'));self.assertFalse(self.admitted())

if __name__=='__main__':unittest.main()
