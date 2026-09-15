import unittest
from benchmarks.vla.realtime import assess,decision
from benchmarks.vla.util import ConfigurationError
class BudgetTests(unittest.TestCase):
 def setUp(self):
  self.measurement={'synthetic':False,'hardware':{'gpu_name':'NVIDIA Thor'},'executed_actions':50,'samples_ms':[100,200,1200]}
  self.budget={'target_device':'Thor','control_hz':50,'executed_actions':50,'scheduling':'pipelined','max_deadline_miss_fraction':0}
 def test_pipelining_does_not_override_reaction_deadline(self):
  r=assess(self.measurement,self.budget);self.assertEqual(r['deadline_ms'],1000);self.assertEqual(r['deadline_misses'],1)
  self.budget['max_observation_to_action_ms']=150
  self.assertEqual(assess(self.measurement,self.budget)['deadline_misses'],2)
 def test_blocking_controller_has_only_one_control_period(self):
  self.budget['scheduling']='blocking';r=assess(self.measurement,self.budget)
  self.assertEqual(r['deadline_ms'],20);self.assertEqual(r['deadline_misses'],3)
 def test_wrong_device_nonfinite_samples_and_geometry_refuse(self):
  for update in [{'samples_ms':[float('nan')]},{'hardware':{'gpu_name':'H100'}},{'executed_actions':10},{'synthetic':True}]:
   with self.assertRaises(ConfigurationError):assess(dict(self.measurement,**update),self.budget)
 def test_fast_numeric_candidate_does_not_authorize_loss(self):
  self.assertEqual(decision([{'mode':'stock','assessment':{'meets_observed_budget':True}}]),'baseline_meets_observed_budget_no_lossy_optimization_needed')
  result=decision([{'mode':'runtime_default','tier':'NUMERIC','assessment':{'meets_observed_budget':True}}]);self.assertIn('minimum_quality_loss',result)
 def test_empty_or_nonstring_target_refused(self):
  for target in ['', ' ', None, 123]:
   with self.assertRaises(ConfigurationError):assess(self.measurement,dict(self.budget,target_device=target))
 def test_cli_reads_explicit_budget_and_preserves_existing_output(self):
  import contextlib,io,json,tempfile
  from pathlib import Path
  from benchmarks.vla.cli import main
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);measurement=root/'measurement.json';budget=root/'budget.json';output=root/'report.json'
   measurement.write_text(json.dumps(self.measurement));budget.write_text(json.dumps(self.budget))
   args=['realtime-report','--measurement',str(measurement),'--budget',str(budget),'--output',str(output)]
   with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
    self.assertEqual(main(args),0);original=output.read_bytes()
    with self.assertRaises(SystemExit) as refused:main(args)
    self.assertEqual(refused.exception.code,2)
   self.assertEqual(output.read_bytes(),original)
   report=json.loads(original);self.assertEqual(report['deadline_misses'],1);self.assertFalse(report['deployment_certified'])
if __name__=='__main__':unittest.main()
