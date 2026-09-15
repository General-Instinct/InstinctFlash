"""Independent endpoint pairs overlap; stateful episodes and resume settings do not."""
import copy
import json
import os
import time
from unittest.mock import patch
import sys
import tempfile
import unittest
from pathlib import Path
from benchmarks.vla.plan import build_plan
from benchmarks.vla.registry import Registry
from benchmarks.vla.runner import execute_plan, _execution_groups
from benchmarks.vla.util import ConfigurationError, sha256_json
from tests.test_vla_benchmark_pipeline import _unit_registry, _arms, ROOT

class PairedRunnerTests(unittest.TestCase):
    def make_plan(self, directory, fail=False, delay=.15):
        registry = _unit_registry()
        raw = copy.deepcopy(registry.raw)
        suite = next(s for s in raw['suites'] if s['id'] == 'model_contract')
        suite.update(kind='closed_loop', tasks=['task_a','task_b'])
        raw['profiles']['unit'].update(suites=['model_contract'], limits={'tasks':2,'seeds_per_task':{'closed_loop':1}},arm_repeats=1)
        registry = Registry(raw,sha256_json(raw),registry.path)
        wrapper = Path(directory)/'driver.py'
        wrapper.write_text('''import json,sys,time,os
from pathlib import Path
from benchmarks.vla.reference_driver import main
request=Path(sys.argv[sys.argv.index('--request')+1]);job=json.loads(request.read_text())
Path(str(request)+'.pid').write_text(str(os.getpid()))
started=time.time();time.sleep(DELAY)
result=main()
Path(str(request)+'.timing').write_text(json.dumps([started,time.time()]))
if FAIL and not (request.parent.parent/'allow-success').exists() and job['request']['arm']['role']=='treatment':sys.exit(7)
sys.exit(result)
'''.replace('FAIL',repr(fail)).replace('DELAY',repr(delay)))
        arms = _arms()
        for i,arm in enumerate(arms['arms']):
            arm['driver']['command']=[sys.executable,str(wrapper)]
            arm['driver']['environment']['PYTHONPATH']=str(ROOT)
            arm['operating_point']['remote']={'endpoint':f'ws://localhost:{30000+i}'}
        return build_plan(registry,arms,'unit',['lerobot/pi05_base'])

    def test_overlap_pair_barrier_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_plan(directory);root=Path(directory)/'run'
            first=execute_plan(plan,root,repo_root=ROOT,paired_workers=2)
            self.assertEqual(first['completed'],4);self.assertFalse(first['failed'])
            groups=_execution_groups(plan,2)
            times=[[json.loads((root/'requests'/(job['job_id']+'.json.timing')).read_text()) for _,job in group] for group in groups]
            for pair in times:self.assertLess(max(x[0] for x in pair),min(x[1] for x in pair))
            self.assertGreaterEqual(min(x[0] for x in times[1]),max(x[1] for x in times[0]))
            resumed=execute_plan(plan,root,repo_root=ROOT,paired_workers=2)
            self.assertEqual(resumed['resumed'],4)
            with self.assertRaisesRegex(ConfigurationError,'environment drifted'):
                execute_plan(plan,root,repo_root=ROOT,paired_workers=1)

    def test_fail_fast_finishes_current_pair_only(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_plan(directory,True);root=Path(directory)/'run'
            result=execute_plan(plan,root,repo_root=ROOT,paired_workers=2,fail_fast=True)
            self.assertEqual(result['completed'],1);self.assertEqual(len(result['failed']),1)
            self.assertEqual(len(list((root/'requests').glob('*.json'))),2)
            failed_id=result['failed'][0]['job_id']
            previous_log=(root/'logs'/(failed_id+'.log')).read_bytes()
            trace=root/'results'/('.'+failed_id+'.pending.trace');trace.mkdir()
            (trace/'trace.json').write_text('{"complete":false}')
            (trace/'00000.request.msgpack').write_bytes(b'recorded request before failure')
            retry=execute_plan(plan,root,repo_root=ROOT,paired_workers=2,fail_fast=True)
            self.assertEqual(retry['resumed'],1)
            archived=root/'failures'/'attempts'/failed_id/'0001'
            self.assertEqual((archived/'logs'/(failed_id+'.log')).read_bytes(),previous_log)
            self.assertTrue((archived/'results'/('.'+failed_id+'.pending.json')).exists())
            inventory=json.loads((archived/'attempt.json').read_text())
            self.assertIn('failures/'+failed_id+'.json',inventory['files'])
            relative='results/'+trace.name+'/00000.request.msgpack'
            self.assertEqual((archived/relative).read_bytes(),b'recorded request before failure')
            self.assertIn(relative,inventory['files']);self.assertFalse(trace.exists())
            (root/'allow-success').touch()
            recovered=execute_plan(plan,root,repo_root=ROOT,paired_workers=2,fail_fast=True)
            self.assertEqual(recovered['completed'],4);self.assertFalse(recovered['failed'])
            self.assertFalse((root/'failures'/(failed_id+'.json')).exists())
            self.assertTrue((root/'failures'/'attempts'/failed_id/'0002'/'attempt.json').exists())

    def test_validation_exception_terminates_concurrent_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_plan(directory,delay=30);root=Path(directory)/'run'
            first,second=plan['jobs'][:2]
            marker=root/'requests'/(first['job_id']+'.json.pid')
            def existing(path,job):
                if job['job_id']==second['job_id']:
                    deadline=time.monotonic()+5
                    while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
                    self.assertTrue(marker.exists())
                    raise ConfigurationError('injected invalid completed evidence')
                return False
            with patch('benchmarks.vla.runner._valid_existing',side_effect=existing):
                with self.assertRaisesRegex(ConfigurationError,'refusing to overwrite'):
                    execute_plan(plan,root,repo_root=ROOT,paired_workers=2)
            pid=int(marker.read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_refuses_shared_endpoints_and_non_quality_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_plan(directory)
            for job in plan['jobs']:job['request']['arm']['operating_point']['remote']['endpoint']='ws://localhost:1'
            with self.assertRaisesRegex(ConfigurationError,'independent remote'):_execution_groups(plan,2)
            plan['jobs'][0]['request']['suite']['kind']='latency'
            with self.assertRaisesRegex(ConfigurationError,'closed-loop'):_execution_groups(plan,2)

if __name__=='__main__':unittest.main()
