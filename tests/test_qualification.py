import copy,json,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from benchmarks.vla.qualification import requested_profile_id,read_startup,report
from benchmarks.vla.execution_evidence import execution_profile
from benchmarks.vla.registry import load_registry
from benchmarks.vla.util import ConfigurationError
from test_execution_selection import receipt

class Qualification(unittest.TestCase):
    def startup(self):
        r=receipt('runtime_default');r['execution'].update(capture_required=True,graph_stats={'captured':True})
        r['execution']['transforms'][0]['params']['self_check']={'passed':True,'n':6,'tolerance':.05,'repeats':3}
        r['startup']={'protocol':'seeded-eight-call-admission-v1','attempt_id':uuid.uuid4().hex,'seed':0,'calls':8,'status':'ready'}
        return r
    def test_request_groups_acceptance_and_rejection_but_keeps_guard_strength(self):
        a=self.startup();b=copy.deepcopy(a)
        b['execution']['graph_stats']['captured']=False;b['execution']['transforms'][0]['params']['self_check']['passed']=False
        self.assertNotEqual(execution_profile(a),execution_profile(b))
        self.assertEqual(requested_profile_id(execution_profile(a)),requested_profile_id(execution_profile(b)))
        b['execution']['transforms'][0]['params']['self_check']['repeats']=1
        self.assertNotEqual(requested_profile_id(execution_profile(a)),requested_profile_id(execution_profile(b)))
    def test_receipt_status_must_match_executed_capture(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.json';r=self.startup();p.write_text(json.dumps(r));read_startup(p)
            r['startup']['status']='rejected';p.write_text(json.dumps(r))
            with self.assertRaises(ConfigurationError):read_startup(p)
    def test_duplicate_attempt_cannot_inflate_sample_size(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.json';p.write_text(json.dumps(self.startup()))
            with self.assertRaisesRegex(ConfigurationError,'duplicate startup'):
                report(load_registry(),target_device='Thor',startup_receipts=[p,p])
    def test_fault_drill_cannot_count_as_ordinary_admission(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fault.json';r=self.startup()
            r['startup']['fault_injection']={'IFL_VLA2_SELFCHECK_FAULT':'1'}
            p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ConfigurationError,'fault-injected'):
                read_startup(p)
    def test_va_history_protocol_is_checkpoint_specific(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'va.json';r=self.startup()
            r['model_id']='robbyant/lingbot-va-posttrain-libero-long'
            r['execution'].update(capture_required=False,graph_stats={})
            r['startup'].update(protocol='seeded-eight-predict-seven-commit-v1',
                                commit_calls=7,commit_frames=[12]+[16]*6)
            p.write_text(json.dumps(r));read_startup(p)
            r['startup']['commit_frames']=[4]+[8]*6;p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ConfigurationError,'commit history'):read_startup(p)
    def test_contracts_are_checkpoint_specific_and_variants_separate(self):
        value=report(load_registry(),target_device='Thor');rows=value['checkpoints']
        base=next(r for r in rows if r['checkpoint']=='lerobot/pi05_base')
        ft=next(r for r in rows if r['checkpoint']=='lerobot/pi05_libero_finetuned_v044')
        self.assertEqual(base['adapters'],[]);self.assertTrue(ft['adapters'])
        variants=[r for r in rows if r['checkpoint']=='nvidia/GR00T-N1.7-LIBERO']
        self.assertEqual({r['checkpoint_subdir'] for r in variants},{'libero_10','libero_spatial','libero_object','libero_goal'})
        self.assertTrue(all(not r['bound_executions'] for r in rows));self.assertFalse(value['deployment_certified'])
    def test_missing_device_and_invalid_sample_requirement_refuse(self):
        for device,count in [('',3),('Thor',0),('Thor',True)]:
            with self.assertRaises(ConfigurationError):report(load_registry(),target_device=device,minimum_startups=count)

if __name__=='__main__':unittest.main()
