"""Evidence bindings and conservative deployment choice; no GPU or fabricated release claims."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.vla.execution_evidence import execution_profile,verify_profile,build_record,verify_record
from benchmarks.vla.configuration_select import select
from benchmarks.vla.util import ConfigurationError,sha256_json


def receipt(mode='stock'):
    return dict(synthetic=False,model_id='test/model',model_revision='a'*40,
        checkpoint_sha256='1'*64,upstream_sha256='2'*64,runtime_source_sha256='3'*64,
        pipeline_sha256='4'*64,adapter_sha256=None if mode=='stock' else '5'*64,
        hardware={'gpu_name':'NVIDIA Thor','capability':[11,0]},packages={'torch':'test'},
        numeric_environment={'matmul_tf32':True},
        execution={'mode':mode,'precision':'native','backend':'stock' if mode=='stock' else 'in_process',
                   'declared':{'nfe':{'action':10},'guidance':{'action':'none'}},'action_shape':[50,14],
                   'transforms':[] if mode=='stock' else [{'name':'capture','tier':'NUMERIC','params':{}}]})


class Bindings(unittest.TestCase):
    def test_identity_changes_with_weights_device_precision_schedule_code_and_numeric_flags(self):
        original=receipt();base=execution_profile(original)['profile_id']
        changes=[('checkpoint_sha256','6'*64),('runtime_source_sha256','7'*64),
                 ('hardware',{'gpu_name':'H100'}),('numeric_environment',{'matmul_tf32':False})]
        for key,value in changes:
            changed=copy.deepcopy(original);changed[key]=value
            self.assertNotEqual(base,execution_profile(changed)['profile_id'])
        changed=copy.deepcopy(original);changed['execution']['declared']['nfe']['action']=4
        self.assertNotEqual(base,execution_profile(changed)['profile_id'])
        changed=copy.deepcopy(original);changed['execution']['precision']='fp8'
        with self.assertRaises(ConfigurationError):execution_profile(changed)
    def test_counters_and_self_check_timing_are_not_new_executions(self):
        a=receipt('runtime_default');a['execution']['graph_stats']={'captured':True,'replays':2}
        a['execution']['transforms'][0]['params']['self_check']={'passed':True,'tolerance':.05,'seconds':1,'max_abs_delta':.01}
        b=copy.deepcopy(a);b['execution']['graph_stats']['replays']=100
        b['execution']['transforms'][0]['params']['self_check'].update(seconds=7,max_abs_delta=.02)
        self.assertEqual(execution_profile(a),execution_profile(b))
    def test_legacy_receipt_and_tampered_profile_refuse(self):
        a=receipt();del a['runtime_source_sha256']
        with self.assertRaises(ConfigurationError):execution_profile(a)
        p=execution_profile(receipt());p['facts']['checkpoint_sha256']='9'*64
        with self.assertRaises(ConfigurationError):verify_profile(p)
    def test_resigning_a_claim_does_not_create_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'receipt.json';path.write_text(json.dumps(receipt()))
            record=build_record(path);record['quality_status']='certified'
            record.pop('record_sha256');record['record_sha256']=sha256_json(record)
            with self.assertRaisesRegex(ConfigurationError,'recomputed'):verify_record(record)
    def test_receipt_timing_binding_and_later_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);a=receipt();(r/'receipt.json').write_text(json.dumps(a))
            timing={'synthetic':False,'execution_profile':execution_profile(a),'hardware':a['hardware'],
                    'executed_actions':50,'samples_ms':[100,200],'iterations':2,'warmup':1}
            (r/'timing.json').write_text(json.dumps(timing))
            record=build_record(r/'receipt.json',measurement_path=r/'timing.json')
            self.assertEqual(record['quality_status'],'unmeasured');verify_record(record)
            timing['execution_profile']=execution_profile(receipt('runtime_default'))
            (r/'timing.json').write_text(json.dumps(timing))
            with self.assertRaises(ConfigurationError):build_record(r/'receipt.json',measurement_path=r/'timing.json')
            with self.assertRaises(ConfigurationError):verify_record(record)


class CampaignIdentity(unittest.TestCase):
    def test_runtime_metadata_is_not_geometry_but_precision_and_steps_are(self):
        from benchmarks.vla.campaign import _validate_paired_identities
        a=receipt();b=receipt('runtime_default')
        b['execution'].update(graph_stats={'captured':True},capture_required=True,tier_ceiling='numeric')
        _validate_paired_identities([a,b])
        for field,value in [('precision','fp8'),('action_shape',[10,7]),
                            ('declared',{'nfe':{'action':4},'guidance':{'action':'none'}})]:
            altered=copy.deepcopy(b);altered['execution'][field]=value
            with self.assertRaises(ConfigurationError):_validate_paired_identities([a,altered])
    def test_capture_requirement_refuses_live_fallback(self):
        import numpy as np
        from types import SimpleNamespace
        from benchmarks.vla.joint_policy_server import Policy
        class Arm:
            _runtime=SimpleNamespace(_backend=SimpleNamespace(_impl=SimpleNamespace(graph_stats={'captured':False})))
            def new_episode(self,prompt):pass
            def predict(self,obs):return np.zeros((50,14))
        identity={'execution':{'action_shape':[50,14]}}
        policy=Policy(Arm(),identity,lambda seed:None,require_capture=True)
        policy.infer({'reset':True,'prompt':'test','benchmark_seed':1,'benchmark_identity_sha256':sha256_json(identity)})
        with self.assertRaisesRegex(ConfigurationError,'capture'):policy.infer({})


class Selection(unittest.TestCase):
    def setUp(self):
        self.base=self.record(receipt(),[100,200]);self.candidate=self.record(receipt('runtime_default'),[20,30])
        self.budget={'target_device':'Thor','control_hz':50,'executed_actions':50,
                     'scheduling':'pipelined','max_deadline_miss_fraction':0}
        self.policy={'baseline_profile_id':self.base['profile']['profile_id'],'tier_ceiling':'numeric',
                     'required_suites':['clean'],'required_tasks':{'clean':['task']},'max_quality_loss':.05,'quality_registry_sha256':'a'*64}
    def record(self,identity,samples):
        return {'profile':execution_profile(identity),'quality_status':'unmeasured','quality':[],
                'latency':{'synthetic':False,'hardware':identity['hardware'],'executed_actions':50,'samples_ms':samples}}
    def run_select(self):
        # Only the decision unit is isolated here; binding/rebuild has separate filesystem tests.
        with patch('benchmarks.vla.configuration_select.verify_record') as verify:
            result=select([self.base,self.candidate],self.budget,self.policy)
            self.assertEqual(verify.call_count,2)
            return result
    def certify(self,reference=None):
        c=self.candidate
        c['quality_status']='certified';c['quality']=[{'status':'certified','registry_sha256':'a'*64,'matched_arms':['candidate'],
            'comparisons':[{'control':'stock','treatment':'candidate','closed_loop':[{'suite_id':'clean','verdict':'PASS'}],
                            'paired_success':[{'suite_id':'clean','per_task':{'task':{'pairs':100}}}]}],
            'reference_profiles':{'stock':reference or self.base['profile']},
            'success_gates':{'candidate':{'margin':-.05}}}]
    def test_baseline_first_even_when_candidate_is_faster(self):
        self.assertEqual(self.run_select()['selected_profile_id'],self.base['profile']['profile_id'])
    def test_screening_does_not_authorize_numeric_upgrade(self):
        self.budget['max_observation_to_action_ms']=50;self.candidate['quality_status']='screen'
        self.assertIsNone(self.run_select()['selected_profile_id'])
    def test_matching_certificate_can_select_when_baseline_misses(self):
        self.budget['max_observation_to_action_ms']=50;self.certify()
        self.assertEqual(self.run_select()['selected_profile_id'],self.candidate['profile']['profile_id'])
    def test_self_repeat_other_reference_and_wrong_suite_do_not_authorize(self):
        self.budget['max_observation_to_action_ms']=50;self.certify(self.candidate['profile'])
        self.assertIsNone(self.run_select()['selected_profile_id'])
        self.certify();self.policy['required_suites']=['randomized'];self.policy['required_tasks']={'randomized':['task']}
        self.assertIsNone(self.run_select()['selected_profile_id'])
    def test_strict_permission_and_quality_margin_are_enforced(self):
        self.budget['max_observation_to_action_ms']=50;self.certify();self.policy['tier_ceiling']='bitexact'
        self.assertIsNone(self.run_select()['selected_profile_id'])
        self.policy['tier_ceiling']='numeric';self.policy['max_quality_loss']=.01
        self.assertIsNone(self.run_select()['selected_profile_id'])
    def test_other_registry_protocol_cannot_authorize_selection(self):
        self.budget['max_observation_to_action_ms']=50;self.certify()
        self.policy['quality_registry_sha256']='b'*64
        self.assertIsNone(self.run_select()['selected_profile_id'])
    def test_numeric_permission_does_not_authorize_fp8(self):
        self.budget['max_observation_to_action_ms']=50
        identity=receipt('runtime_default');identity['execution'].update(precision='fp8',backend='engine',
            artifacts={'engine_sha256':'8'*64,'calibration_sha256':'9'*64})
        self.candidate=self.record(identity,[20,30]);self.certify()
        self.assertIsNone(self.run_select()['selected_profile_id'])
        self.policy['allowed_precisions']=['native','fp8']
        self.assertEqual(self.run_select()['selected_profile_id'],self.candidate['profile']['profile_id'])
    def test_certified_suite_does_not_cover_an_unmeasured_task(self):
        self.budget['max_observation_to_action_ms']=50;self.certify()
        self.policy['required_tasks']={'clean':['unmeasured-task']}
        self.assertIsNone(self.run_select()['selected_profile_id'])
        self.policy['required_tasks']={'clean':['task']}
        self.candidate['quality'][0]['comparisons'][0]['paired_success'][0]['per_task']['extra']={'pairs':100}
        self.assertIsNone(self.run_select()['selected_profile_id'])
    def test_different_checkpoint_and_step_change_refuse(self):
        self.budget['max_observation_to_action_ms']=50;self.certify()
        self.candidate['profile']['facts']['checkpoint_sha256']='9'*64
        self.assertIsNone(self.run_select()['selected_profile_id'])
        self.candidate['profile']['facts']['checkpoint_sha256']=self.base['profile']['facts']['checkpoint_sha256']
        self.candidate['profile']['facts']['execution']['declared']['nfe']['action']=4
        self.assertIsNone(self.run_select()['selected_profile_id'])


class CliTests(unittest.TestCase):
    def test_real_file_binding_selection_exit_and_overwrite_guard(self):
        import contextlib,io
        from benchmarks.vla.cli import main
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=receipt();profile=execution_profile(a)
            def write(name,value):
                p=root/name;p.write_text(json.dumps(value));return p
            rp=write('receipt.json',a)
            mp=write('timing.json',{'synthetic':False,'execution_profile':profile,'hardware':a['hardware'],
                'executed_actions':50,'samples_ms':[100,200],'warmup':1,'iterations':2})
            out=root/'record.json'
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['execution-report','--receipt',str(rp),'--measurement',str(mp),'--output',str(out)]),0)
                policy=write('policy.json',{'baseline_profile_id':profile['profile_id']})
                budget={'target_device':'Thor','control_hz':50,'executed_actions':50,'scheduling':'pipelined','max_deadline_miss_fraction':0}
                bp=write('budget.json',budget);selection=root/'selection.json'
                args=['select-configuration','--record',str(out),'--budget',str(bp),'--policy',str(policy),'--output',str(selection)]
                self.assertEqual(main(args),0)
                saved=selection.read_bytes()
                with self.assertRaises(SystemExit):main(args)
                self.assertEqual(selection.read_bytes(),saved)
                budget['scheduling']='blocking';write('budget.json',budget)
                args[-1]=str(root/'miss.json');self.assertEqual(main(args),3)

if __name__=='__main__':unittest.main()
