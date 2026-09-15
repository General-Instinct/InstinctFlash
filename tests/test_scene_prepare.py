import unittest
from benchmarks.vla.scene_prepare import task_shards,merge_manifests
from benchmarks.vla.util import load_json,ConfigurationError
from benchmarks.vla.plan import validate_plan
from pathlib import Path
from unittest.mock import patch
import tempfile
from benchmarks.vla.campaign import build_campaign
import sys

class ScenePreparationTests(unittest.TestCase):
    def plan(self):
        spec={'schema_version':1,'adapter_id':'groot-libero-v1','driver_python':sys.executable,
              'environment':{'LIBERO_ROOT':'/tmp/libero'},'tasks':2,'seeds':2,'candidate_tier':'BITEXACT'}
        with tempfile.TemporaryDirectory() as d, patch('benchmarks.vla.campaign.subprocess.check_output',return_value='a'*40+'\n'):
            output=Path(d)/'plan.json';build_campaign(spec,output);return load_json(output)
    def test_task_shards_preserve_all_paired_jobs(self):
        plan=self.plan();shards=task_shards(plan)
        self.assertEqual(len(shards),2)
        self.assertEqual(sum(len(s['jobs']) for _,s in shards),len(plan['jobs']))
        for _,shard in shards:
            validate_plan(shard)
            self.assertEqual(shard['preparation_parent_plan_id'],plan['plan_id'])
            self.assertEqual(len(shard['jobs']),4)
    def test_reused_manifest_keeps_scene_identity_separate_from_preparation(self):
        plan=self.plan()
        scenes={f"{j['request']['task']}/{j['request']['requested_seed']}":{} for j in plan['jobs']}
        manifest={'protocol':'test','sources':{'sha':'a'},'scenes':scenes}
        reused=dict(manifest,preparation={'parent_plan_id':'older-plan'})
        self.assertEqual(merge_manifests(plan,[reused]),manifest)

    def test_merge_refuses_missing_duplicate_or_changed_source_scenes(self):
        plan=self.plan();scenes={f"{j['request']['task']}/{j['request']['requested_seed']}":{} for j in plan['jobs']}
        manifest={'protocol':'test','sources':{'sha':'a'},'scenes':scenes}
        self.assertEqual(merge_manifests(plan,[manifest]),manifest)
        with self.assertRaises(ConfigurationError):merge_manifests(plan,[dict(manifest,scenes={})])
        with self.assertRaises(ConfigurationError):merge_manifests(plan,[manifest,manifest])
        with self.assertRaises(ConfigurationError):merge_manifests(plan,[manifest,dict(manifest,sources={'sha':'b'})])
if __name__=='__main__':unittest.main()
