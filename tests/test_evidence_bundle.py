import tempfile,unittest
from pathlib import Path
from benchmarks.vla.bundle import verify_bundle, _verify_scene_artifacts
from benchmarks.vla.util import sha256_json,sha256_file,write_json_atomic,ConfigurationError

class BundleTests(unittest.TestCase):
    def manifest(self,root,files):
        value={'schema_version':1,'plan_id':'p','files':files,'scene_path_mapping':{}}
        value['bundle_sha256']=sha256_json(value)
        write_json_atomic(root/'bundle.json',value)
    def test_corrupted_evidence_is_refused_before_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'report.json';p.write_text('{}')
            self.manifest(root,{'report.json':sha256_file(p)})
            p.write_text('{"modified":true}')
            with self.assertRaisesRegex(ConfigurationError,'changed'):verify_bundle(root)
    def test_relative_path_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.manifest(root,{'../outside':'0'*64})
            with self.assertRaisesRegex(ConfigurationError,'unsafe'):verify_bundle(root)
    def test_rehashed_inventory_cannot_substitute_another_scene(self):
        plan={'arms':[{'operating_point':{'scene_manifest':{'path':'/original/scenes.json','sha256':'a'*64}}}]}
        manifest={'files':{'artifacts/scene.json':'b'*64},'scene_path_mapping':{'/original/scenes.json':'artifacts/scene.json'}}
        with self.assertRaisesRegex(ConfigurationError,'frozen plan hash'):
            _verify_scene_artifacts(plan,manifest)
        manifest['files']['artifacts/scene.json']='a'*64
        _verify_scene_artifacts(plan,manifest)
    def test_missing_scene_mapping_is_refused(self):
        plan={'arms':[{'operating_point':{'scene_manifest':{'path':'/original/scenes.json','sha256':'a'*64}}}]}
        with self.assertRaisesRegex(ConfigurationError,'cover the frozen plan'):
            _verify_scene_artifacts(plan,{'files':{},'scene_path_mapping':{}})

if __name__=='__main__':unittest.main()
