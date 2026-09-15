import contextlib,io,json,sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from instinctflash.cli import main
from benchmarks.vla.coverage import coverage,run_evidence
from benchmarks.vla.registry import load_registry
from benchmarks.vla.util import ConfigurationError

class Tests(unittest.TestCase):
    def test_public_cli_exposes_adapter_catalog(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):code=main(['eval','adapters'])
        self.assertEqual(code,0)
        entries=json.loads(out.getvalue())['adapters']
        self.assertTrue(any(a['backbone']=='pi05' for a in entries))
        self.assertTrue(any(a['backbone']=='lingbot_vla_v2' for a in entries))
    def test_routes_never_count_as_completed_evaluations(self):
        registry=load_registry();r=coverage(registry)
        self.assertEqual({x['backbone'] for x in r['families']},{m['backbone'] for m in registry.models.values()})
        self.assertTrue(all(not f['validated_runs'] for f in r['families']))
        cosmos=next(f for f in r['families'] if f['backbone']=='cosmos3_policy')
        self.assertEqual(cosmos['status'],'adapter_missing');self.assertTrue(cosmos['route'])
    def test_missing_evidence_cannot_count_as_measured(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ConfigurationError):run_evidence(d)

if __name__=='__main__':unittest.main()
