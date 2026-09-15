import hashlib,struct,unittest
from benchmarks.vla.result import _validate_controller_trace
from benchmarks.vla.util import ConfigurationError

class ControllerTraceTests(unittest.TestCase):
    def metrics(self):
        values=[0.,-0.,1.,2.,3.,4.,5.]
        return {'executed_steps':1,'action_values':values,'action_digest':hashlib.sha256(b''.join(struct.pack('!d',v) for v in values)).hexdigest()}
    def test_signed_zero_change_is_detected(self):
        metrics=self.metrics();_validate_controller_trace(metrics,'groot-libero-paused-v1','test')
        metrics['action_values'][1]=0.
        with self.assertRaises(ConfigurationError):_validate_controller_trace(metrics,'groot-libero-paused-v1','test')
    def test_controller_step_count_is_verified(self):
        metrics=self.metrics();metrics['executed_steps']=2
        with self.assertRaises(ConfigurationError):_validate_controller_trace(metrics,'pi05-libero-schedule-paused-v1','test')
if __name__=='__main__':unittest.main()
