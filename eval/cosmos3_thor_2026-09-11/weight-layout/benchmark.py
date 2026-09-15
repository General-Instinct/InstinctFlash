"""Native Cosmos weight-storage ablation; original dtype and schedules retained."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService
from benchmarks.regression import cosmos

stats={'scope':'Gen-tower Nano gate/up weight storage only; no arithmetic fusion or quantization', 'weights':[]}
original_init=RobolabPolicyService.__init__

def initialize(self,args):
    original_init(self,args)
    if os.environ.get('IFL_WEIGHT_LAYOUT')!='1':return
    targets=[]
    for path,module in self.model.net.named_modules():
        if '.mlp_moe_gen.' not in path or not path.endswith(('.up_proj','.gate_proj')):continue
        assert type(module) is torch.nn.Linear and module.bias is None
        w=module.weight
        assert type(w) is torch.nn.Parameter and w.is_cuda and w.dtype==torch.bfloat16
        assert tuple(w.shape)==(12288,4096), (path,w.shape)
        assert w.is_contiguous(), (path,w.stride())
        targets.append((path,module,w))
    assert len(targets)==72, len(targets)
    with torch.no_grad():
        packed=[w.t().contiguous().t() for _,_,w in targets]
        for (path,module,w),p in zip(targets,packed):
            assert torch.equal(w.view(torch.uint8),p.contiguous().view(torch.uint8)),path
            stats['weights'].append(dict(path=path,shape=list(w.shape),old_stride=list(w.stride()),new_stride=list(p.stride())))
            module.weight=torch.nn.Parameter(p,requires_grad=w.requires_grad)
    self._layout_targets=[module for _,module,_ in targets]
    stats['installed']=len(targets)
    # Retain a validation callable only; no hook enters the compiled/graph path.
    stats['check_live']=lambda: all(tuple(m.weight.stride())==(1,12288) for m in self._layout_targets)

RobolabPolicyService.__init__=initialize
output=Path(sys.argv[3])
assert not output.exists()
code=cosmos.main()
check=stats.pop('check_live',None)
if check:
    stats['live_layout_passed']=check()
    assert stats['live_layout_passed']
report=json.loads(output.read_text())
report['weight_layout']=stats
report['weight_layout_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
output.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(code)
