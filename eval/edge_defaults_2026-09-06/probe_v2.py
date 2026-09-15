"""Native production Runtime validation, with an explicit precision ceiling and real noise."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.vla.instinctflash_driver import RuntimeArm
from benchmarks.vla.latency_probe import measure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tier-ceiling',choices=['bitexact','numeric'],required=True)
    a=p.parse_args()
    outputs=[]
    original=RuntimeArm.predict
    def predict(self,request):
        action=original(self,request)
        outputs.append(np.asarray(action).copy())
        return action
    RuntimeArm.predict=predict
    try:
        measure(a.trace,'robbyant/lingbot-vla-v2-6b-robotwin',
                '0451855729ec904f970600e0aec8b84661423afe','runtime_default',a.output,
                warmup=8,iterations=64,tier_ceiling=a.tier_ceiling)
    finally:
        RuntimeArm.predict=original
    np.savez_compressed(str(a.output)+'.actions.npz',actions=np.stack(outputs))

if __name__=='__main__':
    main()
