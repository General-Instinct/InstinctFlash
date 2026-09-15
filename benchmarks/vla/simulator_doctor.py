"""Read-only local simulator hardware preflight; no quality certification."""
import subprocess

ISAAC_REQUIREMENTS='https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html'

def inspect_simulator(simulator, gpu_names=None):
    if simulator not in {'libero','robotwin','robolab','droid_sim_evals'}:
        raise ValueError('unknown simulator')
    error=None
    if gpu_names is None:
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
            gpu_names=[s.strip() for s in result.stdout.splitlines() if s.strip()]
        except (OSError,subprocess.SubprocessError) as exc:
            gpu_names=[];error=str(exc)
    isaac=simulator in {'robolab','droid_sim_evals'}
    unsupported=[g for g in gpu_names if any(x in g.upper() for x in ('A100','H100','H200','H800','H20','A800'))]
    # Unknown GPUs are not declared supported just because they were not denied.
    blocked=isaac and bool(gpu_names) and len(unsupported)==len(gpu_names)
    return {'schema_version':1,'simulator':simulator,'scope':'Local rendering hardware only; a remote renderer may be used. Does not verify dependencies/assets or produce quality evidence.',
            'gpu_names':gpu_names,'status':'blocked_local_renderer' if blocked else 'requires_runtime_validation',
            'reason':'Isaac Sim requires RT Cores; all detected GPUs lack supported RTX rendering.' if blocked else 'Run pinned simulator dependency, asset and reset checks before evaluation.',
            'resolution':'Use a supported RTX machine for simulation/rendering; policy inference can remain on H100.' if blocked else None,
            'source':ISAAC_REQUIREMENTS if isaac else None,'probe_error':error}
