"""Offline narrow experiment: retain UniPC control timesteps on the CPU."""
import inspect

def install(enabled):
    from cosmos_framework.model.generator.diffusion.samplers import unipc
    original=unipc.FlowUniPCMultistepScheduler
    signature=inspect.signature(original.set_timesteps)
    stats={'set_timesteps_calls':0,'devices':[],'enabled':bool(enabled)}
    class HostTimesteps(original):
        def set_timesteps(self,*args,**kwargs):
            bound=signature.bind(self,*args,**kwargs)
            if enabled:
                if bound.arguments.get('num_inference_steps')!=4 or self.solver_p is not None:
                    raise ValueError('Experiment only admits four-step native UniPC without solver_p')
                bound.arguments['device']='cpu'
            result=original.set_timesteps(*bound.args,**bound.kwargs)
            stats['set_timesteps_calls']+=1
            stats['devices'].append(self.timesteps.device.type)
            return result
    unipc.FlowUniPCMultistepScheduler=HostTimesteps
    def restore():
        if unipc.FlowUniPCMultistepScheduler is HostTimesteps:
            unipc.FlowUniPCMultistepScheduler=original
    return stats,restore
