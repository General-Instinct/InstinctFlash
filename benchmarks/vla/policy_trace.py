"""Portable, hash-checked wire recordings for real fixed-input policy replay."""
from pathlib import Path
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic

class PolicyTrace:
    def __init__(self, root, identity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.manifest={'schema_version':1,'identity':identity,'calls':[],'complete':False}
        self._save()
    def _save(self):write_json_atomic(self.root/'trace.json',self.manifest)
    def append(self, request, response):
        index=len(self.manifest['calls']);paths={}
        for kind,value in [('request',request),('response',response)]:
            path=self.root/f'{index:05d}.{kind}.msgpack';path.write_bytes(value)
            paths[kind]={'path':path.name,'sha256':sha256_file(path)}
        self.manifest['calls'].append(paths);self._save()
    def close(self):
        self.manifest['complete']=True;self._save()

def read_trace(root):
    from instinctflash.serving.msgpack_numpy import unpackb
    root=Path(root);manifest=load_json(root/'trace.json')
    if manifest.get('schema_version')!=1 or not manifest.get('complete') or not manifest.get('calls'):
        raise ConfigurationError('incomplete policy recording')
    calls=[]
    for record in manifest['calls']:
        values=[]
        for kind in ('request','response'):
            ref=record[kind];path=Path(ref['path'])
            if path.is_absolute() or len(path.parts)!=1 or path.name in {'.','..'}:
                raise ConfigurationError('unsafe trace path')
            file=root/path
            if sha256_file(file)!=ref['sha256']:raise ConfigurationError('changed policy recording')
            values.append(unpackb(file.read_bytes()))
        calls.append(tuple(values))
    return manifest,calls

class V2Noise:
    """Capture or inject the native sampler's actual initial tensor, without RNG reseeding."""
    def __init__(self, arm):
        server=getattr(arm,'_server',None)
        if server is None:server=arm._runtime._backend._impl._server
        self.server=server;self.original=server.sample_actions_fn;self.pending=None;self.last=None
        server.sample_actions_fn=self.sample
    def sample(self,*args,**kwargs):
        import torch
        state=args[4];config=self.server.vla.model.config
        shape=(state.shape[0],config.n_action_steps,config.max_action_dim)
        if self.pending is None:
            noise=torch.randn(shape,device=state.device,dtype=state.dtype)
        else:
            noise=torch.as_tensor(self.pending,device=state.device,dtype=state.dtype)
            if tuple(noise.shape)!=shape:raise ConfigurationError('replay noise shape mismatch')
            self.pending=None
        if not torch.isfinite(noise).all():raise ConfigurationError('nonfinite replay noise')
        self.last=noise.detach().float().cpu().numpy()
        return self.original(*args,noise=noise,**kwargs)
