"""Materialize a small declared view over a pinned native checkpoint, without copying weights."""
from pathlib import Path
import os
from .util import ConfigurationError, sha256_file, sha256_json, write_json_atomic

GROOT_VARIANTS={'libero_10':'groot_libero_10','libero_spatial':'groot_libero_spatial',
                'libero_object':'groot_libero_object','libero_goal':'groot_libero_goal'}

def create_view(model,revision,subdir,output):
    from huggingface_hub import snapshot_download
    if model!='nvidia/GR00T-N1.7-LIBERO' or subdir not in GROOT_VARIANTS:
        raise ConfigurationError('no reviewed native checkpoint-view contract for this model/subdirectory')
    from .adapters import load_adapters
    contract=next(a for a in load_adapters() if a['id']=='groot-libero-v1')
    if {'id':model,'revision':revision} not in contract['checkpoints']:
        raise ConfigurationError('checkpoint revision is not covered by the adapter contract')
    source=Path(snapshot_download(model,revision=revision,local_files_only=True))/subdir;output=Path(output).resolve()
    files={f.name:f for f in source.iterdir() if f.is_file() and f.suffix in {'.json','.safetensors'}}
    if 'config.json' not in files or not any(k.endswith('.safetensors') for k in files):
        raise ConfigurationError('checkpoint lacks native config or weights')
    if output.exists():raise ConfigurationError('checkpoint view already exists')
    hashes={k:sha256_file(v) for k,v in files.items()};output.mkdir(parents=True)
    for name,path in files.items():os.symlink(path,output/name)
    write_json_atomic(output/'instinctflash.json',{'instinctflash_schema':1,'execution':{
        'model_id':model,'backbone':'groot_n17','servable':True,'guidance':{'action':'none'},
        'nfe':{'backbone':1,'action':4},'base_weights':str(source),'embodiment_tag':'libero_sim',
        'fast_decode':True,'backbone_fastpath':True,
        'param_bytes':sum(p.stat().st_size for p in files.values() if p.suffix=='.safetensors')}})
    # Receipt stays next to, not inside, the native checkpoint view.
    receipt={'model':model,'revision':revision,'subdir':subdir,'checkpoint_sha256':sha256_json(hashes),
             'files':hashes,'view':str(output),'suite':GROOT_VARIANTS[subdir]}
    write_json_atomic(output.with_suffix('.receipt.json'),receipt);return receipt
