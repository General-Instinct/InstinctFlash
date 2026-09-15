"""CPU byte-level checkpoint mapping audit; no model execution."""
import hashlib,json,struct,mmap
from pathlib import Path
from contextlib import ExitStack
root=Path.home()/'.cache/huggingface/hub'
converted=next((root/'models--lerobot--lingbot_va_robotwin/snapshots').glob('*/model.safetensors'))
raw=next((root/'models--robbyant--lingbot-va-posttrain-robotwin/snapshots').glob('*/transformer'))
def header(p):
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];return 8+n,json.loads(f.read(n))
def digest(mm,base,offsets):
 h=hashlib.sha256()
 for start in range(base+offsets[0],base+offsets[1],8*1024*1024):h.update(mm[start:min(start+8*1024*1024,base+offsets[1])])
 return h.hexdigest()
rows=[]
with ExitStack() as stack:
 cb,ch=header(converted);cm=stack.enter_context(mmap.mmap(stack.enter_context(converted.open('rb')).fileno(),0,access=mmap.ACCESS_READ))
 index=json.loads((raw/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map'];sources={}
 for filename in set(index.values()):
  path=raw/filename;b,h=header(path);mm=stack.enter_context(mmap.mmap(stack.enter_context(path.open('rb')).fileno(),0,access=mmap.ACCESS_READ));sources[filename]=(b,h,mm)
 for key,entry in ch.items():
  if key=='__metadata__':continue
  native=key.removeprefix('transformer.');b,h,mm=sources[index[native]];orig=h[native]
  same_meta=entry['dtype']==orig['dtype'] and entry['shape']==orig['shape']
  lhs=digest(cm,cb,entry['data_offsets']);rhs=digest(mm,b,orig['data_offsets'])
  rows.append({'key':key,'native_key':native,'same_meta':same_meta,'same_bytes':lhs==rhs,'converted_sha256':lhs,'native_sha256':rhs})
report={'converted':str(converted),'native':str(raw),'tensor_count':len(rows),'native_count':len(index),'ok':len(rows)==len(index) and all(r['same_meta'] and r['same_bytes'] for r in rows),'tensors':rows}
Path('/home/ubuntu/ifl_eval/framework_comparison_20260913/lerobot-va-conversion-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='tensors'})
