"""Apply the existing T2 normalized-action gate to the current public FP8 engine."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from instinctflash import Runtime
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
api=Runtime.from_pretrained('robbyant/lingbot-vla-4b-posttrain-robotwin',precision='fp8',device='cuda:0',tier_ceiling='numeric');api.reset(prompt='pick up the object')
loop=api._backend._loop;generator=loop._server.vla.model
ref=torch.load(a.reference,weights_only=False,map_location='cpu');rows=[]
for i,s in enumerate(ref['samples']):
 ids=s['lang_tokens'].reshape(1,-1).cuda();mask=torch.arange(ids.shape[1],device='cuda')[None,:]<s['n_real']
 with torch.inference_mode():
  output=generator.sample_actions(s['pixel_values'].cuda(),torch.ones(1,3,dtype=torch.bool,device='cuda'),ids,mask,s['state'].reshape(1,75).cuda(),noise=s['noise'].reshape(1,50,75).cuda(),num_steps=10).float().cpu().reshape(50,75)
 reference=s['actions'].float().reshape(50,75);delta=(output-reference).abs()
 rows.append({'i':i,'mae':float(delta.mean()),'max_abs':float(delta.max()),'cosine':float(torch.nn.functional.cosine_similarity(output.flatten(),reference.flatten(),dim=0)),'finite':bool(torch.isfinite(output).all())})
 print(rows[-1],flush=True)
result={'standard':'existing thor_t2/p4_thor_e2e.py: mean normalized action MAE <= .045 and mean action cosine >= .99; speed threshold separate','reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'rows':rows,'mean_mae':float(np.mean([r['mae'] for r in rows])),'mean_cosine':float(np.mean([r['cosine'] for r in rows]))}
result['quality_pass']=all(r['finite'] for r in rows) and result['mean_mae']<=.045 and result['mean_cosine']>=.99
result['scope']='12 frozen numerical reference inputs; no closed-loop certificate or new threshold'
a.output.write_text(json.dumps(result,indent=2)+'\n');api.close()
