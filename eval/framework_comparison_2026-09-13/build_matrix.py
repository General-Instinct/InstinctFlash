"""Build the 8x3 candidate matrix only from successful, finite Thor receipts."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.receipts
models=[('va','LingBot-VA'),('vla4','LingBot-VLA-4B'),('vla2','LingBot-VLA-V2-6B'),('edge','Cosmos3 Edge DROID'),('nano','Cosmos3 Nano DROID'),('pi05','pi05'),('groot','GR00T N1.7'),('dreamzero','DreamZero DROID')]
frameworks=['LeRobot','vLLM-Omni','InstinctFlash']
support={'LeRobot':{'va','pi05','groot'},'vLLM-Omni':{'edge','nano','dreamzero'},'InstinctFlash':{m for m,_ in models}}
def candidates(m,fw):
 if fw=='InstinctFlash':
  found=[('native/'+m+'-current.json','native, default schedule'),('initial/'+m+'-native.json','NUMERIC, default schedule'),('initial/'+m+'-fp8.json','FP8, default schedule'),('recovery/'+m+'-fp8.json','FP8, default schedule')]
  if m=='va':found += [('initial/va_2v4a-native.json','NUMERIC, 2V/4A'),('recovery/va_2v4a-fp8.json','FP8, 2V/4A')]
  if m=='pi05':found += [('flash-pi05-nfe1-native-v1.json','native, NFE1')]
  return found
 if fw=='LeRobot':
  return {'pi05':[('lerobot-pi05-compiled-v2.json','compiled, NFE10'),('lerobot-pi05-nfe1-compiled-v2.json','compiled, NFE1')],'groot':[('lerobot-groot-v3.json','native, NFE4')],'va':[('lerobot-va-default-v1.json','native, 25V/50A'),('lerobot-va-2v4a-v1.json','native, 2V/4A')]}.get(m,[])
 return [(f'omni-{m}-eager-v1.json','eager, UniPC4'),(f'omni-{m}-compiled-v1.json','compiled, UniPC4')] if m in ('edge','nano') else [('omni-dreamzero-eager-v6.json','eager, upstream step cache'),('omni-dreamzero-compiled-v1.json','compiled, upstream step cache')]
rows=[]
for m,label in models:
 cells={}
 for fw in frameworks:
  if m not in support[fw]:cells[fw]={'status':'unsupported_in_pinned_registry'};continue
  ok=[];failed=[]
  for rel,config in candidates(m,fw):
   path=root/rel
   if not path.exists():continue
   r=json.loads(path.read_text());lat=r.get('p50_ms');count=sum(c.get('phase')=='measured' for c in r.get('calls',[]))
   if not r.get('ok') or not isinstance(lat,(float,int)) or not math.isfinite(lat) or lat<=0 or count<30:
    failed.append({'receipt':rel,'error':r.get('error'),'measured_count':count});continue
   selected=[c for c in r['calls'] if c.get('phase')=='measured' and (m not in ('va','dreamzero') or c.get('cycle',c['i']%3) in (1,2))]
   if len(selected)<(20 if m in ('va','dreamzero') else 30):
    failed.append({'receipt':rel,'error':'insufficient aligned samples'});continue
   ok.append({'p50_ms':statistics.median(c['ms'] for c in selected),'raw_all_cycle_p50_ms':lat,'configuration':config,'receipt':rel,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'measured_count':count,'selected_count':len(selected),'selected_call_indices':[c['i'] for c in selected],'timing_scope':'continuation cycles 1 and 2' if m in ('va','dreamzero') else 'stateless action generation'})
  cells[fw]={'status':'measured_candidate' if ok else 'pending','fastest_measured':min(ok,key=lambda x:x['p50_ms']) if ok else None,'candidates':ok,'failed':failed,'quality_admission':'separate checkpoint-specific gate; speed does not imply pass'}
 rows.append({'model':m,'label':label,'frameworks':cells})
report={'hardware':'Jetson Thor','table_shape':[8,3],'scope':'fastest measured candidates, not yet a complete quality-admitted ranking','rows':rows,'runnable_cells':sum(len(x) for x in support.values()),'measured_cells':sum(c['status']=='measured_candidate' for r in rows for c in r['frameworks'].values())}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
lines=['# Thor measured candidates','','p50 milliseconds; lowest observed completed candidate per cell. History models use continuation calls ([protocol](HISTORY_TIMING.md)). [Quality admission](quality_admission_inventory.json) is separate; these are not all quality-qualified winners.','','| Model | LeRobot | vLLM-Omni | InstinctFlash-internal |','| --- | ---: | ---: | ---: |']
for r in rows:
 values=[]
 for fw in frameworks:
  c=r['frameworks'][fw];v=c.get('fastest_measured');values.append(f"{v['p50_ms']:.2f} · {v['configuration']}" if v else 'Unsupported' if c['status'].startswith('unsupported') else 'Pending')
 lines.append('| '+r['label']+' | '+' | '.join(values)+' |')
lines += ['', 'Unsupported = no matching native action policy in the [pinned registry](framework_support_audit.json). Configurations can differ in steps, precision and caching; these are not same-computation speedups. [Setup and compatibility fixes](COMPATIBILITY.md) · [Raw receipts](receipt_inventory.json).']
a.output.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
