import json,os,signal,time
from pathlib import Path
root=Path('/home/guanming/ifl_eval/native_total_20260910_dreamzero_trim')
progress=root/'thor-0/progress.json'
while True:
    if progress.exists():
        status=json.loads(progress.read_text())
        if status['status']!='running':break
        jobs=status.get('jobs',[])
        if jobs and jobs[-1]['status']=='running' and jobs[-1]['family']=='dreamzero':
            memory={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','MemTotal:'))}
            if memory['MemAvailable']<2*1024*1024:
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit():continue
                    try:args=(p/'cmdline').read_bytes().split(b'\0')
                    except (FileNotFoundError,PermissionError):continue
                    if str(root/'source/eval/native_total_2026-09-10/benchmark.py').encode() in args:
                        os.kill(int(p.name),signal.SIGTERM)
                        with (root/'memory-stops.jsonl').open('a') as f:f.write(json.dumps({'pid':p.name,'job':jobs[-1],'memory_kb':memory,'reason':'native upstream initialization exhausted available unified memory reserve'})+'\n')
    time.sleep(2)
