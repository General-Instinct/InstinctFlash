"""Isolated, resumable scene preparation; renderer lifetime is bounded to one task."""
import copy
import concurrent.futures
import fcntl
import os
import re
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

from .plan import validate_plan
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic

MODULES = {
    'benchmarks.vla.joint_robotwin_driver': 'benchmarks.vla.joint_robotwin_driver',
    'benchmarks.vla.groot_libero_driver': 'benchmarks.vla.groot_libero_driver',
    'benchmarks.vla.pi05_libero_driver': 'benchmarks.vla.pi05_scenes',
    'benchmarks.vla.wan_va_libero_driver': 'benchmarks.vla.wan_va_libero_driver',
}


def task_shards(plan):
    validate_plan(plan)
    if len(plan.get("selected_models", [])) != 1:
        raise ConfigurationError("prepare scenes for one checkpoint campaign at a time")
    groups = defaultdict(list)
    for job in plan['jobs']:
        request = job['request']
        if request['suite']['kind'] != 'closed_loop':
            raise ConfigurationError('scene preparation accepts only closed-loop jobs')
        groups[(request['model_id'],request['suite_id'],request['task'])].append(job)
    shards = []
    for key, jobs in sorted(groups.items()):
        child = copy.deepcopy(plan)
        child['jobs'] = jobs
        child['preparation_parent_plan_id'] = plan['plan_id']
        child.pop('plan_id')
        child['plan_id'] = sha256_json(child)
        validate_plan(child)
        shards.append((sha256_json(key)[:16],child))
    return shards


def merge_manifests(plan, manifests):
    if not manifests:
        raise ConfigurationError('no scene manifests produced')
    expected = set()
    for job in plan['jobs']:
        r = job['request']
        joint = r['model']['backbone'] in {'lingbot_vla','lingbot_vla_v2'}
        expected.add(f"{r['suite_id']}/{r['task']}/{r['requested_seed']}" if joint else f"{r['task']}/{r['requested_seed']}")
    header = {k:v for k,v in manifests[0].items() if k not in {'scenes', 'preparation'}}
    scenes = {}
    for manifest in manifests:
        if {k:v for k,v in manifest.items() if k not in {'scenes', 'preparation'}} != header:
            raise ConfigurationError('scene shards disagree on protocol/source/assets; cannot merge')
        if scenes.keys() & manifest['scenes'].keys():
            raise ConfigurationError('duplicate scene across preparation shards')
        scenes.update(manifest['scenes'])
    if set(scenes) != expected:
        raise ConfigurationError('prepared scenes do not cover exactly the requested tasks/seeds')
    return dict(header,scenes=scenes)


def prepare_scenes(plan, output, repo_root, *, workers=1, gpus=(), reuse_scenes=()):
    if type(workers) is not int or workers < 1:
        raise ConfigurationError('workers must be a positive integer')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',str(g)) for g in gpus):
        raise ConfigurationError('GPU selectors must be ordinals or GPU UUIDs')
    output = Path(output).resolve()
    if output.exists():
        raise ConfigurationError('refusing to replace frozen scenes')
    shards = task_shards(plan)
    reused = [load_json(Path(path)) for path in reuse_scenes]
    reused_keys = set()
    for manifest in reused:
        if reused_keys & manifest['scenes'].keys():
            raise ConfigurationError('duplicate scenes in reuse inputs')
        reused_keys.update(manifest['scenes'])
    pending = []
    for name, child in shards:
        keys = set()
        for job in child['jobs']:
            r = job['request']
            joint = r['model']['backbone'] in {'lingbot_vla','lingbot_vla_v2'}
            keys.add(f"{r['suite_id']}/{r['task']}/{r['requested_seed']}" if joint else f"{r['task']}/{r['requested_seed']}")
        if keys & reused_keys and not keys <= reused_keys:
            raise ConfigurationError('reuse must cover all seeds/arms of a task shard')
        if not keys <= reused_keys:pending.append((name,child))
    total_shards = len(shards)
    shards = pending
    work = output.with_suffix(output.suffix+'.prepare')
    work.mkdir(parents=True,exist_ok=True)
    with (work/'lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ConfigurationError('another process owns this preparation directory') from error
        identity = {'parent_plan_id':plan['plan_id'],'repo_root':str(Path(repo_root).resolve()),
                    'gpus':list(gpus),'workers':workers,
                    'reused_scene_files':{str(Path(p).resolve()):sha256_file(Path(p)) for p in reuse_scenes}}
        identity_path = work/'identity.json'
        if identity_path.exists() and load_json(identity_path) != identity:
            raise ConfigurationError('preparation resume identity changed')
        write_json_atomic(identity_path,identity)
        def run(item):
            index,(name,child) = item
            request_path = work/(name+'.plan.json')
            result_path = work/(name+'.scenes.json')
            receipt_path = work/(name+'.receipt.json')
            write_json_atomic(request_path,child)
            if result_path.exists() and receipt_path.exists():
                receipt = load_json(receipt_path)
                if receipt != {'plan_id':child['plan_id'],'sha256':sha256_file(result_path)}:
                    raise ConfigurationError('changed completed scene shard')
                return load_json(result_path)
            if result_path.exists():
                raise ConfigurationError('scene shard exists without a completion receipt')
            driver = child['jobs'][0]['driver']
            contract = child['jobs'][0]['request']['adapter']['contract']
            module = MODULES.get(contract['driver_module'])
            if module is None:
                raise ConfigurationError('this adapter has no supported scene preparation entrypoint')
            # All workers use the selected frozen execution checkout, not the orchestrator imports.
            env = dict(os.environ,**driver.get('environment',{}))
            env['PYTHONPATH'] = str(Path(repo_root).resolve())
            env['HF_HUB_OFFLINE'] = '1'
            if gpus:env['CUDA_VISIBLE_DEVICES'] = str(gpus[index % len(gpus)])
            command = [driver['command'][0],'-m',module,'--prepare-plan',str(request_path),'--output',str(result_path)]
            with (work/(name+'.log')).open('w') as log:
                process = subprocess.Popen(command,cwd=repo_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code = process.wait(timeout=driver.get('timeout_seconds',3600))
                except BaseException:
                    from .runner import _terminate_process_group
                    _terminate_process_group(process)
                    raise
            if code:
                raise ConfigurationError(f'scene task shard {name} failed ({code}); see {work/(name+".log")}')
            manifest = load_json(result_path)
            write_json_atomic(receipt_path,{'plan_id':child['plan_id'],'sha256':sha256_file(result_path)})
            print(f'prepared {name}',file=sys.stderr,flush=True)
            return manifest
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            manifests = list(pool.map(run,enumerate(shards)))
        merged = merge_manifests(plan,reused+manifests)
        merged["preparation"] = {**identity, "orchestrator_sha256":sha256_file(Path(__file__)),
                                 "task_shards":total_shards, "reused_task_shards":total_shards-len(shards)}
        write_json_atomic(output,merged)
        return {'output':str(output),'sha256':sha256_file(output),'scenes':len(merged['scenes']),'task_shards':total_shards,'reused_task_shards':total_shards-len(shards)}
