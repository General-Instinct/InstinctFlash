"""Separate implemented contracts from actual validated simulation evidence."""
from pathlib import Path
from .adapters import load_adapters
from .plan import validate_plan
from .registry import Registry
from .result import validate_result
from .util import ConfigurationError, load_json, sha256_json


def run_evidence(root):
    root = Path(root).resolve()
    plan = load_json(root/'plan.json'); validate_plan(plan)
    report = load_json(root/'report.json')
    payload = {k:v for k,v in report.items() if k != 'report_sha256'}
    if sha256_json(payload) != report.get('report_sha256') or report.get('plan_id') != plan['plan_id']:
        raise ConfigurationError('report identity mismatch')
    if not report.get('complete') or report.get('synthetic') or report.get('issues'):
        raise ConfigurationError('coverage requires a complete, real run without issues')
    grouped = {}
    for job in plan['jobs']:
        req = job['request']; result = load_json(root/'results'/f"{job['job_id']}.json")
        validate_result(result, job)
        if result['provenance']['synthetic']: raise ConfigurationError('synthetic result is not real evidence')
        if req['suite']['kind'] != 'closed_loop': continue
        checkpoint = req['model']['checkpoint']
        key = (req['model']['backbone'], checkpoint['id'], checkpoint['revision'], req['suite_id'], checkpoint.get('subdir'))
        entry = grouped.setdefault(key, {'backbone':key[0], 'checkpoint':key[1], 'revision':key[2],
            'suite':key[3], 'episodes':0, 'arms':{}, 'plan_id':plan['plan_id'], 'run':str(root),
            'screening':report.get('screening',False), 'quality_reportable':report['reportable']})
        if key[4] is not None: entry['checkpoint_subdir'] = key[4]
        entry['episodes'] += 1
        arm = entry['arms'].setdefault(req['arm']['id'], {'episodes':0, 'successes':0})
        arm['episodes'] += 1; arm['successes'] += int(result['metrics']['success'])
    return list(grouped.values())


def coverage(registry: Registry, runs=()):
    routes = load_json(Path(__file__).with_name('config')/'simulator_routes.json')['routes']
    contracts = load_adapters(); evidence = []
    seen = set()
    for path in runs:
        for row in run_evidence(path):
            key = (row['plan_id'],row['checkpoint'],row['revision'],row['suite'],row.get('checkpoint_subdir'))
            if key not in seen: evidence.append(row);seen.add(key)
    rows = []
    for backbone in sorted({m['backbone'] for m in registry.models.values()}):
        route = next((r for r in routes if r['backbone']==backbone), None)
        adapters = [a for a in contracts if a['backbone']==backbone]
        observed = [e for e in evidence if e['backbone']==backbone]
        rows.append({'backbone':backbone,
                     'registered_checkpoints':[m['id'] for m in registry.models.values() if m['backbone']==backbone],
                     'implemented_adapters':[{'id':a['id'],'simulator':a['simulator'],'checkpoints':a['checkpoints']} for a in adapters],
                     'route':route, 'validated_runs':observed,
                     'status':'real_evidence_supplied' if observed else 'adapter_implemented_unmeasured' if adapters else 'adapter_missing'})
    return {'schema_version':1, 'scope':'Coverage of supplied runs, not a universal accuracy certification; routes are not implemented adapters',
            'families':rows}
