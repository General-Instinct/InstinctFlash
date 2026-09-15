"""Two real A/B repeats supply A/A and B/B controls without inventing new episodes."""
import argparse
from collections import defaultdict
from pathlib import Path
from benchmarks.vla.bundle import verify_bundle
from benchmarks.vla.execution_evidence import execution_profile
from benchmarks.vla.util import ConfigurationError,load_json,sha256_file,write_json_atomic
from instinctflash.verify.certify import _tango_paired_score_bounds


def compare(left,right):
    if left.keys()!=right.keys():raise ConfigurationError('repeat task/seed/setting coverage differs')
    grouped=defaultdict(list)
    for key in sorted(left):
        a,b=left[key],right[key]
        if a['resolved_seed']!=b['resolved_seed']:raise ConfigurationError('resolved scenes differ across repeats')
        am,bm=a['metrics'],b['metrics']
        grouped[key[0]].append({'task':key[1],'requested_seed':key[2],'resolved_seed':a['resolved_seed'],
            'left_job_id':a['job_id'],'right_job_id':b['job_id'],
            'left_success':am['success'],'right_success':bm['success'],
            'same_executed_actions':am['action_digest']==bm['action_digest'],
            'left_steps':am['executed_steps'],'right_steps':bm['executed_steps']})
    result=[]
    for suite,rows in sorted(grouped.items()):
        outcomes=[(r['left_success'],r['right_success']) for r in rows]
        n=len(rows);a=sum(x for x,y in outcomes);b=sum(y for x,y in outcomes)
        result.append({'suite_id':suite,'pairs':n,'left_successes':a,'right_successes':b,
            'delta':(b-a)/n,'left_only_successes':sum(x and not y for x,y in outcomes),
            'right_only_successes':sum(y and not x for x,y in outcomes),
            'tango_central95':list(_tango_paired_score_bounds(outcomes,z=1.959963984540054)),
            'identical_action_pairs':sum(r['same_executed_actions'] for r in rows),'rows':rows,
            'scope':'Descriptive matched episodes on fixed tasks; task clustering is not modeled. No quality certificate.'})
    return result


def summarize(root):
    raw={};profiles={};bundles={};scenes=None
    for repeat in ('repeat1','repeat2'):
        bundle=root/repeat/'evidence';verified=verify_bundle(bundle);plan=load_json(bundle/'plan.json')
        if len(plan['jobs'])!=80:raise ConfigurationError('expected 10 tasks x 2 seeds x 2 settings x 2 arms')
        by_arm=defaultdict(dict)
        for arm in plan['arms']:
            identity=arm['operating_point']['remote']['identity'];p=execution_profile(identity)
            previous=profiles.setdefault(arm['id'],p)
            if previous!=p:raise ConfigurationError('execution changed across repeats')
            current=arm['operating_point']['scene_manifest']['sha256']
            if scenes is None:scenes=current
            if scenes!=current:raise ConfigurationError('frozen scene manifest differs')
        for job in plan['jobs']:
            r=job['request'];key=(r['suite_id'],r['task'],r['requested_seed'])
            arm=r['arm']['id']
            if key in by_arm[arm]:raise ConfigurationError('duplicate task/seed/setting')
            by_arm[arm][key]=load_json(bundle/'results'/(job['job_id']+'.json'))
        for rows in by_arm.values():
            for suite in {k[0] for k in rows}:
                counts=defaultdict(int)
                for k in rows:
                    if k[0]==suite:counts[k[1]]+=1
                if len(counts)!=10 or set(counts.values())!={2}:raise ConfigurationError('incomplete declared coverage')
        raw[repeat]=by_arm;bundles[repeat]={'bundle_sha256':verified['bundle_sha256'],'plan_id':plan['plan_id']}
    comparisons={'ab_repeat1':compare(raw['repeat1']['stock'],raw['repeat1']['candidate']),
                 'ab_repeat2':compare(raw['repeat2']['stock'],raw['repeat2']['candidate']),
                 'aa':compare(raw['repeat1']['stock'],raw['repeat2']['stock']),
                 'bb':compare(raw['repeat1']['candidate'],raw['repeat2']['candidate'])}
    return {'schema_version':1,'complete':True,'synthetic':False,'unique_executed_episodes':160,
            'profiles':profiles,'bundles':bundles,'scene_sha256':scenes,'comparisons':comparisons,
            'analysis_source_sha256':sha256_file(Path(__file__)),
            'startup':load_json(root/'startup-findings.json'),
            'limitations':['AA/BB reuse the two AB repeats; comparisons are dependent and are not pooled as new seeds.',
                           '10 tasks and two seeds per setting are screening, not a five-percentage-point noninferiority certificate.',
                           'Capture quality is conditional on the accepted endpoint; a separate startup failed the unchanged self-check.',
                           'H100 execution evidence does not certify Thor or another checkpoint, precision, schedule or source revision.',
                           'Paused simulation does not establish realtime deployment performance.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ConfigurationError('refusing to overwrite expanded analysis')
    write_json_atomic(a.output,summarize(a.root))
