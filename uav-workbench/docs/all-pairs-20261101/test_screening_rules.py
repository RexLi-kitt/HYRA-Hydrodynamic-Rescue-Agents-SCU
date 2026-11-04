"""Metadata-only checks: fair64 ceiling and complete C(6,2) candidate coverage."""
import collections,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from all_pairs import ALGORITHMS,PAIRS,rank_at,e
run=Path(__file__).parent;fixtures=run/'screening-test-fixtures';folder=fixtures/'example';folder.mkdir(parents=True,exist_ok=False)
tests=[]
def check(name,condition):
    assert condition,name
    tests.append({'test':name,'passed':True})
check('exactly15_distinct_unordered_pairs',len(PAIRS)==15 and len({frozenset(p) for p in PAIRS.values()})==15)
check('all6_in5_pairs_each',collections.Counter(a for pair in PAIRS.values() for a in pair)==collections.Counter({a:5 for a in ALGORITHMS}))
base={'joint_episode':0,'summary':{'completion':.8,'RCR':.9,'coverage':.95,'ACC':.01,'ST':150},'groups':{'example':{'ACC':.01}},'worst_assignment_RCR':.85}
e.write(folder/'eval-0000.json',base)
later=copy.deepcopy(base);later['joint_episode']=256;later['summary'].update(completion=1.,RCR=1.,coverage=1.,ACC=0,ST=80);later['worst_assignment_RCR']=1.
e.write(folder/'eval-0256.json',later)
missing_rejected=False
try:rank_at(fixtures,64,['example'])
except AssertionError:missing_rejected=True
check('missing64_evaluation_cannot_be_silently_skipped',missing_rejected)
at64=copy.deepcopy(base);at64['joint_episode']=64;at64['summary']['RCR']=.91;e.write(folder/'eval-0064.json',at64)
chosen=rank_at(fixtures,64,['example'])['best_within_budget']['example']['episode']
check('superior256_checkpoint_excluded_from64_ranking',chosen==64)
e.write(run/'screening-rules-audit.json',{'status':'passed','checks':tests,'fixtures_are_not_experiment_results':True})
print(json.dumps({'status':'passed','checks':len(tests)}))
