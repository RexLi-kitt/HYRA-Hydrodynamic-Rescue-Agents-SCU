"""Independent examples for the automatic extension gate, no model execution."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from linked_train import improving,quality

base={'summary':{'n':72,'completion':.8,'RCR':.9,'coverage':.95,'ACC':.01,'ST':150.,'max_conflicts':3},
      'groups':{k:{'ACC':.01} for k in ('legacy/sample','legacy/fixed','new/sample','new/fixed')},
      'worst_assignment_RCR':.88}
tests=[]
def check(name,condition):
    assert condition,name
    tests.append({'name':name,'passed':True})
check('plateau_does_not_extend',not improving(base,base))
x=copy.deepcopy(base);x['summary']['RCR']+=.01
check('material_rescue_gain_can_extend',improving(x,base))
x['groups']['new/fixed']['ACC']=.03
check('one_conflict_stratum_blocks_extension',not improving(x,base))
check('conflict_gate_precedes_average_rescue_rank',quality(base)>quality(x))
x=copy.deepcopy(base);x['summary']['ST']-=6
check('time_gain_with_equal_task_quality_can_extend',improving(x,base))
x['summary']['RCR']-=.001
check('faster_but_fewer_serviced_blocks_extension',not improving(x,base))
x=copy.deepcopy(base);x['summary']['coverage']+=.01;x['summary']['max_conflicts']=11
check('one_large_conflict_tail_blocks_extension',not improving(x,base))
Path(__file__).with_name('gate-rules-audit.json').write_text(json.dumps({'status':'passed','checks':tests},indent=2),encoding='utf8')
print(json.dumps({'status':'passed','checks':len(tests)}))
