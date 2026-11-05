from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'hyra-uav-pr-20261101/uav-workbench/experiments/flood-drop-convlstm20-v1'))
from long_environment import Config,make_scenario
cfg=Config(duration=120,window=60,people=1,traps=0,trigger_depth=2.,trigger_speed=1.,step=1)
rows=[]
for name,h,u,v,expected in [('depth_equal',2.,0.,0.,'threshold'),('speed_norm_equal',.1,.6,.8,'threshold'),('both_below',1.999,.99,0.,'random')]:
 w=np.broadcast_to(np.array([h,u,v],np.float32)[None,:,None,None],(25,3,100,100))
 sc=make_scenario(w,2026112903,cfg,pool_size=1)
 assert sc['events'][0]['trigger']==expected
 rows.append(dict(name=name,depth=h,u=u,v=v,actual=sc['events'][0]['trigger'],expected=expected))
w=np.broadcast_to(np.array([0.,2.,0.],np.float32)[None,:,None,None],(25,3,100,100))
try:make_scenario(w,2026112903,cfg,pool_size=1)
except ValueError as ex:
 assert 'No wet' in str(ex);rows.append(dict(name='dry_velocity_not_water_exposure',passed=True))
else:raise AssertionError('Dry water produced a fall')
path=ROOT/'uav-frozen-rigor-20261104/run01/initial-fall-contracts.json'
path.write_text(json.dumps(dict(checks=rows,scope='Four manufactured input-contract tests, not field safety thresholds'),ensure_ascii=False,indent=2),encoding='utf-8')
print('4 initial-fall contracts passed')
