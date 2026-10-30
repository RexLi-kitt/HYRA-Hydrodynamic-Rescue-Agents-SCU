import os,sys,json,time
from pathlib import Path
import numpy as np
os.environ['HYRA_FLOOD_BACKEND']='numba'
LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
import map_sessions,server
from multilayer import json_safe
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
source=OUT.parent/'round21/runs/f8-n0.042-layered-v1'
r=json.loads((source/'result.json').read_text(encoding='utf-8'));config=json.loads((source/'config.json').read_text(encoding='utf-8'))
world=map_sessions.create(LAB,server.describe(config['terrain_id']),config['scenario']);s=world.solver
with np.load(source/'final.npz') as a:
 for saved,current in [('bed','bed'),('n','manning'),('solid','solid'),('wall_x','wall_x'),('wall_y','wall_y')]:np.testing.assert_array_equal(a[saved],getattr(s,current))
 for saved,current in [('depth','h'),('mx','mx'),('mz','mz'),('maximum_depth','maximum')]:getattr(s,current)[:]=a[saved]
replay=json.loads((LAB/'generated'/r['replay_id']/'maps/replay.json').read_text(encoding='utf-8'));last=replay['process']['frames'][-1]
s.time=r['duration_s'];s.steps=last['steps'];s.balance=dict(r['balance']);s.diagnostics=dict(last['diagnostics'])
for k in ['external_volume_by_id']:
 if k in last:setattr(s,k,dict(last[k]))
points=[p for p in r['points'] if p['id'] in ['16-K','19-J']];rows=[];start=time.monotonic()
def record():
 rows.append(dict(time_s=s.time,volume=float(s.h.sum()*s.dx**2),points=[dict(id=p['id'],depth=float(s.h[p['grid_row'],p['grid_col']]),maximum_depth=float(s.maximum[p['grid_row'],p['grid_col']])) for p in points]))
record()
for end in np.arange(850,1601,50):
 try:world.advance_to(float(end),cancelled=lambda:time.monotonic()-start>180)
 except InterruptedError:break
 record();print(json.dumps(rows[-1]),flush=True)
result=dict(kind='late-wetting diagnostic, never replaces 0-800s scores',start_s=800,end_s=s.time,complete=s.time==1600,wall_s=time.monotonic()-start,records=rows,balance=s.summary()['balance'],finite=bool(np.isfinite(s.h).all()),nonnegative=bool((s.h>=0).all()))
(OUT/'continuation.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False),encoding='utf-8')
np.savez_compressed(OUT/'continuation-final.npz',depth=s.h,mx=s.mx,mz=s.mz,maximum_depth=s.maximum)
