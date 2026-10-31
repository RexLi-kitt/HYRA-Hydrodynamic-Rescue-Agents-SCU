"""Read-only solver audit; writes evidence only. Never opens running result fields."""
from pathlib import Path
import hashlib, json, struct, shutil
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs')
R=O/'round91'; A=D/'round91-artifacts'; S=O/'round86/lisflood8-isotropic-friction'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((R/'protocol.json').read_text())
assert all(sha(R/'full_event'/n)==h for n,h in protocol['configuration_hashes'].items())
par=(R/'full_event/case.par').read_text()
keys={line.split()[0].lower() for line in par.splitlines() if line.strip() and not line.lstrip().startswith('#')}
assert not keys.intersection({'depththresh','maxint','saveint_max','mint_hk'})
snippets=[]
for name,lo,hi in [('lisflood.cpp',165,170),('lisflood.cpp',224,227),('lisflood.cpp',252,256),('lisflood2/sgm_fast.cpp',3203,3217),('lisflood2/sgm_fast.cpp',3944,3964),('lisflood2/sgm_fast.cpp',4628,4660),('lisflood2/sgm_fast.cpp',4704,4718),('lisflood2/sgm_fast.cpp',4858,4867),('lisflood2/lis2_output.cpp',506,516)]:
 p=S/name; lines=p.read_text(errors='replace').splitlines()
 snippets.append({'file':name,'sha256':sha(p),'lines':[{'line':i+1,'text':lines[i]} for i in range(lo-1,hi)]})
def grid(p):
 with p.open('rb') as f:
  nc,nr=struct.unpack('<ii',f.read(8)); hdr=struct.unpack('<4d',f.read(32)); h=np.fromfile(f,'<f8')
 assert (nc,nr)==(951,611) and hdr[:3]==(338500.,554700.,5.) and h.size==nc*nr
 return h.reshape(nr,nc)
completed=O/'round90/cost7200_omp4/results'
mx=grid(completed/'check.maxb'); checks=[]
for p in sorted(completed.glob('*.wdb')):
 h=grid(p); eligible=h>.001
 checks.append({'file':p.name,'sha256':sha(p),'cells_above_accumulation_floor':int(eligible.sum()),'recorded_max_below_saved_depth_cells':int((eligible&(mx+1e-12<h)).sum())})
assert len(checks)==5 and all(x['recorded_max_below_saved_depth_cells']==0 for x in checks)
audit={'purpose':'audit event-maximum output semantics before full-event scoring, without reading running simulation fields','configuration_hashes_unchanged':True,'depth_accumulation_floor_m':.001,'zero_recorded_max_meaning':'never recorded above 0.001 m; may contain thinner water; NOT proof of mathematically zero depth or observed dry land','maximum_updated':'every accepted step under this configuration; massint 1e-9 and mint_hk OFF','maximum_resets':'both maxint and saveint_max OFF; six-hour regular saves do not reset maximum','scoring_thresholds_unchanged_m':[.01,.1,.3],'not_a_model_change':True,'existing_completed_7200s_regression':checks,'source':snippets,'full_event_unscored':True,'production_adopted':False}
(R/'maximum-output-semantics.json').write_text(json.dumps(audit,indent=2))
shutil.copy2(__file__,R/Path(__file__).name)
for n in ['maximum-output-semantics.json',Path(__file__).name]:shutil.copy2(R/n,A/n)
print(json.dumps({k:v for k,v in audit.items() if k!='source'}))
