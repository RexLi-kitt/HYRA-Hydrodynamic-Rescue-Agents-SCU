from pathlib import Path
import json,hashlib,struct
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round83';C=R/'carlisle_2400'
status=json.loads((R/'status.json').read_text())
if status['state']=='running':raise SystemExit('Not complete; verification not executed.')
def series(path):
    lines=[s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    blocks={};k=0
    while k<len(lines):
        name=lines[k];n,unit=lines[k+1].split();assert unit=='seconds';n=int(n)
        a=np.array([[float(v) for v in line.split()] for line in lines[k+2:k+2+n]])
        assert np.all(np.diff(a[:,1])>0);blocks[name]=a;k+=n+2
    return blocks
def integral(a,t):
    knots=np.unique(np.r_[0,a[(a[:,1]>0)&(a[:,1]<t),1],t])
    vals=np.interp(knots,a[:,1],a[:,0])
    return float(np.sum((vals[1:]+vals[:-1])*.5*np.diff(knots)))
rec={'state':status['state'],'field_passes_added':0}
rejected=np.loadtxt(C/'rejected_rk_attempts.txt',ndmin=2)
rec['rejection_stage_counts']={str(i):int((rejected[:,3]==i).sum()) for i in (1,2)}
rec['stage2_retry_observed']=bool((rejected[:,3]==2).any())
steps=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2)
old=np.loadtxt(O/'round81/archive_conditional/accepted_timesteps.txt',ndmin=2)
n=min(len(steps),len(old));equal=np.all(steps[:n]==old[:n],axis=1);diff=np.flatnonzero(~equal)
rec['matching_original_attempt_prefix_rows']=int(diff[0]) if len(diff) else n
rec['steps_contiguous']=bool(np.allclose(steps[1:,0],steps[:-1,0]+steps[:-1,1],rtol=0,atol=1e-10))
rec['positive_steps']=bool((steps[:,1]>0).all())
end=float(steps[-1].sum());blocks=series(C/'carlisle.bdy')
expected=sum(integral(blocks[n],end)*10 for n in ['upstream1','upstream2','upstream3'])
v=np.loadtxt(C/'accepted_source_volumes.txt',ndmin=2)
rec.update(independent_source_expected_m3=expected,actual_source_m3=float(v[:,2].sum()),source_integral_error_m3=float(v[:,2].sum())-expected)
rec['rejection_snapshot']=None
f=C/'last-rejection-prestate.bin'
if f.exists():
    data=f.read_bytes();nx,ny,nf=struct.unpack_from('iii',data);t,dt=struct.unpack_from('dd',data,12)
    a=np.frombuffer(data,dtype=np.float64,offset=28).reshape(nf,ny,nx)
    rec['rejection_snapshot']={'nx':nx,'ny':ny,'fields':nf,'t_s':t,'attempted_dt_s':dt,'all_coefficients_finite':bool(np.isfinite(a).all()),'mean_nonnegative':bool((a[0]>=0).all()),'bytes':len(data),'not_an_executable_restart':True}
rec['regression_export_bytes_equal']={}
for name in ['closed','drain_E','point_and_outlet']:
    new=R/'controls'/name/'results';base=O/'round77/candidate'/name/'results'
    rec['regression_export_bytes_equal'][name]={f.name:f.read_bytes()==(base/f.name).read_bytes() for f in new.glob('check-0000.*')}
rec['independent_gate_pass']=bool(status.get('internal_gate_pass') and rec['steps_contiguous'] and rec['positive_steps'] and abs(rec['source_integral_error_m3'])<1e-6 and all(all(v.values()) for v in rec['regression_export_bytes_equal'].values()))
(R/'independent-verification.json').write_text(json.dumps(rec,indent=2))
print(json.dumps(rec),flush=True)
