from pathlib import Path
import json,hashlib,numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round87';C=R/'carlisle_2400_cfl033';B=O/'round83/carlisle_2400'
s=json.loads((R/'status.json').read_text());assert s['state']!='running'
if not s.get('internal_gate_pass'):raise SystemExit('Preserve failure; do not score partial fields.')
rows=[];arrays={}
for idx,t in enumerate(range(300,2401,300)):
    a=np.loadtxt(B/f'results/check-{idx:04d}.wd',skiprows=6);b=np.loadtxt(C/f'results/check-{idx:04d}.wd',skiprows=6);d=b-a;row={'time_s':t,'max_abs_depth_change_m':float(abs(d).max()),'L1_depth_change_volume_m3':float(abs(d).sum()*100),'net_volume_change_m3':float(d.sum()*100),'thresholds':[]}
    for th in [1e-6,.001,.01]:
        x=a>th;y=b>th;row['thresholds'].append({'h_m':th,'old_wet':int(x.sum()),'candidate_wet':int(y.sum()),'both_wet':int((x&y).sum()),'old_only_wet':int((x&~y).sum()),'candidate_only_wet':int((~x&y).sum()),'changed_cells':int((x!=y).sum())});arrays[f't{t}_h{th}']=x.astype('uint8')+2*y.astype('uint8')
    arrays[f'depth_change_t{t}']=d;rows.append(row)
steps=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2);continuous=bool(np.allclose(steps[1:,0],steps[:-1].sum(axis=1),atol=1e-10,rtol=0));assert continuous
rec={'type':'same-CFL algorithm change, not observational correctness and not new candidate late-window time sensitivity','field_passes_added':0,'accepted_steps_contiguous':continuous,'frames':rows,'fine_candidate_2400_available':False}
(R/'extension-comparison.json').write_text(json.dumps(rec,indent=2));np.savez_compressed(R/'full-extension-comparison.npz',**arrays)
print(json.dumps({'last_frame':rows[-1],'accepted_steps_contiguous':continuous}),flush=True)
