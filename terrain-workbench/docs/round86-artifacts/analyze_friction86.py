from pathlib import Path
import json,numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round86'
assert json.loads((R/'status.json').read_text())['state']=='completed'
proto=json.loads((R/'protocol.json').read_text());rows=[];arrays={}
def compare(a,b):
    d=b-a
    return {'max_abs_depth_m':float(abs(d).max()),'L1_depth_volume_m3':float(abs(d).sum()*100),'net_volume_change_m3':float(d.sum()*100),'masks':[{'threshold_m':th,'both_wet':int(((a>th)&(b>th)).sum()),'a_only_wet':int(((a>th)&~(b>th)).sum()),'b_only_wet':int((~(a>th)&(b>th)).sum()),'changed_cells':int(((a>th)!=(b>th)).sum())} for th in proto['thresholds_m']]}
for idx,t in enumerate(proto['times_s']):
    cases=[O/'round83/carlisle_2400',O/'round85/carlisle_2400_cfl_half',R/'cfl033',R/'cfl0165']
    hs=[np.loadtxt(c/f'results/check-{idx:04d}.wd',skiprows=6) for c in cases]
    row={'time_s':t,'old_cfl_pair':compare(hs[0],hs[1]),'new_cfl_pair':compare(hs[2],hs[3]),'change_at_cfl033':compare(hs[0],hs[2]),'change_at_cfl0165':compare(hs[1],hs[3])}
    row['sensitivity_change']={'max_abs_depth_reduction_fraction':1-row['new_cfl_pair']['max_abs_depth_m']/row['old_cfl_pair']['max_abs_depth_m'],'L1_reduction_fraction':1-row['new_cfl_pair']['L1_depth_volume_m3']/row['old_cfl_pair']['L1_depth_volume_m3'],'changed_cells_before_after':[[a['changed_cells'],b['changed_cells']] for a,b in zip(row['old_cfl_pair']['masks'],row['new_cfl_pair']['masks'])]}
    rows.append(row)
    for k,h in enumerate(hs):arrays[f'h{t}_case{k}']=h
    for label,a,b in [('old',hs[0],hs[1]),('new',hs[2],hs[3])]:
        for th in proto['thresholds_m']:arrays[f'{label}_t{t}_th{th}']=(a>th).astype('uint8')+2*(b>th).astype('uint8')
np.savez_compressed(R/'coupled-full-comparisons.npz',**arrays)
result={'evidence':'paired numerical sensitivity, not independent observations','frames':rows,'field_passes_added':0,'production_adopted':False}
(R/'coupled-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
