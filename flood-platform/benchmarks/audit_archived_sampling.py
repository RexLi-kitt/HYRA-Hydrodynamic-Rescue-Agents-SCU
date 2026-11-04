"""Independently verify final-cell sampling; do not invent missing transient fields."""
from pathlib import Path
import argparse,json,math
import numpy as np

def run(base,output):
    results=[]
    for case in ['round47/exact_full','round48/fixed_domain_018','round51/east_plus6']:
        folder=base/case;cfg=json.loads((folder/'configuration.json').read_text());dx=cfg['dx']
        with np.load(folder/'final_state.npz',allow_pickle=False) as f:
            h=f['h'];bed=f['bed'];stage=h+bed;ny,nx=h.shape;gauges=[]
            for y in [1.196,1.696,2.196]:
                rr=ny-.5-y/dx;cc=4.521/dx-.5;i,j=math.floor(rr),math.floor(cc);fy,fx=rr-i,cc-j
                weights=np.array([[(1-fy)*(1-fx),(1-fy)*fx],[fy*(1-fx),fy*fx]])
                gauges.append({'x_m':4.521,'y_m':y,'row_col':[rr,cc],
                    'native_cells':[[i,j],[i,j+1],[i+1,j],[i+1,j+1]],'weights':weights.tolist(),
                    'stage_m':float((weights*stage[i:i+2,j:j+2]).sum()),
                    'min_local_depth_m':float(h[i:i+2,j:j+2].min()),
                    'bed_m':bed[i:i+2,j:j+2].tolist(),'inside_domain':bool(0<=i<ny-1 and 0<=j<nx-1)})
        with np.load(folder/'timeseries.npz') as f:
            err=np.asarray([g['stage_m'] for g in gauges])-f['predicted_m'][-1]
            assert f['time_s'][-1]==22.5
        with np.load(folder/'checkpoint.npz',allow_pickle=False) as f:
            m=json.loads(str(f['metadata']));dt=f['steps_dt']
            stepinfo={'actual_steps':m['audit']['steps'],'listed_dt_count':len(dt),
                'listed_dt_sum_s':float(dt.sum()),'actual_time_s':m['audit']['time_s'],
                'excess_count':len(dt)-m['audit']['steps']}
        assert max(abs(err))<1e-12
        results.append({'case':case,'final_time_s':22.5,'gauges':gauges,
            'max_abs_final_sampling_difference_m':float(max(abs(err))),
            'legacy_step_list_defect':stepinfo,'transient_cell_history_available':False})
    result={'scope':'Independent manual four-cell interpolation of stored final native fields only; no claim about unavailable 21--22s native samples',
        'cases':results,'limitation':'Correct final sampling cannot rule out transient interpolation error; missing native transient fields are not invented.'}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.output)
