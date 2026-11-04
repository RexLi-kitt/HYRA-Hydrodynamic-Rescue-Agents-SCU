"""Decode original T07 unit-width point hydrographs without inventing a model."""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np


def prepare(source, output):
    source, output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    points=collections.defaultdict(list)
    boundaries=[]
    for raw in (source/'carlisle.bci').read_text().splitlines():
        row=raw.split()
        if not row or row[0].startswith('#'): continue
        if row[0]=='P':
            assert len(row)==5 and row[3]=='QVAR',raw
            points[row[4]].append([float(row[1]),float(row[2])])
        else: boundaries.append(raw)
    rows=[s.split() for s in (source/'carlisle.bdy').read_text().splitlines()
          if s.strip() and not s.lstrip().startswith('#')]
    offset=0; records=[]
    while offset<len(rows):
        name=rows[offset][0]; count=int(rows[offset+1][0])
        assert rows[offset+1][1]=='seconds'
        values=np.array(rows[offset+2:offset+2+count],dtype=float)
        assert values.shape==(count,2) and np.isfinite(values).all()
        q,t=values.T
        assert (q>=0).all() and (np.diff(t)==900).all() and t[0]==0 and t[-1]==245700
        n=len(points[name]); Q=q*n*5.
        integral=float(np.sum(.5*(Q[1:]+Q[:-1])*np.diff(t)))
        rec={'name':name,'point_count':n,'original_grid_dx_m':5.,'unit_width_values_m2_s':q.tolist(),
             'time_since_model_start_s':t.tolist(),'physical_total_Q_m3_s':Q.tolist(),
             'point_coordinates_EPSG27700':points[name],
             'peak_Q_m3_s':float(Q.max()),'peak_time_relative_s':float(t[np.argmax(Q)]),
             'piecewise_linear_integral_m3':integral,
             'naive_same_points_50m_Q_ratio':10.,'same_point_count_50m_unit_width_multiplier':.1}
        records.append(rec); offset+=2+count
    assert {x['name'] for x in records}==set(points)
    assert sorted(x['point_count'] for x in records)==[6,6,11]
    # All times and original series are preserved. No coarse bci is generated.
    result={'evidence_class':'model input from documented 1D routing, not independent discharge observation',
      'source_files_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir() if f.is_file()},
      'relative_time_only':True,'absolute_start_utc':None,'grid_contract':'DG2 P QVAR/QFIX source volume is q * dx * accepted_dt; summed over points',
      'coarse_grid_rule':'For preserved total Q and new point count n_new, q_new=Q/(n_new*dx_new). This does not identify correct spatial distribution.',
      'source_geometry_warning':'Same-cell points are summed by the reviewed solver; merging requires preserving total flux. No point was moved or dropped.',
      'other_boundary_records':boundaries,'hydrographs':records,
      'unresolved':['absolute event origin','restricted original5m bed','50m channel/opening representation and source distribution','observed dry areas and independent same-time field comparison']}
    (output/'source-contract.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps([{k:r[k] for k in ('name','point_count','peak_Q_m3_s','peak_time_relative_s','piecewise_linear_integral_m3')} for r in records],indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output')
    a=p.parse_args();prepare(a.source,a.output)
