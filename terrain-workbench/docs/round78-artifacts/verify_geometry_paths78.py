from collections import deque
import json
import numpy as np
from audit_carlisle_geometry78 import R,DATA,faces,dump,sha

def connected_at(mask,fx,fy,source,level):
    ny,nx=mask.shape;seen=set();todo=deque()
    for j in np.flatnonzero(mask[:,0] & (fx[:,0]<=level)):
        seen.add((int(j),0));todo.append((int(j),0))
    while todo:
        j,i=todo.popleft()
        if (j,i)==source:return True
        for jj,ii,h in [(j,i-1,fx[j,i]),(j,i+1,fx[j,i+1]),(j-1,i,fy[j,i]),(j+1,i,fy[j+1,i])]:
            if 0<=jj<ny and 0<=ii<nx and mask[jj,ii] and h<=level and (jj,ii) not in seen:
                seen.add((jj,ii));todo.append((jj,ii))
    return False

if __name__=='__main__':
    a=np.load(R/'geometry-arrays.npz');m=a['mask'];zero=np.zeros_like(a['raw'])
    rawfaces=faces(a['raw'],zero,zero)
    c=np.load(R/'mean-preserving-geometry/geometry.npz');newfaces=faces(c['mean'],c['sx'],c['sy'])
    old=json.loads((R/'geometry-coverage-audit.json').read_text('utf-8'));candidate=json.loads((R/'mean-preserving-geometry/verification.json').read_text('utf-8'))
    checks=[]
    for k,s in enumerate(old['sources']):
        point=(s['row'],s['col'])
        for name,f,level in [('raw',rawfaces,s['raw_path_threshold']),('archive_dg2',(a['dg2_face_x'],a['dg2_face_y']),s['dg2_path_threshold']),('candidate',newfaces,candidate['sources'][k]['candidate_barrier_m'])]:
            at=connected_at(m,*f,point,level+1e-12);below=connected_at(m,*f,point,level-1e-9)
            checks.append(dict(source=s['name'],representation=name,threshold_m=level,connected_at_threshold=at,disconnected_below_threshold=not below,passed=at and not below))
    assert all(x['passed'] for x in checks)
    lines=(DATA/'carlisle.bdy').read_text().splitlines();i=next(i for i,s in enumerate(lines) if s.strip()=='downstream');n=int(lines[i+1].split()[0]);arr=np.array([list(map(float,l.split())) for l in lines[i+2:i+2+n]])
    diagnostic={'curve_rows':n,'curve_value_min':float(arr[:,0].min()),'curve_value_max':float(arr[:,0].max()),'original_time_span_s':arr[[0,-1],1].tolist(),'western_raw_bed_min_m':float(a['raw'][:,0].min()),'western_channel_rows':np.flatnonzero(m[:,0]).tolist(),'western_channel_beds_m':a['raw'][m[:,0],0].tolist(),'note':'HVAR spans both below and above lowest western bed; cannot dismiss specified level as always dry or assume it equivalent to free outlet. Absolute datum/epoch still unresolved.'}
    dump(R/'independent-path-verification.json',{'checks':checks,'all9_passed':True,'method':'Thresholded breadth-first graph reachability independently checks minimax Dijkstra results, still static channel-mask geometry only.','boundary_diagnostic':diagnostic})
    print(json.dumps({'all9_passed':True,'boundary_diagnostic':diagnostic},ensure_ascii=False))
