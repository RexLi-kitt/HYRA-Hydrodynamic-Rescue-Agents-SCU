"""Prepare one isolated raster interpretation candidate; never fit observations."""
from pathlib import Path
import json,shutil,sys
import numpy as np
from audit_carlisle_geometry78 import OUT,R,DATA,SRC,grid,dump,sha,faces,bottleneck,stats

def mm(a,b):
    return np.where((a>0)&(b>0),np.minimum(a,b),np.where((a<0)&(b<0),np.maximum(a,b),0.))
def slopes(z):
    x=np.zeros_like(z);y=np.zeros_like(z)
    x[:,1:-1]=mm(z[:,1:-1]-z[:,:-2],z[:,2:]-z[:,1:-1])/(2*np.sqrt(3))
    y[1:-1,:]=mm(z[:-2,:]-z[1:-1,:],z[1:-1,:]-z[2:,:])/(2*np.sqrt(3))
    return x,y

if __name__=='__main__':
    case=R/'mean-preserving-geometry';case.mkdir(exist_ok=False)
    protocol={
      'status':'geometry-only candidate prepared, no hydrodynamic run or adoption',
      'baseline':'author2023 archived DG2 mean/slope arrays',
      'single_factor':'Raster-to-DG2 bed representation over every cell, same affine coordinates/domain; solver and physical inputs untouched.',
      'hypothesis':'Using raw cell entries as corner nodes changes cell mean and channel face levels. Alternative interprets original cell value as DG2 mean, not certified true cell-average bathymetry.',
      'formula':'Z0=raw cell value; Z1x=minmod(Zi-Zi-1,Zi+1-Zi)/(2sqrt3); Z1y=minmod(Znorth-Zi,Zi-Zsouth)/(2sqrt3); respective perimeter slopes=0. No tuning coefficient.',
      'not_changed':['source positions and hydrographs','domain, grid header and observation coordinates','roughness','solver formulas and defaults','archive and every old bad result'],
      'criteria':['Mean arrays exactly original raw raster','Finite, planar slopes exact on interior affine field','Each face value stays between the central and axial-neighbour extrema','All3 channel-mask source-to-outlet static thresholds no higher than baseline, at least one strictly lower; otherwise reject this geometric mechanism'],
      'cost':'One static preprocessing and full mask audit; no long solve. Follow with limited static-water/dynamic compatibility gates only if geometry criteria pass.',
      'rejection':'No real-world improvement claim from lower geometric barrier. Do not alter source cells alone, widen channels by guesswork, move gauges, or fit thresholds. Poor hydrodynamic conservation/well-balance rejects candidate.',
      'open_issue':'This does not restore measured river widths or resolve author-cell-vs-node intent. Geometry candidate is not production or a verified field setup.'}
    dump(case/'protocol.json',protocol)
    z,h=grid('CarlislefixDEMwithbed.asc');w,_=grid('width.asc');bed,_=grid('bed.asc');m=w>0
    assert np.all(z!=h['nodata_value'])
    x,y=slopes(z);fx,fy=faces(z,x,y);bar,prev=bottleneck(m,fx,fy)
    old=json.loads((R/'geometry-coverage-audit.json').read_text('utf-8'))
    header='\n'.join((DATA/'CarlislefixDEMwithbed.asc').read_text().splitlines()[:6])+'\n'
    shutil.copy2(DATA/'CarlislefixDEMwithbed.asc',case/'bed.asc')
    for name,a in [('bed.asc1x',x),('bed.asc1y',y)]:
        with (case/name).open('w') as f:f.write(header);np.savetxt(f,a,fmt='%.17g')
    assert np.array_equal(np.loadtxt(case/'bed.asc',skiprows=6),z)
    ztest=3+np.arange(6)[:,None]*3+np.arange(7)[None,:]*2
    tx,ty=slopes(ztest.astype(float))
    affine_ok=bool(np.allclose(tx[1:-1,1:-1],1/np.sqrt(3),rtol=0,atol=1e-15) and np.allclose(ty[1:-1,1:-1],-1.5/np.sqrt(3),rtol=0,atol=1e-15))
    checks=[]
    for a,b in [(z+np.sqrt(3)*x,np.c_[z[:,1:],z[:,-1:]]),(z-np.sqrt(3)*x,np.c_[z[:,:1],z[:,:-1]]),(z+np.sqrt(3)*y,np.r_[z[:1,:],z[:-1,:]]),(z-np.sqrt(3)*y,np.r_[z[1:,:],z[-1:,:]])]:
        checks.append(bool(np.all(a<=np.maximum(z,b)+1e-12) and np.all(a>=np.minimum(z,b)-1e-12)))
    sources=[]
    for s in old['sources']:
        j,i=s['row'],s['col'];sources.append(dict(name=s['name'],row=j,col=i,baseline_barrier_m=s['dg2_path_threshold'],raw_piecewise_constant_barrier_m=s['raw_path_threshold'],candidate_barrier_m=float(bar[j,i]),candidate_minus_baseline_m=float(bar[j,i]-s['dg2_path_threshold'])))
    geometric_gate=bool(all(s['candidate_minus_baseline_m']<=1e-12 for s in sources) and any(s['candidate_minus_baseline_m']< -1e-12 for s in sources))
    inputs=json.loads((R/'geometry-coverage-audit.json').read_text())['all_input_hashes']
    assert all(sha(Path(p))==v for p,v in inputs.items())
    result={'checks':dict(mean_file_byte_identical=(case/'bed.asc').read_bytes()==(DATA/'CarlislefixDEMwithbed.asc').read_bytes(),finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),affine_interior_exact=affine_ok,all_faces_no_axial_overshoot=all(checks),static_source_path_gate=geometric_gate,original_hashes_unchanged=True),
      'sources':sources,'geometry_only_passed':affine_ok and all(checks) and geometric_gate,
      'channel_mean_delta_m':stats((z-z)[m]),'channel_widths_exceeding_grid_width':int((w[m]>h['cellsize']).sum()),
      'sgc_bankfull_from_this_archive_and_local_code_m':stats((z-np.minimum(bed,z-.01))[m]),
      'sgc_note':'In the locally read SGC implementation, raw DEM and SGCbed differ only by rounding at channel cells, so CalcSGCz enforces its .01m minimum bankfull depth. This is an input/code compatibility observation, not proof that SGC is wrong; wide-channel cells have distinct volume treatment. Author improved-code provenance is not established.',
      'generated_files':{p.name:sha(p) for p in case.glob('bed.asc*')},'production_adopted':False,'hydrodynamic_validation':'not run','field_passes_added':0}
    dump(case/'verification.json',result)
    np.savez_compressed(case/'geometry.npz',mean=z,sx=x,sy=y,static_path_threshold=bar)
    print(json.dumps(result,ensure_ascii=False),flush=True)
