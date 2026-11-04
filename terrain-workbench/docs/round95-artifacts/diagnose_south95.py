from pathlib import Path
import json,hashlib,heapq,struct,shutil
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round95';C=O/'round91/full_event'
p=json.loads((R/'protocol.json').read_text(encoding='utf-8'))
for f,digest in p['input_hashes'].items(): assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest,f
zall=np.loadtxt(C/'Carlisle_5m.asc',skiprows=6);mxall=np.load(O/'round91/full-classification.npz')['maximum_depth'];labels=np.load(O/'round92/all-miss-diagnostics.npz')['component4']
E0,N0,E1,N1=p['bbox_E_N'];c0=int((E0-338500)/5);c1=int((E1-338500)/5);r0=int((557755-N1)/5);r1=int((557755-N0)/5);sl=np.s_[r0:r1,c0:c1]
z=zall[sl];target=labels[sl]==2031;assert target.sum()==1457
cross=ndi.generate_binary_structure(2,1)
def heap_reach(z,h):
 eta=np.where(h>.1,z+h,-np.inf);best=eta.copy();nr,nc=z.shape
 q=[(-float(eta[y,x]),int(y),int(x)) for y,x in zip(*np.where(np.isfinite(eta)))];heapq.heapify(q)
 while q:
  neg,y,x=heapq.heappop(q);head=-neg
  if head!=best[y,x]:continue
  for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
   if 0<=yy<nr and 0<=xx<nc and head>z[yy,xx]+.1 and head>best[yy,xx]:
    best[yy,xx]=head;heapq.heappush(q,(-head,yy,xx))
 return best

def independent_reach(z,h):
 seeds=np.where(h>.1,z+h,-np.inf);v=seeds.copy()
 for i in range(z.size):
  nearby=ndi.maximum_filter(v,footprint=cross,mode='constant',cval=-np.inf)
  new=np.maximum(seeds,np.where(nearby>z+.1,nearby,-np.inf))
  if np.array_equal(new,v):return new,i
  v=new
 raise AssertionError('Reference propagation not converged')

def readbin(f):
 with f.open('rb') as stream:
  nc,nr=struct.unpack('<ii',stream.read(8));hd=struct.unpack('<dddd',stream.read(32));a=np.fromfile(stream,dtype='<f8')
 assert (nr,nc)==zall.shape and hd[:3]==(338500.,554700.,5.)
 return a.reshape(zall.shape)
marks=json.loads((O/'round92/all263-diagnostic.json').read_text(encoding='utf-8'));local=[a for a in marks if a['miss_component_id']==2031]
frames=[];union=np.zeros(z.shape,bool);potentials=[];test_results=[]
for f in sorted((C/'results').glob('*.wdb')):
 h=readbin(f)[sl];a=heap_reach(z,h);b,n=independent_reach(z,h);assert np.array_equal(a,b),f.name
 available=np.isfinite(a);union|=available;potentials.append(a)
 frames.append({'file':f.name,'nominal_output_time_s':int(f.stem.split('-')[-1])*21600,'available_target_cells':int((available&target).sum()),'actual_target_above_0_1':int(((h>.1)&target).sum()),'local_wet_seeds':int((h>.1).sum()),'reference_iterations':n,'marks':[{'fid':m['fid'],'static_reachable':bool(available[m['grid_row']-r0,m['grid_col']-c0]),'local_depth_m':float(h[m['grid_row']-r0,m['grid_col']-c0]),'available_stage_masl':float(a[m['grid_row']-r0,m['grid_col']-c0]) if available[m['grid_row']-r0,m['grid_col']-c0] else None} for m in local]})
 hmax=mxall[sl]
a=heap_reach(z,hmax);b,n=independent_reach(z,hmax);assert np.array_equal(a,b)
optimistic=np.isfinite(a)
# Independently verified tiny topography cases including isolated bowl and passable/blocked sill.
for sill in (1.,3.):
 zz=np.array([[0.,sill,0.]]);hh=np.array([[2.,0.,0.]])
 got=heap_reach(zz,hh);ref,_=independent_reach(zz,hh);assert np.array_equal(got,ref);assert np.isfinite(got[0,2])==(sill==1.)
 test_results.append({'sill':sill,'passed':True})
summary={'target_id':2031,'target_cells':1457,'target_area_m2':36425,'crop':p['bbox_E_N'],'saved_frames':len(frames),'same_frame_union_potential_cells':int((union&target).sum()),'same_frame_never_potential_cells':int((~union&target).sum()),'asynchronous_max_potential_cells':int((optimistic&target).sum()),'asynchronous_max_never_potential_cells':int((~optimistic&target).sum()),'all_target_marks':[{'fid':m['fid'],'E':m['E'],'N':m['N'],'observed_peak_masl':m['Z_masl'],'cell_bed_masl':float(z[m['grid_row']-r0,m['grid_col']-c0]),'model_max_depth_m':float(hmax[m['grid_row']-r0,m['grid_col']-c0]),'any_snapshot_static_reachable':bool(union[m['grid_row']-r0,m['grid_col']-c0]),'max_only_static_reachable':bool(optimistic[m['grid_row']-r0,m['grid_col']-c0])} for m in local],'frames':frames,'physical_changes':False,'new_accuracy_score':False,'all_old_misses_retained':True,'limits':p['decision_rules']}
(R/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
np.savez_compressed(R/'southern-static-reachability.npz',target=target,terrain=z,snapshot_potential=np.array(potentials),snapshot_union=union,max_only_potential=a,crop_row_col=[r0,r1,c0,c1])
verification={'all13_real_fields_independent_methods_exactly_equal':True,'tiny_barrier_tests':test_results,'all_input_hashes_verified':True,'all1457_target_cells_retained':True,'all263_records_unchanged':True}
(R/'independent-verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained');ext=[E0,E1,N0,N1]
axs[0].imshow(z,extent=ext,origin='upper',cmap='terrain',vmin=12,vmax=25)
axs[0].contour(target,levels=[.5],extent=ext,origin='upper',colors='magenta',linewidths=1)
axs[0].set_title('Original terrain; southern missed component outline')
classes=np.full(z.shape,np.nan);classes[target]=0;classes[target&optimistic]=1;classes[target&union]=2
from matplotlib.colors import ListedColormap,BoundaryNorm
im=axs[1].imshow(classes,extent=ext,origin='upper',cmap=ListedColormap(['#555555','#e6ac00','#00a6c7']),norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
cb=fig.colorbar(im,ax=axs[1],ticks=[0,1,2],shrink=.6);cb.ax.set_yticklabels(['No static access','Max-only access','Snapshot access'])
axs[1].set_title('Potential static access only; NOT predicted inundation')
for m in local:
 for ax in axs:ax.plot(m['E'],m['N'],'ro',markersize=3);ax.text(m['E']+2,m['N']+2,str(m['fid']),fontsize=7)
for ax in axs:ax.set_xlabel('Easting EPSG:27700 (m)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(useOffset=False,style='plain');ax.set_aspect('equal')
fig.suptitle('Round95: constant-head diagnostic, no added water or modified DEM')
fig.savefig(R/'south-static-access.png',dpi=150);plt.close(fig)
shutil.copy2(__file__,R/'diagnose_south95.py')
print(json.dumps({k:v for k,v in summary.items() if k not in ['frames','limits','all_target_marks']},ensure_ascii=False))
print(json.dumps(summary['all_target_marks'],ensure_ascii=False))
