"""Independent input/evidence compatibility audit. Does not read simulation fields."""
from pathlib import Path
import sys,json,sqlite3,hashlib,shutil
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round91-artifacts'
sys.path.insert(0,str(O/'reference-geospatial-libs'))
import shapely
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'round68/data/carlisle_recorded_flood_outlines.gpkg'
db=sqlite3.connect('file:'+str(G)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
records=[];polys=[]
for row in db.execute('select * from carlisle_recorded_flood_outlines order by fid'):
 a=dict(row);b=a.pop('geom');off=8+{0:0,1:32,2:48,3:48,4:64}[(b[3]>>1)&7]
 g=shapely.from_wkb(bytes(b[off:]));assert g.is_valid
 records.append(a);polys.append(g)
db.close()
grid=shapely.box(338500,554700,343255,557755)
mask=np.load(O/'round90/observation-footprint-mask.npz')['mask']
all_masks=np.zeros((5,611,951),dtype='bool');xs=338500+np.arange(951)*5
for k,(a,g) in enumerate(zip(records,polys)):
 for row in range(611):
  y=554700+(610-row)*5
  all_masks[k,row]=shapely.covers(g,shapely.box(xs,y,xs+5,y+5))
 a.update(intersection_area_m2=float(g.intersection(grid).area),fully_covered_cells=int(all_masks[k].sum()))
 a['physical_scope_note']=('Archive labels drainage/surface water; current case has river inflows and FREE outlet, no explicit rainfall or sewer network. Retain all cells; this is an input-process gap, not permission to exclude failures.' if a['flood_src']=='drainage' else 'Archive labels main-river flooding; overtopping/defence geometry remains an input limitation where relevant.')
drain=np.any(all_masks[[i for i,a in enumerate(records) if a['flood_src']=='drainage']],axis=0)
river=np.any(all_masks[[i for i,a in enumerate(records) if a['flood_src']=='main river']],axis=0)
assert np.all(~np.any(all_masks,axis=0)|(mask==1))
dem=np.loadtxt(R/'full_event/Carlisle_5m.asc',skiprows=6)
marks=json.loads((O/'round68/prepared/all-survey-marks.json').read_text())
comparison=[]
for item in marks:
 a=item['source_attributes'];x,y=a['E'],a['N'];col=int((x-338500)//5);row=610-int((y-554700)//5)
 assert 0<=row<611 and 0<=col<951
 z=float(dem[row,col]);comparison.append({'source_attributes':a,'grid_row':row,'grid_col':col,'archived_5m_cell_elevation_m':z,'survey_Z_minus_grid_elevation_m':a['Z']-z,'original_outline_membership':item['inside_outline'],'meaning':'compatibility diagnostic only, NOT measured local water depth; vertical datum and cell-vs-point representation uncertainty remain'})
diff=np.array([a['survey_Z_minus_grid_elevation_m'] for a in comparison])
out={'source_sha256':sha(G),'dem_sha256':sha(R/'full_event/Carlisle_5m.asc'),'mark_source_sha256':sha(O/'round68/prepared/all-survey-marks.json'),'polygons':records,'drainage_fully_covered_cells':int(drain.sum()),'drainage_and_river_covered_cells':int((drain&river).sum()),'drainage_only_covered_cells':int((drain&~river).sum()),'union_known_wet_cells_unchanged':int((mask==1).sum()),'all263_marks_retained':len(comparison)==263,'survey_elevation_below_archived_grid_cells':int((diff<0).sum()),'survey_minus_grid_range_m':[float(diff.min()),float(diff.max())],'did_read_simulation_fields':False,'did_change_scoring_scope':False,'field_passes_added':0,'conclusion':'retain frozen all-polygon score; label causal and elevation representation gaps independently of eventual outcome. Do not change terrain at marks, shift points, clip negative inferred depths, or interpret outside polygons as dry.'}
np.savez_compressed(R/'individual-observation-masks.npz',covered_by_polygon=all_masks,fids=np.array([a['fid'] for a in records]))
(R/'observation-source-compatibility.json').write_text(json.dumps(out,indent=2))
(R/'all263-elevation-compatibility.json').write_text(json.dumps(comparison,indent=2))
shutil.copy2(__file__,R/Path(__file__).name)
for n in ['individual-observation-masks.npz','observation-source-compatibility.json','all263-elevation-compatibility.json',Path(__file__).name]:
 shutil.copy2(R/n,A/n);assert sha(R/n)==sha(A/n)
print(json.dumps(out))
