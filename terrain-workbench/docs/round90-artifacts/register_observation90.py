from pathlib import Path
import sys,json,sqlite3,hashlib
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90'
sys.path.insert(0,str(O/'reference-geospatial-libs'))
import numpy as np
import shapely
from shapely.ops import transform,unary_union
P=O/'round68/data/carlisle_recorded_flood_outlines.gpkg'
protocol={'scope':'all author-archived Carlisle2005 maximum-outline polygons intersecting fixed Kabir5m grid; existing five polygons and all263 marks preserved','registered_before_full_event_simulation':True,'observational_units':'whole 5m cell footprints fully covered by the reported event-maximum outline union; partially intersected cells separately boundary-ambiguous; outside outline is unknown, NOT dry','classification_indicator':'maximum simulated depth >0.1m following archived round68 extent.py analysis indicator; additional 0.01m and0.3m classifications reported as fixed sensitivity, not selected for score','limits':['0.1m is an analysis indicator, not personal/vehicle safety','outline precision has no calibrated uncertainty bound; edge ambiguity is geometric only','source polygons may contain permanent channels; does not prove newly flooded dry land','no known-dry evidence, so cannot infer false alarms, overall accuracy, IoU or verified route accessibility','no observed arrival/recession maps here; maximum map only','other qualitative requirements remain unverified, cannot be deleted for pass'],'pass_rule':'report every pre-registered fully covered cell as supported or missed using fixed main indicator; any miss prevents all-consistent result; boundary and external unknowns preserved, not in correct-count','no_existing_simulation_fields_read':True}
(R/'observation-protocol.json').write_text(json.dumps(protocol,indent=2))
db=sqlite3.connect('file:'+str(P)+'?mode=ro',uri=True)
tab,col,srs=db.execute('select table_name,column_name,srs_id from gpkg_geometry_columns').fetchone()
assert all(c.isalnum() or c=='_' for c in tab+col)
assert int(srs)==27700  # Original polygons already share the DEM CRS; no transformation.
polys=[]
for (b,) in db.execute(f'SELECT "{col}" FROM "{tab}"'):
 env=(b[3]>>1)&7;off=8+{0:0,1:32,2:48,3:48,4:64}[env]
 g=shapely.from_wkb(bytes(b[off:]));assert g.is_valid
 polys.append(g)
db.close();u=unary_union(polys);grid=shapely.box(338500,554700,338500+951*5,554700+611*5)
mask=np.zeros((611,951),dtype='uint8');xs=338500+np.arange(951)*5
for row in range(611):
 y=554700+(610-row)*5;cells=shapely.box(xs,y,xs+5,y+5)
 inter=shapely.intersects(u,cells);cover=shapely.covers(u,cells)
 mask[row,inter]=2;mask[row,cover]=1
np.savez_compressed(R/'observation-footprint-mask.npz',mask=mask)
marks=json.loads((O/'round68/prepared/all-survey-marks.json').read_text());records=[]
for m in marks:
 a=m['source_attributes'];x=a['E'];y=a['N'];inside=338500<=x<343255 and 554700<=y<557755
 records.append({'fid':a['fid'],'E':x,'N':y,'Z_masl':a['Z'],'inside_fixed_grid':inside,'original_outline_membership':m['inside_outline'],'verification':'unscored; surveyed elevation is not depth'})
(R/'all-marks-fixed-domain.json').write_text(json.dumps(records,indent=2))
result={'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'source_srs':srs,'destination_srs':27700,'polygons':len(polys),'polygons_intersecting_domain':sum(p.intersects(grid) for p in polys),'reported_maximum_inundation_fully_covered_cells':int((mask==1).sum()),'edge_intersection_ambiguous_cells':int((mask==2).sum()),'outside_unknown_cells':int((mask==0).sum()),'total_cells':mask.size,'all263_marks_retained':len(records)==263,'marks_inside_fixed_grid':sum(x['inside_fixed_grid'] for x in records),'geometry_no_simulation_read':True,'field_passes_added':0}
assert sum(int((mask==i).sum()) for i in range(3))==581061 and len(polys)==5
(R/'observation-registration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
