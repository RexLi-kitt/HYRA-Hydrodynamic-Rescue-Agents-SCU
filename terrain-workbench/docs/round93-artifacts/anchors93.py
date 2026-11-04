"""Independent 2016 structure-location anchors, NOT historical hydraulic geometry."""
from pathlib import Path
import urllib.request,json,hashlib,math,shutil
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round93';R.mkdir(exist_ok=True)
URL='https://cumbria.gov.uk/elibrary/Content/Internet/536/6357/42684141451.PDF'
names=['ROME STREET RLY NO 6','ROME STREET ARCHES NO 1','CALDEW BRIDGE APPROACH','CALDEW RLY NO 2','VICTORIA VIADUCT RLY NO 252','BOTCHERBY','CALDEW LITTLE','HOLME HEAD F/B','HARRABY','DOW BECK (BEDFORD ROAD)','DOW BECK (DOW BECK ROAD)','DOW BECK (DUNMAIL DRIVE)','DOW BECK (MORTON)','DOW BECK (SCAWFELL ROAD)','DOW BECK (STANHOPE ROAD)','DURRANHILL BECK CULVERT','DURRANHILL CULVERT']
protocol={'source':URL,'source_date':'2016-11-10','evidence_type':'County council structure register, post-event location evidence only','selection_before_spatial_comparison':names,'rules':'Read published eastings/northings, exact name match, no coordinate fitting. Compute distance to all4 prior bottlenecks and fixed missed-component bounding boxes. Coordinates alone never justify DEM edits or source additions. Asset list is not exhaustive.', 'reject':'Absent/unparseable rows remain gaps; do not guess values. No dimensions or2005 identity inferred from2016 locations.'}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
p=R/'Cumbria-FOI-2016-1020.pdf'
if not p.exists():
 try:
  with urllib.request.urlopen(URL,timeout=30) as f:b=f.read(25000001)
  assert len(b)<=25000000 and b.startswith(b'%PDF');p.write_bytes(b)
 except Exception as e:
  (R/'source-status.json').write_text(json.dumps({'state':'failed_no_retry','url':URL,'error':str(e)},indent=2));raise
from pypdf import PdfReader
reader=PdfReader(p); pages=[]
for i in range(22,37):pages.append({'pdf_page':i+1,'text':reader.pages[i].extract_text()})
(R/'source-pages.json').write_text(json.dumps(pages,indent=2),encoding='utf8')
# Parse selected rows from original PDF, never from model locations.
import re
text='\n'.join(a['text'] for a in pages)
blocks=re.split(r'\bCARLISLE\b',text)[1:];rows=[];missing=[]
for name in names:
 hits=[]
 for block in blocks:
  clean=' '.join(block.split())
  if clean.startswith(name):
   m=re.search(r'\b(3\d{5})\s+(5\d{5})\b',clean)
   if m:hits.append({'name':name,'E':int(m.group(1)),'N':int(m.group(2)),'raw_normalized_row':clean[:m.end()+20]})
 unique={(a['E'],a['N']):a for a in hits}
 if len(unique)==1:rows.append(next(iter(unique.values())))
 else:missing.append({'name':name,'candidate_coordinates':list(unique),'status':'missing_or_ambiguous_no_guess'})
paths=json.loads((O/'round92/all32-minimax-paths.json').read_text());targets=[a for a in paths if a['barrier_above_observed_peak']]
south=[a for a in paths if a['fid']==14][0]
targets.append(dict(fid='south-group-14-23',bottleneck_E_N=south['bottleneck_E_N']))
comps=json.loads((O/'round92/all-miss-components.json').read_text());c2031=next(a for a in comps if a['id']==2031);xmin,ymin,xmax,ymax=c2031['bbox_E_N']
for a in rows:
 a['inside_fixed_domain']=338500<=a['E']<343255 and 554700<=a['N']<557755
 a['distance_to_cluster2031_bbox_m']=math.hypot(max(xmin-a['E'],0,a['E']-xmax),max(ymin-a['N'],0,a['N']-ymax))
 a['distances_to_diagnostic_bottlenecks_m']={str(t['fid']):math.hypot(a['E']-t['bottleneck_E_N'][0],a['N']-t['bottleneck_E_N'][1]) for t in targets}
 a['limits']='2016 location; unknown2005 dimensions/condition, grid accuracy and cross-section. Bbox proximity is not hydraulic connectivity.'
result={'source':URL,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected':len(names),'parsed':len(rows),'missing':missing,'anchors':rows,'all_failure_scores_unchanged':True,'simulation_runs':0,'production_modified':False}
(R/'independent-structure-anchors.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({'selected':len(names),'parsed':len(rows),'missing':missing,'south_nearest':sorted([{'name':a['name'],'distance_m':a['distances_to_diagnostic_bottlenecks_m']['south-group-14-23']} for a in rows],key=lambda a:a['distance_m'])[:3]},indent=2))
