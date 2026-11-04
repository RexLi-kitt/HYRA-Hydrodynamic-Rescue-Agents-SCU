from pathlib import Path
import json, hashlib, urllib.request, shutil
S=Path(__file__).parent
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
R=O/'round89'; R.mkdir(exist_ok=False); (R/'original').mkdir()
tree=json.loads((S/'kabir-tree89.json').read_text(encoding='utf-8-sig'))
assert not tree.get('truncated')
shutil.copy2(S/'kabir-tree89.json',R/'repository-tree.json')
protocol={'objective':'Audit separate author-published 5m Carlisle 2005 Run1 input package before any simulation or scoring','revision':tree['sha'],'unchanged_existing_cases':True,'checks':['Git blob and length','finite DEM/grid and CRS','source points and unit-width aggregation','CSV vs bdy value/time identity','event start publication linkage','initial/outlet/configuration completeness'],'rejection':'Do not silently replace original 10m inputs or assign this epoch to other archives; unclosed contracts prevent field scoring','simulation_started':False}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2))
selected=[x for x in tree['tree'] if x['path'] in ['README.md','Data/Carlisle_5m.asc','Data/Upstream_Flows_Run1.csv','Data/carlisle.bci','Data/carlisle_run1.bdy','Data/carlisle_run1.par'] or x['path'].startswith('Data/validation_locations_18Points.')]
manifest=[]
for x in selected:
    url='https://raw.githubusercontent.com/SRKabir/Rapid_FloodModelling_CNN/'+tree['sha']+'/'+x['path']
    with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert len(data)==x['size'] and blob==x['sha'], x['path']
    (R/'original'/Path(x['path']).name).write_bytes(data)
    manifest.append({'path':x['path'],'url':url,'bytes':len(data),'git_blob':blob,'sha256':hashlib.sha256(data).hexdigest()})
    (R/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'downloaded':len(manifest),'revision':tree['sha'],'bytes':sum(x['bytes'] for x in manifest)}))
