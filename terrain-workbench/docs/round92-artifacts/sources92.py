from pathlib import Path
import urllib.request,json,hashlib
R=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs\round92')
sources=[('EA2005-reprinted-in-HIA2011.pdf','https://www.ecrr.org/Portals/27/Publications/Carlisle2005%20FloodHealthImpactAssessment.pdf'),('SEAMLESS-Carlisle.html','https://www.seamlesswave.com/Carlistle_flooding.html')]
rows=[]
for name,url in sources:
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'HYRA-research-source-audit/1.0'})
  with urllib.request.urlopen(req,timeout=35) as f:
   b=f.read(15000001)
  assert len(b)<=15000000
  if name.endswith('.pdf'):assert b.startswith(b'%PDF')
  (R/name).write_bytes(b)
  rows.append({'file':name,'url':url,'status':'saved','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 except Exception as e:rows.append({'file':name,'url':url,'status':'failed_no_retry','error':str(e)})
evidence={
 'downloads':rows,
 'claims':[
 {'source':'EA2005-reprinted-in-HIA2011.pdf','locator':'Appendix1 PDF pages49-51 (1-based), original EA2005 section4.3.3; page51 FigureA1',
 'claim':'Event included obstruction at South Vale footbridge, damaged wall/overtopping at gasworks, manhole surcharging downstream of Victoria Bridge, Dow Beck culvert and surface-water inputs, later Eden backwater.',
 'evidence_type':'EA2005 event account reproduced in 2011 report; not a new independent survey',
 'limits':'No numerical blockage geometry, breach dimensions, local-source hydrographs, or grid registration available. Figure uses warning zones; colored regions must not be mistaken for measured wet pixels.'},
 {'source':'SEAMLESS-Carlisle.html','locator':'Setup paragraphs, inflow point table and DEM description',
 'claim':'Published idealized benchmark uses Bristol processed5m DEM with trees/bridges removed, three1D-routed hydrographs and FREE west outlet. This is a model specification, not event truth.',
 'limits':'It is provenance for the benchmark family; byte identity to our separate Kabir archive is not asserted. Existing local config independently proves no rain/sewer/breach inputs.'}],
 'decision':'Register local geometry/omitted flood mechanisms as supported candidates, not confirmed per-cell causes. Do not add guessed water or edit elevations from observation residuals.',
 'not_downloaded':'Building archive TLS failure retained; no retry. ScienceDirect returned403, no alternate access attempted.',
 'validation_changed':False}
(R/'source-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(evidence,ensure_ascii=False))
