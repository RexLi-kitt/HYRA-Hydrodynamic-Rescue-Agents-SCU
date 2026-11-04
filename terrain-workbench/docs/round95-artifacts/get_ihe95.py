from pathlib import Path
import urllib.request,json
S=Path(r'C:\Users\Lenovo\.codex\visualizations\2026\10\23\01a16912-ff90-73e3-a2db-13d580b278d3')
u='https://cdm21063.contentdm.oclc.org/digital/api/singleitem/collection/masters2/id/20'
try:
 with urllib.request.urlopen(u,timeout=30) as r: data=r.read()
 (S/'ihe-record95.json').write_bytes(data)
 d=json.loads(data);print(json.dumps(d,ensure_ascii=False)[:10000])
except Exception as e:print(type(e).__name__,str(e))
