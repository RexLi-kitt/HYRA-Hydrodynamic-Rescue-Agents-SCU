from pathlib import Path
import json,hashlib
from pypdf import PdfReader
R=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs\round88')
p=R/'Rong-2024-thesis.pdf';r=PdfReader(p);hits=[]
for i,page in enumerate(r.pages):
    text=page.extract_text() or ''
    if any(term in text.lower() for term in ['carlisle','sheepmount','hvar']):
        (R/f'thesis-page-{i+1:03d}.txt').write_text(text,encoding='utf-8')
        hits.append({'pdf_page1':i+1,'terms':[x for x in ['carlisle','sheepmount','hvar'] if x in text.lower()],'preview':text[:140]})
rec={'url':'https://research-information.bris.ac.uk/ws/portalfiles/portal/430852101/Final_Copy_2024_10_15_Rong_Y_T_PhD.pdf','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'pages':len(r.pages),'hits':hits,'method':'pypdf text extraction; selected relevant pages to be visually checked; no source edits'}
(R/'thesis-index.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
