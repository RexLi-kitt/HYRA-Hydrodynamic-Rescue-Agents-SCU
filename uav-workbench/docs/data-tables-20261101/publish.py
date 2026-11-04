import hashlib,json,shutil,zipfile
from pathlib import Path
from xml.etree import ElementTree as E
BASE=Path(__file__).parent
BOOK=BASE/'outputs/uav-data-20261101/无人机训练与组合实验完整数据-20261101.xlsx'
data=json.loads((BASE/'data.json').read_text(encoding='utf8'))
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(BOOK) as z:
    assert z.testzip() is None
    w=E.fromstring(z.read('xl/workbook.xml'));names=[x.attrib['name'] for x in w.findall('s:sheets/s:sheet',ns)]
    assert len(names)==7
    cells=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
    def cell(ref):return cells.find(f".//s:c[@r='{ref}']/s:v",ns).text
    assert abs(float(cell('G12'))-3072)<1e-12
    assert abs(float(cell('E18'))-.99375)<1e-12
    for name in [n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')]:
        tree=E.fromstring(z.read(name));assert not tree.findall(".//s:c[@t='e']",ns),name
    for sheet,count in [(3,145),(4,1545),(5,3073),(6,61)]:
        rows=E.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml')).findall('s:sheetData/s:row',ns)
        assert len(rows)==count,(sheet,len(rows),count)
audit=dict(**data['audit'],workbook_sheets=names,formula_errors=0,exported_headline_values_verified=True,
           visual_review='all seven sheets; date formatting and headline rounding corrected',
           workbook_sha256=hashlib.sha256(BOOK.read_bytes()).hexdigest(),workbook_bytes=BOOK.stat().st_size)
outputs=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs\uav6-tables-20261101')
docs=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\docs\data-tables-20261101')
for target in [outputs,docs]:
    target.mkdir(parents=True,exist_ok=False)
    shutil.copy2(BOOK,target/BOOK.name)
    (target/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    (target/'source-index.json').write_text(json.dumps(data['sources'],ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['prepare.mjs','build.mjs','publish.py']:shutil.copy2(BASE/name,target/name)
print(json.dumps(dict(output=str(outputs/BOOK.name),bytes=BOOK.stat().st_size,sheets=names,training=data['audit']['main_training_rows'],validation=data['audit']['validation_rollouts']),ensure_ascii=True))
