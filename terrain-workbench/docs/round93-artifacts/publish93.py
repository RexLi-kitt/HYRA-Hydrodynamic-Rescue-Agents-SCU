from pathlib import Path
import json,hashlib,shutil
S=Path(__file__).parent;O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round93';D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round93-artifacts'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((R/'independent-structure-anchors.json').read_text());a={v['name']:v for v in d['anchors']}
checks={'17_preselected_rows_all_parsed':d['selected']==d['parsed']==17,'original_foi_pdf_hash_matches':sha(R/'Cumbria-FOI-2016-1020.pdf')==d['source_sha256']}
for name,xy in {'ROME STREET RLY NO 6':(340113,555150),'ROME STREET ARCHES NO 1':(340145,555204),'BOTCHERBY':(341733,555900),'CALDEW RLY NO 2':(339609,556048),'HOLME HEAD F/B':(339843,554543)}.items():checks[name]=(a[name]['E'],a[name]['N'])==xy
assert all(checks.values())
(R/'coordinate-verification.json').write_text(json.dumps({'checks':checks,'independent_method':'Five selected coordinates cross-checked against separately retrieved public PDF text in web tool; parser reads local original PDF.','input_hashes':{str(p):sha(p) for p in [O/'round92/all32-minimax-paths.json',O/'round92/all-miss-components.json',O/'round91/protocol.json']},'simulation_checks':'Skipped: no solver/input modifications.'},indent=2),encoding='utf8')
report='''# 第93轮：桥涵位置证据核查（2026-10-31）

基线为第92轮全部漏项及路径。完成了一个有限的数据补充单元：取得Cumbria County Council公开FOI2016-1020原PDF（2016-11-10），解析预先按名称选定的17处桥涵/跨线结构坐标，逐一计算到既有瓶颈与最大南部漏片边框的距离。未启动模拟，原88.56%受淹正例覆盖和全部失败保持；没有精度提升。

来源：https://cumbria.gov.uk/elibrary/Content/Internet/536/6357/42684141451.PDF

## 发现与适用边界

采用原表E/N米坐标，不按模拟位置移动。南部14–23点组瓶颈(340152.5,555117.5)距Rome Street Rly No6记录位置51.15m、距Rome Street Arches位置86.82m。45/47/48共享瓶颈距本次名录中最近Botcherby记录537.65m；107瓶颈距Caldew Rly No2记录144.75m。因此不能把这些瓶颈直接叫作桥梁堵塞点；近邻只用于进一步查图。

17处位置全部保存，名单并非区域完整资产清单。表为2016年，不能证明2005年断面/标高/状态或全部结构位置未变。坐标未注明测量精度，距离保留计算值但不代表厘米级定位。没有把名为Victoria Viaduct的记录等同事件叙述的Victoria Bridge，也没有把Holme Head或近年照片标签当作旧South Vale桥的可靠对应。South Vale和gasworks受损墙精确定位仍未闭合。

## 更改日志、原因与回退

新增anchors93.py：下载一次公开PDF、以预登记17名称解析原表、计算与固定第92轮对象距离；新增publish93.py：独立坐标核验、来源/输入哈希、双份报告与日志。未改水动力函数、地形、边界、糙率、阈值、时窗、生产默认、网页和PR。回退是停止使用本轮诊断，不需恢复任何物理参数。旧建筑TLS下载没有重试。

原因：上一轮区域性桥堵/排水线索缺精确位置。县议会原始结构名录提供独立坐标，可减少按地名猜测，但它不能提供历史水力参数。使用原表位置优于看漏淹图手动挑桥；仍拒绝由观测残差推断应削掉的地形高度。

## 可行性、下一项及否决条件

Q4-R92-G1/H1仍待数据。下一步优先查Rome Street一带2005前后工程图/原EA事件图和06/1473、2008调整工程的公开图纸：核实旧结构、地面通道、墙与通行孔洞的对应；对537m和145m偏离的瓶颈不强行指认为桥。只有取得独立旧高程/通道尺寸或事件时序/流量，才登记单因素水力候选。现有资料不支持强加堵塞、破口或排水源；不运行无输入依据的完整长跑。

目标为减少原固定域漏淹且保留原已吻合证据，原263点、全部区域、未知干区及评分继续保留。若公开图纸仍只描述灾后新防洪工程，则仅保存线索，否决把其当2005地形修补。当前成本为465567字节PDF和轻量坐标计算，未新增重负载。

## 检查及局限

17预选行均成功解析、PDF哈希复核、另用网页原表文本核对5处坐标，全部通过。源PDF、抽取页文本、全17坐标/距离、协议、代码和结果双份归档。物理回归及完整运行跳过（无物理修改）。没有新增现场通过项，也没有确认具体漏淹根因。当前仅原网页服务进程，工作树全部既有修改保留。
'''
name='audit-round93-structure-anchors-20261031.md'
for root in [R,O,D]:(root/name).write_text(report,encoding='utf8')
for n in ['anchors93.py','publish93.py']:shutil.copy2(S/n,R/n)
entry='''## 2026-10-31 第93轮：独立桥涵位置补充

取得县议会FOI2016-1020原PDF，预选17处结构坐标全部解析且5处另行复核。南部瓶颈距Rome Street铁路结构51.15m、拱道86.82m；45/47/48瓶颈距名录Botcherby537.65m，107距Caldew Rly No2为144.75m。只能定位查图线索，不能把瓶颈直接判作桥堵；2016名录非2005历史断面。South Vale/旧受损墙对应仍未闭合，不将同名或灾后工程强配。

下一步查南部Rome Street一带旧工程图/EA事件图及06/1473、2008调整公开图纸，取得独立旧通道/墙高程、尺寸或时序后才跑单因素候选。全量结果和输入不改、覆盖仍88.56%非总体准确率，无新精度收益，无重负载；旧建筑TLS不重试。报告audit-round93-structure-anchors-20261031.md与原PDF/坐标/代码/哈希双份round93及docs/round93-artifacts。生产、网页、PR不变。

'''
for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
 for root in [O,D]:
  p=root/n;t=p.read_text(encoding='utf-8-sig')
  if not t.startswith('## 2026-10-31 第93轮：'):p.write_text(entry+t,encoding='utf8')
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ['artifact-hashes.json','mirror-verification.json']}
(R/'artifact-hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');A.mkdir(exist_ok=True)
for p in R.iterdir():
 if p.is_file() and p.name!='mirror-verification.json':shutil.copy2(p,A/p.name)
verify={n:sha(R/n)==sha(A/n) for n in manifest};assert all(verify.values())
for root in [R,A]:(root/'mirror-verification.json').write_text(json.dumps(verify,indent=2))
print(json.dumps({'dual_saved_files':len(verify),'checks':checks,'physics_changed':False,'new_field_passes':0}))
