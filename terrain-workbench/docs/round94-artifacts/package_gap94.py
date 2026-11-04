from pathlib import Path
import json,hashlib,shutil,zipfile
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round94';R.mkdir(exist_ok=True)
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round94-artifacts';A.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
urls={
 '2008_april_schedule':'https://cumberland.moderngov.co.uk/Data/CaCC%20Development%20Control%20Committee/200804251000/Agenda/%24Schedule%20of%20Decisions.doc.pdf',
 '2008_march_schedule':'https://cumberland.moderngov.co.uk/Data/CaCC%20Development%20Control%20Committee/200803071000/Agenda/%24Schedule%20of%20Decisions.doc.pdf',
 '2017_planning_history':'https://cumberland.moderngov.co.uk/Data/CaCC%20Development%20Control%20Committee/201702101000/Agenda/Item%2010%20-%2016-1005%20-%20Old%20Brewery%20Halls%20Of%20Residence%2C%20Bridge%20Street%2C%20Carlisle%2C%20CA2%205SR.pdf'}
retrieval={'queries':['Carlisle 06/1473 flood drawings','Carlisle Rome Street flood wall2005','Caldew South Vale2008 flood wall drawing'],
 'results':[{'url':urls['2008_april_schedule'],'search_index_available':True,'full_document_status':'web open403 Forbidden','retried':False,'usable_scope':'Search-indexed planning-history lead only, not full drawings or hydraulic dimensions.'},
 {'url':urls['2008_march_schedule'],'search_index_available':True,'full_document_status':'web open cache miss','retried':False,'usable_scope':'Application08/0112 lead only; no acquired cross-sections.'},
 {'url':urls['2017_planning_history'],'search_index_available':True,'full_document_status':'not downloaded','usable_scope':'Index points to approvals06/1473(2007),07/1389(2008),08/0112(2008); post2005 improvement plans, not proof of2005 dimensions.'}],
 'decision':'Stop broad repeated search for these same documents. Need explicit pre-works/existing2005 survey or historically identified structure drawings; do not use proposed2007-2010 geometry.',
 'source_hash_note':'No raw planning PDFs acquired; hashes below apply to saved retrieval record, not inaccessible original PDFs.',
 'new_field_passes':0,'physical_change':False}
(R/'retrieval-and-gap.json').write_text(json.dumps(retrieval,ensure_ascii=False,indent=2),encoding='utf8')
request={'crs':'EPSG:27700; coordinates metres; elevations require explicit vertical datum matching ODN/mASL.',
 'targets':[
 {'name':'South missed cluster2031','bbox':[339990,554820,340350,555185],'bottleneck':[340152.5,555117.5],'nearby2016_assets':['Rome Street Rly No6 (340113,555150)','Rome Street Arches(340145,555204)'],'needed':['Pre2005 bare-ground/rail embankment and wall cross-sections','Openings and culvert/underpass invert/width/height','Location/time/size of2005 gasworks wall damage if historically linked']},
 {'name':'Marks45/47/48 shared bottleneck','point':[341272.5,555622.5],'raster_crest_m':17.100000381469727,'needed':['Independent historical terrain and passage geometry at this point','Do not identify as Botcherby bridge; selected register bridge is537.65m away']},
 {'name':'Mark107 bottleneck','point':[339702.5,555937.5],'raster_crest_m':15.710000038146973,'needed':['Independent historical terrain and underground/surface links','Do not identify as bridge from proximity alone']}],
 'all78_incompatible_marks':'Included in all263-diagnostic.json via head_compatibility and model wet flag; retain complete263 table.',
 'minimum_acceptable_data':['Original date/event association, horizontal/vertical datum and source location','Existing/pre-works vs proposed/new structure clearly distinguished','Dimensions/ground/crest/invert measurements with uncertainty','For added source: observed or independently justified hydrograph, timing and mass accounting'],
 'reject':['Bed heights chosen from floodmark residuals','Universal water-level or river-flow increase chosen to fit','2016 assets or2007-2010 new defences used as2005 geometry','Excluding buildings/failed points from frozen score'],
 'minimum_followup':'One independently supported local geometry/structure candidate, full fixed-window comparison, all original wet evidence and internal accounting retained. No justified candidate yet.'}
(R/'targeted-data-request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf8')
prompt='''# HYRA漏淹求助包：交接摘要，非聊天逐字稿

目标：请帮助定位2005年Carlisle现有地形/小通道/桥墙几何证据，或指出当前漏淹诊断中可复现的错误。不要为了提高评分改观测、阈值、时间、糙率或凭空增加水源。只做公开资料与技术判断，不代发数据申请。

已经完成：LF8独立参考5m完整68.25小时，原217464实测湿区完整格192580达到0.1m指示/24884未达到，88.56%仅正例覆盖。内部水量/保存场关口通过，非生产HYRA准确率。全部263点，未达126点=78地形兼容性问题+16峰位减模型地形<=0.1m+32仍漏项。32点最低栅格阻隔路径已独立验证，其中45/47/48/107四点最低通路高于实测峰位；这不是已证实的地形错误。

最小复核：读取full-classification.npz的observation_class、maximum_depth及原Carlisle_5m.asc；已知湿格=class1，漏项=depth<=0.1。地形951列611行，左下338500/554700，5m；数组第0行为北侧。调查点采样列floor((E-338500)/5)，行610-floor((N-554700)/5)。all32-minimax-paths.json保存全路径及瓶颈，all263-diagnostic.json保存所有点。path_probe92.py是原计算脚本，原绝对路径需映射到包内文件；不可把包内局部问题当完整水动力复现。full-classification文件只是结果，非独立观测。

具体问题和拒绝条件见targeted-data-request.json。最大南部漏片bbox339990..340350/554820..555185；Rome Street结构是独立2016位置线索，非2005断面。原事件EA记录包含桥堵、受损墙、涵洞/井口冒水，当前理想化3河源模型没有这些过程；尚不能证明对应每个漏片。后期规划文件多为2007-2010防洪改建，不能替换2005几何。

希望输出：1)可公开取得的原始资料具体链接、页码、日期和标高基准；2)是否足以支持单因素修改；3)若不足，最少还缺哪些测量；4)若认为诊断算法有误，给出固定输入下反例。没有证据请明确，不推荐盲目全域NS、扫糙率或移测点。

包仅供用户手动转交；没有自动发送。此前全聊天ZIP保持不变。
'''
(R/'给Dot的提示词与摘要.md').write_text(prompt,encoding='utf8')
files={
 'Carlisle_5m.asc':O/'round91/full_event/Carlisle_5m.asc',
 'full-classification.npz':O/'round91/full-classification.npz',
 'all263-diagnostic.json':O/'round92/all263-diagnostic.json',
 'all32-minimax-paths.json':O/'round92/all32-minimax-paths.json',
 'path_probe92.py':O/'round92/path_probe92.py',
 'independent-verification92.json':O/'round92/independent-verification.json',
 'independent-structure-anchors.json':O/'round93/independent-structure-anchors.json',
 'full-event-protocol.json':O/'round91/protocol.json',
 'targeted-data-request.json':R/'targeted-data-request.json',
 'retrieval-and-gap.json':R/'retrieval-and-gap.json',
 '给Dot的提示词与摘要.md':R/'给Dot的提示词与摘要.md'}
manifest={n:{'original_path':str(p),'sha256':sha(p)} for n,p in files.items()}
(R/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
z=R/'HYRA漏淹定点求助包-20261031.zip'
with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED) as f:
 for n,p in files.items():f.write(p,n)
 f.write(R/'package-manifest.json','package-manifest.json')
with zipfile.ZipFile(z) as f:
 assert f.testzip() is None
 assert all(hashlib.sha256(f.read(n)).hexdigest()==v['sha256'] for n,v in manifest.items())
report='''# 第94轮：历史输入证据门槛与定点求助包（2026-10-31）

接第93轮进行一次有界检索，找到06/1473、07/1389、08/0112等灾后工程线索。搜索索引显示防洪方案在2007/2008获批，不能当2005原状态；未取得可直接用来修补地形的原断面。2008年4月原PDF打开返回403、3月文件cache miss，未改入口重试或绕过。完整状态与URL保存在retrieval-and-gap.json；原PDF未下载，不能虚称有原件哈希。

更改日志：新增package_gap94.py，仅保存检索结果、定点数据需求和可手动交接ZIP。基线仍round91完整结果、round92漏项与路径、round93独立资产坐标。求解函数、地形、边界、默认参数/指示、固定域时窗和所有原始点未改。回退是不使用新增求助包，旧文件完全保留。

原因与取舍：区域性机制记录和2016坐标不足以确定2005墙高、孔洞、受损位置/时间/流量。避免反复以相同检索词产生报告或用灾后方案猜补输入，关闭这批链接的重复轮询。将南部2031漏片、45/47/48共享瓶颈、107瓶颈拆为明确坐标和最小数据项。NS或全域流量调节不能替代此输入缺口。

可行性：当前独立资料只支持进一步查证，不支持物理候选。最小可区分实验需原年代地面/结构断面或可靠局地源时序，再进行单因素完整对照；目标仍减少固定漏淹且不新增原已知湿区漏项，守恒/有限非负通过，误淹证据不足继续保留。无数据时不启动约46分钟的无依据长跑。可手动交接ZIP含全域DEM、完整分类、全部263点、32条路径、原协议和源码路径映射说明；摘要明确非逐字聊天、非完整水动力复现。

前后定性与必要数值：正例覆盖仍192580/217464=88.56%，24884漏格/126点未达指示未改，新现场通过0，求解运行0。无任何拟合改善声明。

检查：ZIP解压CRC及所有包内文件SHA256核验通过，来源/结果/配置已在包清单定位；报告与包双份保存。物理回归跳过（未修改物理），没有外发、推送或删除。所有既有失败保留。当前瓶颈是可用历史输入证据，不是计算仍在运行。

下一步不能再次重复检索同批规划链接。可优先复核已有2005原EA事件图中的明确地标及受损点，只有独立控制点足够才配准；新公开断面出现或用户转来资料/Dot建议时核验年代、坐标、单位和证据独立性。需要转交时使用本包，不假定Dot已有结论。
'''
name='audit-round94-historical-data-gap-20261031.md'
for root in [R,O,D]:(root/name).write_text(report,encoding='utf8')
shutil.copy2(__file__,R/Path(__file__).name)
entry='''## 2026-10-31 第94轮：历史断面缺口与定点交接已保存

一次有界规划检索仅定位2007/2008灾后防洪改建线索，不能作2005几何；4月原PDF403、3月cache miss，未绕过或重复下载。停止同批链接/关键词循环。南部2031及45/47/48、107瓶颈的最小历史地形/孔洞/墙高/事件时序数据需求和本地复核包已双份round94/docs。HYRA漏淹定点求助包-20261031.zip供用户手动转交，未自动发送，摘要非逐字聊天，未假定外部结论。

未改物理输入、不新跑完整模拟，覆盖仍88.56%正例覆盖、24884漏格全部保留，无精度收益。下一项仅在原2005事件图有独立地标时做可审查配准，或等待新原断面/外部建议核验；不能凭当前资料削地形、加水或换NS。报告audit-round94-historical-data-gap-20261031.md。网页及既有工作树保留。

'''
for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
 for root in [O,D]:
  p=root/n;t=p.read_text(encoding='utf-8-sig')
  if not t.startswith('## 2026-10-31 第94轮：'):p.write_text(entry+t,encoding='utf8')
allhash={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name!='artifact-hashes.json'}
(R/'artifact-hashes.json').write_text(json.dumps(allhash,indent=2),encoding='utf8')
for p in R.iterdir():
 if p.is_file():shutil.copy2(p,A/p.name)
assert all(sha(R/n)==sha(A/n) for n in allhash)
print(json.dumps({'files_dual_saved':len(allhash),'zip_bytes':z.stat().st_size,'zip_sha256':sha(z),'new_simulations':0,'not_sent':True}))
