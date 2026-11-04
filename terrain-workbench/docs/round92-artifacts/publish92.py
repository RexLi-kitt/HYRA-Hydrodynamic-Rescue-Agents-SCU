from pathlib import Path
import json,hashlib,shutil
S=Path(__file__).parent
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round92'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round92-artifacts'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((R/'summary.json').read_text());v=json.loads((R/'independent-verification.json').read_text());assert all(v['checks'].values())
for n in ['diagnose_misses92.py','path_probe92.py','verify92.py','sources92.py','publish92.py']:shutil.copy2(S/n,R/n)
current=(S/'diagnose_misses92.py').read_text(encoding='utf8')
old=current.replace("handles,legend_labels=axs[1].get_legend_handles_labels()\nfig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,fontsize=9)\nfig.get_layout_engine().set(rect=(0,.06,1,.94))", "axs[1].legend(loc='lower center',bbox_to_anchor=(.5,-.24),fontsize=8)")
registered=R/'registered-diagnose_misses92.py';registered.write_text(old,encoding='utf8',newline='\n')
protocol=json.loads((R/'protocol.json').read_text(encoding='utf8'));assert sha(registered)==protocol['script_sha256']
candidates=[
 {'id':'Q4-R92-G1','priority':1,'hypothesis':'Some missed survey positions are represented as raised building/obstacle cells, or narrow real paths are blocked in5m raster.',
  'evidence':'78/126 missed marks have observed peak below cell terrain; four compatible marks45,47,48,107 require raster crest0.532..0.852m above observed peak.',
  'gap':'Historical bare-earth levels, building footprints/ground-floor levels, underpasses and wall crests at original coordinates. No per-cell building attribution yet.',
  'minimum_discriminating_experiment':'Overlay independently georeferenced historical geometry on all78 incompatible marks and four path bottlenecks. If a false closure is demonstrated, change only that verified representation in an isolated candidate, preserving original survey/mask.',
  'success':'Independent geometry explains closure; full-window paired run reduces pre-frozen missed area and affected observed points with no lost baseline hits and no new internal failures. Separate unknown false-positive risk remains.',
  'reject':'No independent correction height/location, no change in local connection, greater missed area or regression at existing observed wet evidence. Never lower bed from observed peak residual.',
  'cost':'Current GIS diagnostics completed in seconds; full paired candidate~46min after evidence gate, not launched.', 'status':'awaiting independent historical geometry'},
 {'id':'Q4-R92-H1','priority':2,'hypothesis':'Model omits event-specific obstruction/breach/drainage mechanisms that affected local head and flood pathways.',
  'evidence':'EA2005 account reprinted Appendix1 documents bridge/tree obstruction, damaged wall, manhole and culvert flooding. Current original benchmark config has three river inputs, no rain, sewer or breach functions.137 model-wet points:119 low, median stage residual-.2064m.',
  'gap':'Registration to exact missed clusters; event obstruction size/time, breach crest/time, culvert/network dimensions and local inflow hydrographs. Uniform increase to all3 rivers unsupported.',
  'minimum_discriminating_experiment':'First register South Vale/gasworks/Victoria Bridge/Dow Beck using independent historical map control points. Compare with all-component catalogue and the south cluster2031 (1457cells). Only if connected and dimensions documented, test one hydraulic structure/source mechanism with accounting and untouched validation set.',
  'success':'Correct physical mechanism affects pre-registered local missed area and preserves other supported judgments. Boundary/source-integral and nonnegative checks pass.',
  'reject':'Location/geometry/time/flux inferred solely to fit observations; no local effect; new misses; unjustified water addition or unstable mass.',
  'cost':'New source PDF and setupHTML obtained; spatial registration and forcing not yet available. No heavy run.', 'status':'mechanism supported regionally, per-cluster attribution unverified'},
 {'id':'Q4-R92-E1','priority':3,'hypothesis':'Some apparent misses involve validation representation, including buildings inside broad maximum outlines and16 survey marks with peak-minus-gridbed<=.1m.',
  'evidence':'14839miss cells enclosed by modeled maximum-wet set; this is topology only.16/126 failedmarks have0<=Z-bed<=.1m.',
  'gap':'Observation interior holes, surveyed ground elevation and measurement uncertainty; numeric modelbed is not independent observed depth.',
  'minimum_discriminating_experiment':'Audit raw outline semantics and independent survey ground metadata. Keep original scores, all objects and0.1m rule; any future revised protocol must be prospective and report old failures alongside it.',
  'success':'Clarifies which decision can be verified without concealing failures.',
  'reject':'Deleting buildings/points or lowering threshold just to improve coverage.',
  'cost':'Light documentary/GIS task; no solver rerun.', 'status':'diagnostic only; no rescoring allowed'}]
(R/'repair-decision.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding='utf8')
report='''# 第92轮：完整漏淹归因与下一项修复门槛（2026-10-31）

本轮按最新人工要求检查剩余漏淹。完成全量空间诊断和独立路径复核，尚未改动水动力模型或提高实测覆盖。仍为第91轮独立LF8快速加速后端；生产HYRA、网页服务与PR均未改动。

## 全量结果与前后比较

主规则保持：固定完整68.25h事件、5m域、217464个实测最大受淹完整格，模型累计水深>0.1m。前后均192580格达到、24884格未达到，正例覆盖88.5572%；不是总体准确率，外部354279格和边缘9318格仍未知。全部263点及旧不利项保留。并未重新启动求解器。

- 24884漏项分为2101个四连通区域；八连通诊断为1423个，均保存，不能只保留大区域。最大10片占31.59%。
- 21242格（85.36%）距任一模拟最大湿区<=25m；14839格（59.63%）处在模拟最大湿区包围的孔洞中。漏格相对最近模拟湿格的地形高差中位2.31m；仅说明高地/障碍表达值得核查，不能由外观认定建筑或错误DEM。
- 12个六小时保存帧中，仅10个漏格出现“同一帧相邻湿格水面>目标地形+0.1m”。其余不能据此证明全程没有短暂溢流；采样稀疏，该检查只帮助排查大范围有足够水头却不传播的假设。
- 126个未达湿指示调查点拆为78个实测峰位低于格地形、16个峰位减格地形在0..0.1m之间、32个超过0.1m仍未达指示。差值是地形兼容性诊断，不是独立实测水深；16点仍未通过，未调整阈值。
- 对全部32点求四邻域最低地形阻隔路径。4点（45/47/48/107）最低栅格通路最高处仍高于观测峰位0.532..0.852m。其余28点具备低于观测峰位的静态栅格路径，但不保证真实水力连通。
- 32条所选路径的模拟湿端点中31个峰位低于目标调查峰位，24个低于路径最高地形。端点与目标不同位置，不能作为同点误差或把异步峰位当真实水面。全部路径及端点坐标保存。
- 在原137个模型已湿调查点上，119个模拟峰位低于观测，同点残差中位-0.206418m（范围-0.927228..+0.641627m）。此为辅助原因诊断，不能拿这个子集当总体精度或据此统一抬水位。

## 具体优先位置

最大漏片2031：1457格/36425m²，EPSG27700边框E339990..340350、N554820..555185；并非已证实单一成因。调查点14/15/17/18/21/22/23的低阻隔路径共同连接到模拟湿端点(340107.5,555042.5)，路径最高地形17.09m，而该端点模拟最高水位16.7502m；目标实测峰位17.13..17.537m。提示应查局部水头与真实来水路径，不能按残差把水面强行抬高。

45/47/48共享瓶颈格(341272.5,555622.5)，格高程17.10m；观测峰位16.462/16.568/16.547m。107点瓶颈为(339702.5,555937.5)，最低路径最高处15.71m，高于其14.858m观测峰位。需要核对历史小通道、地下连接和障碍表示；最低路径本身不证明真实水流来自所选端点。

## 独立来源与取舍

本轮保存EA2005事件描述的公开转载：Fewtrell2011 Appendix1，PDF49–51页。原事件记载South Vale桥受树木堵塞、gasworks附近墙受损及漫顶、Victoria Bridge下游井口冒水、Dow Beck涵洞贡献与后来Eden回水。它提供机制线索，不给堵塞断面、破口尺寸、局地流量或精确配准，不能直接参数化。

来源：https://www.ecrr.org/Portals/27/Publications/Carlisle2005%20FloodHealthImpactAssessment.pdf

案例说明记载Bristol处理的5m地形清除了树木/桥梁，使用3条1D演算来水和西侧FREE出口。它是同类基准配置说明，不声称与本次Kabir原始文件逐字节相同；本地case.par/bci独立证实无降雨/管网/破口输入。已有网站说明中的ACC对DG2分数属于软件比较，不当现场证据。

来源：https://www.seamlesswave.com/Carlistle_flooding.html

因此优先核查历史地形通道及遗漏事件机制。不能据区域记录就把全部漏项归于排水；也无证据认为换NS能补齐这些输入。建筑ZIP两次TLS失败仍保留且未重试；ScienceDirect403未绕过。新PDF/HTML下载成功并留哈希。

## 更改日志及回退

基线：第91轮原场和protocol，求解exe/DEM/边界/摩阻/湿阈值/生产默认全部未改。新增diagnose_misses92.py（全漏项、连通片、同时间邻格检查），path_probe92.py（32点最低阻隔路径），verify92.py（独立阈值连通性复核），sources92.py（来源与哈希），publish92.py（双份交付）。代码只在round92诊断目录，不替换生产求解函数。

默认诊断参数：四连通；八连通作敏感性；旧0.1m指示；局部路径100m矩形半宽只作初筛，随后全部32点用全域最低阻隔路径消除该半径限制。没有给原验收增加新通过项。回退为不调用round92脚本，round91及之前均保留。

首次独立验证因Windows默认文本编码读取中文路径JSON失败；明确UTF-8后重做验证通过，未改变数据或容差。图例与坐标标题曾重叠，已调整并视觉检查；原登记版本脚本由哈希核对留存registered-diagnose_misses92.py，最终仅图例布局不同。

## 可行性、最小实验与否决条件

详见repair-decision.json的Q4-R92-G1/H1/E1。G1优先核原78点地形及4点瓶颈：独立历史裸地/建筑/墙/通道资料证实错误后，才建立单因素几何候选。H1把事件堵塞/破口/排水位置独立配准到漏片，必须补尺寸和时间/流量；不得给全域或单点凭空加水。E1审观测语义，仅澄清可判断范围，原失败与分母不改。

候选数据闭合后沿用完整时窗成对比较：漏片与调查点改善、原已吻合证据不得新漏、有限非负及源/边界水量通过，才保留；无收益或退化则拒绝。没有已知干区仍无法验误淹，不能据正例收益直接升级为搜救可靠版本。单次完整候选预计约46分钟，当前未启动。已有诊断约数秒，独立路径验证约几十秒；无必要重复原完整长跑。

## 回归、跳过与局限

6个独立检查全部通过：24884漏格逐格一致、263ID全部唯一保留、连通片计数独立复算、全漏格总数闭合、32条路径由阈值洪泛独立证明最低阻隔、所有输入哈希未变。第91轮守恒/非负检查按未变哈希继承，不重新跑求解器。其他水力案例回归跳过（未改物理）；没有新增真实观测验证项。图像已检查。

不能把拓扑孔洞当建筑、不把全域最低地形路径当真实路径、不把不同位置或不同时间峰位当同一水面、不把峰位减模型DEM当独立测深。测量误差、历史建筑内部高程和观测轮廓内部干岛语义尚缺。道路/干区/涨退/方向/无人机优先顺序仍未新增现场通过。

下一单元：从新事件图与历史地图取得独立控制点，先核最大南部漏片与上述4个瓶颈的通道/墙/桥/管网位置，列出需要的最少尺寸和时序；数据不够时明确缺口，不启动猜测型长跑。此轮完成的是原因范围收窄，没有算法精度提升。
'''
name='audit-round92-missed-regions-20261031.md'
(R/name).write_text(report,encoding='utf8');(O/name).write_text(report,encoding='utf8');(D/name).write_text(report,encoding='utf8')
qa={'visual_checked':True,'legend_overlap_fixed':True,'registered_script_sha256':sha(registered),'final_analysis_script_sha256':sha(R/'diagnose_misses92.py'),'numerical_rules_unchanged':True,'verification_recovery':'Explicit UTF8 decoding after original JSON path decode failure; all6 checks pass.','source_downloads_complete':True}
(R/'delivery-qa.json').write_text(json.dumps(qa,indent=2))
entry='''## 2026-10-31 第92轮：全漏淹区域与地形路径诊断完成

按最新人工要求复用第91轮完整结果：24884漏格全保留、2101个四连通片，85.36%距模型湿区<=25m；126未达指示调查点拆为78峰位低于格地形、16峰位减格地形<=.1m、32兼容但漏项，原评分未改。全部32点全域最低阻隔路径独立验证，45/47/48/107四点通路最高处仍高于实测峰位；最大漏片2031已定位。137已湿点119模拟峰位低，提示局部水头/输入核查，不能统一抬水位拟合。

新公开EA2005转载事件记录支持桥堵、墙受损、涵洞和井口冒水机制，尚缺与漏片配准/尺寸/流量；基准仍3河源且无管网/破口。优先依据独立历史几何验证4瓶颈与最大南部片，再登记单因素修复；无证据不加水、削地形、换NS或跑盲目长试验。建筑TLS下载未重试。详见audit-round92-missed-regions-20261031.md、round92/repair-decision.json及全部原始诊断。

6独立检查通过，图已视觉检查；源数据/配置不变、没有新水动力运行，正例覆盖仍88.56%（非总体准确率），无算法提升。代码/来源/全量表/图/哈希双份round92和docs/round92-artifacts，生产/网页/PR未升级。仅网页服务保留。

'''
for f in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
 for root in [O,D]:
  p=root/f;t=p.read_text(encoding='utf-8-sig')
  if not t.startswith('## 2026-10-31 第92轮：'):p.write_text(entry+t,encoding='utf8')
manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name not in ['artifact-hashes.json','mirror-verification.json']}
(R/'artifact-hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
A.mkdir(exist_ok=True)
for p in R.iterdir():
 if p.is_file() and p.name!='mirror-verification.json':shutil.copy2(p,A/p.name)
mirror={n:sha(R/n)==sha(A/n) for n in manifest};assert all(mirror.values())
for root in [R,A]:(root/'mirror-verification.json').write_text(json.dumps(mirror,indent=2))
print(json.dumps({'dual_saved_files':len(mirror),'all_hashes_match':all(mirror.values()),'report':str(O/name),'summary':'No solver/input edits; full diagnosis complete.'}))
