from pathlib import Path
import json,hashlib,shutil
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round91-artifacts'
rows=json.loads((R/'all263-elevation-compatibility.json').read_text())
d=np.array([x['archived_5m_cell_elevation_m']-x['source_attributes']['dem_5m'] for x in rows])
audit=json.loads((R/'observation-source-compatibility.json').read_text())
audit['independent_archive_dem5m_crosscheck']={'all_points':len(rows),'within_0_0001_m':int((abs(d)<=.0001).sum()),'maximum_absolute_difference_m':float(abs(d).max()),'meaning':'new grid indexing reproduces archived independent per-point dem_5m attributes; this does not validate ground elevations or vertical datum'}
audit['date_correspondence_limit']='All outline records date Jan7-Jan8, no time-of-day of boundary survey; simulation starts Jan7 00:00 as paper and lasts68.25h. Whole-event maximum is conditional comparison, not proven exact observation time span; retain fixed full window and disclose, never choose a better-matching crop.'
(R/'observation-source-compatibility.json').write_text(json.dumps(audit,indent=2))
report='''# 第91轮：现场观测与5米输入兼容性审查（2026-10-31）

## 更改日志
完整事件会话24476仍使用原输入和SGC快速参考后端，本轮未改任何物理函数、求解器参数、生产默认或验收阈值。新增audit_observation_sources91.py以及全5个多边形成因/日期、各自覆盖掩膜、全部263点高程兼容性记录。保留原始字段和坐标，不扣除、不移点、不修地形；仅分析文件可独立移除，不影响模拟。

## 原因及证据
1. 263点中81个实测最高水面高程低于对应5米格高程，差值范围-8.465至4.247m。此差值仅诊断点-格/地形/基准兼容性，不能直接叫实测水深。最突出fid26：调查17.205m、格高25.670000076m，原调查表dem_5m亦25.67m。
2. 全263格采样均与原调查归档dem_5m在0.0001m内一致，最大差约9.16e-7m。因此新读取和行列定位有独立数值交叉支持；尚不能确定是建筑/格代表性、测量位置还是高程基准问题，不能称根因已解决，更不能依观测直接压低格点。
3. 5个原始最大淹没多边形全部保留。域内3个；河流两组分别完全覆盖11637、205827格，合计217464格。另一个drainage/surface-water多边形域内面积585.124m²，但窄于当前完整格规则，没有完整5m格。它的影响保留为边界未知，不能当已通过或无水。当前仅河流来水且无显式雨水/排水网，排水机理验证仍缺数据。
4. 全部边界日期字段为Jan7–Jan8，没有测绘日内时刻；本次模拟依据论文起始Jan7 00:00、时长68.25小时。完整最大范围只作条件性事件对照，不能宣称精确同一观测时窗。完整时窗保持，不能待结果后裁切得分更好的时段。

## 可行性与下一步
本轮只读取地形与独立观测，不读取运行中水深场、不新增水动力试验。主规则仍>0.1m；0.01/0.3m敏感性、所有漏项与未知照旧。区域最大淹没可在完整运行结束后条件性比较；涨退/流向/道路/起降及人员优先级仍未验收。

优先待办：
- 完整结果收尾：假设原输入可形成可审查的事件最大范围；缺口为全场结果。最小工作是一次现有运行完成后的全时窗、质量、有限非负和全部覆盖对照，不重跑。成本为现有运行剩余时间；条件是退出0且内部关口通过，失败则保存并诊断。
- 地形兼容性归因：假设部分漏淹可能与点-格高程表达有关。完成后仅交叉列出81点与全部模拟漏项，不改变总分和对象；随后按作者地形/建筑处理说明或独立地面测量核实。成本先为静态读取。否决条件是位置证据不支持或与失败不关联；不按点调地形，不因差值大直接改NS。
- 时窗与排水证据：缺观测测绘时刻和排水输入，保持未验证。可审查最大值发生时间是否晚于归档日期，作为诊断保留完整结果；缺证据不追认删项通过。

回归：263/263归档地形采样一致；五多边形/217464主观测格及全部阈值未变，源码/输入哈希保留。跳过：未做新的长跑、时间细化、模型改进评分或道路验证。当前新增现场通过0，未声称精度提高。
'''
entry='''## 2026-10-31 第91轮补充：观测—地形兼容性

原完整事件继续运行，勿重启。独立读入发现263调查点中81个最高水面高程低于5米格高程；全263采样与原归档dem_5m一致（最大差9.16e-7m），不能归咎新行列错位，也不能直接由此修地形或删点。域内排水成因多边形585.124m²、无完整5米格，仍保留边界未知；全217464主观测格和阈值不变。原记录边界日期Jan7–8无时刻，完整68.25h最大范围只能作条件性事件对照。完整结果未评分，新增通过0。报告audit-round91-observation-compatibility-20261031.md及全部记录双份保存；完成后交叉列出全部漏项/81点，先核地形表达与时窗，不盲换NS。

'''
for root in [O,D]:
 (root/'audit-round91-observation-compatibility-20261031.md').write_text(report,encoding='utf8')
 for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=root/n;p.write_text(entry+p.read_text(encoding='utf8'),encoding='utf8')
shutil.copy2(__file__,R/Path(__file__).name)
names=['observation-source-compatibility.json','all263-elevation-compatibility.json','individual-observation-masks.npz','audit_observation_sources91.py',Path(__file__).name]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in names:shutil.copy2(R/n,A/n)
manifest={n:sha(R/n) for n in names}
for root in [R,A]:(root/'observation-compatibility-hashes.json').write_text(json.dumps(manifest,indent=2))
assert all(sha(A/n)==h for n,h in manifest.items())
print(json.dumps({'all_original_marks_retained':len(rows),'below_grid_count':audit['survey_elevation_below_archived_grid_cells'],'source_dem_match':audit['independent_archive_dem5m_crosscheck'],'dual_files_verified':len(names),'full_event_scores_added':0}))
