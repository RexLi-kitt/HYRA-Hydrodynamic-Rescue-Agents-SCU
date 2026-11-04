"""Evidence-only comparison of archived point samples, not a DEM replacement."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round91-artifacts'
src=O/'round68/prepared/all-survey-marks.json';p=json.loads(src.read_text());z=np.array([x['source_attributes']['Z'] for x in p])
five=np.array([x['source_attributes']['dem_5m'] for x in p]); rows=[]
for key in ['dem_2m','dem_5m','dem_10m','dem_15m','dem_25m']:
 terrain=np.array([x['source_attributes'][key] for x in p]);delta=z-terrain
 rows.append({'archive_field':key,'count':len(delta),'survey_below_sample':int((delta<0).sum()),'minimum_survey_minus_sample_m':float(delta.min()),'maximum_survey_minus_sample_m':float(delta.max()),'among_81_below_5m_now_not_below':int(((z<five)&(delta>=0)).sum())})
sources=[{'title':'Kabir et al. 2020, A deep convolutional neural network model for rapid prediction of fluvial flood inundation','url':'https://arxiv.org/pdf/2006.11555v2','location':'Section3.1.2 PDFpage15; Figure4 PDFpage17','verified':'5m EA LiDAR further processed by Bristol, including removal of trees and bridges; 2005 flow inputs routed with 1D model, not direct measurement at inlet; time zero Jan7 00:00. Does not identify buildings at the81 suspect point cells.','claim_limit':'CNN targets are LISFLOOD outputs; cannot transfer reported software agreement to real-world accuracy.'}, {'title':'McClean et al. 2020, Implications of Using Global Digital Elevation Models for Flood Risk Analysis in Cities','url':'https://doi.org/10.1029/2020WR028241','location':'Sections2.6.1 and3.2; Figure5 caption; Section3.4','verified':'263 observations are elevations above sea level; paper retains points below DEM in numerical diagnostics. Building/bridge treatment is feature-dependent; this paper does not resolve our specific81 cells.','claim_limit':'No permission to convert Z minus uncertain terrain into exact measured depth, remove high cells, or replace5m with50m.'}, {'title':'Neal et al. 2009 original observation study','url':'https://research-information.bris.ac.uk/en/publications/distributed-whole-city-water-level-measurements-from-the-carlisle/','location':'university bibliographic page','verified':'DOI10.1016/j.jhydrol.2009.01.026 and publication identity only; full original text not obtained in this unit.'}, {'title':'Parkes et al. 2013, Reducing Inconsistencies in Point Observations of Maximum Flood Inundation Level','url':'https://journals.ametsoc.org/view/journals/eint/17/6/2012ei000475.1.xml','verified':'publisher full page returned403; no alternate-entry retry; search snippet not sufficient to assign point-specific uncertainty','claim_limit':'Do not add0.51m uniformly, shift observed elevations, or adopt snippet uncertainty as a new acceptance tolerance.'}]
result={'all263_preserved':len(p)==263,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'sample_comparison':rows,'sources':sources,'not_a_controlled_resolution_experiment':True,'why':'These are archived samples from distinct terrain representations, not a new consistent regridding of one verified raw ground surface. Only point attributes available here.','no_simulation_fields_read':True,'model_changes':[],'acceptance_changes':[],'next_discriminating_work':'After complete result, cross-tab all known misses with elevation flags without exclusion. Obtain matching raw terrain/building-processing provenance before any geometry edit; absent provenance, mark attribution unresolved and retain original input.'}
(R/'terrain-literature-evidence.json').write_text(json.dumps(result,indent=2))
report='''# 第91轮：地形表达证据补充（2026-10-31）

## 更改日志
仅新增terrain_evidence91.py及terrain-literature-evidence.json；原模型、物理参数、阈值、坐标和完整运行依赖不变。所有263点保留。无需求解器回退。

## 原因
此前81个“调查最高水面低于5m格高程”不能自动归因为建筑。核查[原案例论文](https://arxiv.org/pdf/2006.11555v2)确认地形经过树木/桥梁处理，但没有给出81点对应建筑的证据。[独立观测分析论文](https://doi.org/10.1029/2020WR028241)也明确观测为海拔高程并保留低于DEM的点；不能裁掉负差或压低地形来改善结果。原调查归档2/5/10/15/25m采样统计见附表，它们不是受控分辨率试验，不能单凭表格决定改用2m。

## 可行性
候选假设：点与格的地形表达可能解释部分漏淹；缺口是原地面/建筑处理及定位证据。最小区分工作为整场结束后全部漏项与81点交叉列表，再查对应地形原始资料；不按评分移动采样。成本是轻量静态核验。若无位置/处理证据或与漏项不关联，拒绝盲改地形，继续已支持的输入归因。原3个固定深度指示和217464主观测格保持。

回归/跳过：本轮不新跑水动力，不将替代地形采样当模拟结果。资料网页403已记录，未绕过。未拿到原观测研究全文，不把检索摘要中误差数当本项目校正量。现有完整事件仍继续；没有新现场吻合率或生产升级。
'''
report+='\n|归档样本|调查水面低于样本的点数|原81点中在此样本不再低于的点数|\n|---|---:|---:|\n'+''.join(f"|{a['archive_field']}|{a['survey_below_sample']}|{a['among_81_below_5m_now_not_below']}|\n" for a in rows)
for root in [O,D]:(root/'audit-round91-terrain-evidence-20261031.md').write_text(report,encoding='utf8')
entry='''## 2026-10-31 第91轮补充：地形表达文献核验

完整事件仍在原会话24476运行。已核查Kabir2020原案例及McClean2020观测论文，明确81点异常不能直接归因为建筑或通过压低地形修复；归档2/5/10/15/25m全部263点采样对照见round91/terrain-literature-evidence.json。它们不是受控加密试验，不能替换主5m输入。原观测论文全文尚缺，403资料未绕过，不采用摘要偏差作统一校正。下一步仍先完成全事件评分和全部漏项归因，阈值/观测域不变，现场新增通过0。

'''
for root in [O,D]:
 for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  f=root/n;f.write_text(entry+f.read_text(encoding='utf8'),encoding='utf8')
shutil.copy2(__file__,R/Path(__file__).name)
files=['terrain-literature-evidence.json',Path(__file__).name]
manifest={}
for n in files:
 shutil.copy2(R/n,A/n);h=hashlib.sha256((R/n).read_bytes()).hexdigest();assert h==hashlib.sha256((A/n).read_bytes()).hexdigest();manifest[n]=h
for root in [R,A]:(root/'terrain-evidence-hashes.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'sample_comparison':rows,'field_passes_added':0,'dual_files_verified':len(files)}))
