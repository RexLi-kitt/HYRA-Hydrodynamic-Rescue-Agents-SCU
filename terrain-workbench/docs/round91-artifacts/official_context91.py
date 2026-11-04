from pathlib import Path
import urllib.request,json,hashlib,shutil
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91';D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round91-artifacts'
items=[('EA-2005-retrospective.html','https://environmentagency.blog.gov.uk/2015/01/08/10-years-on-from-the-cumbrian-and-carlisle-floods-of-2005/'),('Carlisle-recovery-debrief.pdf','https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/78990/carlisle-recovery-phase-debrief-report.pdf'),('FD2321-TR1.pdf','https://assets.publishing.service.gov.uk/media/602bbc768fa8f50383c41f80/Flood_risks_to_people_-_Phase_2_The_flood_risks_to_people_methodology_technical_report.pdf')]
acquired=[]
for name,url in items:
 p=R/name;assert not p.exists(),'Do not duplicate acquisition'
 try:
  with urllib.request.urlopen(url,timeout=30) as r:
   data=r.read(16*1024*1024+1)
  assert len(data)<=16*1024*1024
  assert data[:4]==b'%PDF' if name.endswith('.pdf') else b'<html' in data[:5000].lower()
  p.write_bytes(data);shutil.copy2(p,A/name)
  acquired.append({'name':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'downloaded'})
 except Exception as ex:acquired.append({'name':name,'url':url,'status':'failed_no_retry','error':str(ex)})
locations=[('A','Willow Holme industrial area, Eden north and Caldew east'),('B','Immediately south of Bridge Street, west bank of Caldew'),('C','City centre near Corporation Road'),('D','Warwick Road near Botcherby Bridge'),('E','Residential neighbourhood around Warwick Road')]
checks=[{'id':'future-area-'+letter,'location_text':place,'observation':'report describes inundation in this area','evidence_type':'government research synthesis of incident notes and photographs, not new instrument measurements','source':items[2][1],'locator':'Section6.3.5, printed32/PDF42','precise_geometry':None,'status':'unscored; freeze independently mapped boundaries before comparison','prohibited':'do not invent boundaries from model wet cells or substitute nominal report velocities as measurements'} for letter,place in locations]
checks += [{'id':'future-macro-onset','observation':'report describes gradual inundation from early morning toward a lunchtime peak','source':items[2][1],'locator':'printed33/PDF43','status':'unscored; qualitative time bins and geographic scope still require registration','prohibited':'do not turn this narrative into exact per-station arrival timestamps'}, {'id':'jan9-context','observation':'EA page caption dates a flooded Warwick Road area to9Jan2005; recovery timeline lists flooding8/9Jan','sources':[items[0][1],items[1][1]],'locator':'EA image caption; recovery report timeline printed3/PDF4','status':'additional event-duration context, not a georeferenced observed mask','prohibited':'do not equate caption date with model-specific maximum times or verified road closure duration'}]
result={'acquisition':acquired,'observational_candidates':checks,'not_replacing_round90_protocol':True,'no_model_fields_inspected':True,'new_passes':0,'scope_limits':['Five zones are report-defined contextual units, with no precise vectors acquired here. Do not claim5/5tested.','0.5m/s in FD2321 is an assumption inferred from photos; nominal depths and population counts include analytical assumptions, not complete measured maps.','Jan9 contextual corroboration does not establish exact times of outline measurement.','No per-link accessibility, landing safety or UAV priority validation follows from these sources.']}
(R/'official-event-context.json').write_text(json.dumps(result,indent=2));shutil.copy2(R/'official-event-context.json',A/'official-event-context.json')
shutil.copy2(__file__,R/Path(__file__).name);shutil.copy2(__file__,A/Path(__file__).name)
report='''# 第91轮：搜救相关宏观现场证据补充（2026-10-31）

## 更改日志
仅新增官方来源与待配准定性证据表。模型、完整事件、主阈值和217464格主观测范围不变。没有把新证据写成已通过。

## 原因
[Defra/EA FD2321/TR1](https://assets.publishing.service.gov.uk/media/602bbc768fa8f50383c41f80/Flood_risks_to_people_-_Phase_2_The_flood_risks_to_people_methodology_technical_report.pdf)第6.3.5节叙述五处受淹区和逐渐上涨过程，可为未来宏观核对提供独立定位线索。报告的统一流速是分析假设，不能补作实测流向/流速。五区精确边界尚未获取，不按模拟结果画区。

[EA回顾页](https://environmentagency.blog.gov.uk/2015/01/08/10-years-on-from-the-cumbrian-and-carlisle-floods-of-2005/)的照片说明标注Warwick Road在1月9日受淹，[恢复阶段报告](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/78990/carlisle-recovery-phase-debrief-report.pdf)时间线也列8/9日。这补充事件持续性背景，但不是配准洪水图或道路关闭时刻，不能关闭原最大范围时窗不确定性。

## 可行性
已完成：官方文本证据定位、来源下载尝试和哈希记录。部分完成：五区及涨水阶段的候选对照项。未完成：独立地理配准、事先时段规则、模型对照、路段连通性。最小下一步是依据原地图/地名矢量冻结区域后复用完整结果，不新增求解、不看模型画区；缺精确定位则仍未判断。成本为轻量数据配准，否决条件为无法独立确定范围或来源只提供假设值。

原下载失败建筑条目不重试。新官方来源逐条下载一次，下载状态逐项保存在official-event-context.json，失败不虚称已归档。回归/跳过：没有数值改动，不重跑完整实验，不将文献人数/流速作为救援验证。当前无新增现场通过或准确率。
'''
for root in [O,D]:(root/'audit-round91-official-context-20261031.md').write_text(report,encoding='utf8')
entry='''## 2026-10-31 第91轮补充：宏观现场证据定位

获得官方文本线索：FD2321/TR1第6.3.5节五区受淹/逐渐上涨描述；EA照片说明与恢复报告支持8/9日仍有洪水背景。候选表round91/official-event-context.json尚未配准/评分，不能称5/5通过；报告统一流速为假设，不当实测。完整模拟仍会话24476，原协议不变，结束后执行finish_full91.py并视觉检查地图。所有新来源下载状态/哈希双份保存，原建筑TLS失败不重试。现场新增通过0。

'''
for root in [O,D]:
 for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=root/name;p.write_text(entry+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps({'acquisition':acquired,'candidates_not_passes':len(checks),'new_passes':0}))
