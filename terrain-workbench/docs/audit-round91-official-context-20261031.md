# 第91轮：搜救相关宏观现场证据补充（2026-10-31）

## 更改日志
仅新增官方来源与待配准定性证据表。模型、完整事件、主阈值和217464格主观测范围不变。没有把新证据写成已通过。

## 原因
[Defra/EA FD2321/TR1](https://assets.publishing.service.gov.uk/media/602bbc768fa8f50383c41f80/Flood_risks_to_people_-_Phase_2_The_flood_risks_to_people_methodology_technical_report.pdf)第6.3.5节叙述五处受淹区和逐渐上涨过程，可为未来宏观核对提供独立定位线索。报告的统一流速是分析假设，不能补作实测流向/流速。五区精确边界尚未获取，不按模拟结果画区。

[EA回顾页](https://environmentagency.blog.gov.uk/2015/01/08/10-years-on-from-the-cumbrian-and-carlisle-floods-of-2005/)的照片说明标注Warwick Road在1月9日受淹，[恢复阶段报告](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/78990/carlisle-recovery-phase-debrief-report.pdf)时间线也列8/9日。这补充事件持续性背景，但不是配准洪水图或道路关闭时刻，不能关闭原最大范围时窗不确定性。

## 可行性
已完成：官方文本证据定位、来源下载尝试和哈希记录。部分完成：五区及涨水阶段的候选对照项。未完成：独立地理配准、事先时段规则、模型对照、路段连通性。最小下一步是依据原地图/地名矢量冻结区域后复用完整结果，不新增求解、不看模型画区；缺精确定位则仍未判断。成本为轻量数据配准，否决条件为无法独立确定范围或来源只提供假设值。

原下载失败建筑条目不重试。新官方来源逐条下载一次，下载状态逐项保存在official-event-context.json，失败不虚称已归档。回归/跳过：没有数值改动，不重跑完整实验，不将文献人数/流速作为救援验证。当前无新增现场通过或准确率。
