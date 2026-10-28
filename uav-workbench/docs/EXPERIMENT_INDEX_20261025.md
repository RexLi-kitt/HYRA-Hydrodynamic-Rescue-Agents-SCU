# 本地实验与工程运行总索引

规范阶段共 261 条完成回放。这个数字包含不同策略、起点、时间预算和重用过程，不能称作 261 个独立事件。模型与缓存哈希见各阶段输入索引，完整路径、输入摘要、动作数和时刻见机器可读总表。

| 目录 | 用途 | 完成回放数 | 留出评分完整 |
|---|---|---:|---|
| p2-pilot-20261025-final | P2 development pilot | 54 | 是 |
| p2-formal-20261025 | P2 formal scale selection | 81 | 是 |
| p3-20261025 | P3 historical linear screening | 18 | 是 |
| p3-nonlinear-round1 | P3 nonlinear round 1 | 54 | 是 |
| p3-nonlinear-round2 | P3 nonlinear round 2 | 54 | 是 |

## 独立学习贯通实验

原P3收尾后另完成QMIX与Sur-QMIX适配版各12训练回合/720联合步，合计24回合；各3条贪心评估、共360联合评估步。检查点回放和6条评估索引单列，不并入上面的261条原阶段回放。只有一个种子和已见fixture，不是正式物理辅助结果。详见[贯通报告](SUR_PREFLIGHT_20261025.md)。

## 独立物理辅助世界模型首轮

四组×三个种子×2000回合，计划24000回合/1440000联合训练步；当前完成12/12个任务。实际水情索引：拟合9、选型3、确认留出9。
本段为生成索引时的快照，进程存活须另外检查。资源试跑与CLI拟合集验证单列，不并入确认性回放或独立事件数量。
正式训练分片、每次恢复与十次选模检查点保留在generated/sciml-round1/formal；逐分片哈希见训练审计。
[协议与证据入口](SCIML_PROTOCOL_20261025.md)。最终结论需要全部留出评价和独立审计，不由完成计数推断。

## 历史、预检与工程产物

- `imports`：engineering demonstration or test artifacts; not independent research events。
- `p1-final-demo`：engineering demonstration or test artifacts; not independent research events。
- `p1-fixed-demo`：engineering demonstration or test artifacts; not independent research events。
- `p2-pilot-20261025`：superseded pilot, excluded from canonical totals。
- `p2-pilot-20261025-v2`：superseded pilot, excluded from canonical totals。
- `p3-fit-density-preflight-20261025`：fit-only relation preflight, not another independent mechanism round。
- `p3-fit-preflight-20261025`：fit-only relation preflight, not another independent mechanism round。
- `p3-weighted-preflight-20261025`：fit-only relation preflight, not another independent mechanism round。
- `p3-weighted-scale-preflight-20261025`：fit-only relation preflight, not another independent mechanism round。
- `sur-preflight-20261025`：fit-only relation preflight, not another independent mechanism round。
- `test-temp`：engineering demonstration or test artifacts; not independent research events。
- `web`：engineering demonstration or test artifacts; not independent research events。

- [逐条运行、摘要与全部阶段路径](evidence/20261025-experiment-index/index.json)
- [验收表](CONTINUOUS_ACCEPTANCE.md)
- [两轮 P3 结论](P3_FINAL_20261025.md)
- [复现命令](REPRODUCE.md)

完整过程、回放和模型保留在本机 generated 中；该目录不随代码入库。新克隆按复现命令重建，不能把本地缓存路径当作已经发布的数据包。
