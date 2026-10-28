# P3 第二轮冻结协议说明

本页为已登记机器协议的可读说明，于评分期间整理，不冒充在本文创建时才完成预登记。原始登记为 `generated/p3-nonlinear-round2/plan.json`；模型冻结文件绑定其 SHA256，留出水情在模型冻结后生成。此页不修改任何方法、预算或门槛。

## 唯一问题与改动

原 B3、强历史基线及有效 A2 的方法和预算保持第一轮设定；在全新母地形上检验重复方向，并以严格最新观测 A1 回答变化历史是否必要。没有根据第一轮留出选择新的图尺度、树容量、聚合形式或评价门槛。

A1 对目标与全部64个邻区都只保留最后一次实际观测，移除更早标签、时间槽和推导变化，保留最新标签、年龄、质量及相同静态先验。A1 的信息减少是明确消融；其维数和模型预算一致，但不能说五组拥有完全相同原始历史。B1/B2/B3/A2 仍共享相同原始信息。

## 独立划分与预算

- 拟合：cross-basin、diagonal-trough；选型：split-ridge；留出：radial-notch、warped-valley、oblique-steps。均为第二轮新配方，与第一轮不重叠。
- 每配方覆盖 slow、rapid、async 三机制，各自使用固定独立种子产生母地形。共18事件、54条R0/R1/R2回放。留出9事件属于3家族；路线和训练种子不增加独立事件数。
- 拟合仅在每决策时刻共同抽至多32个合法旧记录，选型和留出评价全部合法旧记录。观察刚刷新的格点和未知格不参与旧记录失准排序。
- 五组各两个叶节点候选，64棵树选型；最终各3种子×128棵树，最大深度18、max_features=0.25、n_jobs=2。共10次选型、15个最终模型，所有最终训练只用拟合集。
- 选型按独立选型家族的事件等权 Precision@10%，并列取首个配置；不将选型集并入最终拟合，不在留出重选检查点。

## A2 与判读门槛

A2 在同一目标、相同精确格点距离及正支持内置换地形因子，支持、未加权度数和分距离向外质量匹配；向内加权质量差异单列披露。有效扰动要求总变差至少0.05且正权重改变至少50%，留出至少8/9事件通过。

B3 相对 B1、B2 的平均 Precision@10% 均至少提高0.5个百分点，至少7/9事件同时正向，每机制至少2/3事件正向，且各机制均值对两基线都为正；相对A2至少+0.5个百分点，相对A1非负。此为合成开发登记门槛，不是普适阈值或独立推广的统计证明。

## 源码与中断

原始模型与全部拟合输入摘要冻结后才计算留出水情；冻结源码包含相应水动力依赖，保存为 frozen-sources.zip。原科学代码没有因归档和审计工具增加而改变。

一次会话中断后的接续保留 interruptions 原始记录。核实原进程退出后复用已保存模型，重新进行未保存的同配置拟合；不增加候选、不挑留出、不另开机制轮。完成拟合耗时不包括中断前未保存尝试的未知额外开销。

第二轮是原P3最后一轮，不自动追加第三轮。用户另行授权的 Sur-QMIX 与物理辅助世界模型为独立扩展，不能改变本轮判读。

## 复现

在独立克隆恢复第二轮冻结源码及依赖，并完成第一轮登记后执行：

```sh
python -m hyra_uav.strong_study prepare generated/p3-nonlinear-round2 --round 2
python -m hyra_uav.strong_study fit generated/p3-nonlinear-round2
python -m hyra_uav.strong_study evaluate generated/p3-nonlinear-round2
python tools/engineering_audit.py --source generated/p3-nonlinear-round2 --output generated/p3-nonlinear-round2/input-audit-final --expected 54
python tools/study_input_audit.py generated/p3-nonlinear-round2 --output generated/p3-nonlinear-round2/input-audit-final/cache-model-audit.json --expected 54 --models
```

已有本地研究使用其原路径和冻结模型续跑；评分进程存活时不要并行执行 evaluate。第一次与第二次 prepare 使用不同冻结版本，应分别恢复对应归档；不能用第二轮源码冒充第一轮重建。
