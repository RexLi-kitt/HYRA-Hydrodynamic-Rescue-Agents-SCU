# Sur-QMIX 适配基线小预算贯通报告

两机四动作、合法共享观测、CNN/GRU、QMIX混合器、Sur训练期增强、清洁执行与目标分支、完整回放及检查点恢复已在实际洪水开发过程上贯通。本次不是原论文全部实验复现，也不是物理辅助学习的正式结果。

## 实际预算与检查

| 组 | 完成训练回合 | 联合训练步 | 优化器更新 | 贪心评估回合/步 | actor参数 | mixer参数 |
|---|---:|---:|---:|---:|---:|---:|
| QMIX | 12 | 720 | 9 | 3/180 | 61508 | 32769 |
| Sur-QMIX-adapted | 12 | 720 | 9 | 3/180 | 61508 | 32769 |

两组均按预先记录的6回合保存→独立进程恢复→12回合结束执行，没有额外训练回合和检查点挑选。全部1440个训练联合步的images、poses、集中训练state、动作、奖励和截止标志均重建核对；全部6条评估轨迹的贪心动作、实际帧观测和共享快照与保存结果一致。目标网络在第8次优化后同步，也已通过真实检查点核对。

37项NumPy工程/输入权限测试与8项PyTorch学习测试通过。后者包括有限差分梯度、Sur填充掩码、执行关闭、QMIX单调性、GPU反向传播，以及恢复后的下一次采样、损失、参数更新和随机状态完全一致。

## 资源实测

| 组 | 两段训练/保存耗时合计s | CUDA峰值allocated MiB | CUDA峰值reserved MiB | 进程原生峰值GiB |
|---|---:|---:|---:|---:|
| QMIX | 6.22 | 74.1 | 94.0 | 1.82 |
| Sur-QMIX-adapted | 5.79 | 74.1 | 94.0 | 1.93 |

时间仅覆盖两段训练循环及逐回合保存，不含进程启动、模型初始化、恢复加载、输入读取和评估。原生内存峰值是进程级累计值，可能包含前一方法；CUDA数字为PyTorch分配器，不包括全部驱动占用。本次硬件为RTX5060 Laptop约8GB显存、约16GB内存。

## 学习质量的边界

| 组 | slow当前正确覆盖时间均值 | rapid | async |
|---|---:|---:|---:|
| QMIX | 10.00% | 11.37% | 9.05% |
| Sur-QMIX-adapted | 14.62% | 21.14% | 16.85% |

所有事件均为此前看过的工程fixture，只有一个训练种子和每组9次更新，结果总体很低；不能由此宣布Sur有效、策略已收敛或达到规则基线水平。正式研究需保留R0/R1/R2同环境竞争基线、足够训练预算、至少3种子和新的母地形及来水留出。

## 已明确的适配假设

执行观测为8通道局部7×7图与8通道池化共享图，加6维姿态/队友信息；不是原文9通道。原文八移动动作改为物理前进/左右转/悬停。两机共享可靠观测，CNN16/16、嵌入32、GRU64。Sur按有效批次代理行min-max归一化，固定ones单位方向，幅度停止梯度，目标和执行分支关闭增强。有限600秒任务在TD中不跨截止bootstrap，虽然API将截止标为truncated。奖励是实际物理时间当前正确覆盖积分减实际运动/转向和碰撞成本，未按模型自报风险发奖。

## 证据和复现

- [原文—代码一致性核查](SUR_QMIX_IMPLEMENTATION_AUDIT_20261025.md)
- [实际数据及动作审计](evidence/20261025-sur-adaptation/preflight-audit.json)
- [冻结可执行协议](evidence/20261025-sur-adaptation/plan.json)
- [实际完整摘要](evidence/20261025-sur-adaptation/summary.json)
- [6回合检查点与恢复记录](evidence/20261025-sur-adaptation/planned-resume.json)
- [学习测试日志](evidence/20261025-sur-adaptation/network-training-tests-8.log)

在安装了匹配PyTorch的解释器下，新输出目录可运行：

```sh
python learning.py preflight generated/new-sur-preflight --stop-after 6
python learning.py preflight generated/new-sur-preflight --resume
python -m unittest discover -s learning_tests -v
```

现有本机输出禁止重训或覆盖。`tools/audit_sur_preflight.py`只读审计原始本机目录；源码有新变化时先恢复本轮冻结包到独立副本。执行阶段只需actor，不携带mixer、训练真水情或未来强迫。

下一步先定义并校验物理算子、预测标签和时间尺度，测量辅助分支资源，再冻结独立四组正式协议。当前整体目标仍未完成；不因这次贯通测试而跳过物理辅助世界模型实验。
