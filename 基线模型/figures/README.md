# 图表说明

四张架构图基于发布源码与配置整理，使用科研绘图流程和内置图像生成工具绘制，经过结构人工核对；生成器未返回可核验的后端型号，故不标注具体模型名。图是结构概览，精确维度、残差作用范围、训练冻结与触发条件见 [完整报告](../实验报告.md)。

| 图片 | 对应内容 |
|---|---|
| [MAPPO](mappo-framework.png) | 公开／局部观测、冻结搜索actor、救援残差、critic与训练分支 |
| [四种Q方法](q-family-framework.png) | 共用FC-GRU及搜索／救援残差；DQN、VDN、QMIX、QPLEX分别采用的训练混合方式 |
| [MAT](mat-framework.png) | 时间记忆、Transformer编码／因果解码、四类动作分数修正、训练与部署边界 |
| [V44](v44-framework.png) | 公开水情路由、两类搜索、学习评分和人员服务；下游交会是条件备选 |
| [对比图PNG](model-comparison.png) / [SVG](model-comparison.svg) | 固定发布版本、主与额外开发面板的送达率；没有把三种子合计与单种子混画 |

数值图由真实 [比较记录](../evidence/comparison.json) 用Matplotlib绘制，没有使用图像生成器绘制数据。安装 `matplotlib` 后，在仓库根目录运行：

```powershell
python 基线模型/figures/build_comparison.py
```

中文字体优先Microsoft YaHei，其次Noto Sans CJK SC、SimHei或Arial Unicode MS。SVG已转字形路径以方便分享。各PNG的SHA-256及生成提示见 [provenance.json](provenance.json)。
