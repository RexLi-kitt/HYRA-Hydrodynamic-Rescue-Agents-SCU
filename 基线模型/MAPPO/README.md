# MAPPO：冻结搜索续训至 32768 联合步

公开观测编码、局部覆盖与联合动作掩码，actor 输出 9×65 联合动作概率；critic 和 ValueNorm 随权重保留。 续训冻结搜索 actor 骨干，仅更新 rescue 分支；critic 继续训练。

本目录有三个固定种子的权重，默认 77126、模式 `sampled`。默认分支事先固定，未按评测成绩挑选。逐种子主面板、额外面板、源 checkpoint 与导出权重 SHA-256、逐 tensor 核验见 [manifest.json](manifest.json)。

这是**权重发布包**，不包含优化器、环境快照或 replay，不能原样断点续训。`training_source/` 是历史源码归档，包含原机器路径；训练数据及完整训练依赖没有打包，不能直接复现实验。独立推理只使用本目录的 `src/` 和 `weights/`。

验证环境：Python 3.12.9、PyTorch 2.8.0+cu128、NumPy 2.2.6。PyTorch 安装应根据 CUDA 环境选择对应官方版本。另需 HYRA 官方 SDK 和评测面板，均不在本包中。

在本目录运行以下 PowerShell 示例，将 SDK、面板及输出路径换成自己的路径：

```powershell
$env:HYRA_SDK = "<SDK目录>"
$env:HYRA_MODEL_SEED = "77126"
$env:HYRA_MODE = "sampled"
$env:HYRA_DEVICE = "cuda:0"
$env:HYRA_SEED = "622004"
$agent = (Join-Path (Get-Location) "policy.py") + ":make_policy"
python (Join-Path $env:HYRA_SDK "run.py") eval --panel "<面板目录>" --agent $agent --out "<新结果.json>" --save-actions
```

`HYRA_MODEL_SEED` 选择目录内权重，`HYRA_MODE` 可选 sampled、greedy；`HYRA_CHECKPOINT` 可指定其他同算法导出权重。精确复现已保存随机轨迹须使用相同设备、模式及该场 `evaluation_seed`。`policy.py` 校验模型类别和权重格式。

[第三方来源与许可](NOTICES.md)。公开开发集不是盲测。
