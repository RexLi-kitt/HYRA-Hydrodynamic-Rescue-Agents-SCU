# QPLEX：hard 8192 联合步

GRU64 个体价值与 QPLEX DMAQ mixer；推理由单机个体网络选择动作。 冻结 GRU 与搜索分支，联合更新 rescue 分支与 mixer；训练使用额外困难示范和成功经验。

本目录有三个固定种子的权重，默认 88126、模式 `epsilon_0.05`。默认分支事先固定，未按评测成绩挑选。逐种子主面板、额外面板、源 checkpoint 与导出权重 SHA-256、逐 tensor 核验见 [manifest.json](manifest.json)。

`weights/` 是独立推理的轻量权重，不含优化器、环境快照或 replay；本模型目录可单独复制进行推理。完整原始训练检查点通过 Release 下载到 `training/checkpoints/`，续训还需共享 `training_runtime/` 和训练输入资产。见 [训练与续训说明](training/README.md)。`training_source/` 为早期历史源码归档，并非可移植训练入口。

验证环境：Python 3.12.9、PyTorch 2.8.0+cu128、NumPy 2.2.6。PyTorch 安装应根据 CUDA 环境选择对应官方版本。另需 HYRA 官方 SDK 和评测面板，均不在本包中。

在本目录运行以下 PowerShell 示例，将 SDK、面板及输出路径换成自己的路径：

```powershell
$env:HYRA_SDK = "<SDK目录>"
$env:HYRA_MODEL_SEED = "88126"
$env:HYRA_MODE = "epsilon_0.05"
$env:HYRA_DEVICE = "cuda:0"
$env:HYRA_SEED = "622004"
$agent = (Join-Path (Get-Location) "policy.py") + ":make_policy"
python (Join-Path $env:HYRA_SDK "run.py") eval --panel "<面板目录>" --agent $agent --out "<新结果.json>" --save-actions
```

`HYRA_MODEL_SEED` 选择目录内权重，`HYRA_MODE` 可选 epsilon_0.05、greedy；`HYRA_CHECKPOINT` 可指定其他同算法导出权重。精确复现已保存随机轨迹须使用相同设备、模式及该场 `evaluation_seed`。`policy.py` 校验模型类别和权重格式。

[第三方来源与许可](NOTICES.md)。公开开发集不是盲测。
