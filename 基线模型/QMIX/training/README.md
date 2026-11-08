# QMIX 训练与继续训练

本目录的 `checkpoints.json` 列出三个原始完整训练检查点的 SHA-256。原始 checkpoint 含环境、优化器及随机状态；Q 系列还含目标网络和 replay，hard 含成功经验与辅助 RNG，QMIX/QPLEX 含混合网络；MAPPO 保留 actor、critic、两个优化器和 ValueNorm。它们是受信任的 PyTorch pickle，必须按清单校验后使用。

本包从**所选本轮的原父模型**开始重现该轮训练，或从其原始终点继续；不宣称从零重建更早期搜索训练。本轮 MAPPO 从 stable 8192 开始、冻结搜索 actor 到 32768；Q 系列从各 seed 的 BC 父模型开始到 8192。硬示范分支使用共享缓存示范数据；均使用四个公开训练场景，输入在共享资产内。

在 `基线模型` 目录下（联网下载只需第一次）：

```powershell
python training_runtime/fetch_assets.py --inputs
python training_runtime/fetch_assets.py --model QMIX --seed 88126
# 从固定终点额外训练；输出目录必须是新目录
python QMIX/training/train.py --seed 88126 --steps 64 --output "<新的输出目录>"
# 从本轮的原始父模型起步
python QMIX/training/train.py --seed 88126 --start parent --steps 64 --output "<另一个新目录>"
# 从上一步输出继续，必须显式给出 result.json 中的 checkpoint_sha256
python QMIX/training/train.py --seed 88126 --resume "<先前输出目录>/checkpoint.pt" --resume-sha256 "<完整SHA256>" --steps 64 --output "<再一个新目录>"
```

上面 128/64 步是快速功能核验。要从所选原父模型重现本轮完整预算，请运行：

```powershell
python QMIX/training/train.py --seed 88126 --start parent --steps 8192 --output "<新的完整训练输出目录>"
```

默认 GPU `cuda:0`。MAPPO 的步数须为 128 的倍数；Q 系列最早在第 64 步开始每 32 步一次更新，终点续训保持原有 8192 步 ε 调度，ε=0.05。一次输出包括新的完整 `checkpoint.pt`、`trajectory.json`、`result.json` 和运行时副本，绝不覆盖原始检查点。使用自定义输入目录可加 `--inputs <已解压original目录>`。

`training_runtime/original_manifest.json` 将旧路径哈希映射到包内原始文件并记录 SHA-256；运行前逐条核验原字节、父模型、示范及场景，之后仅修改新输出目录中的路径适配副本。运行结果记录修改前后源码 SHA；不跳过旧锁。每种算法应单独启动进程，避免旧训练模块同名冲突。历史源码位于共享 `training_runtime/source/`；通过上面的可移植入口运行。

准确复现已在原记录的 Windows Python 3.12.9、PyTorch 2.8.0+cu128、NumPy 2.2.6、CUDA 12.8/cuDNN 91002 和 `cuda:0` 环境验证；Q 原检查点还严格核验软件版本及设备配置。跨平台、不同 GPU/驱动的数值位级一致性没有保证；不能静默关闭原配置或来源检查。

若仅用于跨环境继续训练，可显式增加 --allow-runtime-change：在内存副本中迁移运行时及设备元数据，保留网络、优化器、目标网络、replay、环境和 RNG，并在输出记录原/当前环境与不保证逐位复现。原 checkpoint 与全部来源 SHA 仍校验。为完整恢复 CUDA RNG，运行时必须有与保存时相同数量的 CUDA 设备；不满足时会拒绝继续。未加该选项时保留原有严格运行时/设备检查。

