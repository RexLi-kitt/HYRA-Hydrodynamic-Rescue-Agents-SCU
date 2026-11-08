# 原训练与续训运行时

五个模型的训练入口位于各自 `training/train.py`，具体命令见各自 `training/README.md`。原始 15 个训练终点及共享输入通过 [assets.json](assets.json) 对应的 Release 资产分发；首次使用运行：

```powershell
python training_runtime/fetch_assets.py --inputs
python training_runtime/fetch_assets.py --model DQN --seed 88126
```

下载器对传输压缩包和解压出的原始 checkpoint 分别核对字节数、SHA-256；共享输入解压后逐文件核验。原始 checkpoint 保留 optimizer、target、replay、环境和 RNG；不是推理权重，也不应加载不可信来源的 pickle。资产未下载时不会自动访问网络，运行时直接提示缺失。

共享输入包括四张训练场景、固定示范缓存、对应原父模型和来源锁中的文件。`source/` 是便于阅读的源码镜像，实际执行来自下载后逐文件 SHA 验证的 `original/` 副本；只修改 `source/` 不会改变训练行为。`original_manifest.json` 以旧路径哈希索引原始文件，避免将历史绝对路径当成本机必需路径。`run.py` 在新的输出目录中生成路径适配副本：验证原锁中每条来源 SHA，再逐项验证输入文件；适配前后源码哈希写入 `result.json`。旧裁判源码保持原样，运行时使用经校验的四张训练场景。不要把 `source/` 中的历史脚本直接视为新机器上的运行入口。

训练从提供的原父模型起步，或从原始终点继续，不包含更早期搜索模型的从零训练过程。MAPPO 使用 128 步 rollout；Q 系列预热 64 步、此后每 32 步更新。checkpoint 续训保持原状态和 ε 进度。自定义 checkpoint 续训需显式提供 `--resume-sha256`。

默认 Q 恢复保留原始运行时和设备严格比较。跨环境仅在显式指定 `--allow-runtime-change` 时迁移内存中的运行时/设备元数据，输出记录原始与当前环境及非逐位复现标记；原始 checkpoint、输入 SHA 和其他配置仍验证。CUDA RNG 要求相同设备数量。不同 GPU、驱动或系统不保证轨迹或数值逐位一致。

`validation_checkpoint_state.json` 记录 15 个原始 checkpoint 的状态与锁检查；`validation_roundtrip.json` 记录五个代表分支的保存、恢复与不间断训练逐项对照。验证数据均仅属于本机原始环境。SDK 与 QPLEX 组件来源及许可状况请参阅各模型 `NOTICES.md` 及 SDK 自身文件；SDK 原始文件及来源已保留，目录中未发现单独 LICENSE。

## 安装依赖

```powershell
python -m pip install numpy==2.2.6
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

原验证环境是 Python 3.12.9、PyTorch 2.8.0+cu128、NumPy 2.2.6；具体 GPU 驱动需与 CUDA 12.8 兼容。训练原始源码、PyMARL、MAPPO on-policy 和官方 SDK 文件的来源及许可边界见各模型 `NOTICES.md`；SDK 未发现独立 LICENSE。

[代表模型恢复与隔离验证](validation_reproducibility.json) 汇总了五算法父模型、原终点、禁止旧路径访问、跨环境开关及压缩资产核验；它是原验证环境的记录。

