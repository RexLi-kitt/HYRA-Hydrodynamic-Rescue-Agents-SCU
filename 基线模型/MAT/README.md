# MAT 近距 PPO 1024

本目录发布 HYRA 的选定 MAT 版本，训练种子 99131。模型读取官方裁判的四机公开观测和实际执行的历史动作，每 5 秒为四架无人机依次选择移动方向和救援目标。动作共 9 种移动 × 65 种目标选择，包含无目标。推理保留公开几何特征、循环记忆和合法动作掩码。

| 固定评测组 | 送达 / 总人数 | 有效指定后进入 1 米 / 有效指定 | 裁判冲突 |
|---|---:|---:|---:|
| 主面板 12 场 | 407/768 | 440/561 | 237 |
| 额外面板 12 场 | 236/768 | 254/375 | 233 |
| 留出组 6 场 | 100/384 | 118/185 | 116 |
| 固定单图 | 21/64 | 26/48 | 18 |

主、额外面板使用固定公开开发场景。留出组用于选检查点。上表使用 CPU、采样模式和各场固定动作种子。模型改进、对照及指标口径见 [改进记录](IMPROVEMENTS.md)，权重与逐场复核见 [清单](manifest.json)。

## 运行推理

在本目录使用 Python 3.12。安装所需 PyTorch 构建后，可用下面的命令运行官方 SDK。SDK 和评测面板按本地位置填写。首次运行会从原作者仓库下载三个固定版本的 MAT 文件并校验 SHA-256。也可按 [上游清单](upstream.json) 将已校验的文件放入 `_upstream/`，再运行 `python src/upstream.py --check`。

```powershell
python -m pip install -r requirements.txt
python src/upstream.py

$env:HYRA_SDK = "<官方 SDK 目录>"
$env:HYRA_MODE = "sampled"
$env:HYRA_DEVICE = "cpu"
$env:HYRA_SEED = "623015"
$agent = (Resolve-Path "policy.py").Path + ":make_policy"
python (Join-Path $env:HYRA_SDK "run.py") eval --panel "<评测面板目录>" --agent $agent --out "<新的结果文件.json>" --save-actions
```

`HYRA_SEED` 设置每次创建策略时的动作采样种子。上面的 SDK 命令对每场使用同一个种子。逐场复现历史动作时，请用 [清单](manifest.json) 中该场的 `evaluation_seed` 单独运行对应场景，并保持设备和模式一致。`HYRA_MODE` 也支持 `greedy`。默认权重是 `weights/mat-contact1024-99131.pt`，可通过 `HYRA_CHECKPOINT` 指定同格式权重。

此检查点仅保存推理参数和内存配置。续训所需的优化器、环境和随机状态保存在原训练恢复点中。[来源与依赖](NOTICES.md)
