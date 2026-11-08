# V44 无人机洪水搜救与物资投放

本目录包含默认 V44、提前送达候选、终局派单候选和强纯规划对照。独立算法通过 `runtime/uav-downstream-meet-v44/model.py` 的 `load_model()` 加载。显式传入 `fleet_policy="terminal_dispatch"` 或 `"early_delivery"` 可复现相应候选。

## 场景与结果

每场四机、64 人、2400 秒，首次落水后个人窗口 720 秒，站立者也可投放。视野为前向 180°、20 米，每 5 秒决策一次。同机对同人在 1 米内连续投放 10 秒，每秒检查。V44 将漂移到地图边缘的人员截停在接触点，仍可在个人截止前获投放。主指标为实际送达人次除以全部预设人次。

| 原始测试组 | 总人次 | 强纯规划 | 默认 V44 | 提前送达候选 | 终局派单候选 |
|---|---:|---:|---:|---:|---:|
| 混合开发复核，40 场 | 2560 | 1844 | 1881 | 1886 | 1874 |
| 既有普通图补测，24 场 | 1536 | 1341 | 1396 | 1371 | 1383 |
| 新普通图，8 场 | 512 | 464 | 468 | 450 | 466 |

表中均为最终送达人次。混合组默认 V44 为 73.48%，纯规划为 72.03%。新普通图是单独的补测。综合送达率达到 85% 且领先纯规划 15 个百分点的目标仍待证明。

## 复核一场

已验证的运行平台为 Windows x64，Python 3.12.14、NumPy 2.3.5。以下命令在本目录运行。

```powershell
python -m pip install -r requirements.txt
python replay.py --case 72110001-ordinary-r0
python replay.py --case 88520001-rewet-r0
```

程序按原始观测规则跑满一场，核对四种策略的逐人首次发现、投放完成时间、冲突数和分时送达总数。示例普通图的四组结果依次为 58、62、62、62 人。研究候选的 Windows 运算库附有 C++ 源码和编译脚本。若要自行构建，在安装 Visual C++ 工具链的 Windows 机器上进入 `runtime/uav-downstream-meet-v44` 运行 `build.cmd`。

## 全量输入、训练与评测

`原始数据/` 用 Git LFS 保存三份原始水场和人员轨迹归档。训练与选型、普通图补测及新图、混合复核各一份。在安装 Git LFS 的仓库工作区执行以下命令可直接恢复原始数组，跳过数值重建。恢复脚本先校验压缩包的 SHA-256 和条目清单。

```powershell
git lfs pull --include="基线模型/V44/原始数据/*.zip"
python restore_inputs.py --group all
python replay.py --stage all --prepare-only
```

`runtime/inputs.json` 收录终局派单研究登记的全部 120 场，包括拟合 32 场、选型 16 场、既有普通图补测 24 场、混合复核 40 场、新普通图 8 场。每条记录给出地图种子、名单、障碍位置、随机数、原始水场文件哈希及人员轨迹数组哈希。压力水场随源码提供，普通图由随包的数值求解器重建，人员轨迹由同一模拟程序重建。校验原始水场和人员轨迹后才进入评测。首次全量重建需要额外磁盘空间和 CPU 时间。

```powershell
python replay.py --stage all --prepare-only
python replay.py --stage mixed
python replay.py --stage repeat
python replay.py --stage fresh
```

训练和选型脚本会写入同名结果。要从头训练终局派单模块，请在单独的项目副本里先执行全量输入准备，再从本目录运行以下命令。

```powershell
python runtime/uav-downstream-meet-v44/study_terminal44.py fit select repeat mixed fresh
python runtime/uav-downstream-meet-v44/audit_terminal44.py
```

## 文件位置

- `runtime/uav-downstream-meet-v44/` 包含模型入口、C++ 源码、Windows 运算库、训练与选型脚本、权重、逐人对照和研究报告。最新汇总见 `terminal-summary.json` 及《V44换目标终局训练与新图复核.md》。各阶段必须分开解读。
- `runtime/` 还包含 V44 依赖的搜索、人员运动、水场生成和裁判模块。保留原相对目录名以便导入；早期目录名用于追溯共用模块来源。
- `环境与裁判/` 为 20261106.1 版接入包，含三张合成示例、投放与避碰裁判、输入格式及回归样例。该包独立运行时按 11 月 5 日协议处理出界。复现本表请使用本目录的 `replay.py`，它在同一裁判上使用 V44 碰边停留的人员轨迹。
