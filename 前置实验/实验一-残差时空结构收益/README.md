# 实验一：残差时空结构收益

研究问题：相同原始历史和训练预算下，显式残差及空间结构能否改善预测诊断和模拟搜救服务？
主实验采用修正后的 V2 五组：B0_raw、B_local、B1_distance、B2_structure、B2_uniform。
已修复服务失败后移动权恢复，保留旧轮更正说明 `docs/FIRST_RUN_CORRECTION.md`。
旧轮错误完成率不再作为机制证据。

## 运行与预期结果

```sh
python run.py replay --check-reference
```

在随包 6训练/3验证/3内部诊断家族上重新拟合五组线性诊断读出，重建训练图并与历史逐项核对。
这一步是信息诊断，不计算策略服务完成率。结果见 `results/latest_reproduction.json` 指向的新目录。

重新训练和评价五组策略：

```sh
python run.py train --device cpu --smoke-only
python run.py train --device cpu
# 有兼容 NVIDIA 环境时可改用 --device cuda
```

`--smoke-only` 只训练两个静态组件各8回合，验证训练链路；不能当作完整实验。
完整模式另训练五组×三个种子×120回合，并进行共享场景评价，可能耗时较长。
原轮用 CUDA，默认 CPU 是便于跨机器运行的适配，不能承诺得到逐位相同的策略或排名。
历史参考完成率依次为32.2%、40.0%、41.1%、36.7%、35.6%；主要配对区间包含零，未证明稳定任务收益。
历史值和统计见 `reference/closed_loop/summary.json`，它们不是本次重新训练的结果。

从头生成相同种子的数据后再重放：

```sh
python run.py generate
python run.py replay
```

生成器写入新 results 目录并更新 `data/manifests/pilot.json`；随包 `data/events` 保留。
数据重新生成后不再要求原始字节校验相同；`verify` 会提示 manifest 已改变。

本目录还保留收益来源、本地残差消融、服务遗憾等原研究辅助源码与协议；它们不是 `replay` 自动执行的步骤。
本实验使用已经查看过的家族，是诊断重评，不是新盲测。

## 环境与安装

推荐 Python 3.14；本机验证的完整版本见 `VERIFIED_ENVIRONMENT.json`。代码最低声明为 Python 3.10，但较旧 Python 是否有这些精确依赖的 wheel 需要自行确认。
在本文件所在目录运行（Windows PowerShell、Linux/macOS 均可；入口也支持绝对路径调用）：

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py verify
```

固定数据重放不需要 ANUGA、CUDA、外部 warehouse 或联网下载源码。安装依赖需要网络。
`requirements.txt` 固定直接依赖；运行时还会保存完整 `pip freeze`。传递依赖以实际安装环境为准。
如需从头生成水动力数据或运行包含求解器的测试，再执行：

```sh
python -m pip install -r requirements-solver.txt
python run.py test
```

测试覆盖因果时间、家族隔离、环境服务规则和相应机制；不是训练效果显著性的检验。
所有入口使用当前 Python 解释器，自动切到包目录，BLAS 单线程。数据划分和随机种子沿用原实验。
CPU 与 GPU、不同 BLAS/操作系统之间可能有浮点差异，训练也不保证逐位相同。
`--check-reference` 按 1e-9 绝对/相对容差比较固定数据重算的完整数值结果；超差即报错。

## 文件与可复现边界

- `data/`：随包提供的真实 ANUGA 仿真数据及原始求解器场，不是现场生成的替代样本。
- `src/`、`scripts/`：原实验实现及其内部辅助脚本；日常使用统一入口 `run.py`。
- `configs/`、`docs/`：冻结参数、原协议、更正说明及历史结果；设计草案不等于已执行实验。
- `reference/`：历史参考结果，用于数值比对；历史工作区路径统一标记为 `historical_workspace/`，仅作来源信息。
- `results/`：新运行、日志、源码指纹、环境与产物，Git 默认忽略。每次运行建立新目录。
- `third_party/`：运行用的固定源码与原许可证，见 `THIRD_PARTY.md`。
- `SHA256SUMS.json`：发布包源码、数据和参考结果的逐文件校验；改动代码后 `verify` 失败是预期行为。

两个文件夹各自独立，可以放在同一个仓库，也可以分别上传。不要把原工作区的大型 vendor、warehouse、tmp、历史交付压缩包一起上传。
包内历史报告仅供证据追溯，日常运行方式以本 README 为准。仓库尚未选定自有代码许可证。

核验说明：历史图元数据称验证“事件”，现实现称验证“家族”；固定数据上一家族对应一个事件。核验比较全部图数值、家族ID和读出指标，仅不比较这一说明字符串，历史文件原文保留。
