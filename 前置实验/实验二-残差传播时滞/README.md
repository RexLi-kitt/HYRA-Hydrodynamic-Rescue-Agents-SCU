# 实验二：残差传播时滞

研究问题：上游残差中的方向和物理时滞是否提供下游未来误差的独立增量信息，能否超越匹配的新近性对照？
主实验保留12家族、36条件事件，6训练/3验证/3内部诊断，固定截面和轮询观测，噪声0/0.02米、延迟0/10秒。

## 运行与预期结果

```sh
python run.py replay --check-reference
python run.py timing --check-reference
```

第一条重训六组岭回归读出，比较自身、共同变化、未对齐、物理时滞、错时滞、反方向对照。
第二条执行11种观测设置、10/20/30秒提前量与8种表征，共264个读出，并重算覆盖率、家族配对区间和门槛。
使用随包冻结的名义物理图，全部特征由原始事件重算；输出写入各自新目录。
`results/latest_reproduction.json` 指向最后一次运行。
预期阶段门槛均未通过；固定条件下20秒物理时滞核相对匹配新近性核的误差高约0.34%，轮询相位平均差接近零。
结果限定当前尺度和表征，不能据此否定所有物理先行关系。

从求解器重新校准网格、生成家族、拟合名义物理图并重训阶段一：

```sh
python run.py generate
```

该命令独立输出新运行，不覆盖随包数据。对新数据继续时滞诊断：

```sh
python scripts/run_lead_timing_diagnostic.py --source-run results/runs/<新生成的lead_stage1目录>
```

## 10月16日后续开发结果

```sh
python run.py development-summary
```

从12个验证家族的完整候选统计重新计算选中模型的配对对比，并逐项核对历史汇总。
这是统计汇总复现，不是重跑1692份水动力场和79个模型。
对应开发证据为：物理时滞短期误差有改善，但相对强原始历史基线的服务增量仅约0.150个/100任务，区间跨零。
24训练/12验证用于选模，不是独立确认，不能与前述探索轮混为同一批证据。

完整开发轮也提供从头执行入口：

```sh
python run.py development-full
```

它顺序生成24训练/12验证家族、审核数据、构建和审核特征、训练、评价323候选、重算配对及信息干预。
此入口沿用原训练实现，需要兼容 NVIDIA CUDA 环境、大量磁盘空间与较长计算时间；本次整理未完整重跑该昂贵流程。
原始大型场和模型未随 Git 包重复分发，全部由该入口生成；已随包提供统计复核所需的家族级结果。
没有打开先导或独立确认，也不自动开展新研究轮次。

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
