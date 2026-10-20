# 实验三：多无人机闭环搜救

从桌面方案开始的新实验，独立于实验一、二。当前处于 **正式测试后诊断阶段**：3024回合主测试及全量独立审计、配对统计复核均已完成。2500格零延迟主实验中，P与冻结基线H完成率均为99.9074%，增量0个百分点、95%区间[0,0]，未达到增加3个百分点的工程目标。整体实验尚未完成，改派干预、数值方向复跑、报告与独立复现包仍在后续流程中；实时计数见`RUNNING.json`。

实际执行与未完成项见 [启动与预检记录](docs/results/20261018_启动与预检记录.md)。原始方案按字节保存于 [方案快照](docs/protocols/实验三原始方案.md)，开发选择与未定事项记录在 `configs/experiment3.v1.json`。

完整实验执行已经登记于 `manifests/full_execution_plan.v1.json`，当前任务与未完成要求见 `RUNNING.json`、`COMPLETION.json`。正式条件保留原方案第0秒启动、每回合1200秒；12种水情在24训练／12验证／36测试家族内均衡预分配，三个尺度继承家族条件，延迟扩展为H／E／P，正式测试总数3024。450／600秒预运行保留为开发诊断。主测试及统计复核已完成，正在执行后续诊断流程。证据见 [训练完成与检查点审计](docs/results/20261019_训练完成与检查点审计.md)、[验证回放与敏感性核验](docs/results/20261019_验证回放与敏感性核验.md)、[前置检查完成与正式测试登记](docs/results/20261019_前置检查完成与正式测试登记.md) 和 [主测试结果与独立审计](docs/results/20261019_主测试结果与独立审计.md)。

五组已共用一个37,070参数的可变网格骨干，接收相同的60秒逐机原始观测、名义预测历史、位置、年龄和缺失标记。H对共享区域库保留不受物理候选限制的注意力；E／P改变稀疏联系组织，P错联系采用预定置换，时间消融保留时滞与过期条件。联系拟合仅使用训练家族，并在其中留出6家族验证；12个验证家族用于检查点和B*选择。代码实现不等于模型已训练或实验已完成。

后续已完成 [216场景／576回合设计集检查](docs/results/20261018_设计集机会空间检查.md)、[28回合启动阶段候选诊断](docs/results/20261018_任务启动阶段候选诊断.md) 和 [450／600秒完整候选核验](docs/results/20261018_450与600秒候选完整核验.md)。原始设计集全部完成目标；候选核验共432阶段场景、2304回合，当时主规模未出现稳定的理想预测增量。该轮定位到预测未参与探索、仍需多访问规划，后续改动见下文；正式阶段尚未冻结。

预测前沿探索与有限候选路线的工程回放，在全部1152组保留场景新增5760回合，包括探索／路线改动对照、合用的名义／真值诊断，以及同候选、同路线规划的去前沿水情权重对照。该历史回放中主尺度延迟场景整体完成率从83.98%变为85.02%，其中名义前沿水情权重本身仅贡献0.14个百分点、仅1/6家族改善，小尺度存在负结果。完整结果与归因限制见 [预测探索与多访问回放](docs/results/20261018_预测探索与多访问规划回放.md)；这些结果不作为当前训练模型的正式效果结论。

## 运行

在本目录执行：

```powershell
python run.py audit
python run.py test
python run.py preflight --side 70
python run.py numerical --side 50
python run.py report
```

完整执行的当前阶段入口如下，数值加密与记录采集均支持精确配置下恢复；训练会等待全部训练／验证记录完成：

```powershell
python scripts/refine_design.py --workers 6
python scripts/collect_common_records.py --workers 2
python scripts/check_planner_sensitivity.py --workers 2
python scripts/train_models.py --data-run results/runs/<共同记录运行> --wait-for-data
```

模型训练需要PyTorch（`pip install -e .[learning]`）。实际验证环境为本机PyTorch 2.14.0+cu130及RTX 5060 Laptop GPU，环境快照保存在运行目录中；其它版本不在当前验证范围内。

`audit` 登记全新6设计／24训练／12验证／36测试家族并核对回合数，仅写计划和划分清单。`test` 检查时钟、因果可见性、完整服务、实际ANUGA调用和窗口评分。`preflight` 独立生成真值与名义场并运行一个名义预测贪心基准；默认1200秒、0.9米涨幅／600秒历时／0.1米扰动。可用 `--side 32` 或 `--side 50` 改变规模，`--delay 10` 检查延迟。只能使用登记的设计家族种子。

完整设计集机会空间检查使用以下入口，默认4个进程，可选1—6个。6家族×12水情×3尺度为216个场景；两套家族内固定布局／噪声配置，加上2500格10秒延迟，共576个未训练名义基准回合：

```powershell
python run.py design-check --workers 6
python run.py design-check --workers 6 --resume results/runs/<设计检查运行>
python scripts/report_design.py --source-run results/runs/<设计检查运行>
python scripts/audit_design.py --source-run results/runs/<设计检查运行>
```

每个job只拥有一个家族／尺度目录，跨真值扰动共享独立名义场。每个场景与回合独立落盘；恢复运行会核对执行代码与固定场景清单，保留中断的原始文件。查看 `design_progress.json` 与各job的 `progress.json` 可获得进度，`design_summary.json` 记录家族平均与回合明细。全部设计场景都会保留；该检查不会训练模型或访问独立测试真值。

另有未冻结的任务启动阶段诊断：`python scripts/probe_mission_phase.py`。它在一个设计家族、2500格、同一水情下，预先固定0—900秒、间隔150秒的水动力预运行阶段；每次任务仍完整运行1200秒，真值和名义场独立积分到2100秒。该开发变体不会改写原始方案，也不选择正式条件。

450／600秒候选的完整设计集核验使用独立入口：

```powershell
python scripts/validate_phase_candidates.py --workers 6
python scripts/validate_phase_candidates.py --workers 6 --resume results/runs/<候选核验运行>
python scripts/audit_phase_candidates.py --source-run results/runs/<候选核验运行>
python scripts/report_phase_candidates.py --source-run results/runs/<候选核验运行>
```

两档候选完整交叉6设计家族、12水情、3尺度、两套布局及主规模延迟，共432阶段场景、1152对运行。每对包含未训练名义基准与仅查询已发现目标、保持60秒预测有效期的真值预测诊断。后者使用特权未来水情，明确标为非训练数据、非H／E／P、非最优联合上界；正负任务增量全部保留。真值和独立名义场实际积分至1800秒，每个任务仍运行1200秒。恢复执行同时核对执行代码和数值环境。

预测探索与多访问规划使用这些已保存场进行完整配对回放，再执行同候选去权重对照：

```powershell
python scripts/replay_planner.py --workers 6
python scripts/control_frontier_weights.py --workers 6
python scripts/audit_planner_replay.py --source results/runs/<规划回放运行> --workers 4
python scripts/audit_planner_replay.py --source results/runs/<同候选对照运行> --workers 4
python scripts/report_planner_replay.py
```

新控制器仅获得公共网格、已到达覆盖和已发现目标；前沿查询位置由几何候选产生，不读取隐藏目标图。真值诊断在同一接口内使用特权未来水情，仍非H／E／P。路线由有限宽度搜索产生，二元优化选择不冲突的路线，只执行首段并重规划；不是全局最优保证。所有回合保存原始观测、查询角色、发布版本、路线和服务事件，逐回合重算完整服务及因果约束。

`numerical` 在同一设计家族和固定任务中心比较10米／5米三角水动力网格，以及独立ANUGA求解的1秒／2秒输出；保存水深差和完整12秒服务窗口标签差异。结果是初步诊断，未自动宣布收敛或冻结参数。

已有场可直接重跑闭环，路径指向对应运行目录：

```powershell
python run.py episode --truth results/runs/<运行>/truth/field.npz --nominal results/runs/<运行>/nominal/field.npz --delay 10
```

每次运行创建独立 `results/runs/<时间_随机ID_阶段>/`，保存源码快照及SHA-256、环境、配置、原始统计和状态。`observations.npz` 无损保存观测值、格坐标、无人机ID、采样和到达时刻；轨迹与服务事件位于 `episode.json`，存储实测位于 `episode_storage.json`。错误保留为 `error.txt`，修正后创建新运行，不覆盖旧结果。`results/progress.json` 汇总实际完成的运行。

## 环境

本机可用 Python 3.14.2、NumPy 2.4.6、SciPy 1.18.1、ANUGA 4.0.1、pytest 9.1.1。推荐在兼容Python的独立虚拟环境安装 `requirements.txt`；具体运行环境以各次 `environment.json` 为准。ANUGA预检使用CPU，不需要CUDA。

当前Windows Python的 `sys.prefix` 为相对盘符 `D:`，导致遗漏自身 `Lib/site-packages`。`run.py` 仅在本次进程找不到NumPy时，尝试从解释器所在发行目录补充路径；不修改系统安装或全局环境。普通环境不需要此兼容逻辑。

## 实现边界

- 任务格固定10米，水动力三角网格独立设置；有限涨水在120秒开始，结束后保持。名义场低估涨幅25%、无局地扰动，独立初始化，不复制真值状态。
- 飞行按米制欧氏距离连续计时；每5秒在真实途中位置采样5×5窗口。共享信息按到达时间可见，发现目标与服务失败触发滚动规划。
- 完整12秒服务成功才扣一份载荷，失败耗完整服务时间。评分检查分段线性深度曲线的端点和全部内部输出节点。输出间未解析的峰值需继续做数值核验。
- `WindowEstimate` 绑定查询格、预测版本、发布时间、曲线、有效期、完整服务的合法开始区间和证据标记；超出发布有效期返回未知。开发名义预测适配器的有效期暂定60秒。
- 覆盖与探索使用线性存储和局部足迹计算，没有4900×4900稠密矩阵。保留原单步规划器作对照；新增预测探索与有限候选多访问滚动优化，正式参数尚未冻结，候选不穷尽，路线末端覆盖是代理量。
- 五组容量匹配骨干、逐机完整原始输入和因果预测接口已实现，训练、固定回放、闭环验证、3024回合主测试及P对B*区间的审计已通过；改派服务收益归因、数值方向诊断与完整交付仍待完成。旧工程报告中的名义适配器不学习观测，不能与训练模型结果混用。

## 冻结前的协议差异

方案的3024回合等于：36测试家族×2布局／噪声配置×3训练种子×（主测试11个方法尺度组合＋延迟3方法）。完整执行登记`manifests/full_execution_plan.v1.json`已按这一数量预先平衡分配水情：每个家族一种条件，训练/验证/测试的12种条件各覆盖2/1/3个家族；同一家族跨尺度继承划分和条件。划分与条件分配不参考P的收益。旧`experiment3.v1.json`保留为初始草案，正式测试只接受前置检查通过后生成的`configs/frozen_experiment3.v1.json`。

先完成全部设计家族的数值、输入合法性、窗口分布和可挽回空间检查，再开发模型、进行固定记录回放与验证选模。保持所有失败、空机会和负收益场景；不能按HyRA的有利表现筛选。

完整衔接流程由`scripts/continue_experiment.py`执行：等待共同数据、15模型训练、设计数值加密和规划敏感性计算完成，独立复核后依次运行固定记录验证、闭环验证、B*选择与配置冻结、单次3024回合测试、改派干预、数值方向复跑、记录审计及报告打包。登记后的独立测试只能从原目录精确续跑。`COMPLETION.json`只有全部交付证据与复现包完整性通过时才标记完成；正增益不是完成条件，收益数值方向改变会保留未完成状态并要求继续核查。

本次执行将固定回放和闭环验证与其余设计检查并行推进，输入均为已经审计的共同记录和模型；冻结与正式测试仍要求全部前置证据通过。阶段互斥执行并复用同一份结果，未新增测试访问或更改场景、模型及规划预算。

移动独立复现包后，各入口会读取包内`BUNDLE_ORIGIN.json`，将原始记录中的绝对路径在读取时映射到新目录，原JSON字节和散列保持不变。先运行`python scripts/verify_bundle.py --audit-primary`核验冻结证据与全部正式服务评分。执行环境与依赖版本分别保存在各运行目录的`environment.json`及项目`requirements.txt`。
