# HyRA V1 四机洪水搜救

四架无人机在原生 1 公里浅水洪水地图上协同搜索、发现人员并完成连续施救。本仓库发布 V1 优化实验中选出的 **Gaps 完整规划策略**、物理环境、地图生成器与独立测试记录。

## 已有结果

在 50 张新父水场、每图 4 套人员布局上，每场 20 人均匀布点。

| 方法 | 回合数 | 获救 / 总人次 | 救援率 | 平均覆盖率 | 全救回合 |
|---|---:|---:|---:|---:|---:|
| 原 R2，3 个冻结 DQN 种子 | 600 | 9,863 / 12,000 | 82.19% | 83.25% | 17 / 600 |
| Gaps 协同规划 | 200 | 3,748 / 4,000 | **93.70%** | **94.51%** | **59 / 200** |

两种方法共用相同地图和布局。新方法为确定性策略，每布局运行一次；R2 使用三个已有冻结种子。父图内先平均布局及 R2 种子，再对 50 张父图配对重采样 10,000 次，救援率提高 **11.51 个百分点，95% 区间为 [10.47, 12.52]**。新策略救援率区间为 [92.95%, 94.45%]。

本轮目标为严格超过 95%，实测还差 53 人次。252 个未救人次中，221 个未被发现，26 个在发现后没有连续两步合格水情，5 个发现后有合格区间却未获救。后两项使用真值事后分类，不等同于当时一定可赶到。

## 方法

1. 四机共享真实步末采样记录，用 16 步、束宽 48 的路线搜索选择接下来的移动，执行最多 5 步后重新规划。
2. 提高小块漏测区域的搜索权重，扣除队友计划覆盖的格子，减少重复搜索。
3. 使用 24 张训练图的水情统计先验，优先访问可能较早失去施救机会的区域。随包提供固定先验及哈希。
4. 已发现人员根据距离、最近实测水情和已有服务进度分配无人机，执行接近、连续悬停及失败重试。环境最终检查水情与碰撞。

这轮发布的 Gaps 使用规划器和固定训练统计。结果代表完整部署策略的表现，新增 DQN 训练次数为零。

## 环境协议

| 项目 | 设置 |
|---|---|
| 地图 | 100×100 格，每格 10 米；合成河谷地形与原浅水求解器 |
| 飞机和人员 | 4 架机从四角格心出发，20 人固定于非起始格、均匀无放回抽取 |
| 决策 | 每 5 秒同步选八方向移动或悬停；轴向和斜向最大航程均为 50 米 |
| 感知 | 初态和每步末端采样，25 米、180° 前向扇形，边界计入 |
| 施救 | 从外侧进入已知目标的 10 米圈时截停，同机主动连续悬停两步 |
| 水情 | 两个完整区间内水深 ≤0.08 米且速度 ≤0.3 米/秒，按求解器内部步检查 |
| 避碰 | 同步完整轨迹检查 5 米间距，冲突机停留并复查 |
| 结束 | 300 联合步，或已覆盖全部 10,000 格且救齐所有人员 |

控制接口仅暴露共享实测、飞机状态和已发现人员。隐藏坐标及未来水情用于环境与事后评分。合成水场结果用于该生成器下的实验评估。

## 安装与运行

已验证环境为 Python 3.14.2、NumPy 2.4.6、Node.js 24.22.1。需要可从 PATH 调用 `node`。Python 依赖仅为 NumPy。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd uav-workbench
python release.py verify
python release.py smoke
```

Linux/macOS 使用 `source .venv/bin/activate` 激活。首次真实回放会按原种子生成完整水场，计算与磁盘开销明显高于 smoke。

```powershell
# 重新生成水场并运行新策略，默认 189001、布局 0
python release.py run --parents 189001 --layouts 0

# 将归档动作提交给原物理环境，逐项核对终局指标
python release.py replay-archive --parents 189001 --layouts 0

# PowerShell 下复跑完整 200 场
python release.py run --parents (189001..189050) --layouts 0 1 2 3 --workers 4
```

完整运行输出到 `uav-workbench/generated/v1-rescue95-v1/`。`run` 的汇总只包含本次命令指定的场景。已发表的完整结果存于 `results/`，避免与本机新运行结果混淆。

## 目录

```text
uav-workbench/
  release.py                 独立发布入口
  release_io.py              日志读写
  flood_dqn/                 原物理环境、原地图生成器
  v1_rescue95/                冻结 Gaps 规划器与水情训练先验
  coordinated_search_dqn/     几何工具
  rescue_assignment_dqn/      导航工具
terrain-workbench/terrain-lab/ 原浅水求解器及数值函数
results/
  test-summary.csv           200 场指标汇总
  test-episodes.csv          逐场结果
  test-people.csv            4,000 人次事件
  r2-test-episodes.csv       原 R2 的 600 场对照结果
  r2-test-paired-parents.csv 父图配对差
  r2-test-people.csv         人员配对得失
  actions/                  200 场完整提交动作、人员事件和指标
  maps.json                 50 张水场种子、配置和哈希
  seal.json                 原实验封存记录
SOURCE_MANIFEST.json         原始文件与发布文件哈希及路径映射
```

`results/实验报告.md` 中的开发命令属于原研究工作区。独立发布包的执行命令以本 README 为准。开发时试验过的其他候选仅保留汇总记录，发布代码集中于最终策略。原 R2 权重及其全部训练依赖另行管理，本包提供 R2 结果、权重哈希和人员对照表。

## 指标与数据字段

ST 为执行联合步数，ACC 为每步原始意图冲突机对数，RCR 为获救比例。CSV 中 FRT、CRT 使用联合步，乘 5 为秒，分别表示首救和全救时刻；空值表示未发生。覆盖率为实际采样格数 / 10,000。`timeout` 按环境的“全覆盖且全救”结束条件判定，因此救齐 20 人仍可能记超时。

人员表中 `found`、`saved` 也是联合步，`-1` 表示未发生。不同布局的人次按父图聚类分析，三个 R2 种子不作为三批独立水场。

## 来源、许可与发布范围

源码整理自 HyRA 主项目，沿用随主项目提供的 Apache-2.0 许可，见 LICENSE。求解器保留原文中的数值方法参考。本次整理只替换实验日志 IO 的跨实验依赖，并从数据集模块提取原样的种子参数函数。原始与发布哈希见 SOURCE_MANIFEST.json。

该包包含自有源码、固定统计先验及自有合成实验结果。水场数组可以按种子生成，论文全文、外部 Word、下载的第三方仓库及本机环境未打包。

GitHub 可以使用仓库描述：`Four-UAV flood search and rescue with endpoint sensing, coordinated coverage planning and reproducible shallow-water simulations.`

建议 topics 为 `multi-uav`、`search-and-rescue`、`flood-simulation`、`coverage-path-planning`、`shallow-water`、`reproducible-research`。

## 上传 GitHub

在 GitHub 创建空仓库，进入本目录后执行以下命令。将 `<仓库地址>` 替换成自己的 HTTPS 或 SSH 仓库地址。

```powershell
git init
git add .
git commit -m "Release HyRA V1 rescue planning and independent evaluation"
git branch -M main
git remote add origin <仓库地址>
git push -u origin main
```

随包 `.gitignore` 会排除新生成的水场和 Python 缓存。上传目录中的源文件与 results 文件夹即可。
