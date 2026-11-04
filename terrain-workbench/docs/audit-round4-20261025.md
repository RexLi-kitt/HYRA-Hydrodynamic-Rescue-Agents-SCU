# 第四轮：外源漏洞修复与真实实验对照

日期：2026-10-25。项目位于桌面“无人机大创/HYRA”，基于本地分支 fix/flood-validation-20261024、提交 5c9caf9 开展。本轮修改尚未提交或推送；上次打包文件不包含本轮修改。

## 已复现和修复的问题

1. **外源落入建筑仍会丢水**：前轮修复了基流和变化水源，但手动外源是独立接口。旧版接收 1 m³，建筑水深随后清零，残差 -1 m³。本轮统一校验手动外源和边界外源，涵盖两种设置顺序。网页已有部分拦截，此项首先证明底层接口缺陷，不证明所有网页场景受影响。
2. **调用方能改写已设置的来水**：设置 1 m³/s 后修改外部列表为 4 m³/s，旧版实际注入约 4 m³；现在保存深拷贝，外部修改不改变模拟。
3. **运行中设置环境会损坏水量账目**：先降雨产生 0.16 m³，再设置相同的零排水图层，旧版把初始水量重设为 0.16 m³，同时保留累计降雨，残差变成 -0.16 m³。现在明确拒绝运行中重配环境，保持已有水量、峰值和账目；需要改变固定环境时新建模拟。原有按时间启停堤坝机制不受此设置入口保护影响。
4. **外源配置缺少统一校验和事务性**：新增唯一编号、事件过程、格索引、权重和为 1、边界位置/宽度、单位入流方向、建筑和墙冲突等校验。失败时不覆盖旧设置；上层地图也在求解器接受后才更新外源列表。重新设置/禁用水源保留已累计的逐源水量。

修改文件：flood_solver.py、river_inlets.py；新增 forcing_validation.py、tests/test_forcing_validation.py。所有路径均相对 terrain-workbench/terrain-lab/。before.json 保存三种旧行为的实际复现数值。

## 有针对性的算法试验：镜像反射墙

原内部固体面仅施加两侧静水压力，封闭外边界则使用镜像状态的 Riemann 通量。本轮在 flood_solver.py 添加可选 wall_flux_mode='reflective'，对内部墙两侧分别求镜像反射通量，同时保持穿墙质量和切向输运为零。其目的在于更一致地表达法向动量撞墙后的反射。

默认仍为 wall_flux_mode='hydrostatic'，未替换全局算法。新模式仅在 Python 求解器接口开放，使用 NumPy 路径；网页参数面板暂未接入，Numba 内核未实现此选项。此模式仍是二维浅水方程，并非三维 NS，也不是工程撞击荷载计算模块。

```python
solver = FloodSolver(bed, depth, dx, boundary='closed',
                     flux_method='rusanov', wall_flux_mode='reflective')
```

新增 tests/test_reflective_wall.py 验证两通量、两方向、多种法向速度下内部墙与封闭外边界的通量一致性，并验证静水保持、碰撞时非负与守恒。强三维撞击、强退水/脱离墙面等情况仍需进一步验证，不能将此项试验推广为所有墙边界的最终解法。

## 真实实验来源与复现口径

来源为 ANUGA 官方仓库的 Yeh–Petroff 溃坝冲击方柱实验验证目录，不是本轮自行生成的“真值”，也不是 Merewether 40 个速度点。官方说明指出快速时空变化可能超出浅水近似的适用范围。

- 固定源版本：7c1b454f8513378877766fe4f9acd11207d9a89a。
- 数据入口：https://github.com/anuga-community/anuga_core/tree/7c1b454f8513378877766fe4f9acd11207d9a89a/validation_tests/experimental_data/dam_break_yeh_petroff
- 水槽 1.60 × 0.61 m；闸门 x=0.4 m；上游水深 0.30 m，下游 0.01 m；方柱范围 x=0.9–1.02 m、y=0.25–0.37 m、高 0.75 m。
- 四周封闭；Manning n=0.03；计算 3 s，每 0.01 s 采样；比较 0.01 m 和 0.005 m 网格。
- 通量比较 Rusanov 与 KNP；默认墙算法另比较建筑 solid 和抬高床面 raised_bed 表示。此次几何恰与网格对齐，不能代表斜建筑或窄通道普遍表现。
- 对照量为柱前后相邻流体格水深积分得到的净静水力代理：0.5 ρ g ∑(h前²-h后²)Δy；后处理使用来源脚本的 ρ=1024 kg/m³、g=9.8 m/s²，求解器仍用 g=9.81。它不包含垂向冲击压力，因此与实测总力存在物理口径局限。
- 原数据存在重复/非单调时刻及负读数；只排序，不删负值、不合并重复点、不筛除误差大的点。按原始观测逐点计算 RMSE/MAE，不能将其当作均匀时间加权积分。
- 原 ANUGA 作图脚本将模拟时刻固定加 0.1 s。本轮同时给出未平移及该固定平移指标，没有搜索最优时间偏移。未平移使用 0–3 s 内 416 个观测；固定平移使用 0.1–3.1 s 内 403 个观测。两列时窗不同，不应直接将跨列降低当作算法改善；算法之间在同一列比较。
- 本轮没有运行 ANUGA 求解器；只读取其公开实验数据和设置，独立运行 HYRA。

## 实际对照结果

|通量|格宽 m|建筑表示|墙算法|未平移 RMSE N|固定平移 RMSE N|峰值 N|
|---|---:|---|---|---:|---:|---:|
|rusanov|0.01|solid|hydrostatic|7.6033|4.7936|42.7886|
|knp|0.01|solid|hydrostatic|7.7627|4.8899|46.6605|
|rusanov|0.005|solid|hydrostatic|7.8196|5.1809|46.9210|
|knp|0.005|solid|hydrostatic|7.8667|5.1547|43.3714|
|rusanov|0.01|raised_bed|hydrostatic|7.6033|4.7936|42.7886|
|knp|0.01|raised_bed|hydrostatic|7.7627|4.8899|46.6605|
|rusanov|0.01|solid|reflective|6.7933|4.2186|30.8494|
|knp|0.01|solid|reflective|6.9235|4.1962|31.7803|
|rusanov|0.005|solid|reflective|7.5008|4.7668|35.4318|
|knp|0.005|solid|reflective|7.5254|4.6978|35.1779|

本次已完成 10 组模拟。原始观测全文件最大值为 33.3000 N。曲线见 force-comparison.png，逐组指标见 comparison.csv，逐时间输出在 yeh-results/、yeh-reflective/、yeh-reflective-fine/。

粗网格 Rusanov 的固定平移 RMSE 从 4.7936 N 降到 4.2186 N（约 12.0%），KNP 从 4.8899 N 降到 4.1962 N（约 14.2%）。这支持继续验证镜像反射墙，尚不足以调整全局默认。默认墙算法下单纯加密或更换 KNP 并不保证实测指标下降，说明不能把“更细网格/更高阶”直接等同于“更符合实验”。

## 回归与交付

最后一次全量测试 201 项：197 通过、4 跳过（Numba 与 RivGraph 可选环境）。新增 6 项外源测试、3 项反射墙测试含多个子情形。日志为 regression-final.log。本轮没有浏览器交互验收；算法试验通过 Python 运行。

benchmarks/run_yeh_petroff.py 为可复现入口。将本交付中的 data/yeh_petroff 目录作为数据目录，在 terrain-workbench/terrain-lab 下运行：

```powershell
python benchmarks/run_yeh_petroff.py --data "实验数据目录" --output "结果目录"
python benchmarks/run_yeh_petroff.py --data "实验数据目录" --output "反射墙结果目录" --wall-flux-mode reflective --solid-only
```

默认会运行两档网格和两通量，并另做粗网格抬高床面对照；--coarse-only 可只跑粗网格两通量。输出指标、采样曲线及源码/数据哈希。计算时间受并行进程影响，不用于性能排名。

## 官方环境署案例准备情况

已从英国环境署下载 Benchmarking_Model_Data.zip，并核实 Test 6A/6B 说明。6A 是实验室尺度溃坝绕建筑；6B 是其尺寸放大 20 倍的版本，不是真实山谷事件。包内 Test6Aoutput.csv 与 Test6Boutput.csv 是测点坐标，不是观测时间序列。本轮未将这些坐标当作实测值，也未完成 Test 6 的标准时长运行；它们仅属后续案例准备。

官方入口：https://www.gov.uk/flood-and-coastal-erosion-risk-management-research-reports/2d-benchmarking-evaluating-the-latest-generation-of-the-hydraulic-models-for-fcrm

Merewether 原始输入、五组配置与评分脚本仍未补齐，未新增该案例速度误差结论。下一步应优先验证反射墙对第二组独立实验及更细网格的表现，并针对明显三维撞击局部引入 NS 对照，而不是直接替换全域浅水模型。

## 最终复核与细网格结论

- rusanov，0.005 m：固定平移 RMSE 5.1809→4.7668 N，下降 7.99%；峰值 46.9210→35.4318 N。
- knp，0.005 m：固定平移 RMSE 5.1547→4.6978 N，下降 8.86%；峰值 43.3714→35.1779 N。

以上共 10 个不同配置，另运行 2 次当前默认模式复核；两次粗网格复核的完整采样序列与初始结果逐项完全相同。初始批次的源码哈希在运行结束时读取，期间加入了可选墙模式；manifest 已明确记录这个采样时点，不能将它当作初始模块加载时的哈希。后续批次均在启动时采样哈希。

图中回流阶段约 1.5–1.8 s 仍有明显偏差，不能仅凭峰值更接近就称整个过程验证通过。观测文件来自公开验证库，不是本轮取得的原始仪器记录；重复时刻的来源和测量不确定性仍需回到原实验文献核查。

两档网格下新墙算法均作了对照，但尚不构成完整网格收敛或独立案例泛化证明。推荐保持默认，使用可选模式开展后续独立验证。
