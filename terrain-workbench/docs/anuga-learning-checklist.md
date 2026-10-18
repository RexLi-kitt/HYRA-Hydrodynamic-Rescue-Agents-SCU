# ANUGA 水动力代码学习与方格迁移清单

实施记录：本文描述的是升级前的项目现状与学习计划。随后完成的方格数值升级及实测结果见[水动力对比实验](flood-numerics-comparison.md)；具体配置见[数值方法说明](../terrain-lab/MULTILAYER.md#数值方法与实验配置)。

研读日期：2026-10-15。目标：保持 HYRA 原生规则方格、逐格状态和公共边通量，学习或提取 ANUGA 中有实际证据支持的水动力实现。

源码基线：[anuga-community/anuga_core，提交 e2e7bf67b5ed](https://github.com/anuga-community/anuga_core/commit/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b)，该提交日期为 2026-09-21。本次取得固定版本的 61 份源码、测试和文档，重点研读下列函数及其调用关系；没有逐行审计全部仓库。

本清单是源码研读与迁移评估，没有安装或运行完整 ANUGA，没有测量两者的精度或性能，也没有修改现有求解器。文中的验收项是后续工作要求，不是已经获得的结果。

## 1. 研读后的判断

ANUGA 最值得借鉴的是一条完整的计算链：单元平均状态 → 带限制的边状态重构 → 地形与干湿修正 → 公共边通量 → 压力/地形源项 → 时间推进与摩阻 → 外部水源、结构交换 → 水量和误差诊断。单独取出其中一个公式，不能继承整条链的数值性质。

方格约束仍然允许局部代码复用。最有希望提取的是不依赖三角邻接表的单边通量函数、局部水工结构公式，以及以水面/高程/面积数组为输入的水量分配函数。梯度重构、三边遍历、三角形半径、网格并行和 Domain 数据结构需要重新设计。当前最值得优先投入的是验证基线、通量与地形压力的配套关系、二阶重构，以及有限高度堤防和涵洞。

优先级：P0 为后续修改必须具备的基础；P1 为近期重点；P2 为场景需要时开展；P3 为计算规模或功能需求明确后开展。难度是本项目迁移的相对判断，不是工期承诺。

复用类型：A＝原版局部函数提取候选，尚未验证独立编译或调用；B＝借鉴算法并按方格改写；C＝借鉴测试、配置或工程设计。一个条目可以同时属于多类。

## 2. 我们已有的基础

| 本地实现 | 当前已经具备 | 本次应补什么 |
| --- | --- | --- |
| `terrain-lab/flood_solver.py::_flux` | 一阶 Rusanov、hydrostatic reconstruction、地形压力修正 | 单边通量对照、二阶边重构及配套验证 |
| `FloodSolver.step` | 水量更新、半隐式 Manning、干格清动量、微小负水深修正计量 | 多阶段一致性、量化精度、阈值敏感性 |
| `boundary_forcing` | 河口宽度加权的指定流量和法向平流动量 | 物理边界闭合与回水条件的独立验证 |
| `_scheduled_rate`、`advance` | 分段常量过程、事件切步、独立来源计量 | 持续保留事件语义；连续曲线另设插值选项 |
| `set_environment` | 不透水建筑、公共边墙、逐格排水容量 | 有限堤顶、漫顶和内部水工连接 |
| `river_baseflow.spin_up` | 基流预热、水深/速度收敛检查、预热收支 | 真实或标准河道的稳态基线 |
| `tests/test_flood.py` 等 | 静水、干床溃坝行为、转置对称、收支和入口测试 | 解析误差、实验数据、网格收敛和真实观测 |
| `map_sessions.py`、`multilayer.py` | 真实计算帧、最大水深、回放、ZIP 和数据字典 | 算法完整配置、最大速度/动量、可恢复计算状态 |

本地源码入口：[求解器](../terrain-lab/flood_solver.py)、[物理图层](../terrain-lab/multilayer.py)、[场景运行](../terrain-lab/map_sessions.py)、[现有物理测试](../terrain-lab/tests/test_flood.py)。这里的现状判断来自代码阅读，未在本次重新执行测试。

## 3. 核心数值方法：01—10

### 01. 状态变量、单位与位置语义

**P0 · B/C · 难度低。** ANUGA 以 `stage, xmomentum, ymomentum` 表达状态，水深来自水面高程与床面高程之差；我们主要保存 `h,mx,mz`。先明确水面高程、水深、单宽流量、总流量和速度的区别，以及单元中心值和边值的区别。

学习产出：写清 `stage=z+h`、`mx=h*u`、`mz=h*v`，以及本地 `[z,x]` 水平轴与 ANUGA `(x,y)` 的对应关系。保持现有 SI 单位契约；不要把 m²/s 的动量变量当成 m³/s 的河口总流量。

验收：单格体积、断面总流量、方向正负和高程基准都有明确换算；提高绝对高程但保持相对水深后，结果不应发生不合理变化。

源码：[Domain 状态定义](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L453)；[边状态取值](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L994)。

### 02. 单边法向旋转与统一通量接口

**P1 · A/B · 难度中。** `gpu_rotate` 把动量转换为法向/切向分量，计算后再旋回。这个局部运算与三角形无直接绑定，方格可使用四个坐标轴法向。

学习产出：定义一个与网格存储独立的边函数，输入左右状态、单位法向和床面信息，输出质量通量、动量通量、压力贡献及最大波速。X/Z 两类边共享同一数学实现，拓扑遍历仍由我们的方格代码负责。

验收：旋转再旋回恢复原值；交换左右状态并反向法向满足通量方向一致性；保留现有转置对称测试，另加入有解析参考的斜向传播检查。

源码：[gpu_rotate](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L148)；[test_rotate](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/tests/test_flux.py#L25)。

### 03. KNP 中央迎风通量及原版函数复用入口

**P1 · A/B · 难度中高。** ANUGA 根据左右速度与重力波速构造双向传播速度，计算中央迎风通量。它是替代当前 Rusanov 通量的候选，是否减少我们场景的误差要实测。

具体可复用入口是 `gpu_flux_function_central`；现有 Cython 还公开了 `flux_function_central`，原版 `test_flux.py` 正在直接调用它。因此可先通过已编译 ANUGA 的接口建立函数级参考，再评估抽取 C helper。名称含 `gpu` 不表示它只能用于 GPU。

必须保留其输入语义：`ql/qr[0]` 是水面高程；`h_left/h_right` 是修正后水深；`hle/hre` 是重构边水深；`ze` 是公共床面高程。`edgeflux` 与 `pressure_flux` 分开返回，当前 `_flux` 里的完整压力项不能直接叠上去。

验收：原版给定的单边测试向量逐项一致；增加双干、单干、正反法向及平床常量流；然后才进入完整方格域的比较。确认无重复压力或遗漏压力。

源码：[gpu_flux_function_central](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L210)；[Python/Cython 可调用入口](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/sw_domain_openmp_ext.pyx#L1172)；[原版单边测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/tests/test_flux.py#L49)。

### 04. 地形源项与静水保持

**P0 · B · 难度高。** `core_compute_fluxes_central` 先取左右床面最大值，计算有效水深，再结合单元中心/边的高程和水深处理压力梯度。学习重点是通量与源项怎样在静水下抵消。

我们已有 hydrostatic reconstruction 和静水测试，不能把这部分当成空白重写。引入 ANUGA 通量时，要在现有静水配套方案与移植 ANUGA 方案之间作明确选择，并验证一致性，不能拼接两套压力补偿。

验收：斜床静水、陡岛干岸、床面台阶、建筑邻接和大高程偏置；静水速度与水深误差应维持合理数值量级，记录水量修正。

源码：[公共床面与有效水深](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L1035)；[压力梯度配套项](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L1142)。

### 05. 二阶空间重构和单调限制器

**P1 · B · 难度高。** `core_extrapolate_second_order_edge` 从单元中心重构到边，按邻域水深调整限制强度；`gpu_limit_gradient` 限制三个边上的增量。它显式使用三个邻居和三个边，不能把 3 改成 4 就得到正确方格算法。

学习产出：按方格邻域重写有界梯度与四边状态重构，确定水面/水深/速度的重构组合，处理建筑附近的有效邻域和单侧边界，并与第 04 项一起验证。

验收：光滑波的空间误差随网格细化降低；溃坝不出现不可接受的超调或负水深；窄通道不因错误邻域跨过建筑取梯度。激波附近不能要求处处二阶收敛。

源码：[二阶重构完整入口](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L24)；[限制器](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L63)。

### 06. 浅水速度重构与多种干湿阈值

**P1 · B/C · 难度中。** 源码在过浅单元清零动量，可先将单元动量换成速度重构，再乘边水深恢复动量。应学习如何避免极薄水层形成巨大速度。

我们已有 `DRY` 和干格处理；需要进一步区分求解保护阈值、速度统计阈值、淹没判定阈值和显示阈值，记录其物理含义。ANUGA 不同算法的阈值也不同，不能机械复制一个常数。

验收：薄水层退水、干床突然来水、粗糙率变化下速度有界；报告阈值变化对淹没面积和最大速度的影响。

源码：[中心状态与速度重构](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L64)；[边速度干湿处理](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L156)；[DE1 参数组](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L1104)。

### 07. 负水深修正与新增水量诊断

**P0 · A/B/C · 难度中。** `core_protect` 和 `core_fix_negative_cells` 会清动量并将负水深夹到零；`core_negative_cells_volume` 在夹零前计算因此增加的水量。成熟软件同样需要诊断数值修正，并非依靠夹零就证明严格守恒或正性。

我们已有 `numerical_adjustment_m3`。应保留它，补充修正位置、出现次数和相对累计输入水量的量级，区分浮点微差与明显不稳定。不要用修正量抵消后收支为零来掩盖过大的原始误差。

验收：干湿交界、极陡床面、强源汇下记录未修正最小水深和修正体积；修正应随合理细化或步长收紧而得到解释。

源码：[core_protect](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L499)；[修正前体积统计](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L585)；[体积警告测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/tests/test_negative_cells_warning.py#L48)。

### 08. RK2 时间推进与源项时间一致性

**P1 · B · 难度中高。** `evolve_one_rk2_step` 备份状态，进行两次更新再加权，第二阶段重新计算边状态和边界。源码还专门修正了阶段时间污染后续外部算子的错误。

学习产出：把空间残差计算从状态更新中拆开，统一管理阶段时间。继续保证当前分段常量来水在事件点切步。注意：输运采用 RK2，并不自动使分裂执行的降雨、结构等全过程都达到二阶时间精度。

验收：无源光滑解的时间收敛；时变边界在正确阶段取值；相同来水曲线在不同最大步长下累计量一致；子阶段不得重复计入水量账本。

源码：[RK2 阶段与时间恢复](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L3547)；[时变算子回归测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/tests/test_operator_timing.py#L25)。

### 09. CFL、输出时刻与异常小步长诊断

**P1 · B/C · 难度中。** ANUGA 根据单元几何尺度和边波速估计局部步长，再施加 CFL 与输出时间截断。三角形半径不能直接当作方格间距使用，其 CFL 数也不能照搬。

我们已有波速、来水增长和事件切步限制。应学习显式跟踪下一输出时刻、避免浮点取模产生极小余步，以及定位到底是哪一格/哪一物理条件控制步长。

验收：最终时间和输出时间准确；细化网格后稳定性合理；极小步长能定位到单元和原因。不要直接照搬隔离高速三角形后清零动量的启发式处理。

源码：[局部几何步长](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L1150)；[输出时刻截断](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/abstract_2d_finite_volumes/generic_domain.py#L2577)；[diagnose_timestep](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L5689)。

### 10. Manning 摩阻与低弗劳德数修正

**P2 · A/B · 难度中。** 平床半隐式 Manning 是我们已有的内容，优先对照量纲、浅水保护和时间耦合；斜床摩阻需要用方格高程梯度替代三角边几何。

另一个独立选项是 `gpu_compute_local_froude`：它只调整通量中的动量扩散，质量通量不使用该系数。适合把缓慢回水/浅滩作为单独研究案例，不能默认所有洪水都应启用，更不能把它与摩阻混为一项物理作用。

验收：均匀流或摩阻衰减参考；零糙率极限；低速回水场景比较动量误差和振荡，保留开启/关闭对照。

源码：[平床半隐式摩阻](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L611)；[斜床摩阻](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L713)；[局部 Froude 修正](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L175)。

## 4. 边界与外部驱动：11—15

### 11. 反射墙的法向与切向处理

**P1 · B/C · 难度低中。** `Reflective_boundary.evaluate` 只反转法向动量，切向保留。这是不可穿透的滑移反射处理；不要因注释里的类比把它误当成完整黏性无滑移墙。

我们墙面已有零质量通量及两侧压力保留。学习重点是用斜入射、沿墙流和静水检验其物理含义，延续建筑邻接下的压力平衡。

验收：墙不漏水、沿墙运动没有无依据地归零、静水墙面没有虚假加速。

源码：[Reflective_boundary.evaluate](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/boundaries.py#L99)。

### 12. 开放边界、固定水位与特征边界

**P1/P2 · B · 难度中高。** 研究水位指定边界、透射边界和 Flather 类弱反射边界的差别。后者结合内外部特征信息，源码明确按亚临界条件处理，不能泛化到所有流态。

当前普通边缘是只出流，海洋边缘是固定水位静止外部水库。扩展时应新增独立边界类型，并保留旧场景语义。指定幽灵状态的水位/动量，也不等于精确指定离散边界通量。

验收：脉冲离域的反射量、涨落水位下双向交换、回水和干岸重新淹没；记录真实数值边通量。

源码：[水位与法向动量边界](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/boundaries.py#L316)；[Flather 类边界](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/boundaries.py#L1184)。

### 13. 来水过程积分与边界/内部源的区别

**P0/P1 · B/C · 难度中。** `Inlet_operator` 给区域增加总体积，可设置入口速度；它的默认时间积分采用端点流量平均。我们现有河口则直接输入边质量通量和法向动量，内部点才是体积源，两者必须继续区分。

学习其请求量/实际量统计和入口速度选项，但保留我们分段常量水文过程的切步积分。对流量突变直接使用端点梯形平均会改变当前场景的注水体积；连续或线性曲线应作为另一个明确模式。

验收：阶跃、矩形脉冲、多点和角点总量准确；水不重复加入；请求流量、实际边通量、实际注入体积有分别记录。

源码：[流量积分与实际应用](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/inlet_operator.py#L198)；[请求/实际抽排量测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/tests/test_inlet_operator.py#L138)。

### 14. 空间降雨与可组合源汇算子

**P2 · B/C · 难度中。** `Rate_operator` 接收时间函数、空间函数或数组；Towradgi 示例把不同雨量站的时间序列施加到不同多边形区域。

学习产出：把目前全域统一降雨扩展为同方格的降雨率数组或分区曲线，统一 m/s 与 mm/h 的转换。每个源汇保留独立累计量、启停与时间定义，避免引入完整 Domain 依赖。

验收：均匀雨退化回当前结果；两片不同降雨区域的总体积等于面积积分；雨停时刻精确；建筑屋顶雨水是否转排另行建模。

源码：[Rate_operator](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/operators/rate_operators.py#L268)；[分区降雨案例](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/case_studies/towradgi_simulation/setup_rainfall.py#L22)。

### 15. 限量抽排与动量随水移除

**P1 · A/B/C · 难度低中。** 负 `Rate_operator` 将抽排量限制在可用水深内，同时按剩余水量缩放动量；否则水减少而动量不变会产生极大速度。

我们排水和入渗已经这样处理，近期任务是增加与原版行为一致的测试及请求/实际抽排统计。指定一个负降雨率并不等于已经拥有土壤入渗、饱和或地下水模型。

验收：抽干不产生负水深；移除水时速度处理有明确假设；排水容量超过存水时实际排量受限，账本记录实际值。

源码：[负源率和动量缩放](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/operators/rate_operators.py#L442)；[超额抽排限制](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/inlet_operator.py#L276)。

## 5. 提升场景物理表达：16—20

### 16. 有限堤顶高度、漫顶与淹没堰流

**P1（存在堤防时）· A/B · 难度中高。** `gpu_adjust_edgeflux_with_weir` 根据堤顶以上水头计算堰流，进行淹没修正，并与浅水通量过渡。当前 `wall_x/wall_y` 只有完全封闭布尔值，无法表达水位升高后漫顶。

学习产出：在方格公共边保存堤顶高程与水力参数；明确 m²/s 单宽通量乘边宽后才是总流量；同步更新动量与稳定步长估计。保留完全封闭墙作为独立类别。

验收：低于堤顶无漫顶；上下游互换时方向正确；淹没程度变化时流量连续；两边质量收支相抵。

源码：[堰流局部函数](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L315)；[通量中的堤防耦合](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L1075)；[堤防数据与参数](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/riverwall.py#L57)。

### 17. 箱涵、圆管及入口/出口控制

**P1/P2 · A/B · 难度中高。** `boyd_box_function`、`boyd_pipe_function` 独立接收结构尺寸、水头、损失和糙率等标量，返回流量和出口状态，比完整三角网格求解器更容易局部提取；仍依赖 ANUGA 的常数等上下文。

把涵洞表示成两个方格交换区之间的水量连接，处理堵塞、反向水头、入口和出口控制。它能补足道路/堤防下通水，但不能据此宣称已模拟完整地下管网。

验收：零水头差、完全堵塞、不同淹没状态、反向流；结构出流量受上游可用水量限制。照搬的经验系数要保留来源与适用条件。

源码：[boyd_box_function](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/boyd_box_operator.py#L328)；[boyd_pipe_function](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/boyd_pipe_operator.py#L296)；[箱涵测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/tests/test_boyd_box_operator.py#L1)。

### 18. 交换区按水面调平、按水深分配动量

**P1（引入结构时）· A/B · 难度中。** `level_stages_to_average` 以水面、高程、面积和平均水深变化为输入，用体积约束调平。`set_average_momenta` 按水深分配动量，避免浅格分到同样动量后速度发散。前者是很清楚的原版 Python 函数提取候选。

应把这套处理限定在采用充分混合假设的结构交换区；不能把整条河、所有内部点水源或不连通洼地都强行调平。现有逐格指定源也无需统一替换。

验收：零交换严格不变；斜床静水不被扰动；指定增减体积准确；干格不获得巨大速度；加水不压低原有高水位格。

源码：[level_stages_to_average](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/inlet.py#L12)；[按水深分配动量](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/inlet.py#L266)；[调平和守恒测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/tests/test_structure_operator.py#L310)。

### 19. 内部水工关系及守恒交换框架

**P2 · B/C · 难度中高。** `Internal_boundary_operator` 接收上下游水位决定流量的函数；隐式路径把交换流量和两边水位变化联立求解。可借鉴为闸门、可控泵或非标准结构的扩展接口。

`Structure_operator` 同时处理上游可用水量、下游增加体积和动量出口选项。学习框架时要审查区域充分混合、水头采样、出口射流和流量平滑的假设，而不是只取 Q 公式。

验收：一处损失等于另一处增益；闸门关闭、两侧同水位和方向反转有正确结果；减小步长后流量与水位变化稳定。

源码：[隐式水头—流量耦合](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/internal_boundary_operator.py#L224)；[结构交换及水量限制](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/structure_operator.py#L360)。

### 20. 建筑的不可穿透与可漫过表示

**P2 · B/C · 难度中。** Merewether 案例显式区分建筑作为网格孔洞加反射边界，或抬高建筑区域的高程。两者代表不同物理假设，有限高程建筑可能被淹没。

我们的 `solid` 当前是完全不透水，继续保持该语义。若未来增加有限建筑/道路高程，应新增类型，不能静默用抬高地形代替已有建筑掩膜。方格楼间通道是否存在取决于原始分辨率。

验收：不透水建筑内部不存水；有限高程障碍在超过阈值后可过水；降雨和排水与建筑类型一致。

源码：[建筑两种建模方式](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/case_studies/merewether/runMerewether.py#L77)。

## 6. 水量、灾害指标与独立验证：21—25

### 21. 积分真实数值通量的水量账本

**P0 · B/C · 难度中。** `boundary_flux_integral_operator` 按 Euler、RK2、RK3 或 ADER 的阶段权重累计实际通量。`compute_boundary_flows` 自己明确说明，用边动量估算的瞬时流量并不等于计算中采用的精确数值通量。

我们已使用实际面通量做收支，这是应保留的设计。改成 RK2 后，累计量必须按阶段加权；当前 `qx/qy` 最后子步瞬时值不代表整个输出间隔的平均通量。需要时另加区间累计输水量字段，避免改变旧字段含义。

验收：初始水量＋外部输入－输出＋数值修正＝当前水量；内部结构交换不计为域外来水；稀疏输出不改变总收支。

源码：[分阶段通量积分](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/operators/boundary_flux_integral_operator.py#L44)；[近似流量与精确积分的区别](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L5498)；[整体水量诊断](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L5613)。

### 22. 最大水深、最大速度和最大水深×速度

**P1 · A/B/C · 难度低中。** `Collect_max_quantities_operator` 维护最大水面、最大水深、最大速度和动量模，且浅水速度有独立阈值。我们已有每步累计最大水深，可扩展其余量。

学习产出：在真实求解步更新最大速度和 `h*|u|`，与播放帧抽样分开。洪水到达时间、超过水深阈值的持续时间可以作为我们的进一步扩展，本次没有把它们算成该算子现有功能。

验收：降低导出帧率不应漏掉累计峰值；同一时刻计算 `h*|u|` 后再取最大，不能用最大水深乘最大速度冒充；清楚记录干湿阈值。

源码：[最大量更新](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/operators/collect_max_quantities_operator.py#L122)。

### 23. 解析解与网格收敛

**P0 · C · 难度中。** 优先移植的是案例定义和解析参考，而非三角网格脚本：干/湿床溃坝、静水含地形突起、抛物盆地振荡、跨临界含激波渠道。当前干床溃坝测试主要检查流动方向、守恒和对称，还没有解析误差曲线。

学习产出：固定物理域，在至少三档方格间距下报告水深、动量及湿区速度的误差，分开观察波前位置和整体误差；有激波的题目不要求全域二阶。原版误差门槛只适用于其具体网格和配置，不直接成为我们的通过线。

验收：每个案例有方程、边界、初始条件、参考解、采样方式和误差定义；改变算法之后能说明误差为什么变化。

源码：[多时刻水面/动量/速度误差](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/analytical_exact/dam_break_dry/validate_dam_break_dry.py#L58)；[盆地解析参考](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/analytical_exact/parabolic_basin/analytical_parabolic_basin.py#L1)；[跨临界激波验证](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/analytical_exact/transcritical_with_shock/validate_transcritical_with_shock.py#L20)。

### 24. 实验数据与真实观测校验

**P1/P2 · C · 难度中高。** Yeh–Petroff 溃坝实验、Okushiri 水槽波浪和 Towradgi 历史洪水分别提供不同层次的物理对照。学习重点是实测边界输入、测站坐标、水位时间序列、峰值及到达时刻。

对项目先选与河流洪水相关的实验。数据单位、实验比例尺和垂直基准必须核对；真实案例需要原始资料、边界条件和校准。合成地形的空间形态合理，不代表能校准某场真实洪水。

验收：独立观测与拟合数据分开；使用测站水位过程、峰值误差和时刻误差；数据缺失写明。本次只检查仓库脚本，未下载或验证外部大数据集。

源码：[实验溃坝对照](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/experimental_data/dam_break_yeh_petroff/plot_results.py#L1)；[观测和模拟时间序列对齐](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/experimental_data/okushiri/test_results.py#L76)；[实测洪水位对照](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/case_studies/towradgi/Compare_results_with_fieldObs.py#L15)。

### 25. 单元测试、数值回归、跨后端一致性与物理验证分层

**P0 · C · 难度中。** 仓库把局部函数测试、快照回归和解析/实验验证分开。CPU/GPU 或新旧实现一致只能证明实现一致，两个版本也可能共享错误；快照相同同样不能代替物理正确性。

学习产出：保留现有功能测试，增加独立物理基准，并建立固定算法配置的回归样本。对每个验证脚本检查实际断言：部分旧实验脚本把若干断言放在 verbose 条件下，不能把脚本能运行当成验证已通过。

验收：每条通过结论能对应真实断言、参考来源、阈值和运行配置；优化后既检查旧结果一致性，也检查独立基准。

源码：[固定算法/计算模式的快照](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/tests/test_regression_snapshots.py#L22)；[需核对执行条件的实验统计](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/experimental_data/okushiri/test_results.py#L179)；[验证类型说明](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/docs/source/reference/validation.rst#L90)。

## 7. 工程与可复现性：26—30

### 26. 配置算法组合与完整参数留痕

**P0/P1 · C · 难度低中。** `get_algorithm_parameters` 和各 `DE*` 配置把重构阶数、时间法、限制器、阈值等作为整体管理。我们目前的求解器元数据主要是一条描述字符串。

学习产出：导出求解器版本/提交、通量类型、空间和时间阶数、CFL、干湿阈值、摩阻方案、边界类型、地形尺度和源项积分规则。对照时显式选择算法。

验收：同一份场景可恢复相同配置；参数改变有可追溯差异。特别注意源码 `set_flow_algorithm` 默认参数是 DE0，而算法文档把 DE1 描述为通用默认；运行时应显式设置并导出实际值。

源码：[完整算法参数](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L998)；[源码默认算法](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L1783)；[算法文档](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/docs/source/setup_anuga_script/flow_algorithms.rst#L1)。

### 27. 状态、边值、残差和物理算子的接口分层

**P1 · B/C · 难度中。** ANUGA 的 Quantity 区分中心/边/顶点值以及显式/半隐式更新；算子有统一注册与调用流程。当前 `BaseMap.configure` 会直接写求解器数组，可借此机会梳理接口。

建议形成 `重构边状态 → 边界求值 → 通量/残差 → 时间推进 → 源汇/结构 → 诊断` 的清晰接口，保持方格数组。无需复制其全部 Domain/Quantity 体系。算子顺序会影响结果，应固定顺序并测试。

验收：通量函数可独立验证；源汇不重复计量；环境只在允许阶段更新；时间推进不被前端帧率控制。

源码：[Quantity 状态组织](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/abstract_2d_finite_volumes/quantity.py#L1)；[Operator 接口](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/operators/base_operator.py#L1)；[算子顺序及后端同步](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/shallow_water_domain.py#L5214)。

### 28. 共享计算内核与 CPU 优先的性能改进

**P3 · A/B/C · 难度高。** `gpu/core_kernels.c` 同时被 CPU 扩展编译使用；Meson 对 CPU 构建定义 `CPU_ONLY_MODE`。学习其物理实现共用、连续数组、工作区复用和减少主机/设备往返。

原版网格遍历仍硬编码每个三角形三个边，不能整体搬来处理方格。先在我们的网格上做正确性与性能剖析，确认瓶颈再选择向量化、C/Cython 或并行；不要先复制 MPI/GPU 基础设施。

验收：同物理域、相近空间误差、相同输出量下比较总耗时和内存；单列启动、计算与导出耗时。不能用单元数量相等直接当作精度公平。

源码：[CPU/GPU 共用物理内核](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L1)；[CPU 编译复用同一内核](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/meson.build#L4)；[显式设备同步](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_domain_core.c#L1178)。

### 29. 保存真实状态、输出元数据与可恢复运行

**P2 · B/C · 难度中。** SWW 写入时间、地理参考、物理量及版本信息；checkpoint 保存可恢复的 Domain。当前项目已能重启后回放，但回放文件与能继续推进的完整计算状态是两件事。

学习产出：保持我们的 NPZ/JSON 与方格接口，补全第 26 项元数据；若需要长时计算恢复，另存完整 h/mx/mz、时间、源项累计量、最大值、事件位置和配置。

验收：中途保存恢复后与连续计算在合理容差内一致；稀疏播放帧或插值帧不当作求解状态；输出结构有版本号。

源码：[SWW 元数据写入](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/file/sww.py#L578)；[恢复检查点](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/checkpoint.py#L24)。

### 30. 自动验证、错误案例留存与开源来源管理

**P0/P1 · C · 难度低中。** ANUGA 单独运行耗时的物理验证，并保留日志与图；这与每次改动只跑快速单元测试的用途不同。我们可设快速验证与定期完整验证两层。

如提取原版函数，应固定提交、保留适用的版权/许可信息，并记录提取范围、本地方格改动和对应测试。仓库采用 Apache-2.0；具体复用时逐项检查相关文件和第三方依赖的声明。

验收：数值修改能关联到基准结果；失败有可重现场景；提取文件能追溯上游版本。本文只提出验证流程，不创建定时自动化或提交代码。

源码：[独立验证运行器](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/validation_tests/run_auto_validation_tests.py#L1)；[验证日志和结果留存](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/.github/workflows/validation.yml#L100)；[上游许可证](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/LICENSE.txt#L1)。

## 8. 暂不直接照搬的内容

| 内容 | 研读中看到的具体原因 | 处理建议 |
| --- | --- | --- |
| `Inflow_boundary` | 源码明示未完成、关联测试禁用；坡度仍被赋为 0 | 不作为现有精确河口流量边界的替代 |
| 三角形梯度、三边数组与半径 | 索引、边数量和几何尺度直接参与计算 | 方格四边与邻域重写，保留数学目标 |
| `Cd_through` 穿墙流 | 当前 helper 只追加质量通量，未加入对应动量；采用简化孔口关系 | 作为可选经验功能研究，不称作完整渗流或涵洞模型 |
| 整套 GPU/MPI 域管理 | 三角拓扑、幽灵单元和设备状态耦合 | 在 CPU 方格正确性确立后再研究 |
| `DE_ader2` 与多种旧兼容算法 | 学习成本高于先建立空间二阶＋RK2 基线；文档仍提示新法验证状态 | 先显式固定 DE1 做参考，之后再评估 |
| 土壤/地下管网/结构破坏的推断 | 指定负源率、涵洞或动态堤顶并不自动包含这些完整物理系统 | 按实际需求另建模型和验证 |

对应证据：[未完成的入流边界](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/boundaries.py#L938)；[穿墙流的质量/动量处理](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h#L389)；[硬编码三边遍历](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c#L989)。

## 9. 建议学习顺序与具体交付物

| 阶段 | 优先条目 | 完成后应拿到什么 |
| --- | --- | --- |
| 第一步：建立证据基线 | 01、04、07、21、23、25、26 | 当前方格求解器的解析误差、水量误差及完整配置记录 |
| 第二步：做最小原版函数复用 | 02、03、04 | 原版通量函数的单边调用/提取验证，以及压力配套明确的平床、静水与溃坝对照 |
| 第三步：提高离散精度 | 05、06、08、09 | 方格二阶重构与 RK2；空间/时间收敛、干湿和事件积分检查 |
| 第四步：补场景物理 | 11—20、22、24 | 根据目标场景加入开放边界、漫顶、涵洞或空间降雨，并配相应独立验证 |
| 第五步：工程化扩展 | 27—30 | 稳定接口、可复现导出、恢复运行，以及有测量依据的加速 |

第 03 项可以先在平床一阶方格里研究，不必等二阶重构完成；进入变床地形时必须同时解决第 04 项。若当前研究关注道路/堤防对救援路径的影响，第 16—19 项可以在基线后提前，其收益不必等到所有数值算法升级完成。

最值得优先尝试复用的三个代码单元：`gpu_flux_function_central` 及所需 helper、`level_stages_to_average`、`boyd_box_function/boyd_pipe_function`。它们分别对应方格边输水、交换区水位分配和涵洞过水；后两类仅在引入相应结构时需要。

## 10. 本次结论的证据边界

已核对：固定提交的目录、关键函数、调用链、相关测试、验证脚本、构建方式和许可证；并对照本地求解器、图层、运行及测试代码。

尚未完成：独立提取编译、原版函数的本地数值一致性运行、完整 ANUGA 安装和基准运行、真实案例大数据获取，以及本地方格修改后的精度/性能比较。因此 A 类表示有源码依据的提取候选，不表示已形成可用的替换模块。

需要更正此前讨论中的两点：ANUGA 并非只能作为整套三角求解器或验证工具，它确实公开了可独立测试的边通量接口；但提取该接口后，也必须配好压力、地形和干湿处理。另一个是不能把文档里的 DE1 推荐值直接当作源码默认值；所有对照均应显式保存实际配置。
