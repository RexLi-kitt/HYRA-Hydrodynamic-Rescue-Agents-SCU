# 地形生成工作台

为洪水模拟准备合成网格地形。页面可以调整种子、地形起伏、大陆尺度、河流、湖泊、湿地和气候参数，查看二维高程、水系、地表类型与群系。

![工作台预览](docs/preview.png)

## 打开页面

需要 Python 3.9 或以上。仓库附带一张由原版完整区块流程生成的示例地图，不安装 Java 也可以查看。

```bash
cd terrain-workbench
python3 terrain-lab/launcher.py
```

也可以双击 macOS 的 `terrain-lab/启动页面.command`。页面地址为 <http://127.0.0.1:8766/>；保持启动服务的终端运行。

## 高程可视化

点击地图右上角的“高程可视化”，或打开 <http://127.0.0.1:8766/elevation.html>。界面读取当前地图，无新地图时使用内置示例。

- 三维地形支持拖动旋转、滚轮缩放、方向键旋转与加减键缩放；高程夸张只改变显示比例。
- 二维高程图支持点击查点和方向键移动选中点，显示等高线与剖面位置。
- 统计最低、最高、平均高程与地形高差；沿 X 方向查看指定行的原始高程剖面。
- 支持下载当前视图 PNG 和原始二维高程 JSON 矩阵。

完整地图读取实际地面高程；快速预览读取连续地形高程。所有高程沿用方块单位，未标定为米。浏览器不支持 WebGL 时自动使用二维视图。生成新地图后，点击“读取最新地图”更新。

## 动态洪水

完整的地图与观测结构可从 `terrain-lab/maps.html` 进入，[多层地图说明](terrain-lab/MULTILAYER.md)包含固定环境、真实／已知水情、探索与人员位置、更新规则、编辑、导出和执行接口。

安装洪水计算依赖后，运行同一个页面服务，点击地图右上角的“动态洪水”，或打开 <http://127.0.0.1:8766/flood.html>：

```bash
python3 -m pip install -r requirements.txt
python3 terrain-lab/launcher.py
```

内置示例无需 Java 即可模拟。设置降雨强度、降雨时长和模拟时长；可点击地图指定一个来水位置，输入立方米/秒的来水量。降雨结束后仍可继续模拟退水；指定来水持续整个模拟过程。可设置统一入渗率、Manning 糙率及海平面/封闭/自由出流边界。

- 默认 **1 方块＝1 米**，是可修改的合成场景设定，同时换算高程、初始水深和网格间距，不代表真实地形标定。
- 默认计算网格为 **128 × 128**；可选 32、64、128、256。降采样按块平均地面高程和初始水深，保持初始水量；粗网格会改变狭窄河道与地形屏障。
- 完整地表读取 `ground_height` 和 `water_surface`（干地的 NaN 水面转为零水深）；快速预览用连续高程、海平面和河流/湖泊水位估算初始水体。可以取消“保留生成器的初始水体”，从干地开始。
- 默认海平面边界：生成器标记为海洋且位于地图最外侧的边界面，连接固定水位的静止外海，水位为地图 `sea_level × metres_per_block`，允许双向交换。内陆水深照常演化，其他边缘仅出流；无海岸时等同自由出流。降采样按各方向实际外侧面识别海岸，避免把同一粗格内部的海水误认为另一方向的外海。即使关闭初始水体，外海仍可流入。当前水位恒定，不含潮汐或风暴增水。
- 自由出流边界仅允许水离开网格，会把海洋当作有限水体；封闭边界反射法向流动。指定来水是内部格子的体积源，不自动给所有河流补水。
- 生成器初始水体不一定满足水动力平衡，开始模拟后可能重新分配。需要区分初始调整与降雨效果时，可用相同初始状态运行无雨、无来水对照。

默认启用河流日常基流，总量 **5 m³/s**，是可修改的合成场景参数。按原始网格中 `river` 的八邻域连通河段，选最高有效河流水面附近作为上游补给；同水位时优先浅河床，按河段原始面积分配总流量，再映射并分散到计算网格。只补给有有效水体的河段，湖泊、湿地和海洋不会被当作河流源；无河流地图跳过补给。此方法是河源识别的启发式，未使用真实集水面积或完整支流拓扑，狭窄河道在粗网格下仍可能断开。

启用基流后默认先做最多 10 分钟无雨、无指定洪水来水的预热，保留设定入渗。每 60 秒检查水深和速度的最大变化；两次连续变化均不超过 0.01 m 和 0.01 m/s，且已预热至少 120 秒，才标为达到稳定阈值并提前结束。达到时间上限后仍有变化会明确显示“尚未达稳定阈值”，可延长到最多 60 分钟；封闭地图持续补水也可能无法稳定。预热过程中可取消。洪水的 t=0 保留预热后的水深和动量，并重新开始累计洪水收支；新增淹没相对这一状态计算。基流在整个洪水过程中持续补给，不重置河流水位，也不移除多余洪水。预热收支单独保存。

计算期间可查看已完成的时间片，也可取消。完成后使用播放、暂停、时间进度和“回到初始状态”观察演化。图层包含当前水深、流速和累计最大水深。流向箭头按方向、长度和颜色标出速度，数值单位为 m/s，可选择疏密或关闭数值。每片区域取实际最快湿格的向量，避免规则采样漏掉窄河道；数值标签避免互相重叠。箭头长度采用固定速度尺度，2.58 m/s 以上封顶，精确速度仍由数值和悬停提示给出。青色圈标记基流源，橙圈是指定洪水来水。悬停查点不会改变来水位置。当前水深达到 0.01 米计为淹没，“新增淹没”只统计洪水起点水深低于此阈值的计算格。水量收支分别显示起点水量、日常基流、累计降雨、指定来水、外海补水、排入外海、入渗、其他边缘排水和当前水量；收支误差包含基流及所有外海交换。

可下载当前时刻 JSON、PNG，或包含所有时间片的完整 ZIP。二进制时间片按 `depth, u, v, maximum_depth` 四个平面排列，为 little-endian float32；每个平面为 `shape=[nz,nx]`、`[z,x]` 行序。水深与地面高程为米，`u` 为沿 +X 的速度，`v` 为沿 +Z 的速度，单位米/秒。`metadata.json` 包含形状、时间、计算格中心坐标、单位、场景参数、水量收支以及 `boundary_conditions` 中的外海水位与四向边界面掩码；`final-state.npz` 保存最终状态、初始水深、累计最大水深、`sea_level_m` 和 `sea_edge_west/east/north/south`，适合后续救援智能体读取。

`river_baseflow` 保存实际总流量、补给河段及源格坐标；`initial_conditions` 保存预热时长、稳定检查和独立收支。`initial_depth` 是洪水 t=0 的水深，`generated_initial_depth` 是未经预热的生成器水深，`baseflow_rate_m_s` 是每格持续基流的水深增加率。各帧 `balance.baseflow_m3` 为洪水开始后的累计日常来水。

```python
import numpy as np
state = np.load('final-state.npz', allow_pickle=False)
depth, vx, vz = state['depth'], state['u'], state['v']
```

计算采用二维浅水方程的一阶有限体积 Rusanov 通量与 hydrostatic reconstruction，并使用自适应 CFL 时间步、半隐式 Manning 摩阻和干湿处理。方法依据 [Audusse et al. (2004)](https://doi.org/10.1137/S1064827503431090)。当前是合成地形上的原型，未加入建筑、桥涵、堤坝、地表分类糙率、土壤入渗模型或真实洪水校准；一阶格式与粗网格的数值扩散会影响洪峰和淹没边界。

洪水数据保存在 `terrain-lab/generated/flood-<编号>/flood/`，不提交到仓库。物理不变量和适配器检查可运行：

```bash
cd terrain-lab
python3 -m unittest discover -s tests -v
```

## 生成新地图

安装 Java 21，并设置 `JAVA_HOME`，或者保证系统能找到 Java 21。首次生成需要联网下载 Gradle、Minecraft、Fabric 等构建依赖。

调整页面参数后点击“生成地图”：

- **快速预览地形**：读取原版连续地形与河网；不生成实际方块和群系。
- **完整地表与群系**：执行原版完整区块流程，再读取地面方块、水面和生物群系；耗时更长。

每次生成的数据保存在 `terrain-lab/generated/<生成编号>/raw/`；`terrain-lab/current.json` 指向最新地图。这些运行数据和构建缓存不会提交进仓库。

模拟地图关闭建筑结构、火山与蘑菇岛，并跳过为游戏世界高度设计的山峰压缩。完整地表按生成地形自动扩大垂直空间；超出底层方块引擎容量时明确报错，保留未截断的二维高程。图例显示当次地图的实际高程范围。

## 导出高程矩阵

导出当前地图；尚未生成新地图时导出内置示例：

```bash
python3 terrain-lab/export_height_matrix.py
```

结果位于 `terrain-lab/exports/height-matrix.json`，文件本身是二维数值数组。`height-metadata.json` 记录种子、形状、坐标、间距与高程定义。

如需 NumPy 格式：

```bash
python3 -m pip install -r requirements.txt
python3 terrain-lab/export_height_matrix.py --npy
```

```python
import numpy as np
H = np.load('terrain-lab/exports/height-matrix.npy', allow_pickle=False)
height_at_cell = H[0, 0]
```

矩阵顺序为 `H[行, 列] = H[z, x]`：行从地图上方到下方，列从左到右。索引从 0 开始；坐标满足 `x = origin_x + 列 × step`、`z = origin_z + 行 × step`。水域格子记录水底高程，不是水面高度。单位沿用生成器的方块单位，尚未标定为米。

## 目录与来源

- `terrain-lab/`：本地页面、服务、参数预设与矩阵导出。
- `freeterraforged-reproduction/`：Java 原版生成器与导出适配层。
- `terrain-lab/example/`：裁剪为页面所需字段的实际生成示例；后续生成仍保存全部原版 Cell 字段。

上游 FreeTerraForged 固定在提交 `c5d322f5e44539ca6d3a6059c5de440af9abcfe6`，保留完整源码及 MIT 许可证。`freeterraforged-reproduction/upstream-manifest.json` 记录逐文件校验值，可检查：

```bash
python3 freeterraforged-reproduction/run.py verify-source
```

生成器的更多用法见 [生成器说明](freeterraforged-reproduction/README.md)。项目原有 Apache-2.0 许可证保留；上游第三方源码继续遵循其 [MIT 许可证](freeterraforged-reproduction/upstream/LICENSE)。
