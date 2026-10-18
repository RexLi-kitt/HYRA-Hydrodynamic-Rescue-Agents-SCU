# FreeTerraForged 完整地形复现

基准：FreeTerraForged `1.21.1` 分支，提交 `c5d322f5e44539ca6d3a6059c5de440af9abcfe6`。

实现采用原版 Java 源码与网格导出接口。Minecraft/Fabric 提供原版注册表、密度函数、洞穴、地表和区块运行环境；Python 入口负责运行配置和读取结果。`upstream/` 中保留了 635 个原版文件及 MIT 许可证，逐文件校验清单位于 `upstream-manifest.json`。

## 已接入的原版流程

- 四种大陆模式、大陆边界、海岸和海底。
- 地形分区、过渡、平原、丘陵、谷地、高原、荒地、山脉、火山及群岛。
- 主河、递归支流、河线扭曲、河床、河岸、谷底、湖泊、湿地与水位控制。
- 固定 tile 及边界、水滴侵蚀、平滑、坡度、海滩与高度上限处理。
- 气候、地貌和生物群系相关的原版字段。
- 完整 Minecraft 区块生成：三维密度、洞穴、含水层、地表规则、岩层、矿石、地形特征与结构。
- 原版 8 个预设，以及通过原版 Codec 加载完整自定义预设 JSON。

生成步骤直接调用原版 `GeneratorContext`、`TileGenerator`、`WorldFilters` 和完整区块生成流程。适配层负责采样、导出与进程生命周期，并可按模拟请求跳过游戏高度压缩、扩展实际方块生成空间。

## 运行

需要 Java 21。首次构建需要联网下载项目依赖。可通过系统 Java 21 或 `JAVA_HOME` 提供运行时，也可用 `--java-home /路径/到/JDK21` 指定运行时。

在本目录运行：

```bash
python3 run.py export
python3 run.py export --request request-voxels.json
python3 run.py export --request request-validation.json
python3 run.py test
python3 run.py build
python3 run.py verify-source
```

`build` 生成带导出入口的 Fabric 模组。`export` 在无图形界面的原版运行环境中执行，使用固定版本算法。三维导出通过本地测试服务器运行完整区块流程。

## 配置地图

默认请求位于 `request.json`：

```json
{
  "seed": "42",
  "preset": "ftf-default",
  "origin_x": -512,
  "origin_z": -512,
  "nx": 128,
  "nz": 128,
  "step_blocks": 8,
  "output": "generated/default-grid",
  "export_datapack": true
}
```

坐标与间距采用 Minecraft 方块单位；第 `row, col` 个采样点为 `(origin_x + col × step_blocks, origin_z + row × step_blocks)`。算法先按原版 tile 生成与过滤，再采样；改变输出网格间距不会改变原版侵蚀使用的空间尺度。

可选 `fields` 数组指定导出的原版Cell字段名，例如 `["height","terrain","riverWaterLevel"]`；不指定时保留全部字段。工作台固定 `step_blocks=1`，并令 `nx=nz=地图边长`，只导出显示和洪水需要的7个字段。导出器按tile窗口遍历采样点，避免为每个格子保存额外的装箱索引。

种子使用字符串保存，保留完整 64 位整数精度。输出路径相对于请求文件所在目录。

预设名称：`ftf-default`、`modern-default-with-rivers`、`legacy-default`、`legacy-vanillaish`、`legacy-beautiful`、`legacy-lite`、`legacy-huge-biomes`、`community-1`。

可将 `preset_file` 设置为原版预设 JSON 路径，直接配置原版所有参数。每次导出都会保存实际使用的 `preset.json`。内置预设可用 `continent_type` 选择 `MULTI`、`SINGLE`、`MULTI_IMPROVED`、`UPLIFT`。

添加 `voxels` 即可导出完整三维方块。例如：

```json
"voxels": {
  "origin_x": -16,
  "origin_z": -16,
  "nx": 32,
  "nz": 32
}
```

默认包含该预设的全部垂直范围；也可设置 `min_y`、`max_y`，其中 `max_y` 不包含在输出范围内。三维方块按每个方块采样，完整生成后输出；每次运行创建新的世界，避免复用旧种子的区块。

## 读取结果

安装 NumPy 后：

```python
from load_grid import load_grid, load_volume

maps, metadata = load_grid("generated/default-grid")
height = maps["height_blocks"]
terrain = maps["terrain"]
river_zone = maps["riverZone"]

volume, volume_metadata = load_volume("generated/voxel-validation/voxels")
blocks = volume["blocks"]
```

二维输出包含原版 `Cell` 的全部 34 个实例字段。`grid.json` 记录字段类型、数组形状和类别编号；二进制文件保留原版数值精度。数组顺序为 `[z, x]`。

`height` 是原版连续地形值；`height_blocks` 是按原版高度比例换算后的方块坐标。它与执行洞穴、地表、结构和植被之后的实际方块表面分别保存。

三维数组顺序为 `[z, y, x]`，`volume.json` 提供方块状态和生物群系编号表。另有原版 `world_surface`、`ocean_floor`、`motion_blocking`、`motion_blocking_no_leaves` 四种高度图，其含义沿用 Minecraft。

转换为 NumPy 压缩文件：

```bash
python3 load_grid.py generated/default-grid --npz generated/default-grid.npz
python3 load_grid.py generated/voxel-validation/voxels --volume --npz generated/voxel-validation.npz
```

## 验证记录

运行 `request-validation.json` 可重新生成预设、大陆模式、相同种子和跨 tile 的验证结果。历次运行日志与生成世界不包含在源码仓库中。二维一致性验证逐字段比较导出值与原版游戏缓存查询，并覆盖负坐标及 tile 边界。

原版可选联动代码随完整源码保留；本次运行环境使用 Minecraft 1.21.1、Fabric Loader 0.19.3、Fabric API 0.102.0 和 Architectury 13.0.8。

源码来源：https://github.com/ETcodehome/FreeTerraForged/tree/c5d322f5e44539ca6d3a6059c5de440af9abcfe6

## 模拟地图高度

请求设置 `unbounded_height: true` 可通过适配层跳过原版 `TerrainCeiling` 山峰压缩，不修改上游源码。二维连续高程不受游戏建造高度限制。默认不设置时保持原版行为。完整地表或体素导出按已生成 tile 的高程（含边界）预留 64 单位并自动扩大垂直空间，实际使用的预设另存为 `surface-preset.json`；底层引擎容量不足时明确报错，保留未截断的二维数据。`grid.json` 的 `height_policy`、`height_ceiling_enabled` 与地表元数据记录实际策略。
