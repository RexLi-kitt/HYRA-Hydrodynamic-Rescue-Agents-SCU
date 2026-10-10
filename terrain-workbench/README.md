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
