# 地形生成中的河网检查

地形生成、原有类型验收、河口标注后，自动执行 RivGraph；检查完成才替换 current.json。没有独立按钮或页面。无河地图记录不适用；RivGraph 报错、超时或取消均保留原地图。孤立河流和对角连接标为待核查，不以未经校准的阈值判定地形无效。不修改原始高程。

## 安装

固定上游提交 f8745a645d822377c75e16f97ae995da32587695 要求 Python >=3.12、NumPy <2。使用独立 Python 3.12 环境，不改变洪水环境。从仓库根目录运行：

```powershell
py -3.12 -m venv terrain-workbench/terrain-lab/.runtime/rivgraph-env
& terrain-workbench/terrain-lab/.runtime/rivgraph-env/Scripts/python.exe -m pip install -r terrain-workbench/requirements-rivgraph.txt
```

其他位置可设置 HYRA_RIVGRAPH_PYTHON 为兼容环境的 Python 绝对路径。Linux 默认读取 .runtime/rivgraph-env/bin/python。

## 检查含义

先保留原始河道的四邻域分区，再分别调用 RivGraph skeletonize_mask/skel_to_graph；不使用仅保留最大连通块的河流工作流。摘要保存到 raw/river-network.json，包含四/八邻域分区、孤立格、无初始水深格、节点和河段数；无方向图保存到 raw/river-network-graph.json。骨架可能调整局部像素，因此与源掩膜分区分别保存。

这是几何拓扑诊断，不是河床、河岸或水动力合格证。湖泊连接、河流离开后重新进入裁剪范围等需要进一步判断，当前河道分区不一律算错误。上游资料：https://github.com/VeinsOfTheEarth/RivGraph ，BSD-3-Clause；本项目未复制其源码。

## 合并网格

无端点的闭环骨架显式保存为 unresolved_components 并提示待核查，避免固定上游 walker 在这类输入上长时间不返回；不会删除环路来制造通过结果。

每次创建 current 洪水场景，在图层、建筑、墙和 t=0 有效堤防映射后自动检查实际计算网格，保存 scenario_metadata.river_network_grid_check，在已有格宽说明中展示：原河道格因类别众数或建筑消失、原分区分裂、原独立分区合并，以及实际床面不低于原始水面的位置数量。

诊断不改变地形，不拒绝用户有意修筑的建筑或堤防。当前对比分区成员关系，尚不能检测同一分区内部的所有捷径、环路或断面变化。分类连续不保证水力畅通，仍应比较不同格宽的洪峰、到达时间和流量；没有自动完成网格精度验收。

## 范围与计算预算

候选起点为 512 m × 512 m、2 m 计算格宽（256²格），随后验证 1024 m × 1024 m、2 m（512²格）。这不是已通过性能与精度验收的默认值，本次不修改已有范围或4倍预览设置。局部河段和洪泛区可通过上下游边界承接域外作用，无需把整个流域纳入搜救地图。

格宽由需保留的河道、路堤和过水口决定。地形范围、计算格宽、后续搜索划分应分别管理。不宜仅为扩大范围而缩放 metres_per_block，因为当前实现同时缩放高程和水深。同一范围格宽减半，格数约4倍；若CFL限制令时间步也减半，同一时长工作量约8倍，仅为尺度估计。
