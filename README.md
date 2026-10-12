# HYRA-Hydrodynamic-Rescue-Agents-
项目仓库

## 地形生成工作台

[进入地形实验台](terrain-workbench/README.md)：生成二维高程、水系、地表类型与群系图，并导出用于洪水模拟的高程矩阵。

工作台使用固定版本的 FreeTerraForged 原版地形算法，包含 Python 本地服务、Java 生成器和可直接查看的示例地图。

[多层地图与洪水](terrain-workbench/terrain-lab/MULTILAYER.md)在当前地形上编辑建筑、阻水边、逐格糙率、水源与排水，并整段计算洪水过程。高程、水深、流速和累计最大水深共用网格与时间轴，支持连续播放、暂停、倍速、查点和完整导出。
