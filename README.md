# HYRA-Hydrodynamic-Rescue-Agents-
项目仓库

## 地形生成工作台

[进入地形实验台](terrain-workbench/README.md)：生成二维高程、水系、地表类型与群系图，并导出用于洪水模拟的高程矩阵。

工作台使用固定版本的 FreeTerraForged 原版地形算法，包含 Python 本地服务、Java 生成器和可直接查看的示例地图。

[动态洪水工作台](terrain-workbench/README.md#动态洪水)在当前地形上模拟降雨与指定位置来水，计算水深、流速、流向和累计最大水深，并支持时间播放与完整过程导出。

[多层地图与观测](terrain-workbench/terrain-lab/MULTILAYER.md)把固定环境、水情真值、各无人机观测、探索历史、人员与无人机位置分开保存。支持建筑与边墙、逐格糙率、排水、变化水源、禁飞区、多机共享和信息年龄四图对照。
