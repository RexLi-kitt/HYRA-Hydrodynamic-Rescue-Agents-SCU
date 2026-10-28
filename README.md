# HYRA-Hydrodynamic-Rescue-Agents-
项目仓库

[无人机工作台](uav-workbench/README.md)：现行统一八方向网页，提供100米双任务、三路径信号建模和原始网格回放；[代码导航](uav-workbench/docs/CODE_MAP.md)。历史单机验收与试跑见[P0—P2记录](uav-workbench/docs/P0-P2_20261025.md)。

[开发日志](DEVLOG.md)：记录地图图层、人员布点、河网检查、分类糙率及洪水对照试验的改动、证据与待办。

[真实洪水数据工作台](flood-data-workbench/README.md)：独立回放Harvey 2017真实水位，查看地形概览、空间规模与数据送达时间；保留数据来源和未具备的模拟条件。

## 地形生成工作台

[进入地形实验台](terrain-workbench/README.md)：生成二维高程、水系、地表类型与群系图，并导出用于洪水模拟的高程矩阵。

工作台使用固定版本的 FreeTerraForged 原版地形算法，包含 Python 本地服务、Java 生成器和可直接查看的示例地图。

[多层地图与洪水](terrain-workbench/terrain-lab/MULTILAYER.md)在当前地形上编辑建筑、阻水边、逐格糙率、水源与排水，并整段计算洪水过程。高程、水深、流速和累计最大水深共用网格与时间轴，支持连续播放、暂停、倍速、查点和完整导出。

外源洪水支持生成时标注河流边界入口、手动选点强制来水和多点上游流量过程。外源参数与动态水情分别导出，可供后续强化学习场景配置。

[P1/P2/P3逐项验收](uav-workbench/docs/CONTINUOUS_ACCEPTANCE.md)与[两轮P3阶段结论](uav-workbench/docs/P3_FINAL_20261025.md)：第二轮存在平均增量，但未通过预登记跨机制门槛，原P3不追加第三轮。用户另行授权的Sur-QMIX适配与物理辅助世界模型作为独立首轮已完成实验与审计，结果见下。

[Sur-QMIX适配贯通](uav-workbench/docs/SUR_PREFLIGHT_20261025.md)已完成训练/执行/恢复与权限审计；[独立物理辅助实验](uav-workbench/docs/SCIML_PROTOCOL_20261025.md)已完成全部训练、留出评价与审计；[首轮闭环结果为负](uav-workbench/docs/SCIML_FINAL_20261025.md)，不追加轮次。[运行与复现](uav-workbench/docs/REPRODUCE.md)区分原网页回放、两机clean actor命令行与科学训练入口。
