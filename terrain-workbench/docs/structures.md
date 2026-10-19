# 建筑与有限堤顶防洪设施

## ANUGA 研读与复用

固定版本 `e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b`，本次官方 GitHub API 核对的 `main` 也是该提交。

| 上游文件 | 核对内容与处理 |
| --- | --- |
| `examples/simple_examples/buildings.py` | 建筑用网格内孔与反射边界。沿用本项目已有 `solid`，新增矩形对象，不移植三角网格生成 |
| `anuga/structures/riverwall.py` | 逐边堤顶、插值、参数表、几何匹配、连通检查和动态堤顶。数据组织可借鉴，方格边映射需自实现 |
| `anuga/shallow_water/gpu/gpu_device_helpers.h` | `gpu_adjust_edgeflux_with_weir` 可独立提取。翻译为 `hydraulic_structures.weir_flux`，保留原始默认过渡参数、动量缩放与波速计算 |
| `anuga/shallow_water/gpu/core_kernels.c` | 先用堤顶重构有效水深，再修正堰流；压力单独处理。接入现有供水限制与 RK2 |
| `anuga/structures/tests/test_riverwall_structure.py` | 不漏水、解析漫顶、系数、淹没与动态修改。迁移验证思路，补充方格守恒、旋转、事件与加速一致性测试 |
| `examples/structures/run_riverwall.py` | 演示运行中修改堤顶。临时堤采用显式启用／撤除事件 |
| `validation_tests/behaviour_only/lateral_weir_hecras/channel_floodplain1.py` | 核对不同堤顶连接河道／漫滩的组织；未运行完整 ANUGA 与 HEC-RAS 对比 |
| `anuga/structures/weir_orifice_trapezoid_operator.py` | 依赖涵洞交换区，本次薄堤与建筑不引入 |

固定来源：[数据结构](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/riverwall.py)、[独立 C 函数](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/gpu_device_helpers.h)、[调用链](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/shallow_water/gpu/core_kernels.c)、[原始测试](https://github.com/anuga-community/anuga_core/blob/e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b/anuga/structures/tests/test_riverwall_structure.py)。

原始头文件保持不变，放在 `terrain-lab/third_party/anuga/`，保留上游版权声明与完整 Apache 2.0 许可。
SHA-256：`238ab97be49e0e085c21e1a5b1f1d28052a7dd22d59de7ae16642c7f3c57bd15`。
运行无需完整 ANUGA 或 C 编译器。

## 编辑与物理行为

进入“地物”，选择矩形建筑后拖动绘制；选择永久／临时堤后依次点击折线点，再点“完成堤防”。列表支持选择、更新属性、删除。
修改立即显示草稿，应用或重新模拟后写入计算。未应用时已有水情仍对应上次场景。

建筑是封闭不透水占地，保存高度、基础参考高程与屋顶高程；不改原地形，也不自动定义无人机通行约束。
当前不模拟内部进水、屋顶过流、坍塌或屋面降雨汇流。

永久堤参与预热和洪水；临时堤仅在洪水时间轴 `[start_s,end_s)` 启用，不参与基流预热，撤除时间可空。
事件精确对齐，两个 RK 阶段使用同一状态。部署／撤除改变公共边过流条件，不凭空删除或增加水。
未加入施工能力、材料消耗、渗流、侵蚀与结构失稳模型。

堤顶可选沿地面固定增高（公共边两侧较高地面作为参考），或统一绝对高程。绝对堤顶低于沿线地面会报错。
原有 `wall_x/wall_y` 仍是完全封闭墙，重叠时优先；有限堤重叠时取当前启用对象中最高堤顶，不叠加高度和流量。
棕色为永久堤，橙色为临时堤；当前时间未启用的临时堤显示虚线。

## 存档与网格

顶层 `structures` 存原始米制对象；从显示网格左上外角开始，向右 `x_m`、向下 `z_m`。
建筑字段：`id,name,kind,x_m,z_m,width_m,length_m,height_m`。
堤防字段：`id,name,kind,points_m,crest_mode,crest_m,qfactor,start_s,end_s`。
`M_structures.objects` 额外保存实际建筑格范围、屋顶高程、堤防格边与逐边堤顶，完整过程和快照均导出。

1／2／4倍格宽切换时从原始米制轮廓重新映射；建筑覆盖相交格，窄建筑可能放大。
斜堤是连通阶梯公共边，位置与长度有格宽尺度的近似。过短、沿地图外边界的堤段会报错。
薄堤不创建更小格子，但改变波速仍可能缩短 CFL 步长。

## 验证

`benchmarks/generate_anuga_weir_vectors.py` 编译未修改的原始 C 函数，生成159组固定对照：干堤顶、反流、两类淹没过渡、系数、零原通量、动量缩放上限。
`tests/test_structures.py` 检验移植一致性，并验证实际方格水动力的不漏水、解析漫顶体积、静水、旋转／反向对称、非负水深、水量守恒、事件、跨格宽与导出重载。
这些验证不表示完整方格求解器等价于 ANUGA，也不等于实际堤防工程标定。

```sh
cd terrain-workbench/terrain-lab
python -m unittest discover -s tests
node --test tests/test_structure_editor.mjs
# 有 C 编译器时重建原函数对照数据：
python benchmarks/generate_anuga_weir_vectors.py
```
