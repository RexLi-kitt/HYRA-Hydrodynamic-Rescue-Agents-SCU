# Merewether 物理实验接入

这是带独立流速测量的物理实验验证入口，不是合成世界生成，也不是完整2007年洪水事件复现。当前完成首轮800秒对照，误差仍明显，见[结果与限制](experiments/2026-10-20/merewether/REPORT.md)。

## 数据与配置

采用 Ayog、Sharifian 发布的 [LISFLOOD-DG2 simulation set-up files and additional datasets](https://zenodo.org/records/6914888)，CC BY 4.0；只取 Test 4。读取其0.175米 DEM、边界、采样位置和40个最大流速／方向测量。公开[重建 DEM 记录](https://zenodo.org/records/5069224)提供另一地形发布入口；本轮实际运行统一使用前一数据包，避免混用不同版本。

主配置 n=0.042 来自[论文](https://doi.org/10.1061/JHEND8.HYENG-13244)在水位点4的既有率定，本项目未重新校准；n=0.02 为早期教程敏感性对照。实验路基表面不套用森林、草地等合成群系糙率，建筑与桥墩保留在 DEM。

`merewether.fetch()` 核验 Zenodo 记录中的包 MD5，提取后记录逐文件 SHA-256。缓存位于 `terrain-lab/.runtime/merewether/source/`，原始记录与文件保留。旧 WRL 数据站本次连接超时，因此没有获得水位原始表；不使用 ANUGA 同名历史案例的5个水位点替代它。

## 运行

使用已有兼容环境（Python、NumPy；完整计算另需现有 Numba 加速与 Matplotlib）。从 `terrain-workbench/terrain-lab` 运行：

新环境可从仓库根目录安装 `python -m pip install -r terrain-workbench/requirements-validation.txt`。RivGraph 不是此实验的依赖；下方环境路径沿用本机已有环境，也可换成安装上述依赖的 Python。

```powershell
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/run_merewether.py --factor 4 --duration 800 --wall-budget 600 --activate
# 若达到预算但尚未完成，保留状态续算，避免重跑：
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/run_merewether.py --factor 4 --duration 800 --wall-budget 240 --resume --activate
# 同区域的网格及既有糙率敏感性：
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/run_merewether.py --factor 8 --manning .042 --wall-budget 240
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/run_merewether.py --factor 8 --manning .02 --wall-budget 240
# 检查已完成结果并更新精简报告：
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/report_merewether.py
```

`factor` 是导入时相对0.175米原始 DEM 的整倍合并，不是后续工作台的1／2／4倍选项；4对应0.7米，8对应1.4米。块内最大高程优先保留细障碍，但会扩大部分桥墩和墙体，必须作为精度限制记录。原始高程不取整；作者已换算为原型尺度，不再缩放。

每次预算是单段墙钟上限，达到时保存部分结果；完整观察窗未完成时明确标为诊断，不能作为完整验证。续算校验原始数据、网格、糙率、观测采样和数值内核哈希，恢复水深、动量、最大值与水量账本；不同配置不能混接。

## 查看地图与结果

生成的实验地图及 `scenario.json` 位于 `terrain-lab/generated/merewether-f4-max-n0p042/`，原始地形文件未被覆盖。`--activate` 更新本地 `current.json`，原指针备份在结果目录的 `previous-current.json`。

启动或重启工作台服务后，打开：

```text
http://127.0.0.1:8766/maps.html?fresh=1#merewether-f4-n0p042
```

本次另启动8767端口用于页面验证，已有8766服务未被打断。若8766仍运行旧代码，应重启服务再读取新地图；代码更新不等于已有服务进程自动重载。

页面可读取完整保存的实验过程，播放、拖动、查看实际糙率并导出。导入地图的推荐来水、糙率和边界默认值会自动加载；修改它们后即成为新实验，不应继续沿用原结果的精度结论。

数值结果在 `.runtime/merewether/results/`：每组有 `result.json`、`points.csv`、配置、时间序列、最终场和对照图。逐帧水深／速度及可重放归档保存在 `generated/merewether-f4-n0p042/maps/`；浏览器恢复无需重新计算。

## 可复用的外部高程导入

`raster_import.py` 支持 ESRI ASCII 栅格，保留浮点高程、来源原点与格宽；工作台使用局部 x 向右、z 向下坐标。

```powershell
& .runtime/rivgraph-env/Scripts/python.exe raster_import.py path/to/terrain.asc --name external-demo --manning .035 --activate
```

一般导入默认无积水、无降雨、无基流、统一糙率、开放边界；这些不是自动推断出的真实事件条件。未知高程作为封闭占地，不用插值制造可过水通道；采用角点／中心点原点时做半格换算。可选 `--factor` 与 `--aggregation mean|max`，并在元数据记录补齐尺寸。

导入描述 `raw/imported.json` 保存字段、坐标／单位说明、推荐场景与边界交点。实验专用边界通过 `merewether.make_case()` 写入；通用格式也支持外部提供的静态封闭格、边墙与来水配置。局部 x/y 不能直接冒充投影坐标；源数据未给出的群系、植被、地表材料及自然河网不予生成。

图层创建先加载实验边墙，再允许明确的编辑覆盖。进一步合并计算格会改变边界与障碍表示；已经导入的保形高程处理和工作台的面积平均处理是两次不同操作，评价时应分别记录。

## 本轮验证边界

已做数值守恒、入口实际38米宽度与累计流量、非负水深、来源测点一致性及实际糙率检查。0.7米主配置速度RMSE约0.738 m/s，不能宣布精度通过。完整结果保留所有未进水点和方向无定义点的处理规则。

包内没有实测水位值。逐格核对已确认：0.7米最高值合并将前三个水位点的原始测点地面抬高约3.53、3.72、5.24米，导致测点落在扩大的障碍占地内；不自动平移来匹配，当前水位图仅显示计算结果。未检验0.35米／0.175米网格收敛，也未跑完整 ANUGA 或 LISFLOOD-FP 作独立数值交叉比较。

## 逐测点诊断

[详细报告、全40点CSV及诊断图](experiments/2026-10-20/merewether/diagnostics/REPORT.md)区分持续过快、起涨峰值高估、方向错误和漏淹，保留全部固定点。新增1.4米均值合并对照只改变地形聚合规则，约109秒完成800秒窗口，仍未消除明显偏差，不作为正式修复。

在 terrain-lab 目录复现新增对照及报告：

```powershell
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/run_merewether.py --factor 8 --manning .042 --aggregation mean --duration 800 --wall-budget 240 --output .runtime/merewether/diagnostic-runs
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/analyze_merewether.py
```

分析还需要原三组已保存的结果、逐时记录、最终场和原始DEM。主指标仍为0～800秒最大流速；750～800秒均速、2米邻域和模型到达时间仅作诊断，不能当成实验观测或替换主指标。最大值组沿用原回放编号，均值组文件及回放增加 `-mean` 后缀，避免覆盖。


## 地面与障碍分层（实验版）

新增 `--layered` 导入方式：原始表面保留为来源数据，计算地面与独立不透水障碍分别保存。当前Merewether包没有裸地和独立墙／墩轮廓，障碍由高程特征推断；障碍下方地面为最近已知地面的估计，不能当作实测裸地。推断规则、估计标记、原始分辨率图层随生成结果保存。

粗格地面只平均原始非障碍像素。障碍按覆盖多数取整格，小对象保留一个最大重叠格；这仍是有面积、宽度与连通误差的整格近似。`obstacle_fraction`用于几何检查，不参与比例通量或储水计算。已有 `wall_x/wall_y` 接口可接入可靠墙线，本包没有独立墙线，因此不从建筑边缘另造墙。

在terrain-lab目录运行：

```powershell
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/run_merewether.py --layered --factor 4 --manning .042 --duration 800 --wall-budget 900 --output .runtime/merewether/layered-results
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/inspect_merewether_layers.py --phase geometry
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/inspect_merewether_layers.py --phase patch --patch-budget 120
& .runtime/rivgraph-env/Scripts/python.exe benchmarks/inspect_merewether_layers.py --phase report
```

需要 `requirements-validation.txt` 中的SciPy、Matplotlib及原有加速依赖。`factor=1/2/4/8`对应0.175/0.35/0.7/1.4米；默认实验仍不自动切换到分层。分层结果及回放增加 `-layered-v1` 后缀，保留旧基线。分层地图不允许工作台二次合并，换格宽需要从原始分层数据重新生成。

主结果在 `.runtime/merewether/layered-results/f4-n0.042-layered-v1/`，含实际固体与边墙、最终状态、逐时记录和运行代码快照；生成地图在 `generated/merewether-f4-layered-v1-n0p042/raw/`，其上级有 `native-obstacle-layers.npz`。地图页提供“阻水占地”图层，洪水页同步显示独立地面与不透水占地，提示轮廓推断及格宽限制。

[分层报告、几何检查、原始格宽局部检查和全40点同条件对照](experiments/2026-10-20/merewether/layered/REPORT.md)。局部检查使用人工封闭溃坝边界，不用于实验精度评分，也没有与全域粗格双向耦合。当前没有实现部分通水或全域局部加密算法。
