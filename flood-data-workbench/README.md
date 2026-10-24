# 真实洪水数据工作台

独立于地形生成／洪水模拟工作台，负责核查真实事件资料、显示空间网格与测站位置、按送达规则回放观测。现有求解器和无人机观测调度不作改动。

[2026-10-21成套空间数据筛选](docs/data-screening/README.md)：已实际下载并核验GEOID-Flood的影像／DEM／淹没标签，以及Pakistan 2022同一地区三个日期的淹没栅格；附真实数据预览。尚未接入当前Harvey界面。

## 运行

仓库根目录执行（服务只依赖Python标准库；前端复用同仓库资源）：

```powershell
python flood-data-workbench/server.py --port 8770
```

打开 `http://127.0.0.1:8770/`。服务只读，使用自身事件档案，不读写旧工作台的当前地图指针。导入器更新档案后需重启服务清除缓存。

## 已实际导入

事件为 **Harvey 2017 / Brays Bayou at Houston, USGS 08075000**。

- 2017-08-25 00:00 UTC至2017-09-02 00:00 UTC，含两端769条归档水位，全部有效；相邻间隔900秒，共768个间隔。
- 来源变量00065（相对水尺的水位），原单位ft，保留原始字符串和质量字段；显示米，严格乘0.3048。峰值归档值45.67 ft，2017-08-27 17:30 UTC；本工作台不以另一网页当前的水尺基准改写这些历史数值。
- 原始数据返回的WGS84点位为(-95.4121619737647, 29.6971746884659)，转到UTM 15N（EPSG:32615）。定位于地图第99行、第100列，索引从0开始。
- 2 km × 2 km真实裸地高程概览，200×200格、导出格距10米，40,000格均有效；高程范围约4.35～16.46米。来自现代USGS 3DEP多分辨率动态服务，**不是2017年历史DEM**，不是原生精度统一为10米的产品，也不能用上采样冒充1米地形。
- 没有导入全域逐时水深、流速、多时相淹没范围、2017年建筑／河床／糙率、来水边界；`solver_ready=false`。地形采集年代和垂直基准待逐幅核验，水尺基准也未解析；不做水位减DEM的水深换算。

当前实现完成了真实观测接入、地理网格表示、缺测契约、因果回放和计算规模估算。地图上是DEM、一个观测点与候选计算窗口；全域水情仍未知。尚不等于完整真实洪水场重现或残差校正闭环。

20项数据／接口检查及前端语法检查通过；浏览器检查了规模选择、延迟下的陈旧状态、事件起点、逐步前进和播放／暂停。

## 空间尺度的决定

区分资料范围、救援任务范围和水动力计算域。先保留事件的原始地理范围与分辨率，再按救援场景裁剪；不能为了矩阵小而先裁掉决定入流的上游河段。

| 用途 | 当前建议 | 作用与约束 |
|---|---|---|
| 事件资料 | 原始范围与原生分辨率保留 | 可覆盖数十公里；不代表全域用1米计算 |
| 救援概览 | 先用2×2 km，10米显示采样 | 看位置与数据覆盖，当前仅地图底图 |
| 接口与耗时检查窗口 | 256×256 m，候选1米格宽 | 65,536格；四个float32字段单帧1 MiB，求解器实际内存更大 |
| 下一档检查 | 512×512 m，候选1米格宽 | 262,144格，格数为前一档4倍；实际耗时需要运行测量 |
| 若整片2×2 km采用1米 | 4,000,000格 | 四字段单帧约61 MiB；不能由显示流畅推断求解可实时 |

256米是数据接口与计算负担的初始检查单位，不是最终救援地图大小。最终水动力域要有可解释的入流、出流、降雨和初始状态；不能把多张独立封边小图直接拼成大洪水。此版本没有实现分块耦合、粗细网格嵌套、AMR或新边界求解。

1米是候选**计算格宽**，是否具有1米有效地形信息需另外核验。观测采用原生空间支持：测站仍是点；10米淹没观测不会变成100个独立的1米真值。墙／墩与可通水地面继续沿用现有分层思路，数据包没有障碍时不自动猜出建筑。

## 时间处理

保留UTC绝对时刻，界面同时显示America/Chicago当地时间（含夏令时）。不把本机时区当成事件时区。

三种时间独立：原始观测时刻、回放时钟、求解器内部步长。水位保留15分钟采样；回放可按1分钟定位并加速播放；求解器原有数值时间步不因播放速度或观测间隔改变。本工作台不运行求解器。

- 每条记录保存观测起止时刻、变量、单位、位置支持、质量、来源值、可获得时刻。真实送达时间未知，存null，而不是假称已知。
- API按 `observed_end <= t` 且 `effective_available_at <= t` 释放。默认有效送达为观测后60秒，这只是工程假设；界面可切0秒、15分钟、1小时做敏感性检查。实际观测方式尚未研究，默认值不代表USGS当年的真实通信延迟。
- `last_modified`是数据库刷新时间，不当作2017年传输时间。使用的是现在可下载的历史归档版，可能包含事后审定／修订；因果时间筛选不能消除这种档案版本差异。
- 未来记录不交给回放接口，也不画在曲线上。不作前后帧混合，不插值生成观测。最近记录保留真实观测时刻与年龄；超过30分钟显示陈旧，不把历史值包装成新测量。若有显式缺测记录，则中断上一有效记录。
- 图上的细线仅连接相邻原生采样点供阅读，缺测／断档断线。统计和接口没有增加新观测。
- 事后洪痕、最大淹没范围和最大水深不作为某时刻的在线观测，留作事后评价；后续接入卫星产品必须保留成像时段与发布时刻。

## 复用与数据结构

- Python导入复用 `terrain-workbench/terrain-lab/raster_import.py` 的ASCII解析、北向行序与原点逻辑。
- 前端复用 `map-arrays.mjs` 的二进制矩阵解码和既有`style.css`基础主题，通过只读`/shared/`精确文件白名单访问。
- 未复用原`maps-playback.mjs`的前后帧插值，因为真实观测回放不能混入未来帧。新观测API接受可选变量，不要求伪造完整`h/u/v/max_depth`。
- `data/harvey-2017/manifest.json`记录来源URL、下载文件SHA-256、坐标系、网格、数据能力、缺口与回放规则；`observations.json`是规范化记录，`elevation.f32`和`elevation-valid.u8`是带有效区的北向矩阵。规范化档案随代码保存，小于1 MB，可离线运行。
- 原始API响应、GeoTIFF、服务元数据与中间ASCII保留在自身`.runtime/harvey-2017/`，未写入旧工作台运行目录。接口不给前端暴露原始全部未来记录、运行目录或Python代码。

## 重新导入与检查

服务使用标准库。重新采集地形需要NumPy、rasterio、pyproj；可复用旧工作台已有的`terrain-lab/.runtime/rivgraph-env`环境。

```powershell
python flood-data-workbench/import_harvey.py --terrain
python flood-data-workbench/import_harvey.py --terrain --offline
python -m unittest discover -s flood-data-workbench/tests -v
node --check flood-data-workbench/app.mjs
```

下载设置30秒单次超时、单文件12 MB上限、最多10页；有本地缓存时不重复请求。离线重建使用缓存原始响应。事件窗口和站点固定在导入脚本，其他来源应按同一契约实现单独适配器，不通过改写站点名冒充已核验事件。

## 事件资料核查

| 候选 | 已核查的官方资料 | 可支持什么 | 当前缺口／身份 |
|---|---|---|---|
| Harvey 2017 | USGS连续水位；官方事后2,123处洪痕、19张最大淹没图 | 测站洪水过程与事后洪峰约束；已导入一站连续记录 | 最大图不是逐时范围，尚未匹配当前Brays工作区 |
| Harvey 2017遥感 | NASA/JPL ARIA分别公开不同日期的SAR洪水代理图；NASA记录9月1～4日UAVSAR逐日飞行 | 说明存在多时相空间观测路线 | 不是全域每秒真值；原始科学产品、有效区、Brays覆盖与发布时间待核验，尚未导入 |
| Ahr 2021 / GFZ | 10米DEM、糙率、建筑、边界CSV；直到测站损毁前的实测水位 | 有较完整水动力输入，适合后续对接 | 后段流过程为预测／按估计峰值缩放；最大水深／速度为模型输出，不当实测真值 |

由此可以确认：**确实有真实事件的详细过程记录，但通常是连续点记录、若干遥感快照和事后峰值的组合，不是每格每秒全变量测量。** 后续先匹配同一片区域的这些数据，再判断它能支持哪种残差与验证。

官方来源：

- [USGS Harvey调查](https://www.usgs.gov/centers/oklahoma-texas-water-science-center/science/post-hurricane-harvey-assessment)
- [USGS Harvey空间产品数据发布，CC0](https://www.usgs.gov/data/data-used-characterize-peak-streamflows-and-flood-inundation-resulting-hurricane-harvey)
- [USGS连续数据接口](https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous)
- [3DEP动态高程服务元数据](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer?f=pjson)
- [NASA/JPL 2017-08-27 SAR洪水代理图说明](https://www.jpl.nasa.gov/images/pia21928-extent-of-texas-flooding-shown-in-new-nasa-map/)
- [NASA Harvey响应时间线，含逐日UAVSAR](https://ntrs.nasa.gov/api/citations/20180003185/downloads/20180003185.pdf)
- [GFZ Ahr数据说明及CC BY 4.0许可](https://datapub.gfz-potsdam.de/download/10.5880.GFZ.4.4.2021.005nAEUn9/2021-005_Apel_Data-Description.pdf)

USGS采集与转换由`import_harvey.py`记录具体URL与哈希；本地附带的是公开科学数据的小范围子集。未转载NASA展示图片或GFZ数据。
