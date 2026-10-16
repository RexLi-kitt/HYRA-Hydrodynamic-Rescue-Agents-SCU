# 数据字段草案

实际序列化已实现：每事件NPZ及同名JSON元数据，外加.solver.npz原始质心参考/名义状态。load_events校验形状、有限值、单位、时间次序及家族划分；pipeline核对每次起报初始值来自独立名义状态，保存数据指纹。

NPZ字段：time_s[T]、xy_m[R,2]、truth_depth_m[T,R]、release_s[F]、forecast_valid_s[F,V]、forecast_depth_m[F,V,R]、nominal_time_s[NT]、nominal_depth_m[NT,R]。JSON保存event_id/event_family_id/split、物理单位、参数、求解器版本、映射索引和原始状态文件名。公开manifest中的文件路径相对manifest本身解析。

## 完整事件

| 字段 | 含义 |
|---|---|
| event_id | 当前事件版本的唯一标识 |
| event_family_id | 同一参考洪水及其预测/观测扰动版本的共同标识 |
| source | 原始数据或求解器来源 |
| time_s | 明确参考起点的物理时间，单位秒 |
| region_id / xy_m | 固定区域编号及空间位置，坐标单位米 |
| truth_depth_m | 参考水深，供传感模拟及评分使用 |
| forecasts | 已发布预测记录，保留发布时间、有效时间、版本与水深 |
| metadata | 地形、边界、求解器版本、网格映射和预测输入口径 |

预测水位与水深不能混用。不同事件采用不同网格时，必须先确定一致的区域对应规则。

## 观测记录

每条记录保留 `region_id`、`valid_time_s`、`arrival_time_s`、`observed_depth_m`、`matched_forecast_id`、`prior_depth_m` 及 `residual_m`。

残差定义为同期观测水深减去同位置、同物理量、同有效时刻的先验预测。首轮固定提前量20s，选择发布时间恰为有效时刻减20s的唯一版本；无匹配保持缺失，在线不替换为事后版本。正式生成每10s滚动起报，预测60s。

当前表示仅使用已经到达、有效时间不晚于当前时刻、且未超过最大年龄的记录。无匹配先验的观测保留缺失状态，不使用事后预测补齐。

## 表示和结果

表示应同时返回失配证据、有效性掩码和必要的覆盖信息。无支持的零占位不表示预测准确。区域联系及时间参数应记录拟合事件与冻结版本。

参考真值与允许策略使用的数据分开传递。固定回放的指标属于信息诊断；任务收益需要另行保存动作、服务开始/结束、失败原因和完整任务分母。
