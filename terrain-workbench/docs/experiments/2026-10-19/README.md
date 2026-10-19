# 2026-10-19 短时洪水试验归档

本目录保存本轮合成地图试验的配置、时间序列、统计、数值核验和已检查图表。完整工作记录见[开发日志](../../../../DEVLOG.md)。

- [1／2／4米网格对照](grid-observation/REPORT.md)：同一连续预览地图、统一 n=0.035，比较合并格宽对脉冲响应的影响。
- [完整地表与分类糙率对照](surface-comparison/REPORT.md)：所有计算为1米，区分完整地表流程与分类糙率的影响。

两个试验使用种子3346354700、原点(8512,9536)、256×256米范围。原始地图配置见 `source-preview-config.json`，生成器预设见 `source-preset.json`，完整地表请求见 `full-surface-request.json`。归档请求仍保留当时的相对输出／预设文件名；重新生成时需将其解析到实际工作目录，不应直接覆盖已有世界。

`grid-*.json`、`full_uniform.json` 和 `full_classified.json` 保存时间序列；各组 `summary.json` 和 `validation.json` 保存统计与检查。完整地表的 `comparison-config.json` 还记录七个原始字段一致性和当时的源码哈希。测得耗时受机器与环境影响，不是跨机器性能承诺。

## 运行与复查

脚本位于 `terrain-workbench/terrain-lab/benchmarks/`：

1. `observe_grid_flood.py` 从当前原生1米、边长不超过256格的地图运行带预算的1／2／4米配对试验；`--max-duration 70` 限定本次相同的最大观察时长，机器较慢时仍可能缩短，实际时长以输出配置为准。
2. `compare_surface_roughness.py` 接收完整地表的 `--terrain-id`、上一步本地输出的 `--baseline` 与 `--output`；比较前检查固定原始字段、生成预设和求解器哈希。需使用相同种子／区域的预览与完整图；代码变动导致哈希不符时应重新生成匹配基线。
3. `inspect_surface_comparison.py` 核验本地完整对照输出，并生成地表群系／糙率图。

基础依赖见工作台 requirements，RivGraph 见 `requirements-rivgraph.txt`；观察脚本还使用现有 Numba 加速和 Matplotlib 绘图环境。本次使用本地隔离 Python 3.12、NumPy 1.26.4 环境。

本目录没有生成世界、Java／Python 环境或逐帧 NPZ；原始二进制高程、掩膜和 NPZ 在本地 `generated/`、`.runtime/` 下，被版本控制忽略。因此这里可以直接检查曲线和统计，重新计算必须先生成原始地图并运行配对基线，不能把克隆后的归档目录直接当作完整 `--baseline` 输入。

本次没有真实观测。1米只是数值参照；糙率为未率定初值，20秒预热不保证稳态，70秒观察不覆盖完整退水。
