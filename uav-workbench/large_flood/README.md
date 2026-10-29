# 1.2 公里洪水观测原型

这是独立的规则扫描开发原型，未训练模型，也未修改已封存的100米研究参数。地图为240×240格、5米格距，时长1800秒；默认8架、8 m/s、25米方形感知窗口。机数可选4/8/12，速度5/8/12 m/s，宽度25/50/100 m。水情阈值0.05 m。感知窗口、无风地速是仿真假设，并非真实设备性能。

本机开发测试与计算成本见[开发核验记录](../docs/LARGE_FLOOD_DEVELOPMENT_20261029.md)。

## 使用

使用已安装 `requirements.txt` 的 Python，在 `uav-workbench` 目录：

```powershell
python -m large_flood.generate --output generated/large-flood/development-1200m.zip
python console_app.py
```

从工作台顶部选择“大地图观测”，或打开 `http://127.0.0.1:8767/#large-flood`。留空洪水文件路径使用上述开发场景；也可输入本机洪水工作台导出的过程 ZIP 绝对路径。页面只启动规则扫描，回放和运行数据保存在 `generated/large-flood/runs/`。旧服务如已在运行，需要重启才能载入新入口。

命令行运行与开发组合比较：

```powershell
python -m large_flood.simulate --input generated/large-flood/development-1200m.zip --output generated/large-flood/runs/example.json --agents 8 --speed 8 --footprint 25
python -m large_flood.compare --input generated/large-flood/development-1200m.zip --output generated/large-flood/comparison.json
python -m unittest discover -s tests -p test_large_flood.py -v
```

输出文件不得已存在。生成器的 `--seed 1` 可生成另一张地形；所有场景均为未率定的合成过程。生成器调用地形工作台的浅水求解器，记录计算设置与源码摘要；为控制开发成本使用一阶空间和Euler时间推进。它不是经过实测验证的洪水。

## 语义和限制

导入要求实际帧从0秒开始、每5秒一帧、至少覆盖1800秒；网格、字段、字节长度、单位约定及帧数值均有校验。运动每个5秒决策间隔分成不超过5米的子步，沿途使用采样时刻之前最近的实际帧（不插值、不读取未来），并分别记录观测时间及帧时间。路径按纵向条带确定性扫描，机队同步移动；碰撞检查触发时本子步全体停止。扫描完成后原地驻留。地图假设无禁飞区，不模拟加减速、风、通信延迟、电量或返航。

轨迹每5秒保存一次；观测记录每子步保存范围、二值水情、实际水深（小端float32的base64编码）、帧索引与时刻。首次覆盖是至少观测一次的格子占比；正确覆盖把未知格算为未正确，并按每5秒时刻对水深0.05米阈值下的真实类别比较；汇总为这些时刻的简单平均。观测年龄按最近使用的实际帧时间计算，只对已观测格平均。重复覆盖指格子被同一架飞机离开后再次进入，或被另一架飞机进入；连续观测不累计为多次进入。航程是各机实际路径长度之和。回放显示最新已观测二值地图而不是未来或当前真实水情。

分区与扫描参数未经过优化；对照只用于工程尺度和参数敏感性检查。开发场景、回放、比较摘要属于本地生成文件，不会进入正式实验或Git。大地图对现有学习模型和GMRF的适配、独立确认、真实传感器标定及现场有效性均未完成。
