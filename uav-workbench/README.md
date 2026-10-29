# HyRA 八方向无人机工作台

新增独立的 [1.2 公里水情观测规则原型](large_flood/README.md)，支持洪水过程导入、4/8/12 架无人机扫描与回放；它是未率定的开发原型，不使用已封存的100米模型。

现行网页：<http://127.0.0.1:8767/>。同一页面提供100米双任务、三路径信号建模和原始网格过程，支持八方向运动、规则核查、已发布模型执行及逐帧回放。各模型能否运行，以对应契约的发布检查和页面提示为准。

## 启动

Python 3.10及以上。网页服务需要NumPy，模型运行另需已安装学习依赖的解释器。

```sh
cd uav-workbench
python -m pip install -r requirements.txt
python console_app.py --port 8767
```

也可运行 `python -m hyra_workbench --port 8767`，Windows保留 `启动工作台.ps1` 和原桌面快捷方式。服务只监听本机。

网页进程与模型执行进程分开。模型解释器可通过环境变量 `HYRA_LEARNING_PYTHON` 指向已有Python程序；本机仍优先使用已有的 `D:/Scripts/python.exe`。其他机器未配置该变量且不存在此路径时，使用启动网页的解释器，该环境也需安装学习依赖。不自动安装依赖、不自动训练或重选模型。

## 先看哪里

| 任务 | 入口 |
|---|---|
| 修改现行网页服务、任务分派或执行管理 | [hyra_workbench](hyra_workbench/)、[代码导航](docs/CODE_MAP.md) |
| 修改现行页面 | [workbench-scale100.html](console-static/workbench-scale100.html)、[workbench-scale100.js](console-static/workbench-scale100.js)、[workbench.css](console-static/workbench.css) |
| 查看三路径模型和使用边界 | [三路径入口说明](docs/SIGNAL_ROUTES_WORKBENCH_20261027.md)、[signal_routes](signal_routes/README.md) |
| 查看100米任务与模型发布条件 | [100米工作台说明](docs/SCALE100_WORKBENCH_20261026.md)、[训练说明](docs/SCALE100_TRAINING_20261026.md) |
| 接入洪水过程、启动其他工作台 | [启动与复现](docs/REPRODUCE.md) |
| 查找训练、审计和报告脚本 | [工具导航](tools/README.md)、[代码导航](docs/CODE_MAP.md) |
| 查看原工程验收和研究阶段记录 | [历史README](docs/HISTORICAL_README_20261028.md)、[持续验收台账](docs/CONTINUOUS_ACCEPTANCE.md) |

## 代码结构

```text
uav-workbench/
├─ console_app.py                兼容桌面快捷方式的现行入口
├─ hyra_workbench/               现行网页应用
│  ├─ server.py                  HTTP、页面和下载路由
│  ├─ profiles.py                三类场景的契约与执行入口登记
│  ├─ jobs.py                    校验、互斥、独立进程、日志与结果
│  ├─ large_flood_jobs.py        大范围规则原型的执行与共享互斥
│  └─ paths.py                   不依赖当前目录的路径
├─ console-static/              现行页面与历史页面资源
├─ console_*_contract.py         各任务的参数、模型及来源检查
├─ console_*_runner.py           学习进程中的执行与回放适配
├─ hyra_uav/                    原四动作观测与历史P0—P3研究
├─ hyra_learning/               学习、物理辅助及既有依赖适配
├─ sur_rescue/                  双层GMRF与八方向搜救环境
├─ signal_routes/               已登记三路径研究
├─ scale100_*.py                已登记100米实验、审计和发布脚本
├─ tests/、learning_tests/      可移植检查与学习检查
├─ tools/                       工程、审计和报告工具
├─ fixtures/、research/         随包开发输入与登记文件
├─ docs/                        说明、阶段结论和轻量证据
└─ generated/、.runtime/        本地运行产物和依赖，不提交
```

根目录部分研究脚本继续保留原路径，因为已有模型和封存资料按源码文件名、路径与摘要核验。现行应用已移入 `hyra_workbench`；`console_scale100_app.py` 保留兼容导入。研究源码、训练参数、权重、数据及原始回放不因本次代码整理改变。

## 统一检查

基础检查不依赖已生成的研究场景或模型。部分历史数值测试需要研究依赖；本机也支持已经存在的 `.runtime/research-packages`。

```sh
python -m pip install -r requirements.txt -r research-requirements.txt
python tools/check_workbench.py
```

检查包含观测与评价逻辑、现行HTTP行为、三类任务的进程分派及网页脚本语法。未找到Node.js时会明确跳过脚本语法检查。

已有本地研究资料和模型时，可额外执行真实回放联调：

```sh
python tools/check_workbench.py --integration
```

联调读取既有训练和开发场景，创建新的交互回放，不重新训练、不选模、不添加正式实验结果。`workbench_tests.py`、`console_scale100_checks.py`、`console_scale100_http_checks.py`、`console_signal_routes_checks.py` 的旧检查入口继续可用。

## 研究与历史入口

当前交互页面中的运行不计入正式评价。原始网格的32×32搜救模型属于12回合工程检查点；100米与三路径模型各有独立发布门槛，不跨尺度替代。三路径结果与物理耦合边界见[三路径说明](docs/SIGNAL_ROUTES_WORKBENCH_20261027.md)。

原P3两轮未通过预登记跨机制门槛，见[阶段结论](docs/P3_FINAL_20261025.md)。原四动作物理辅助首轮的负结果见[最终结果](docs/SCIML_FINAL_20261025.md)。这些历史记录与现行八方向模型分别解释。

`python run.py serve` 和 `static/` 属于历史单机四动作界面；日常使用现行入口 `console_app.py`。原 `run.py audit/demo/pilot/report/p3`、`learning.py` 和所有已有实验命令保留，复现时使用对应轮次的冻结源码与依赖，不混用当前应用整理后的源码快照。

2026年10月29日的改动范围、兼容策略和核验记录见[代码整理记录](docs/CODE_CLEANUP_20261029.md)。
