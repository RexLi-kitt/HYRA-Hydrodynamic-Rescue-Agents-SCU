# 无人机工作台代码导航

整理日期：2026年10月29日。日常入口为 `python console_app.py` 或 `python -m hyra_workbench`，默认地址 <http://127.0.0.1:8767/>。

## 现行应用

| 要修改的行为 | 主要文件 | 职责 |
|---|---|---|
| 页面、HTTP状态及下载入口 | [server.py](../hyra_workbench/server.py) | 路由、请求大小与来源检查、错误响应、仅本机监听 |
| 场景属于哪一研究方案 | [profiles.py](../hyra_workbench/profiles.py) | 同一登记表对应场景前缀、原契约和执行脚本；无训练导入 |
| 执行、并发和运行日志 | [jobs.py](../hyra_workbench/jobs.py) | 原契约校验、一次一项任务、独立Python进程、失败日志、回放摘要 |
| 工作区与页面资源位置 | [paths.py](../hyra_workbench/paths.py) | 按文件位置定位，避免依赖调用目录 |
| 桌面快捷方式与旧导入 | [console_app.py](../console_app.py)、[console_scale100_app.py](../console_scale100_app.py) | 保留旧入口，转入同一现行实现 |
| 页面内容、交互和样式 | [HTML](../console-static/workbench-scale100.html)、[JS](../console-static/workbench-scale100.js)、[CSS](../console-static/workbench.css) | 当前实际提供的三份页面资源；本次未更改内容 |

请求流程：页面提交任务 → `profiles` 选择原契约 → `jobs` 校验与启动对应worker → 原worker再次核对任务并保存回放 → `server` 返回回放地址。网页进程不执行训练；模型发布门槛仍在各自原契约内。

## 三类场景的原契约与执行

| 场景标识 | 参数、来源与模型检查 | 独立执行脚本 | 任务范围 |
|---|---|---|---|
| `scale100:` | [console_scale100_contract.py](../console_scale100_contract.py) | [console_scale100_runner.py](../console_scale100_runner.py) | 100米自主探索／探索—搜救 |
| `signal-routes:` | [console_signal_routes_contract.py](../console_signal_routes_contract.py) | [console_signal_routes_runner.py](../console_signal_routes_runner.py) | 三路径及共同对照，仅探索—搜救 |
| 其他已登记场景 | [workbench_api.py](../workbench_api.py)、[workbench_tasks.py](../workbench_tasks.py) | [console_workbench_runner.py](../console_workbench_runner.py) | 原始网格规则核查及匹配的工程检查点 |

场景匹配不代表参数有效；不存在的场景、未发布模型、错误任务与未匹配配置仍由原契约拒绝。三类worker的命令和数据格式保持原状，原始网格worker接收已校验记录，另外两类接收原请求并再次校验。

## 研究代码的位置

| 目录或文件族 | 内容与查找起点 |
|---|---|
| `hyra_uav/`、`run.py` | 原四动作观测、认知记录、规则、评价、P0—P3研究；`run.py serve` 属于历史网页 |
| `hyra_learning/`、`learning.py` | 多机学习、网络、物理辅助、观测与本地依赖适配 |
| `sur_rescue/` | 双层GMRF、八方向环境与学习 |
| `signal_routes/` | 三路径估计、登记、训练和评价，见[研究说明](../signal_routes/README.md) |
| `scale100_env.py`、`scale100_train.py`、`scale100_backend.py` | 100米环境、六组训练及显式训练版本登记 |
| `scale100_aux_*`、`scale100_review_*` | 各轮辅助分支、开发审查和已登记训练变体 |
| `scale100_formal*`、`scale100_evaluate*` | 正式训练队列及开发／留出评价 |
| `scale100_*audit*`、`scale100_*checks*` | 轨迹、发布、训练机制和证据核查 |
| `scale100_publish.py`、`scale100_report.py`、`scale100_round_completion.py` | 已登记发布、报告及当轮证据封装 |
| `tools/` | 按用途查[工具导航](../tools/README.md) |

这些研究文件暂不整体搬目录：已有模型元数据、源码清单和封存归档按相对路径及摘要核对，部分脚本还按自身文件位置寻找输入。移动或改写需要作为新的研究版本登记。本次将可独立整理的现行HTTP实现集中到新包，保留原研究文件的字节与路径。

旧封装脚本服务于对应历史轮次；历史归档里的HTTP实现是当时的完整版本。本次新应用的源码清单另见[核验记录](evidence/20261029-code-cleanup/verification.json)，不要将新包与旧归档混装后称为原轮次复现。

## 检查与依赖

| 检查范围 | 入口 | 是否依赖本地研究产物 |
|---|---|---|
| 统一基础检查 | `python tools/check_workbench.py` | 否；历史数值测试需要研究依赖 |
| 本次应用回归 | `python -m unittest discover -s tests -p test_workbench_app.py -v` | 否；使用临时worker验证真实进程和HTTP行为 |
| 原环境和评价 | `python -m unittest discover -s tests -v` | 否 |
| 原网格与100米联调 | `python -m unittest -v workbench_tests console_scale100_checks console_scale100_http_checks` | 是 |
| 含三路径的全部联调 | `python tools/check_workbench.py --integration` | 是，且需要已有学习解释器 |
| 学习模块检查 | `python -m unittest discover -s learning_tests -v` | 依具体测试而定，需学习依赖 |

基础网页使用 `requirements.txt`，历史研究测试使用 `research-requirements.txt`，学习使用 `learning-requirements.txt`，绘图使用 `report-requirements.txt`。不把所有依赖混装进一个环境；统一联调会按已有配置启动学习解释器，并先调用项目已有的依赖加载函数。

`generated/` 保存本地输入、训练、检查和回放；`.runtime/` 保存本地依赖。二者继续受Git忽略规则管理，整理代码不会清除或重建其中的历史资料。
