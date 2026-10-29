# 工作台运行资料

这些文件随 Git 分发，供现行工作台直接执行既有模型，无需重新训练。

包含36个100米模型、24个三路径模型、两个原始网格工程检查点、登记场景、发布校验资料和三路径封存时的配套源码。正式确认场景保留在数据清单中，界面仅列出训练与开发场景。

`manifest.json` 记录220个文件的大小和SHA-256。JSON、模型及冻结源码保持原始字节；不要为了修改历史绝对路径而重写文件。`runtime_assets.py` 在运行时解析其仓库位置。

这里提供交互运行所需资料，不包含全部训练轨迹、优化器状态及历史回放。交互结果不计入正式评价；模型和合成洪水的科学限制仍以原说明为准。

在 `uav-workbench/` 中运行：

```sh
python -m pip install -r requirements.txt -r learning-requirements.txt
python tools/check_runtime_assets.py --replay
python console_app.py --port 8767
```
