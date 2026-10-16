# 第三方来源

`third_party/pymarl` 来自原工作区已有的 PyMARL 源码，保留 Apache-2.0 许可证。
原快照没有 Git 元数据，不能追认提交号；本包的 SHA256SUMS.json 固定实际使用的逐文件内容。
PyMARL 仅用于 QMIX 组件，本包不依赖 StarCraft II；适配器在内存中兼容现代 PyTorch tuple 索引。

`third_party/filterpy` 来自既有固定下载，提交及归档校验信息见该目录 SOURCE.json，许可证原样保留。
批量普通滤波实现由本包源码提供；原生 FilterPy 用于独立等价核验。
ANUGA 4.0.1 从 Python 包安装，仅在重新生成水动力数据或运行求解器测试时需要。

本项目自有源码尚未选择公开许可证；上传 GitHub 本身不会替作者授予开源使用许可。
