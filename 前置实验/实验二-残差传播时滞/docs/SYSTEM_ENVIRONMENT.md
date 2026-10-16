# 系统本地环境配置

2026-10-14按用户要求配置系统本地环境；随后用户要求继续执行，首轮GPU验证已于2026-10-15完成。项目独立环境已删除。所有命令使用系统 `python`，不激活虚拟环境。实际结果见IMPLEMENTATION_STATUS.md和results/latest_pilot_report.json。

## 解释器与GPU

- 系统命令入口：`D:\Scripts\python.exe`，Python3.14.2，prefix=base_prefix=`D:\`。
- 根目录入口：`D:\python.exe`，同一系统安装。
- NVIDIA GeForce RTX5060 Laptop GPU，约8GB，驱动591.91。
- PyTorch2.14.0+cu130，运行库CUDA13.0，包含sm_120；不需要另装CUDA Toolkit用于预编译包训练。
- ANUGA4.0.1、pymetis2025.2.2、NumPy2.4.6、SciPy1.18.1、pytest9.1.1；完整实际清单见 `results/setup/system_packages_after.json`。

系统Python保留既有用户site-packages，SciPy/pandas等部分包来自用户目录，不属于项目虚拟环境。没有移动或重装这些既有包。

## PyTorch DLL故障与修复

原torch2.14.0+cpu导入报WinError1114，c10.dll初始化失败。改装官方CUDA包后仍报错。只预加载系统新版MSVCP140即可成功，定位为Python目录旧运行库遮蔽。

仅替换以下两个文件为Windows系统现有14.50.35719.0版本：

- `D:\msvcp140.dll`
- `D:\Scripts\msvcp140.dll`

原文件14.29.30154.0的备份：`results/setup/runtime_backup_20261014/D_root_msvcp140.dll` 与 `D_Scripts_msvcp140.dll`。没有修改Windows系统DLL或显卡驱动。最终验收直接import torch，无预加载补丁。

CUDA包来源：https://download.pytorch.org/whl/cu130 ，具体安装日志 `results/setup/system_torch_install.log`。

## ANUGA / pymetis

ANUGA使用官方PyPI4.0.1 Windows包，与warehouse开发commit区别保留。pymetis没有适用Windows wheel，因此从官方PyPI2025.2.2源码构建；本机GCC16.2.0位于`D:\Toolchains\GCC\mingw64\bin`。

源码本机兼容处理（不改变图分割算法）：

1. `gk_arch.h`：MinGW分支不包含不存在的POSIX sys/resource.h。
2. `gk_getopt.h`：参数名 __argc / __argv 改为gk_argc / gk_argv，避免MinGW全局宏冲突。
3. CFLAGS=`-DUSE_GKREGEX -std=gnu99`，CXXFLAGS=`-DUSE_GKREGEX`，使用自带正则及C99。
4. 在ASCII临时路径构建，PYTHONUTF8=1；delvewheel打包MinGW DLL，普通导入无需更改系统PATH。

原源码压缩包、源码与日志位于results/setup/build与results/setup；可安装wheel位于 `results/setup/wheels/repaired/pymetis-2025.2.2-cp314-cp314-win_amd64.whl`。缺mpi4py仅表示使用串行接口，当前任务不需要MPI。

## 旧PyMARL依赖兼容

已核验EpisodeBatch、ReplayBuffer、QMixer和QLearner导入。没有执行PyMARL的StarCraft入口，也未安装SMAC/StarCraft游戏；后续自建搜救环境接入核心组件。

Sacred0.8.7在Python3.14调用已移除的pkgutil.find_loader：系统包utils.py改用importlib.util.find_spec并补充该import，原文件备份 `results/setup/sacred_utils_before.py`。这是本机兼容修改，pip重装Sacred会覆盖它。

setuptools固定80.9.0，保留Sacred依赖的pkg_resources，同时满足torch要求>=77。原版本84.0.0记录在system_packages_before.json。弃用提示不影响导入。

现代TensorBoard2.21.0已安装；本轮试装但与当前protobuf不兼容的tensorboard-logger0.1.0已卸载。旧PyMARL的可选setup_tb仍引用旧包，后续适配为torch.utils.tensorboard，当前核心导入无该依赖。不可据此宣称未经适配的原main.py可直接训练。

## 验收与执行状态

运行 `python scripts/check_system_environment.py`：仅检查导入、CPU张量、CUDA矩阵乘与反向、pymetis四节点分割、ANUGA0.2s常水深求解以及pip check。未加载研究数据或启动训练。输出 `results/setup/system_environment_check.json` 和日志。

在先前暂停前，独立环境官方示例和12个洪水事件生成已完成，保留于 `results/runs/20261014_231607_782244_events/`，不作为正式首轮数据。用户后来要求继续执行，已使用系统环境重新生成18个独立事件，完成四组回放、GPU训练和小规模收益/边界评价。结果以IMPLEMENTATION_STATUS.md和最终报告为准。
