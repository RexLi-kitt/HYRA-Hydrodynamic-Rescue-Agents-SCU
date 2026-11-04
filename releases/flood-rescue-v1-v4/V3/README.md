# HyRA V3 · 贵州三类险情与冻结模型迁移

本目录可以单独作为 GitHub 仓库。V3 是本次发布用的实验分支编号。原实验名称为 `guizhou_mixed_rescue`，两份外部团队 Word 的版本号、模型与成绩归属外部团队。

## 场景与方法

4 架机、40 人、40 米圆形视野，任务 300 秒，水动力从第 900 秒热启动。16 人漂移落水，8 人浅滩固定受困，两个村落各 8 人；每人 240 秒期限。报警在第 0/30/60 秒公开。

H 按报警和公开下游走廊沿河排查；T 使用冻结 T0＋K1；G 使用冻结 TG＋K1。K1 在区域中心及四个象限中心中选点。三组共用报警、外生轨迹、施救调度和避碰。

## 已完成的独立测试

父水场为 184001–184100，每图 2 个布局。所有策略共享同布局的潜在人员轨迹。GRU 的三个训练种子为 11、29、47。

| 方法 | 每场按期投放 | 完成率 |
|---|---:|---:|
|H 报警沿河规则|30.33/40|75.83%|
|T：T0＋K1|32.88/40|82.19%|
|G：TG＋K1|27.25/40|68.11%|

G−T 为 −5.630 人/场，父图配对 95% 区间 [−6.703, −4.558]。本轮不支持迁移后的 GMRF 部署。贵州报道用于启发合成险情，河床、人数和轨迹均为模拟设定。

各组成绩在自身协议内比较。不同版本的传感范围、个人期限和任务难度不同，不能把四版的完成率直接排序。

## 安装与运行

发布核验使用 Python 3.14.2、NumPy 2.4.6、SciPy 1.18.1、PyTorch 2.14.0+cu130。依赖文件使用 PyTorch 公共版本号；冻结模型在 CPU 上核验。安装对应平台的 PyTorch wheel 后执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd uav-workbench
python release.py verify
python release.py smoke
python release.py run --parent 184001 --layout 0 --seed 11 --arms H T G --compare-archive
```

Linux/macOS 使用 `source .venv/bin/activate`。首次运行会由原浅水求解器生成整张 0–1500 秒的水场，耗时高于模型 smoke。结果写入 `uav-workbench/generated/release-runs/`，水场写入 `uav-workbench/generated/flood-dqn-v1/maps/`，原有完整归档位于 `results/`。

`--compare-archive` 核对人员事件、终局指标以及原档包含的决策记录；耗时随硬件变化。`--parent`、`--layout`、`--seed` 可逐场切换，完整确认批次按上述父图范围及布局 0、1、种子 11、29、47 运行。确定性规则每布局只需一次。不同种子、半径和方向分别写入独立日志；同一参数再次运行会覆盖该次本地日志。

V2 支持 `--radius 40/60/100`；V3 的环境固定 40 米；V4 的环境固定 60 米，并支持 `--orientation main/rot/mirror`。V3/V4 的视野由各自协议固定，CLI 的 radius 参数仅对 V2 生效。

## 训练入口

V2/V4 提供原监督训练逻辑。先生成 16 张拟合与校准水场，再生成训练轨迹并训练：

```powershell
python release.py prepare-training
python release.py train
```

V2 原训练实现要求 CUDA PyTorch；V4 支持 CPU 或 CUDA。V3 复用 V2 权重，新增训练为零。V3 训练命令会提示在 V2 目录完成。V2 发布的 A-gate 取自原已完成开发检查，用于原训练采样入口，原检查记录随结果保留。新训练产物写入 generated，并优先于随包冻结模型加载；评估发布模型时使用干净目录。

本次整理核验了冻结权重的 CPU 推理及完整场景回放。训练入口随源码提供，本次没有重新执行完整训练或整批独立测试。用于研究复跑时应记录新权重与运行环境。

## 内容与证据

- `uav-workbench/` 包含本实验和所需公共环境、模型及规划代码。
- `terrain-workbench/terrain-lab/` 包含原浅水求解器及数值函数。
- `models/` 包含冻结权重和模型标准化参数，SOURCE_MANIFEST.json 记录哈希。
- `results/` 包含全部主表、逐场逐人记录、统计、地图种子、封存记录与实验报告。
- `results/cases/` 保留原主测试场景的序列化记录。
- `docs/original-*.md` 保留原实验说明，部分命令依赖原完整研究工作区；独立包运行以本 README 为准。
- `SOURCE_MANIFEST.json` 记录原文件与发布文件哈希。为移除无关实验依赖，发布版重定向了 IO、模型加载和通用保存函数，历史封存记录仍保留原研究代码哈希。
- `PACKAGE_MANIFEST.json` 记录本次完整上传包的文件哈希。

人员记录中的发现、开始服务及完成投放时间以秒计，`-1` 表示未发生，不能按零秒计入时间均值。详细列定义与分母见原实验说明和 results 报告。不同训练种子按父图聚类进行统计。

## 许可与上传

沿用原 HyRA 主项目附带的 Apache-2.0 许可，见 LICENSE。数据为本项目合成实验产物。未打包论文全文、外部 Word、本机环境及大型水场数组。

在 GitHub 创建空仓库后，在本目录运行以下命令，将 `<仓库地址>` 替换成自己的远端地址。

```powershell
git init
git add .
git commit -m "Publish HyRA V3 experiment, models and results"
git branch -M main
git remote add origin <仓库地址>
git push -u origin main
```
