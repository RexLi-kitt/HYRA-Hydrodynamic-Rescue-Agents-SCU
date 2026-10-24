# 实验三：多无人机闭环搜救

正式测试、配对诊断与独立记录复核已执行。
交付包与全部要求是否已核验，以 `COMPLETION.json` 为准；统计正增益不是完成条件。

- [完整实验报告](results/final_reports/20261020_001328/实验三完整报告.md)
- [全部原始统计](results/final_reports/20261020_001328/raw_statistics.json)
- [冻结配置](configs/frozen_experiment3.v1.json)
- [原始方案](docs/protocols/实验三原始方案.md)
- [开发阶段历史说明](README.development.md)

正式测试为2376个主回合和648个延迟回合；改派与数值方向各1728回合，单独标记为诊断。
五方法、三种子的15个模型在独立测试前完成训练与验证；B*只从H/E在验证集选择一次。

## 核验与复现

依赖见 `requirements.txt`，各次实际执行环境见运行目录中的 `environment.json`。移动交付包后，入口自动映射原始记录中的绝对路径，不改写证据字节。

```powershell
python scripts/verify_bundle.py --audit-primary
python run.py test
```

独立测试只能使用冻结配置并从原登记目录精确续跑，禁止据测试结果重新选择条件或基线。
