"""Full-matrix result reporting and byte-preserving reproducible delivery."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np

from .artifacts import write_json,sha256,TZ
from .protocol import load_protocol
from .freeze import verify_frozen
from .statistics import paired_effect


def conclusion(effect,all_at_ceiling,numerically_stable):
    if not numerically_stable:return '收益方向随数值设置改变，尚不能作出方法优劣结论；须继续处理数值问题。'
    if all_at_ceiling:return '各方法均达到100%完成率，当前任务存在天花板，未提供可检验的额外救援收益。'
    if effect['primary_positive']:
        return 'P相对验证集选定的B*具有正增益证据。'+('达到增加3个百分点的工程目标。' if effect['engineering_target_3pp_met'] else '尚未达到增加3个百分点的工程目标。')
    return 'P相对B*的配对区间未能支持稳定正收益，不能宣称HyRA提高了救援完成率。'


def window_timing_context(rows):
    last=[r['summary']['last_successful_service_s'] for r in rows if r['summary']['last_successful_service_s'] is not None]
    initial_safe=0;finished_early=0;margins=[]
    for row in rows:
        windows=[t['true_safe_start_intervals'] for t in row['outcome_analysis']['targets']]
        if not windows or not all(w and abs(w[0][0])<1e-8 for w in windows):continue
        initial_safe+=1
        summary=row['summary']
        if summary['successful_services']!=summary['hidden_targets']:continue
        first_close=min(w[0][1]+12. for w in windows)
        margin=first_close-summary['last_successful_service_s'];margins.append(margin)
        finished_early+=margin>1e-7
    return dict(episodes=len(rows),all_hidden_targets_initially_safe_episodes=initial_safe,
        all_targets_served_before_any_initial_true_window_closed_episodes=finished_early,
        last_successful_service_s_quantiles=np.quantile(last,[0,.5,.95,1]).tolist() if last else None,
        all_targets_served_initial_window_margin_s_quantiles=np.quantile(margins,[0,.5,.95,1]).tolist() if margins else None,
        posthoc_truth_only_not_controller_input=True)


def opportunity_context(method_rows,effect,baseline):
    reference=next(r for r in method_rows if r['method']==baseline)
    total=reference['hidden_targets_total'];served=reference['successful_services_total']
    if total<=0 or not 0<=served<=total:raise ValueError('Invalid baseline service denominator')
    headroom=100*(total-served)/total
    paired=np.asarray(effect['family_seed_differences_pp'],float)
    return dict(baseline_unserved_service_instances=total-served,baseline_hidden_target_instances=total,
        maximum_service_rate_gain_percentage_points=headroom,
        engineering_target_3pp_observed_headroom_available=headroom>=3.,
        all_layout_averaged_family_seed_differences_zero=bool(np.all(paired==0)),
        layout_averaged_family_seed_cells=int(paired.size))


def collect_statistics(frozen,primary,fixed,counter,numerical):
    groups=[];conditions=[]
    for group in primary['groups']:
        side,delay=group['side'],group['delay_s'];rows=[r for r in primary['rows'] if r['side']==side and r['delay_s']==delay]
        numeric=next(g for g in numerical['groups'] if g['side']==side and g['delay_s']==delay)
        reassignment=next(g for g in counter['groups'] if g['side']==side and g['delay_s']==delay)
        method_rows=[]
        for method in group['rates']:
            chosen=[r for r in rows if r['method']==method]
            method_rows.append(dict(method=method,**group['rates'][method],
                successful_services_total=sum(r['summary']['successful_services'] for r in chosen),
                hidden_targets_total=sum(r['summary']['hidden_targets'] for r in chosen),
                true_window_timing=window_timing_context(chosen),
                mean_episode_median_planning_ms=float(np.mean([r['summary']['planning_latency_ms']['median'] for r in chosen])),
                mean_episode_p95_planning_ms=float(np.mean([r['summary']['planning_latency_ms']['p95'] for r in chosen])),
                peak_process_memory_bytes=max((r['execution_memory']['cpu_peak_bytes'] or 0 for r in chosen)),
                peak_gpu_allocated_bytes=max((r['execution_memory']['gpu_peak_allocated_bytes'] or 0 for r in chosen))))
        at_ceiling=all(r['summary']['success_rate']==1. for r in rows)
        groups.append(dict(side=side,delay_s=delay,methods=method_rows,effect=group['primary_effect'],
            mechanism_contrasts=group['mechanism_contrasts'],numerical=numeric,reassignment=reassignment,
            opportunity=opportunity_context(method_rows,group['primary_effect'],frozen['B_star']),
            all_methods_at_ceiling=at_ceiling,conclusion=conclusion(group['primary_effect'],at_ceiling,numeric['gain_direction_unchanged'])))
        keys=sorted({tuple(sorted(r['hydrology'].items())) for r in rows})
        for condition in keys:
            selected=[r for r in rows if tuple(sorted(r['hydrology'].items()))==condition and r['method'] in ('P',frozen['B_star'])]
            conditions.append(dict(side=side,delay_s=delay,hydrology=dict(condition),families=3,
                descriptive_exploratory_only=True,effect=paired_effect(selected,'P',frozen['B_star'])))
    predictions=[]
    for delay in (0.,10.):
        for method in ('H','E','P','P_wrong_links','P_uniform_time'):
            selected=[r for r in fixed['records'] if r['method']==method and r['delay_s']==delay]
            if len(selected)!=72:raise AssertionError('Fixed-route predictive summary is not the full validation matrix')
            predictions.append(dict(method=method,delay_s=delay,records=72,families=12,
                mse_m2=float(np.mean([r['mse_m2'] for r in selected])),mae_m=float(np.mean([r['mae_m'] for r in selected])),
                window_error_rate=float(np.mean([r['window_error_rate'] for r in selected])),
                false_safe_rate=float(np.mean([r['false_safe_rate'] for r in selected])),
                false_unsafe_rate=float(np.mean([r['false_unsafe_rate'] for r in selected]))))
    return dict(B_star=frozen['B_star'],primary_episodes=3024,diagnostic_episodes=3456,groups=groups,
        predictive_validation=predictions,hydrology_subgroups=conditions,all_conditions_retained=True,
        primary_statistics='family and training-seed paired bootstrap, 10000 replicates; average layouts first',
        subgroup_statistics='descriptive exploratory intervals; no selection or primary superiority claim from small subgroups')


def plot_statistics(directory,statistics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    groups=statistics['groups'];labels=[f"{g['side']**2} cells / {g['delay_s']:g}s" for g in groups]
    effects=np.array([g['effect']['delta_percentage_points'] for g in groups]);ci=np.array([g['effect']['ci95_percentage_points'] for g in groups])
    figure,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,g in enumerate(groups):
        names=[m['method'] for m in g['methods']];x=np.arange(len(names))+(i-1.5)*.18
        axes[0].scatter(x,[100*m['success_rate'] for m in g['methods']],label=labels[i],s=35)
    axes[0].set_xticks(range(5),['H','E','P','Wrong links','Uniform time']);axes[0].set_ylabel('Successful hidden targets (%)');axes[0].legend(fontsize=8)
    axes[0].set_ylim(max(0,min(100*m['success_rate'] for g in groups for m in g['methods'])-5),101)
    axes[1].errorbar(effects,np.arange(4),xerr=np.vstack((effects-ci[:,0],ci[:,1]-effects)),fmt='o',capsize=4,color='#126c84')
    axes[1].axvline(0,color='#777777',lw=1);axes[1].axvline(3,color='#c88b17',lw=1,ls='--')
    axes[1].set_yticks(np.arange(4),labels);axes[1].set_xlabel(f"P minus {statistics['B_star']} (percentage points; 95% CI)")
    figure.savefig(Path(directory)/'primary_effects.png',dpi=180);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    names=['H','E','P','P_wrong_links','P_uniform_time'];x=np.arange(5)
    for delay,color in ((0.,'#126c84'),(10.,'#ce7d27')):
        rows=[next(r for r in statistics['predictive_validation'] if r['method']==m and r['delay_s']==delay) for m in names]
        axes[0].plot(x,[100*r['mae_m'] for r in rows],marker='o',label=f'{delay:g}s delay',color=color)
        axes[1].plot(x,[100*r['window_error_rate'] for r in rows],marker='o',label=f'{delay:g}s delay',color=color)
    for axis in axes:axis.set_xticks(x,['H','E','P','Wrong links','Uniform time'],rotation=20);axis.legend()
    axes[0].set_ylabel('Fixed-record validation MAE (cm)');axes[1].set_ylabel('Complete 12s window error (%)')
    figure.savefig(Path(directory)/'fixed_record_errors.png',dpi=180);plt.close(figure)


def report_markdown(statistics,frozen):
    lines=['# 实验三：1k–5k格地图下的多无人机闭环搜救','',
        '## 研究问题','',
        '检验在相同合法原始观测、模型容量、训练预算和中央规划器下，物理约束的信息联系与时间衰减是否使P比强历史或统计联系基线多完成搜救任务。','',
        '## 实验设计','',
        '采用6个设计、24个训练、12个验证和36个独立测试家族。每个训练、验证或测试家族预分配一种水情，12种水情在各划分内平衡；分配不参考方法结果。任务从真实物理时刻0秒开始，持续1200秒。开发阶段的450/600秒起始分析不作为正式条件。','',
        '32/50/70边长对应1024/2500/4900格，配置4/10/20机及12/30/60个隐藏目标。每个家族使用两套目标和噪声配置、三个训练种子。2500格五组比较，端点与10秒延迟扩展比较H/E/P，共2376＋648＝3024回合。','',
        '## 执行方法','',
        f"五组使用相同的37,070参数骨干、完整60秒合法历史与60秒预测有效期。15个模型按同一预算训练，在验证集上选择检查点。H/E中一次选定的主要基线为 **{frozen['B_star']}**；冻结后不按测试结果重选。统计联系和物理联系只使用训练家族拟合，其中18家族拟合、6家族校验联系。",'',
        '独立真值与名义ANUGA场使用5米计算网格、1秒输出；传感在实际途中位置每5秒采样5×5窗口。规划器每10秒及新证据事件更新，以100毫秒为统一时限。服务必须完整12秒不超过0.8米，失败耗完整服务时间，只有成功才扣载荷。规划是有界候选优化，不保证全局最优。','',
        '先在家族内平均两套布局，再对家族和训练种子分别成对重采样10,000次。报告百分点增量及95%区间；格子、时间点和布局均不作为独立家族。2500格、零延迟是主要检验组，端点与延迟扩展分别报告条件性结果，不替代主实验判定。区间下界大于0是主要判定，增加3个百分点是单列工程目标。','',
        '## 结果','',
        '| 格数 | 延迟 | P完成率 | B*完成率 | P多完成服务次数 | P−B*（百分点） | 95%区间 | 达到3百分点 |','|---:|---:|---:|---:|---:|---:|---:|:---:|']
    for g in statistics['groups']:
        e=g['effect'];p=next(m for m in g['methods'] if m['method']=='P');b=next(m for m in g['methods'] if m['method']==statistics['B_star']);lo,hi=e['ci95_percentage_points']
        lines.append(f"| {g['side']**2} | {g['delay_s']:g}s | {100*p['success_rate']:.3f}% | {100*b['success_rate']:.3f}% | {p['successful_services_total']-b['successful_services_total']:+d} | {e['delta_percentage_points']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'是' if e['engineering_target_3pp_met'] else '否'} |")
    lines+=['','服务次数之差是在每个组的216个配对回合中累计的完成差，包含训练种子和布局重复，不代表相同数量的独立现实被救人员。','']
    main=next(g for g in statistics['groups'] if g['side']==50 and g['delay_s']==0)
    opportunity=main['opportunity']
    lines.append(f"主实验B*在{opportunity['baseline_hidden_target_instances']}次隐藏目标执行机会中有{opportunity['baseline_unserved_service_instances']}次未完成；即使全部补足，以本次基线为参照的最大增量也仅为{opportunity['maximum_service_rate_gain_percentage_points']:.4f}个百分点。3个百分点工程目标仍按预定规则单列判定；观察到的高基线完成率限制了本次设计的增益空间。")
    if opportunity['all_layout_averaged_family_seed_differences_zero'] and main['effect']['ci95_percentage_points']==[0.,0.]:
        lines+=['','布局平均后的家族×训练种子配对差值全部为0，因而本次经验重采样区间退化为[0,0]；这不构成其他场景中方法普遍等效的证明。']
    lines+=['','![主要结果](primary_effects.png)','','固定路线误差来自12个验证家族的相同记录回放，两个通信延迟各覆盖全部15模型及两套布局。私有水深标签只用于前向预测之后的评分；下面的误差点不作为独立统计样本。','','![固定记录预测与窗口误差](fixed_record_errors.png)','','### 全部失败与执行代价','',
        '| 格数/延迟 | 方法 | 未发现 | 无真实窗口 | 发现太晚 | 窗口误判 | 执行失败 | 资源/时间耗尽 |','|---|---|---:|---:|---:|---:|---:|---:|']
    keys=['undiscovered','no_true_window','discovered_after_last_window','window_misclassification','execution_failure','resource_or_time_exhausted']
    for g in statistics['groups']:
        for m in g['methods']:
            lines.append(f"| {g['side']**2}/{g['delay_s']:g}s | {m['method']} | "+' | '.join(str(m['failure_targets'][k]) for k in keys)+' |')
    lines+=['','失败类别为互斥的描述性归因，全部隐藏目标均保留在分母中。无真实服务窗口、未发现和负收益条件均未筛除。','','| 格数/延迟 | 方法 | 覆盖率 | 平均飞行(m) | 超时次数 | 平均回合规划中位数(ms) | CPU峰值(MiB) | GPU峰值(MiB) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for g in statistics['groups']:
        for m in g['methods']:
            lines.append(f"| {g['side']**2}/{g['delay_s']:g}s | {m['method']} | {100*m['coverage']:.2f}% | {m['mean_flight_m']:.1f} | {m['planning_timeouts']} | {m['mean_episode_median_planning_ms']:.2f} | {m['peak_process_memory_bytes']/2**20:.1f} | {m['peak_gpu_allocated_bytes']/2**20:.1f} |")
    lines+=['','CPU峰值为执行进程累计峰值，GPU值为执行回合期间分配峰值，包含常驻模型；它们不是每回合独立的内存样本。重复搜索量、回合p95、最大规划耗时和全部逐回合数据见`raw_statistics.json`。','','### 服务时序与机会空间','',
        '| 格数/延迟 | 方法 | 最后成功服务时间中位数(s) | 全部任务在任何目标初始安全窗口关闭前完成的回合 |','|---|---|---:|---:|']
    for g in statistics['groups']:
        for method in ('P',statistics['B_star']):
            t=next(m for m in g['methods'] if m['method']==method)['true_window_timing'];q=t['last_successful_service_s_quantiles']
            median=f'{q[1]:.1f}' if q else '无成功服务'
            lines.append(f"| {g['side']**2}/{g['delay_s']:g}s | {method} | {median} | {t['all_targets_served_before_any_initial_true_window_closed_episodes']}/{t['episodes']} |")
    lines+=['','该表仅在事后用全部隐藏目标的真实完整服务窗口核对机会空间。若大部分任务在窗口普遍收紧前就已完成，信息改善难以转化为额外服务；这用于解释当前条件的天花板，不说明更困难或不同时间尺度的搜救中信息无效。','',
        '### 改派实际服务差与数值方向','',
        '原3024回合之外，P与B*各执行同目标、同噪声和同权重的改派策略干预，共1728回合；另外在独立10米/2秒水动力场上复跑P与B*，共1728回合。改派干预仅针对不与周期规划重合的新证据事件，将仍可行的已有任务改派推迟至下一次周期规划；失效计划的安全处置及周期规划照常进行。因此测量的是这类及时改派的累计策略效果，不是全部水情信息的总贡献，也不累加逐次改派的虚构收益。','',
        '| 格数/延迟 | 方法 | 完整改派−推迟改派（百分点，95%区间） | 干预次数 | 干预前已有差异的配对 |','|---|---|---|---:|---:|']
    for g in statistics['groups']:
        for method,e in g['reassignment']['actual_paired_service_gain_from_reassignment'].items():
            lo,hi=e['ci95_percentage_points'];lines.append(f"| {g['side']**2}/{g['delay_s']:g}s | {method} | {e['delta_percentage_points']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {e['policy_interventions']} | {e['pre_intervention_contaminated_pairs']} |")
    lines+=['','若配对在第一次策略干预前已有执行差异，该组的总体差值不能解释为隔离的改派因果收益。计时超时和候选求解可能造成这种差异；原始对照和前缀检查全部保留。','','| 格数/延迟 | 5米/1秒 P−B* | 10米/2秒 P−B* | 方向一致 |','|---|---:|---:|:---:|']
    for g in statistics['groups']:
        n=g['numerical'];lines.append(f"| {g['side']**2}/{g['delay_s']:g}s | {n['fine_5m1s']['delta_percentage_points']:+.3f} | {n['coarse_10m2s']['delta_percentage_points']:+.3f} | {'是' if n['gain_direction_unchanged'] else '否，需继续核查'} |")
    lines+=['','## 结论','']
    for g in statistics['groups']:
        lines.append(f"- {g['side']**2}格、{g['delay_s']:g}秒延迟：{g['conclusion']}")
    lines+=['','消融比较及全部12种水情的描述性子组区间见`raw_statistics.json`。仅优于错联系或均匀时间权重，不能替代P相对B*的主要判定；三个家族组成的水情子组不用于重新筛选场景或宣布主要优越性。','',
        '本报告保留无增益、天花板、失败和方向不稳定的结果。训练、固定回放、验证、正式测试与诊断的原始统计分别存档；工程真值前沿诊断不与训练模型效果混用。','',
        '## 复核与复现','',
        '独立交付目录保留代码、冻结配置、全部检查点、联系规则、划分清单、原始统计、观测与预测记录以及被引用的水动力场。原始JSON字节不改写，移动目录后在读取时映射原绝对路径。','',
        '在交付目录运行 `python scripts/verify_bundle.py --audit-primary`，先核验每个文件的SHA-256及冻结证据，再独立重算全部正式服务评分。训练命令为 `python scripts/train_models.py --data-run <共同数据目录>`；测试单次登记和原目录精确续跑规则由 `scripts/run_formal_test.py` 执行。','']
    return '\n'.join(lines)


def build_final_package(root,arguments):
    root=Path(root).resolve();paths={k:Path(v) for k,v in arguments.items()};frozen=verify_frozen(root,paths['frozen'])
    primary=load_protocol(paths['primary']/'formal_summary.json');fixed=load_protocol(paths['fixed']/'fixed_validation_summary.json')
    counter=load_protocol(paths['reassignment_counterfactual']/'diagnostic_summary.json');numerical=load_protocol(paths['numerical_direction']/'diagnostic_summary.json')
    audit=load_protocol(paths['primary']/'formal_integrity_audit.json')
    if audit['status']!='passed' or audit['episodes']!=3024 or audit['formal_summary_sha256']!=sha256(paths['primary']/'formal_summary.json'):
        raise ValueError('Complete current independent primary audit required')
    from .formal_audit import audit_diagnostic
    for kind in ('reassignment_counterfactual','numerical_direction'):
        marker=paths[kind]/'diagnostic_integrity_audit.json'
        previous=load_protocol(marker) if marker.exists() else {}
        if previous.get('status')!='passed' or previous.get('summary_sha256')!=sha256(paths[kind]/'diagnostic_summary.json'):
            audit_diagnostic(root,paths[kind],paths['frozen'],workers=3)
    statistics=collect_statistics(frozen,primary,fixed,counter,numerical)
    directory=root/'results/final_reports'/datetime.now(TZ).strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'raw_statistics.json',statistics)
    plot_statistics(directory,statistics)
    (directory/'实验三完整报告.md').write_text(report_markdown(statistics,frozen),encoding='utf-8')
    requirements={
        'design_numerical_and_clock_checks':[paths['refinement']/'refinement_integrity_audit.json',paths['primary']/'formal_integrity_audit.json'],
        'planner_candidate_and_budget_sensitivity':[paths['sensitivity']/'sensitivity_summary.json',paths['sensitivity']/'sensitivity_integrity_audit.json'],
        'common_causal_training_records':[paths['data']/'common_data_integrity_audit.json'],
        'five_parameter_matched_models_three_training_seeds':[paths['training']/'training_summary.json',paths['training']/'training_integrity_audit.json'],
        'training_only_statistical_and_validated_physical_links_and_lags':[paths['training']/'frozen_training_links.json'],
        'fixed_route_identical_record_prediction_and_window_evaluation':[paths['fixed']/'fixed_validation_summary.json'],
        'validation_B_star_and_configuration_freeze':[paths['validation']/'B_star_selection.json',paths['frozen'],root/'registrations/frozen_configuration.v1.json'],
        'single_independent_3024_episode_test':[paths['primary']/'formal_summary.json',paths['primary']/'formal_integrity_audit.json'],
        'paired_family_and_training_seed_95_percent_intervals_and_3pp_target':[directory/'raw_statistics.json'],
        'causal_reassignment_counterfactuals':[paths['reassignment_counterfactual']/'diagnostic_summary.json',paths['reassignment_counterfactual']/'diagnostic_integrity_audit.json'],
        'numerical_direction_stability_of_final_gain':[paths['numerical_direction']/'diagnostic_summary.json',paths['numerical_direction']/'diagnostic_integrity_audit.json'],
        'independent_reproducible_package_frozen_configuration_raw_statistics_final_report':[directory/'实验三完整报告.md',directory/'raw_statistics.json']}
    ledger=load_protocol(root/'COMPLETION.json')
    if set(requirements)!=set(ledger['requirements']):raise AssertionError('Original deliverable scope changed')
    for key,files in requirements.items():
        ledger['requirements'][key]=dict(status='verified',evidence=[dict(path=p.relative_to(root).as_posix(),sha256=sha256(p)) for p in files])
    if not numerical['all_gain_directions_unchanged']:
        ledger['requirements']['numerical_direction_stability_of_final_gain']['status']='requires_numerical_investigation'
    ledger.update(complete=False,formal_test_episodes=3024,models_trained=15,report=directory.relative_to(root).as_posix(),
        statistical_positive_result_is_not_required_for_completion=True,export_status='building')
    write_json(root/'COMPLETION.json',ledger)
    archive=root/'README.development.md'
    if not archive.exists():archive.write_bytes((root/'README.md').read_bytes())
    report_relative=(directory/'实验三完整报告.md').relative_to(root).as_posix()
    status='正式测试与配对诊断已执行，尚需继续处理数值方向问题。' if not numerical['all_gain_directions_unchanged'] else '正式测试、配对诊断与独立记录复核已执行。'
    (root/'README.md').write_text('\n'.join([
        '# 实验三：多无人机闭环搜救','',status,
        '交付包与全部要求是否已核验，以 `COMPLETION.json` 为准；统计正增益不是完成条件。','',
        f'- [完整实验报告]({report_relative})',
        f'- [全部原始统计]({directory.relative_to(root).as_posix()}/raw_statistics.json)',
        '- [冻结配置](configs/frozen_experiment3.v1.json)',
        '- [原始方案](docs/protocols/实验三原始方案.md)',
        '- [开发阶段历史说明](README.development.md)','',
        '正式测试为2376个主回合和648个延迟回合；改派与数值方向各1728回合，单独标记为诊断。',
        '五方法、三种子的15个模型在独立测试前完成训练与验证；B*只从H/E在验证集选择一次。','',
        '## 核验与复现','',
        '依赖见 `requirements.txt`，各次实际执行环境见运行目录中的 `environment.json`。移动交付包后，入口自动映射原始记录中的绝对路径，不改写证据字节。','',
        '```powershell','python scripts/verify_bundle.py --audit-primary','python run.py test','```','',
        '独立测试只能使用冻结配置并从原登记目录精确续跑，禁止据测试结果重新选择条件或基线。','']),encoding='utf-8')
    from .export import export_bundle,verify_bundle
    selected=[p for k,p in paths.items() if k!='frozen']
    bundle=export_bundle(root,selected,directory)
    candidate=dict(ledger,complete=all(r['status']=='verified' for r in ledger['requirements'].values()),export_status='verified',
        portable_manifest='INTEGRITY.json',completion_recorded_at=datetime.now(TZ).isoformat())
    # Finalize the copied completion record, then independently hash the whole
    # package again before the source ledger can say the objective is complete.
    write_json(bundle/'COMPLETION.json',candidate)
    manifest=json.loads((bundle/'INTEGRITY.json').read_text(encoding='utf-8'))
    manifest['files']['COMPLETION.json']=dict(sha256=sha256(bundle/'COMPLETION.json'),bytes=(bundle/'COMPLETION.json').stat().st_size)
    manifest['bytes']=sum(v['bytes'] for v in manifest['files'].values())
    write_json(bundle/'INTEGRITY.json',manifest)
    verified=verify_bundle(bundle);write_json(bundle/'PACKAGE_VERIFICATION.json',verified)
    write_json(root/'COMPLETION.json',candidate)
    write_json(root/'results/latest_final_package.json',dict(package=str(bundle),report=str(directory),objective_complete=candidate['complete'],verification=verified))
    return bundle
