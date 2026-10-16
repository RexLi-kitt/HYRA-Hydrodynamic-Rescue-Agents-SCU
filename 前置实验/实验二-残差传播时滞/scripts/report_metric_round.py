"""Report-only postprocessing of the completed development round. No model runs."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pointer=read(ROOT/'results/latest_metric_development_pipeline.json');assert pointer['completed']
    pipeline=Path(pointer['run']);run=Path(pointer['selection_run']);modelrun=Path(pointer['model_run'])
    data=Path(read(ROOT/'results/latest_metric_development_data.json')['run']);cache=Path(read(ROOT/'results/latest_metric_feature_cache.json')['folder'])
    assert read(pipeline/'completion.json')['completed'];assert not read(pipeline/'completion.json')['pilot_or_confirmation_opened']
    assert read(data/'completion.json')['families']==36;assert read(data/'data_audit/completion.json')['all_development_data_audited']
    assert read(cache/'input_audit/progress.json')['audited_families']==36
    provenance_count=0
    for folder in (pipeline,modelrun,run):
        for rel,checksum in read(folder/'provenance.json')['files'].items():
            assert digest(folder/'frozen_source'/rel)==checksum;provenance_count+=1
    for rel,checksum in read(data/'provenance.json')['files'].items():assert digest(ROOT/rel)==checksum
    registry=read(modelrun/'registry.json');assert len(registry)==79
    for spec in registry:assert digest(spec['path'])==spec['sha256']
    inputs=read(modelrun/'training_inputs.json');assert len(inputs['families'])==24
    for entry in read(data/'family_index.json'):
        family=entry['family'];audit=read(data/'data_audit'/(family+'.json'));features=read(cache/(family+'.json'));causal=read(cache/'input_audit'/(family+'.json'))
        assert audit['passed'] and causal['passed'];assert digest(entry['manifest_path'])==audit['manifest_sha256']==features['manifest_sha256']
        assert causal['cache_sha256']==features['cache_sha256']
    selection=read(run/'selection.json');assert selection['independent_families']==selection['active_families']==12
    specs=read(run/'candidates.json');assert len(specs)==323;names=[s['name'] for s in specs]
    paired=read(run/'paired_comparisons.json');families=paired['families'];assert len(families)==12
    rng=np.random.default_rng(2026101699);bootstrap=rng.integers(0,12,size=(10000,12));statistics={}
    for key in paired['mean_family_robust_contrasts']:
        values=np.array([f['values'][key]['robust'] for f in families]);mean=float(values.mean())
        assert abs(mean-paired['mean_family_robust_contrasts'][key])<1e-15
        low,high=np.quantile(values[bootstrap].mean(axis=1),[.025,.975])
        statistics[key]=dict(mean=mean,descriptive_bootstrap95=[float(low),float(high)],positive_families=int((values>0).sum()),negative_families=int((values<0).sum()),zero_families=int((values==0).sum()))
    p4index=names.index(paired['chosen']['P4']);matched_index=names.index(paired['chosen']['P5_exact_P4'])
    matched_values=[]
    for family in families:
        row=read(run/(family['family']+'.json'))
        matched_values.append(min(m['services'][p4index]-m['services'][matched_index] for m in row['meshes'].values()))
    values=np.array(matched_values);low,high=np.quantile(values[bootstrap].mean(axis=1),[.025,.975])
    statistics['service_gain_vs_exact_matched_P5']=dict(mean=float(values.mean()),descriptive_bootstrap95=[float(low),float(high)],positive_families=int((values>0).sum()),negative_families=int((values<0).sum()),zero_families=int((values==0).sum()))
    headroom=np.array([f['oracle_headroom_vs_Bstar'] for f in families]);lo,hi=np.quantile(headroom[bootstrap].mean(axis=1),[.025,.975])
    statistics['oracle_headroom_vs_Bstar']=dict(mean=float(headroom.mean()),descriptive_bootstrap95=[float(lo),float(hi)])
    information=read(run/'information_action_check.json');info=[s for s in information['summary'] if s['decision_offset_s'] is None]
    assert all(s['robust_completed_service_gain']==0 for s in information['summary'] if s['name']=='P0_nominal' or s['decision_offset_s']==0)
    absolute={}
    for group in ('P0','P1','P2','P3','P4','P5','B_star'):
        name=selection['winners'][group]['name'];i=names.index(name);rows=[]
        for f in families:rows.append(read(run/(f['family']+'.json'))['meshes']['80'])
        absolute[group]=dict(name=name,services=float(np.mean([r['services'][i] for r in rows])),
            window_error=float(np.mean([r['window_error'][i] for r in rows])),mae_6_12_m=float(np.mean([r['mae_6_12'][i] for r in rows])))
    result=dict(stage='development_round_completed_user_requested_stop',objective_achieved=False,training_families=24,validation_families=12,
        primary_active_coverage=1.,candidates=323,trained_models=79,data_fields=1692,statistics=statistics,absolute_primary_mesh80=absolute,
        information_intervention_E1_only=info,chosen=paired['chosen'],selection_run=str(run),model_run=str(modelrun),data_run=str(data),
        bootstrap_seed=2026101699,bootstrap_resamples=10000,interval_interpretation='Descriptive conditional intervals on the same validation families used to select models; selection bias is not corrected; no independent confirmatory claims',
        provenance_hashes_verified=provenance_count,pilot_opened=False,confirmation_opened=False,
        not_executed_in_this_round=['general-population replay','10-second-delay task replay','wrong-link/wrong-lag replay','false-feasible/false-infeasible breakdown','pilot','independent confirmation'],
        stopped_by_user=True)
    write_json(ROOT/'results/metric_round_report_20261016.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,5.2),sharey=True)
    labels=[f['family'].split('_')[-1] for f in families]+['Mean*'];ys=np.arange(13)
    for ax,key,factor,title,xlabel in [
        (axes[0],'H2_MAE_6_12_improvement_vs_matched_P5',1000,'Physical timing vs exactly matched lag removal','MAE reduction at 6 / 12 s (mm)'),
        (axes[1],'H5_completed_service_gain_vs_Bstar',100,'Physical method vs selected strong baseline','Extra completed services per 100 tasks')]:
        values=np.array([f['values'][key]['robust'] for f in families])*factor;s=statistics[key];mean=s['mean']*factor;ci=np.array(s['descriptive_bootstrap95'])*factor
        ax.scatter(values,ys[:12],c=np.where(values>=0,'#206B62','#AE4A3D'),s=28,zorder=3)
        ax.errorbar(mean,12,xerr=np.array([[mean-ci[0]],[ci[1]-mean]]),fmt='D',color='#203E61',capsize=4,zorder=4)
        ax.axvline(0,color='#444444',lw=.9);ax.grid(axis='x',color='#dddddd',lw=.6);ax.set_title(title,fontsize=10,pad=12);ax.set_xlabel(xlabel)
    axes[1].axvline(1,color='#A74C31',linestyle='--',lw=1,label='Prespecified practical target');axes[1].legend(loc='upper right',fontsize=7)
    axes[0].set_yticks(ys,labels);axes[0].set_ylabel('Validation hydraulic family');axes[0].invert_yaxis()
    fig.suptitle('Completed development round: prediction gain does not establish practical action gain',fontsize=12,y=.99)
    fig.text(.08,.02,'* 12-family mean; descriptive bootstrap interval after model selection. Each family uses the worse of mesh 80 / 40. No confirmation data.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95));figure=ROOT.parent/'实验报告/20261016/HyRA_本轮实验结果_20261016.png';fig.savefig(figure,dpi=180);plt.close(fig)
    def line(key,factor):
        s=statistics[key];ci=s['descriptive_bootstrap95'];return f"{s['mean']*factor:.3f}（{ci[0]*factor:.3f}，{ci[1]*factor:.3f}）"
    p4info=next(s for s in info if s['name']==paired['chosen']['P4']);bstarinfo=next(s for s in info if s['name']==paired['chosen']['B_star'])
    report=f'''# HyRA 本轮实验汇报

2026-10-16。本轮已完成，按用户要求停止；不再启动先导、新确认或其他实验。

核心结论：物理传播时滞对短期预测有一致的开发证据；上游信息能够改变选序并带来实际模拟服务收益。但相对最强原始历史基线的额外服务收益很小且不稳定，本轮没有证明预设的实际收益目标。

## 完成范围与证据

24个新训练家族、12个新验证家族全部完成，12/12验证家族有临界任务。共1692份原始数值场；36个家族的状态重启、物理点投影及输入审核全部通过。训练得到79份模型文件，连同名义、普通滤波和直接物理方法，共比较323个候选。固定版本FilterPy的1512条实际完整历史预测曲线已独立核对。

所有方法使用6个共同上游源和全部已到达5秒记录，目标没有本地观测。名义预测从独立名义状态发布；模型不接收真实扰动有无、位置、开始时刻或幅度。任务仍为单机、两个已知目标的初始选序，完整服务12秒、共同6秒中止反馈；0.8米为模拟服务限制。

本轮正式回放为零延迟、名义预测临界任务总体。每项配对增量先在一个家族内取网格80/40的较差值，再对12个家族等权平均。只有12个独立验证单位，重复布局、时刻、噪声和数值精度不能充当额外独立样本。

## 主要结果

|指标（正数为物理方法改善）|家族平均（描述性95%重抽样区间）|单位|
|---|---:|---|
|6/12秒误差降低，相对所选去时滞方法|{line('H2_MAE_6_12_improvement_vs_P5',1000)}|毫米|
|6/12秒误差降低，相对完全匹配的去时滞方法|{line('H2_MAE_6_12_improvement_vs_matched_P5',1000)}|毫米|
|共同完整12秒窗口错误率降低，相对强基线|{line('H4_common_window_error_reduction_vs_Bstar',100)}|个百分点|
|实际完成服务增加，相对强基线|{line('H5_completed_service_gain_vs_Bstar',100)}|个服务/100任务|
|实际完成服务增加，相对所选去时滞方法|{line('service_gain_vs_P5',100)}|个服务/100任务|
|实际完成服务增加，相对完全匹配的去时滞方法|{line('service_gain_vs_exact_matched_P5',100)}|个服务/100任务|

这些区间由12家族、10000次重抽样计算。模型同时由这12个验证家族选出，区间未校正选模偏倚，只作开发描述，不能当作独立确认或宣布统计检验通过。

所选物理方法为`{paired['chosen']['P4']}`：它使用名义响应和未知来水位置/起始年龄的后验混合，最终选择直接物理方法，而非更复杂的混合网络。强基线为`{paired['chosen']['B_star']}`，接收完整原始上游历史；所选去时滞方法为`{paired['chosen']['P5']}`，完全匹配消融为`{paired['chosen']['P5_exact_P4']}`。

完全匹配时滞消融的短期误差改善在12/12家族为正。相对强基线的任务收益仅{statistics['H5_completed_service_gain_vs_Bstar']['positive_families']}/12家族为正，另{statistics['H5_completed_service_gain_vs_Bstar']['negative_families']}/12为负；平均增量只有{statistics['H5_completed_service_gain_vs_Bstar']['mean']*100:.3f}个服务/100任务，未达到预设至少1个服务/100任务。窗口错误率和任务收益的描述性区间均跨零。因此，短期预测更准这一现象尚不足以支持“物理方法获得额外实际任务收益”的最终结论。

## 信息到行动的直接检查

固定同一E1扰动环境和执行反馈，仅把已到达的上游输入在配对E1/E0之间替换，噪声相同，重新预测和选序后实际执行。物理方法的配对输入效应为{p4info['robust_completed_service_gain']*100:.3f}个服务/100个E1任务，强基线为{bstarinfo['robust_completed_service_gain']*100:.3f}个服务/100个E1任务。名义方法效应为零，扰动刚开始、尚无扰动专属输入时效应也为零。

这说明仿真链路中已到达的信息确实能改变动作并改善服务。它比较的是同一模型拿到两种配对信息，分母只含E1任务；不能把上述{p4info['robust_completed_service_gain']*100:.3f}解释成“物理方法比强基线多完成的服务”。物理额外收益仍应看前表的{statistics['H5_completed_service_gain_vs_Bstar']['mean']*100:.3f}。

## 可挽回空间与结论边界

知道未来水情的最优顺序相对强基线还剩平均{headroom.mean()*100:.3f}个服务/100任务，描述性区间为（{lo*100:.3f}，{hi*100:.3f}）。这一开发诊断未达到后续先导方案要求的“可挽回空间下界至少2个服务/100任务”。先导没有开展，不能将其写成正式先导检验；当前结果也不足以支持直接投入收益确认。

本轮未执行一般任务总体和10秒延迟的任务回放、错联系/错时滞回放、窗口假可行/假不可行分项、先导与独立确认。数据中保留了延迟输入，物理库实现了错误联系控制，均不代表相应实验已经完成。按用户停止要求，这些项目不再继续。

数值敏感性只覆盖固定点读取的网格80/40，窗口峰值基于2秒输出的分段线性曲线，不是连续水动力过程的严格误差界。地形、糙率等名义参数已知；实际扰动局限于两个预设边界区间、三个时刻和固定形状/幅值的场景，预测器不接收真实场景标签。以上是模拟水情、直线飞行与给定服务规则下的开发结果，没有验证任意洪水输入、真实洪水、真实无人机动力学、途中重规划或多机调度。

![本轮家族结果]({figure.as_posix()})

## 原始入口

- [完整配对结果]({(run/'paired_comparisons.json').as_posix()})
- [配对信息行动干预]({(run/'information_action_check.json').as_posix()})
- [机器可读汇报]({(ROOT/'results/metric_round_report_20261016.json').as_posix()})
- [本轮完成状态]({(pipeline/'completion.json').as_posix()})

总体研究目标尚未完成。本轮停止后只汇报已有结果，保留全部正负结果和审核记录。

两项额外只读复核已完成：一项检查因果输入、匹配对照和实际服务评分，未发现使本轮主结果失效的目标/未来观测泄漏；另一项独立重算323个候选的选择分数、配对增量及35行信息干预汇总，与家族原始统计的最大算术差为零。复核没有增加实验或更改模型。
'''
    output=ROOT.parent/'实验报告/20261016/HyRA_本轮实验汇报_20261016.md';output.write_text(report,encoding='utf-8')
    print(json.dumps(dict(report=str(output),figure=str(figure),statistics=statistics),ensure_ascii=True))


if __name__=='__main__':main()
