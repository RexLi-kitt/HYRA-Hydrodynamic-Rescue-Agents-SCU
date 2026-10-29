"""Verify and archive the user-confirmed100m training round; no new training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import zipfile
from scale100_publish import validated_bundle, ROOT, FORMAL, EVALUATION, read, digest, require
from console_scale100_contract import model_record, catalog

REPO = ROOT.parent
REPORT = ROOT / 'generated/scale100-report-20261026'
MECHANISM = ROOT / 'generated/scale100-checkpoint-mechanism-20261026'
ACCEPTANCE = ROOT / 'generated/scale100-model-acceptance-20261026'


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def execute(output, browser_path, screenshot_path):
    output = output.resolve()
    require(not output.exists(), 'Refusing to overwrite a completed/partial delivery')
    plan, seal, proof, results, models = validated_bundle()
    require(set(plan['tasks']) == {'E1','E2'} and set(plan['methods']) == {'B0','B1','B2','B3','D','P'}
            and plan['training_seeds'] == [11,29,47] and len(models) == 36,
            'Accepted two-task/six-group/three-seed matrix differs')
    require(plan['episodes_per_job'] == 1024 and plan['updates_per_job'] == 1021
            and plan['development_checkpoints'] == [256,512,1024], 'Budget or selection opportunities differ')
    require(sorted(r['split'] for r in plan['inputs']) == ['development','fit','fit','fit','holdout','holdout'],
            'Required independent-parent split differs')
    preflight = read(ROOT / 'research/scale100_dynamic_preflight_20261026.json')
    require(preflight['actor_water_memory_exactly_reconstructed'] is True
            and preflight['actor_input_channels'] == 21
            and preflight['operator_verified_actual_steps'] == 32
            and preflight['operator_max_abs_error'] < 1e-12, 'Input/operator preflight evidence missing')
    report_manifest = read(REPORT / 'manifest.json')
    require(report_manifest['complete'] and report_manifest['release_audit_sha256'] == digest(EVALUATION/'release-audit.json'),
            'Report belongs to another audited result')
    for name, checksum in report_manifest['files_sha256'].items():
        require(digest(REPORT/name) == checksum, 'Report artifact changed: '+name)
    auxiliary = read(EVALUATION/'auxiliary-forecasts.json')
    require(auxiliary['complete'] and len(auxiliary['rows']) == 2052
            and len(auxiliary['models']) == 12 and len(auxiliary['reference_streams']) == 12,
            'Auxiliary diagnostic incomplete')
    for name in ('scale100-posttraining-20261026.json','scale100-finalize-20261026.json'):
        state = read(ROOT/'generated'/name)
        require(state['status'] == 'completed', 'Background workflow did not complete: '+name)
        for path, checksum in state['bindings'].items():
            require(digest(path) == checksum, 'Bound protocol/source changed')
    mechanism = read(MECHANISM/'summary.json')
    require(mechanism['status'] == 'complete' and mechanism['completed_jobs'] == 12
            and not mechanism['remaining_jobs'], 'Late-checkpoint audit incomplete')
    for row in mechanism['rows']:
        path = ROOT/row['path']
        value = read(path)
        require(digest(path) == row['sha256'] and value['source_sha256'] == digest(ROOT/'scale100_checkpoint_mechanism.py')
                and not value['training_or_selection_changed'] and not value['heldout_fields_loaded'],
                'Late-checkpoint audit changed model or provenance')
        require({r['kind'] for r in value['checkpoint_results']} == {'selected','final'}, 'Missing checkpoint kind')
        for checkpoint in value['checkpoint_results']:
            folder = FORMAL/row['job']/('selected' if checkpoint['kind']=='selected' else '')
            require(checkpoint['actor_sha256'] == digest(folder/'actor.pt') and checkpoint['states'] == 12,
                    'Late audit belongs to another checkpoint')
            require(checkpoint['nonzero_data_shared_states'] == 12
                    and (value['job']['method'] != 'P' or checkpoint['nonzero_physics_shared_states'] == 12),
                    'Report the observed disconnected gradient paths before completion')
    accepted = read(ACCEPTANCE/'acceptance.json')
    require(accepted['complete'] and len(accepted['rows']) == 36
            and accepted['release_registry_sha256'] == digest(ROOT/'research/scale100-workbench-release-20261026.json'),
            'Production-worker acceptance incomplete')
    expected = {m['id']:m for m in models}
    require({r['id'] for r in accepted['rows']} == set(expected), 'Acceptance matrix differs')
    for row in accepted['rows']:
        replay_path = ACCEPTANCE/row['id']/'replay.json'
        replay = read(replay_path)
        require(digest(replay_path) == row['replay_sha256'] and row['audit']['audit_passed'], 'Acceptance replay changed')
        require(replay['execution']['actor_sha256'] == expected[row['id']]['actor_sha256']
                and replay['execution']['actual_group'] == expected[row['id']]['method']
                and replay['execution']['training_seed'] == expected[row['id']]['seed'], 'Worker model attribution differs')
    browser = read(browser_path)
    tasks = {'自主探索':'exploration','探索—搜救':'exploration_rescue'}
    require(len(browser['capabilities']) == 12
            and {(tasks[r['task']],r['group']) for r in browser['capabilities']}
                == {(t,g) for t in tasks.values() for g in plan['methods']}
            and all(r['enabled'] for r in browser['capabilities']), 'Browser capability matrix incomplete')
    browser_runs = []
    for row in browser['runs']:
        match = re.fullmatch(r'/runs/([a-f0-9]{12})/replay.json', row['replay'])
        require(match is not None, 'Invalid observed replay link')
        path = ROOT/'generated/web'/match[1]/'replay.json'
        replay = read(path)
        _, metadata = model_record(row['task'],row['group'],row['training_seed'])
        require(replay['task'] == row['task'] and replay['group'] == row['group']
                and replay['execution']['mode'] == 'model'
                and replay['execution']['training_seed'] == row['training_seed']
                and replay['execution']['scenario_seed'] == row['scenario_seed']
                and replay['execution']['actor_sha256'] == metadata['actor_sha256']
                and replay['audit']['audit_passed'] and len(replay['actions']) == 180
                and len(replay['snapshots']) == 181 and len(replay['observations']) == 362,
                'Observed browser replay is not the requested real model')
        browser_runs.append(dict(**row, run_id=match[1], replay_sha256=digest(path)))
    require({r['task'] for r in browser_runs} == set(tasks.values())
            and any(r['training_seed'] != r['scenario_seed'] for r in browser_runs)
            and browser['restored_default_seeds'] and browser['final_frame_s'] == 900,
            'Two-task or independent-seed browser validation missing')
    require(catalog()['released'] and screenshot_path.is_file(), 'Published workbench/screenshot missing')
    requirements = [
        ('accepted_scope','User-confirmed100x100m,900s,2UAV training round; broader research gates are separate', 'docs/SCALE100_TRAINING_20261026.md'),
        ('matrix_budget_selection','Both tasks,6methods,3seeds,1024episodes/1021updates,3equal selection opportunities', 'generated/scale100-formal-20261026/plan.json'),
        ('frozen_sources_inputs_runtime','Original35scientific sources and current training/data/runtime validated unchanged', 'generated/scale100-heldout-20261026/release-audit.json'),
        ('legal_inputs_eight_actions','21channels,actual sensor logs,eight-neighbor trajectories and synthetic service independently reconstructed', 'generated/scale100-heldout-20261026/release-audit.json'),
        ('numerics_and_data','Actual900s frames, accepted SWE microsteps and independent input/operator preflight', 'research/scale100_dynamic_preflight_20261026.json'),
        ('all_sealed_before_holdout','All36 selected by frozen development rule before216model+36rule trials', 'generated/scale100-heldout-20261026/sealed-models.json'),
        ('all_policy_replays','252actual policies,actions,observations,services and aggregates independently reproduced', 'generated/scale100-heldout-20261026/release-audit.json'),
        ('sciml_mechanism','All12D/P selected/final checkpoints read-only fit gradient probes completed', 'generated/scale100-checkpoint-mechanism-20261026/summary.json'),
        ('separate_physical_diagnostic','2052matched auxiliary/control points; local and volume residuals separated from task benefit', 'generated/scale100-heldout-20261026/auxiliary-forecasts.json'),
        ('report_and_figures','Negative results,paired parent means,CSV,PNG/SVG; scientific figures visually reviewed', 'generated/scale100-report-20261026/manifest.json'),
        ('worker_delivery','All36 real production-worker task/group/seed choices completed audited900s replays', 'generated/scale100-model-acceptance-20261026/acceptance.json'),
        ('browser_delivery','All12task/group controls enabled; both real model tasks and independent seeds verified', 'browser-final-evidence.json')]
    output.mkdir(parents=True)
    shutil.copyfile(browser_path,output/'browser-final-evidence.json')
    shutil.copyfile(screenshot_path,output/'workbench-final.png')
    contrasts = [r for r in results['contrasts'] if r['name'] == 'physics_increment']
    audit = dict(schema='hyra-scale100-round-completion-v1', complete=True,
        verified_at=datetime.now(timezone.utc).isoformat(), scope='user-confirmed100m training round',
        formal_jobs=36, training_episodes=36864, optimizer_updates=36756,
        training_joint_decisions=6635520, policy_trials=252, auxiliary_points=2052,
        late_checkpoint_jobs=12, production_model_checks=36, browser_runs=browser_runs,
        requirements=[dict(id=k,status='verified',evidence=e,what_was_verified=d) for k,d,e in requirements],
        result='Physical auxiliary did not improve primary metrics in this two-parent synthetic study',
        primary_physics_effects=contrasts, independent_holdout_parents=2,
        nonclaims=['strict original-paper performance reproduction','statistical significance','convergence',
                   'long-duration original-large-map learned-policy transfer','real-disaster calibration',
                   'broad research plan and service-threshold sensitivity completed'],
        evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in (FORMAL/'plan.json',
            EVALUATION/'results.json',EVALUATION/'release-audit.json',EVALUATION/'sealed-models.json',
            EVALUATION/'auxiliary-forecasts.json',REPORT/'manifest.json',MECHANISM/'summary.json',
            ACCEPTANCE/'acceptance.json',ROOT/'research/scale100-workbench-release-20261026.json')})
    write(output/'completion-audit.json',audit)
    summary = f'''# HyRA100米训练轮次完成记录

完成范围是用户确认的100×100米、900秒、两机正式训练轮次：2任务×6组×3训练种子，共36项，均执行1024回合和1021次优化。不是整个研究项目所有后续阶段均已完成。

## 核心结果

物理辅助P没有改善本轮主指标：E1首次覆盖AUC为0.8213，D为0.8266，差值−0.54个百分点；E2施救比例为45.56%，D为60.00%，差值−14.44个百分点。三种规则在本轮E2留出场景均为100%。物理残差下降，但水深重建误差和任务效果没有共同改善。

只有两个同一生成器的合成留出父场景，因此这些是配对描述性结果，不作统计显著性、收敛或一般化保证。不通过重选种子、权重或重新调参来回避负面结果。

## 已验证产物

- [完整结果报告]({(REPORT/'report.md').as_posix()})：含逐任务/逐父场景结果、配对消融、训练曲线及物理诊断；同目录保留CSV和PNG/SVG。
- [最终逐项验收]({(output/'completion-audit.json').as_posix()})：12类要求及实际证据。
- [晚期检查点诊断]({(MECHANISM/'summary.json').as_posix()})：全部12个D/P任务，选中和最终权重均在12个fit采样状态检查；所有P保留物理共享梯度。这不是任务收益证明，D的物理潜在梯度不表示D应用了物理损失。
- [工作台](http://127.0.0.1:8767/)：两类任务、六组均已发布；36种模型工作进程组合和真实浏览器回放验收通过，默认训练种子11，场景种子单独记录。

## 复现与归档

本目录`evidence-and-models.zip`保存本轮冻结源码、配置/协议、六个实际洪水数据包、全部36项初始/选中/最终actor、训练历史和开发选模记录、252项原始留出回放、辅助诊断、工作进程验收及最终浏览器证据。内部SHA256清单与ZIP完整性均已检查；同目录`archive-manifest.json`记录归档摘要。

原始训练恢复检查点和优化器大文件继续保留在原研究目录，未重复打包；归档不宣称可在任意系统路径直接恢复训练。冻结记录保留原始本地路径及Python/NumPy/Torch运行时，重放应遵守登记配置与来源校验。旧P3/四方向研究没有被覆盖或改写。

## 保留的边界

原论文严格性能复现、长时原始大图学习策略迁移、服务阈值敏感性、真实洪水率定及无人机动力学没有由本轮完成或证明。旧总体研究计划的这些后续阶段保持单独状态；100米结论不能替代它们。
'''
    (output/'README.md').write_text(summary,encoding='utf-8')
    study_path = ROOT/'research/current-study.json'
    study = read(study_path)
    study.update(status='scale100_round_completed_negative_physics_increment',next_steps=[],
                 final_report=str((REPORT/'report.md').relative_to(ROOT)),
                 round_delivery=str((output/'README.md').relative_to(ROOT)))
    active = study['active_training']
    active['formal_study'].update(complete=True,holdout_evaluated=True,active_workers=0,
        latest_observation=dict(queue_status='completed',completed_jobs=36,failed_jobs=[],workers=[],
            posttraining_status='completed',finalization_status='completed',processes_normally_exited=True))
    active['analysis_delivery'].update(status='completed',completion_audit=str((output/'completion-audit.json').relative_to(ROOT)))
    active['workbench_adapter']['status']='published_and_browser_accepted'
    active['posttraining_continuation'].update(status='completed',phase='completed_and_followup_verified')
    save_plan = read(ROOT/'research/dual_task_plan_20261026.json')
    save_plan['status']='scale100_round_completed_broader_research_gates_separate'
    by_id = {m['id']:m for m in models}
    for job in save_plan['planned_training_jobs']:
        require(job['id'] in by_id,'Unexpected mutable-plan job')
        job.update(status='completed',selected_episode=by_id[job['id']]['selected_episode'])
    save_plan['scale100_training']=active
    save_plan['round_delivery']=study['round_delivery']
    for gate in save_plan['implementation_gates']:
        if gate['id']=='E':
            gate['status']='100m_round_completed;36training,252holdout,allreplays,report,models_and_browser_verified;n2synthetic_limitations'
    write(study_path,study)
    write(ROOT/'research/dual_task_plan_20261026.json',save_plan)
    snapshot_path=ROOT/'research/scale100_execution_snapshot_20261026.json'
    snapshot=read(snapshot_path)
    snapshot['latest_formal_observation']=active['formal_study']['latest_observation']
    write(snapshot_path,snapshot)
    files={}
    def add(path):
        path=Path(path).resolve()
        require(path.is_file(),'Missing archive input: '+str(path))
        try: name='project/'+path.relative_to(REPO).as_posix()
        except ValueError: name='external-sources/'+digest(path)[:12]+'-'+path.name
        require(name not in files or files[name]==path,'Archive name collision')
        files[name]=path
    for name in plan['training_sources']: add(ROOT/name)
    for name in plan['inherited_sources']: add(name)
    for row in plan['inputs']: add(row['path'])
    add(plan['dataset_path'])
    for row in plan['evidence_files'].values(): add(row['path'])
    for name in ('scale100_formal.py','scale100_formal_worker.py','scale100_formal_parallel.py','scale100_backend.py',
        'scale100_run_audit.py','scale100_posttraining.py','scale100_finalize.py','scale100_evaluate_sealed.py',
        'scale100_release_audit.py','scale100_rules.py','scale100_development.py','scale100_formal_forecasts.py',
        'scale100_resilient_forecast.py','scale100_publish.py','scale100_report.py','scale100_model_acceptance.py',
        'scale100_checkpoint_mechanism.py','scale100_round_completion.py','console_app.py','console_scale100_app.py',
        'console_scale100_contract.py','console_scale100_runner.py','workbench_api.py','workbench_tasks.py',
        'console_workbench_runner.py','rescue_console.py'):
        add(ROOT/name)
    for name in ('workbench-scale100.html','workbench-scale100.js','workbench.css'):add(ROOT/'console-static'/name)
    for name in ('generate_scale100.py','generate_scale100_dynamic.py'):add(REPO/'terrain-workbench/terrain-lab'/name)
    for path in (ROOT/'research').glob('*20261026.json'):add(path)
    add(study_path)
    for path in (ROOT/'docs').glob('SCALE100*.md'):add(path)
    add(ROOT/'docs/DUAL_TASK_SCIML_PLAN_20261026.md')
    add(FORMAL/'plan.json');add(FORMAL/'queue.json')
    for job in plan['jobs']:
        folder=FORMAL/job['id']
        for name in ('actor.pt','actor.json','initial-actor.pt','history.json','selection.json','plan.json','progress.json',
                     'selected/actor.pt','selected/actor.json'):
            add(folder/name)
        for episode in plan['development_checkpoints']:
            add(folder/f'development-ep{episode:04d}.json');add(folder/f'actor-ep{episode:04d}.pt')
    for directory in (EVALUATION,REPORT,MECHANISM,ACCEPTANCE,output):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.gz','.md','.csv','.png','.svg'):add(path)
    for row in browser_runs:
        for path in (ROOT/'generated/web'/row['run_id']).glob('*.json'):add(path)
    checksums={name:digest(path) for name,path in files.items()}
    archive=output/'evidence-and-models.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as bundle:
        for index,(name,path) in enumerate(sorted(files.items())):
            require(digest(path)==checksums[name],'Archive input changed during packing')
            bundle.write(path,name)
            if index%100==0:print(json.dumps(dict(archived=index,total=len(files))),flush=True)
        bundle.writestr('SHA256SUMS.json',json.dumps(checksums,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None,'ZIP integrity failed')
        require(len(bundle.namelist())==len(files)+1,'Archive entry count differs')
        import hashlib
        for name,checksum in checksums.items():
            require(hashlib.sha256(bundle.read(name)).hexdigest()==checksum,'Archived content hash differs')
    write(output/'archive-manifest.json',dict(schema='hyra-scale100-evidence-archive-v1',verified=True,
        archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,files=len(files),
        completion_audit_sha256=digest(output/'completion-audit.json'),
        preserves_flood_and_uav_separate_folders=True,optimizer_recoveries_not_duplicated=True))
    print(json.dumps(dict(complete=True,requirements=len(requirements),jobs=36,trials=252,
                         output=str(output),archive_bytes=archive.stat().st_size)),flush=True)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--browser-evidence',type=Path,required=True)
    parser.add_argument('--screenshot',type=Path,required=True)
    args=parser.parse_args()
    execute(args.output,args.browser_evidence,args.screenshot)
