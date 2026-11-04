"""All C(6,2) pairs: equal64 screening -> top5 256 -> top2 1024 -> conditional2048."""
import copy,gc,hashlib,importlib.util,itertools,json,os,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
LEGACY=BASE.parent/'uav-linked-training-20261101';OLD_RUN=LEGACY/'run01'
spec=importlib.util.spec_from_file_location('legacy_linked_engine',LEGACY/'linked_train.py')
sys.path.insert(0,str(LEGACY));e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
from mixed_core import install
ALGORITHMS=('happo','hatrpo','hasac','maac','mat','coma')
PAIRS={'_'.join(p):p for p in itertools.combinations(ALGORITHMS,2)}
e.BASE=BASE;e.ARCHIVE=e.OUT/'uav-all-pairs-20261101';e.DOC=e.REPO/'uav-workbench/docs/all-pairs-20261101';e.PAIRS=PAIRS
for a in ('happo','mat','coma'):e.PARENTS[a]=e.OUT/f'uav6-total256-20261101/{a}/latest.pt'
hooks=install(e);original_evaluate=e.evaluate
ALLOW_ABLATIONS=False

def evaluate(*args,**kwargs):
    if kwargs.get('coverage',True) is False and not ALLOW_ABLATIONS:return {'status':'deferred_until_256_shortlist'}
    return original_evaluate(*args,**kwargs)
e.evaluate=evaluate

def emit(run,status,**kwargs):
    e.write(run/'queue-status.json',dict(status=status,pid=os.getpid(),**kwargs))
    print(json.dumps(dict(status=status,**kwargs)),flush=True)

def make_protocol():
    p=e.read(OLD_RUN/'protocol.json');p['pairs']=PAIRS
    p['authorization']='Latest user requests all15 unordered pairs of all6 allowed algorithms, irrespective of single-model weaknesses; permits64 first if256 all is expensive.'
    p['parents']={a:dict(path=str(path),sha256=e.sha(path),pretrained_episodes=512 if a=='hatrpo' else 256) for a,path in e.PARENTS.items()}
    p['source_hashes'].update({str(path):e.sha(path) for path in (Path(__file__),BASE/'mixed_core.py',BASE/'protocol-addendum.md')})
    p.update(kind='C6choose2 broad screening, not high-level hierarchy',excluded_pairs={},
      adaptation='HAPPO/HATRPO use own value estimates; COMA keeps actual teammates fixed in its counterfactual baseline; MAT decodes the actual preceding joint action, not unused private proposals. Forced/planned decisions have zero policy credit. Adam optimizers skip agents with no owned free decisions.',
      update_schedule='Existing3 pairs use byte-identical legacy controller and update functions. Additional pairs: if one on-policy member, update it per frozen two-episode block; if two, alternate one member per fresh block, never reuse that on-policy batch for the second. Off-policy members get4 batch64 updates with mixed-team targets. Per-member optimizer update counts reported separately from environment episodes.',
      learning='Six explicit research learner mechanisms, with owned-action and per-member value adapters detailed in update_schedule; no substitution by a prohibited method.',
      mat_off_policy='When MAT is a teammate, resample downstream causal actions for each candidate action in HASAC/MAAC actor expectations. This is a stochastic conditional-context adaptation, not a paper reproduction.',
      schedule='All15 reach64 joint episodes. Screening uses only0/64 checkpoints, even where older256 already exists. Top5 to256, top2 to1024, qualifying finalists to2048. Keep best/latest separate. 64 screening omits no-cover ablations; shortlisted pairs add matched0 and256 no-cover ablations. Same72 complete validation matrix.',
      reuse='Old three-pair queue is stopped and preserved. Compatible prefix checkpoints/actions/data/parents reused with explicit provenance, no rerun. Previously completed >64 episodes hidden from64 ranking; reused later only if the pair advances. Existing historical failures retained.',
      ranking='Same frozen lexicographic conflict-group violations, whole-task completion, RCR, worst-assignment RCR, coverage, lowerACC, lowerST. Within each budget use best available0/64/... checkpoint; report when best remains0, no learning gain. Rank all15 together only after every pair reaches64. Expansion is research, not acceptance.',
      budget='Pipeline total ceiling5504 common episodes:15*64+5*192+2*768+2*1024, including reusable compatible prefix episodes. No doubling for2 models. SingleCPU2threads, one heavy workload. Budget boundary4h and8GiB; preserved checkpoint then no automatic retry.',
      hard_stop='Manual STOP_AFTER_CHECKPOINT, numerical/runtime/integrity failure, disk below8GiB or4h at stage boundary; no automatic retry. Stage targets bound comparison budget to5504 common episodes including reused prefixes.',
      references=p['references']+['https://arxiv.org/abs/1705.08926','https://arxiv.org/abs/2205.14953'])
    return p

def load_cases(run):
    assert e.read(OLD_RUN/'queue-status.json')['status'] in ('manual_stopped','completed'), 'Old queue must have stopped'
    shutil.copy2(OLD_RUN/'dataset.json',run/'dataset.json');manifest=e.read(run/'dataset.json');cases={'train':[],'validation':[]}
    for row in manifest['cases']:
        path=Path(row['path']);assert e.sha(path)==row['sha256'];water=np.load(path,mmap_mode='r',allow_pickle=False)
        assert row['finite_nonnegative'] and abs(row['mass_residual_m3'])<=max(1e-6,abs(row['source_volume_m3'])*1e-9)
        if row['split']=='train':cases['train'].append((path.stem,water))
        else:cases['validation'].append(('legacy' if row['seed'] in (2101,2201) else 'new',path.stem,water))
    assert len(cases['train'])==16 and len(cases['validation'])==6
    return cases

def preflight(cases,run):
    checks=[];water=cases['train'][0][1]
    for a in ALGORITHMS:
        e.seed_all(41);m,ck=e.adapted(a);old=e.Learner(a,e.OLD_DIM);old.load_state_dict(ck['model']);obs=e.Env(water,9001).observe();mask=np.ones((4,9),bool)
        for role in e.ROLES:
            e.seed_all(73);x,l,v=old.act(obs,mask);e.seed_all(73);y,k,w=m.act(e.augment(obs,role),mask)
            assert np.array_equal(x,y) and np.allclose(l,k,atol=3e-6) and abs(v-w)<3e-6
        checks.append(dict(test='zero_role_padding_preserves_parent',algorithm=a,passed=True))
    for cid,pair in PAIRS.items():
        e.seed_all(91);models=[e.adapted(a)[0] for a in pair];replay=[];before=copy.deepcopy([m.checkpoint() for m in models])
        model_before=[e.model_hash([m]) for m in models]
        for block in range(2):
            _,rows=e.rollout(models,water,990001+block,e.ROLES[0],True)
            assert any((r['mask'].sum(-1)>1).any() for r in rows)
            e.update_block(models,rows,replay)
        assert all(e.model_hash([m])!=h for m,h in zip(models,model_before)),cid
        assert all(m.updates>0 for m in models),cid
        for j,m in enumerate(models):
            if m.algorithm=='mat':continue # shared encoder changes; loss masks enforce ownership
            for i in range(4):
                if e.ROLES[0][i]!=j:
                    for k,v in m.actors[i].state_dict().items():assert torch.equal(v,before[j]['model'][f'actors.{i}.{k}']),(cid,j,i,k)
        _,rows=e.rollout(models,water,990010,e.ROLES[0],True)
        saved=run/'preflight'/f'{cid}.pt';saved.parent.mkdir(exist_ok=True);e.save(saved,e.checkpoint(models,replay,4,{}))
        loss=e.update_block(models,rows,replay);expected=e.model_hash(models)
        restored,rp=e.restore(torch.load(saved,map_location='cpu',weights_only=False));loss2=e.update_block(restored,rows,rp)
        assert expected==e.model_hash(restored) and loss==loss2,cid
        checks.append(dict(test='both_members_learn_ownership_exact_resume',pair=cid,passed=True))
        e.write(run/'preflight-progress.json',dict(completed_pairs=sum('pair' in c for c in checks),target_pairs=15,current=cid))
        print(json.dumps(dict(preflight=cid,passed=True)),flush=True)
        del models,restored,before,replay,rp;gc.collect()
    # Explicit legacy dispatch equivalence supports reuse, not merely similar metrics.
    for pair in hooks['legacy_pairs']:
        e.seed_all(53);models=[e.adapted(a)[0] for a in pair];env=e.Env(water,9001);obs=env.observe()
        original=hooks['original_controller']();current=e.Controller()
        for _ in range(12):
            state=e.rng_state();x=original.decide(models,obs,env.available(),e.ROLES[0]);e.restore_rng(state)
            y=current.decide(models,obs,env.available(),e.ROLES[0]);assert all(np.array_equal(x[k],y[k]) for k in (0,1,2,4))
            obs,_,_,_=env.step(x[0])
        checks.append(dict(test='legacy_dispatch_action_stream_equivalence',pair=pair,passed=True))
    # Causal decoder: changing current/future actions cannot affect agent i logits.
    m,_=e.adapted('mat');obs=e.augment(e.Env(water,9001).observe(),e.ROLES[0]);o=torch.from_numpy(obs)[None]
    with torch.no_grad():
        encoded=m.policy.encode(o);a=torch.tensor([[0,1,2,3]]);b=a.clone();b[:,2:]=7
        assert torch.equal(m.policy.decode(encoded,a)[:,:3],m.policy.decode(encoded,b)[:,:3])
    checks.append(dict(test='mat_causal_action_prefix',passed=True))
    e.write(run/'preflight.json',dict(status='passed',checks=checks,excluded_pairs=[]))

def import_prefix(cid,limit,run,identity):
    if tuple(PAIRS[cid]) not in hooks['legacy_pairs']:return 0
    src=OLD_RUN/cid;dst=run/cid
    if not src.exists():return 0
    all_old=[p for p in src.glob('checkpoint-*.pt') if int(p.stem.split('-')[1])<=limit]
    if not all_old:return 0
    old_max=max(int(p.stem.split('-')[1]) for p in all_old)
    have=e.read(dst/'latest.json')['joint_episode'] if (dst/'latest.json').exists() else -1
    if old_max<=have:return have
    dst.mkdir(exist_ok=True);provenance=e.read(dst/'imported-prefix.json') if (dst/'imported-prefix.json').exists() else {'source_run':str(src),'checkpoints':[]}
    original_protocol=e.read(OLD_RUN/'protocol.json')
    for a in PAIRS[cid]:assert e.sha(e.PARENTS[a])==original_protocol['parents'][a]['sha256']
    for path,digest in original_protocol['source_hashes'].items():assert e.sha(Path(path))==digest
    for p in sorted(all_old):
        ep=int(p.stem.split('-')[1])
        if ep<=have:continue
        ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['identity']['protocol_sha256']==e.sha(OLD_RUN/'protocol.json') and ck['identity']['dataset_sha256']==identity['dataset_sha256']
        old_identity=ck['identity'];ck['identity']=identity;ck['imported_from']=dict(path=str(p),sha256=e.sha(p),original_identity=old_identity)
        e.save(dst/p.name,ck);provenance['checkpoints'].append(dict(episode=ep,source_sha256=e.sha(p),migrated_sha256=e.sha(dst/p.name)))
    for p in src.glob('eval-*.json'):
        ep=int(p.stem.split('-')[1])
        if ep>old_max or 'no-cover' in p.name:continue
        if (dst/p.name).exists():assert e.sha(dst/p.name)==e.sha(p)
        else:shutil.copy2(p,dst/p.name)
        if (src/p.stem).exists():shutil.copytree(src/p.stem,dst/p.stem,dirs_exist_ok=True)
    old_rows=[json.loads(x) for x in (src/'episodes.jsonl').read_text(encoding='utf8').splitlines() if json.loads(x)['joint_episode']<=old_max]
    (dst/'episodes.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in old_rows),encoding='utf8')
    vals=[e.read(p) for p in dst.glob('eval-*.json') if 'no-cover' not in p.name and int(p.stem.split('-')[1])<=old_max]
    best=max(vals,key=e.quality);ep=best['joint_episode'];latest=dst/f'checkpoint-{old_max:04d}.pt'
    e.write(dst/'best.json',dict(joint_episode=ep,evaluation=f'eval-{ep:04d}.json',file=f'checkpoint-{ep:04d}.pt',quality=e.quality(best)))
    e.write(dst/'latest.json',dict(joint_episode=old_max,file=latest.name,sha256=e.sha(latest)))
    e.write(dst/'status.json',dict(status='reused_compatible_prefix',joint_episode=old_max,target=limit,best_joint_episode=ep,best_summary=best['summary']))
    e.write(dst/'imported-prefix.json',provenance);return old_max

def rank_at(run,limit,ids):
    scores={};best_records={}
    for cid in ids:
        vals=[e.read(p) for p in (run/cid).glob('eval-*.json') if 'no-cover' not in p.name and int(p.stem.split('-')[1])<=limit]
        assert any(v['joint_episode']==limit for v in vals),(cid,limit)
        best=max(vals,key=e.quality);scores[cid]=e.quality(best);best_records[cid]=dict(episode=best['joint_episode'],summary=best['summary'],quality=scores[cid])
    ranked=sorted(ids,key=lambda cid:(scores[cid],cid),reverse=True)
    return dict(budget=limit,ranking=ranked,best_within_budget=best_records,scope='research screening; not field acceptance')

def update_report(run):
    q=e.read(run/'queue-status.json');lines=['# 六模型15组联动筛选进度','',f"队列状态：{q['status']}；当前阶段{q.get('stage','准备')}；当前组合{q.get('current','无')}。",'',
      '|组合|实际共同回合|最佳版本|完整任务|人员服务|覆盖|潜在冲突/步|','|---|---:|---:|---:|---:|---:|---:|']
    for cid in PAIRS:
        d=run/cid
        if not (d/'best.json').exists():lines.append(f'|{cid}|等待|—|—|—|—|—|');continue
        best=e.read(d/'best.json');v=e.read(d/best['evaluation'])['summary'];state=e.read(d/'status.json') if (d/'status.json').exists() else {'joint_episode':0}
        lines.append(f"|{cid}|{state['joint_episode']}|{best['joint_episode']}|{v['completed']}/{v['n']}|{v['RCR']:.2%}|{v['coverage']:.4%}|{v['ACC']:.5f}|")
    lines+=['','全15组先按同一64回合预算比较；HAPPO、MAT、COMA没有因单模型成绩被排除。已训练超过64回合的旧组合，其额外结果不进入64筛选。父水场16训练/6验证、72次回放由布局/分工/动作种子组合产生，不是72个独立事件。','',
      '最佳版本为0意味着未证明训练收益；覆盖与施救规则的收益单独说明。旧单模型/旧组合/失败结果保留。64筛选后前5到256，再前2到1024，2048需满足原有连续改善条件。','',
      '兼容处理：旧三组使用原控制与更新函数，检查后复用完整前缀；新增双在线学习器交替更新，每次仅使用新采集的固定团队轨迹，各自使用自己的价值估计。历史经验型学习器使用混合队友生成目标，含MAT时重算后续因果动作。非负责动作及强制动作不记为自由策略梯度。','',
      '成本：首阶段15×64=960共同任务回合，部分从旧队列复用；不按两成员翻倍。每成员优化更新次数可能不同，记录在检查点。此为一训练种子筛选；封存50、三训练种子复验、现场洪水/人员、通信续航验证未执行。','',
      '代码/数据/权重哈希、启动检查、全部场次与轨迹保留；旧项目源码不改，停用该隔离目录即可回退。数值或运行异常停止，不自动跳过失败组合。无推送或外发。']
    for name in ('selection-0064.json','selection-0256.json','extension-1024.json'):
        if (run/name).exists():lines+=['',name+': '+json.dumps(e.read(run/name),ensure_ascii=False)]
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def publish(run,final=False):
    update_report(run)
    for target in (e.ARCHIVE,e.DOC):
        target.mkdir(parents=True,exist_ok=True)
        for name in ('all_pairs.py','mixed_core.py','protocol-addendum.md'):shutil.copy2(BASE/name,target/name)
        shutil.copy2(LEGACY/'linked_train.py',target/'legacy-linked-engine.py');shutil.copy2(LEGACY/'observed_planner.py',target/'observed_planner.py')
        for p in run.rglob('*'):
            if not p.is_file() or p.suffix in ('.pt','.partial','.npy'):continue
            d=target/p.relative_to(run);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
        e.write(target/'live-run.json',dict(status=e.read(run/'queue-status.json')['status'],live_path=str(run),models=str(run),data=str(OLD_RUN/'dataset.json')))
        e.write(target/'checkpoint-index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=e.sha(p)) for p in run.rglob('checkpoint-*.pt')])
    if final:
        proto=e.read(run/'protocol.json')
        for p,h in proto['source_hashes'].items():assert e.sha(Path(p))==h
        for a,p in e.PARENTS.items():assert e.sha(p)==proto['parents'][a]['sha256']
        audit={cid:e.validate_records(run/cid) for cid in PAIRS if (run/cid).exists()}
        e.write(run/'integrity-audit.json',dict(status='passed',records=audit,source_and_parents_unchanged=True))
        for target in (e.ARCHIVE,e.DOC):shutil.copy2(run/'integrity-audit.json',target/'integrity-audit.json')
        e.update_logs('## 2026-11-01 15组全组合队列阶段结束\n\n实际状态'+e.read(run/'queue-status.json')['status']+'。已按64→前5组256→前2组1024→有连续收益者2048执行，实际轮次与全部失败见outputs/uav-all-pairs-20261101/report.md。原三组被人工新范围取代，兼容前缀复用不重跑；不因单模型差排除组合。封存50未用，结果双份，勿重复启动。\n\n')

def ensure_baseline_ablation(cid,cases,run):
    folder=run/cid
    if (folder/'eval-0000-no-cover.json').exists():return
    ck=torch.load(folder/'checkpoint-0000.pt',map_location='cpu',weights_only=False);models,_=e.restore(ck)
    original_evaluate(models,cases['validation'],folder,0,coverage=False);del models;gc.collect()

def run_stage(cid,cases,run,target,identity,start):
    e.guard_budget(run,start);emit(run,'training',stage=target,current=cid)
    imported=import_prefix(cid,target,run,identity)
    folder=run/cid
    if target>=256:ensure_baseline_ablation(cid,cases,run)
    if imported>=target and (folder/f'eval-{target:04d}.json').exists():
        if target>=256 and not (folder/f'eval-{target:04d}-no-cover.json').exists():
            models,_=e.restore(torch.load(folder/f'checkpoint-{target:04d}.pt',map_location='cpu',weights_only=False))
            original_evaluate(models,cases['validation'],folder,target,coverage=False);del models;gc.collect()
        result=e.read(folder/'status.json')
    else:result=e.train_stage(cid,cases,run,target,identity)
    publish(run)
    if result['status']=='manual_stopped':emit(run,'manual_stopped',current=cid,stage=target);publish(run,True);return False
    return True

def main():
    global ALLOW_ABLATIONS
    run=BASE/'run01';run.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    e.write(run/'protocol.json',make_protocol());emit(run,'preparing',stage=64)
    cases=load_cases(run);publish(run);preflight(cases,run)
    identity=dict(protocol_sha256=e.sha(run/'protocol.json'),dataset_sha256=e.sha(run/'dataset.json'))
    e.update_logs('## 2026-11-01 六模型15组全组合初筛已启动\n\n按最新人工要求全部C(6,2)=15组，不凭单模型表现排除HAPPO/MAT/COMA。启动兼容检查通过；统一64回合预算，前5续到256、前2到1024、连续改善者2048。复用16训练/6验证合成父场，封存50未用。旧3组队列已保存停止，HATRPO+HASAC256及HATRPO+MAAC2保留，仅兼容前缀按公平预算复用。实时入口outputs/uav-all-pairs-20261101/live-run.json；单CPU2线程，不要重复启动。\n\n')
    for cid in PAIRS:
        if not run_stage(cid,cases,run,64,identity,start):return
    screen=rank_at(run,64,list(PAIRS));screen['selected']=screen['ranking'][:5];e.write(run/'selection-0064.json',screen);publish(run)
    ALLOW_ABLATIONS=True
    for cid in screen['selected']:
        if not run_stage(cid,cases,run,256,identity,start):return
    selected=rank_at(run,256,screen['selected']);selected['selected']=selected['ranking'][:2];e.write(run/'selection-0256.json',selected);publish(run)
    for cid in selected['selected']:
        if not run_stage(cid,cases,run,1024,identity,start):return
    decisions={cid:e.trend2048(run/cid) for cid in selected['selected']};e.write(run/'extension-1024.json',decisions)
    for cid,d in decisions.items():
        if d['eligible'] and not run_stage(cid,cases,run,2048,identity,start):return
    emit(run,'completed',elapsed_s=time.perf_counter()-start,completed={cid:e.read(run/cid/'status.json')['joint_episode'] for cid in PAIRS},selected256=screen['selected'],selected1024=selected['selected'],extension=decisions)
    publish(run,True)

if __name__=='__main__':
    try:main()
    except Exception:
        run=BASE/'run01'
        if run.exists():
            e.write(run/'failure.json',dict(traceback=traceback.format_exc()));emit(run,'failed')
            try:publish(run)
            except Exception:pass
        raise
