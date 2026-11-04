"""Continue both selected models to 1024 with no performance early stopping.

The previous learner, environment, controllers, rewards and validation criterion
are imported unchanged. Only orchestration/resume and stopping behavior change.
"""
import argparse,gc,json,os,random,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

EXPERIMENT=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101')
sys.path.insert(0,str(EXPERIMENT/'priority-20261101'))
import priority_train as core


class Session:
    def __init__(self, algorithm, checkpoint, prior_best, cases, identity):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.a=algorithm
        self.model=core.LoggedLearner(algorithm,saved['learner']['dim'])
        self.model.restore(saved['learner'])
        assert self.model.gamma==saved['learner']['gamma']
        self.model.check_finite()
        self.episode=int(saved['total_episode'])
        self.steps=int(saved['phase_steps'])
        self.replay=saved.get('replay',[])
        self.pending=[];self.last_loss=saved.get('last_loss',{})
        self.cases=cases;self.identity=identity
        self.guard=bool(saved['identity']['guard'])
        # A checkpoint is written before its validation. Prefer the explicit,
        # persisted post-validation selection over an older embedded fallback.
        self.best=prior_best if prior_best is not None else saved['selection_state']
        core.restore_rng(saved['rng'])
        del saved

    def consume(self,row):
        assert row['mask'][np.arange(4),row['action']].all()
        assert row['next_mask'][np.arange(4),row['next_action']].all()
        self.steps+=1
        if self.a=='hasac':
            self.replay.append(row)
            if len(self.replay)>4096:self.replay.pop(0)
            if len(self.replay)>=1024 and self.steps%32==0:
                for _ in range(4):self.last_loss=self.model.measured_update(random.sample(self.replay,64),True)
        elif self.a=='hatrpo':self.pending.append(row)
        else:raise ValueError('Only the two authorized selected methods')

    def train_one(self):
        total=self.episode+1
        name,water=self.cases['train'][(total-65)%len(self.cases['train'])]
        record=core.rollout(self.model,water,920000+total-65,self.guard,consume=self.consume)
        self.episode=total
        if self.a=='hatrpo' and total%2==0:
            self.last_loss=self.model.measured_update(self.pending,False);self.pending=[]
        return dict(total_episode=total,case=name,layout=920000+total-65,phase_steps=self.steps,
                    loss=self.last_loss,**record)

    def save(self,path):
        assert not self.pending,'Checkpoint only at an update/episode boundary'
        if path.exists():raise FileExistsError(path)
        tmp=path.with_suffix('.partial')
        torch.save(dict(learner=self.model.checkpoint(),total_episode=self.episode,phase_steps=self.steps,
                        replay=self.replay,rng=core.rng_state(),identity=self.identity,
                        selection_state=self.best,last_loss=self.last_loss),tmp)
        os.replace(tmp,path)

    def validate(self,out,checkpoint):
        sample=core.evaluate(self.model,self.cases['validation'],out,f'validation-{self.episode:04d}-sample',self.guard,False)
        greedy=core.evaluate(self.model,self.cases['validation'],out,f'validation-{self.episode:04d}-greedy',self.guard,True)
        adopted=core.significant_improvement(sample,self.best['validation'])
        if adopted:self.best=dict(episode=self.episode,path=str(checkpoint),validation=sample)
        core.write(out/'best.json',self.best)
        return dict(episode=self.episode,adopted=adopted,best_episode=self.best['episode'],
                    sample=sample['summary'],greedy=greedy['summary'])


def train(a,parent,cases,out,protocol_hash):
    out.mkdir()
    metadata=json.loads((parent/a/'latest.json').read_text(encoding='utf-8'))
    checkpoint=Path(metadata['path']);assert core.sha(checkpoint)==metadata['sha256']
    previous=json.loads((parent/'results.json').read_text(encoding='utf-8'))[a]
    assert metadata['episode']==previous['final_episode']
    prior_best=dict(episode=previous['best_episode'],path=previous['best_checkpoint'],validation=previous['best_validation'])
    identity=dict(algorithm=a,parent_checkpoint=str(checkpoint),parent_hash=metadata['sha256'],
                  protocol_hash=protocol_hash,guard=previous['guard'],target=1024,
                  performance_early_stopping=False,reward='unchanged',
                  training_layout='920000+total_episode-65; unchanged continuation',
                  parent_result_hash=core.sha(parent/'results.json'))
    s=Session(a,checkpoint,prior_best,cases,identity)
    start=s.episode;assert start=={'hasac':448,'hatrpo':704}[a]
    begin=time.perf_counter();history=[]
    core.write(out/'identity.json',identity);core.write(out/'best.json',s.best)
    core.write(out/'resume-audit.json',dict(start_episode=s.episode,phase_steps=s.steps,
        replay_size=len(s.replay),learner_updates=s.model.updates,optimizer_count=len(s.model.optimizers),
        guard=s.guard,parent_hash_verified=True,random_state_restored=True))
    while s.episode<1024:
        core.write(out/'status.json',dict(status='training',completed_episode=s.episode,target=1024,
            best_episode=s.best['episode'],performance_early_stopping=False,elapsed_s=time.perf_counter()-begin))
        row=s.train_one();row['elapsed_s']=time.perf_counter()-begin
        with (out/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if s.episode%32==0:
            cp=out/f'checkpoint-{s.episode:04d}.pt';s.save(cp)
            core.write(out/'latest.json',dict(path=str(cp),episode=s.episode,sha256=core.sha(cp)))
            print(a,'trained',s.episode,'elapsed',round(time.perf_counter()-begin,1),flush=True)
            if (out.parent/'STOP_AFTER_CHECKPOINT').exists():
                reason='manual_stop';break
        if s.episode%64==0:
            core.write(out/'status.json',dict(status='validating',completed_episode=s.episode,target=1024,
                best_episode=s.best['episode'],performance_early_stopping=False,elapsed_s=time.perf_counter()-begin))
            v=s.validate(out,cp);history.append(v)
            core.write(out/'validation-history.json',history)
            # Even a worse validation continues training under the user's new request.
            print(a,'validated',s.episode,json.dumps(v),flush=True)
    else:reason='target_reached'
    assert core.sha(checkpoint)==metadata['sha256']
    result=dict(status='completed' if s.episode==1024 else 'stopped',reason=reason,start_episode=start,
                final_episode=s.episode,added_episodes=s.episode-start,target=1024,
                best_episode=s.best['episode'],best_checkpoint=s.best['path'],best_validation=s.best['validation'],
                guard=s.guard,elapsed_s=time.perf_counter()-begin,validation_history=history,
                performance_early_stopping=False,formal_test_executed=False)
    core.write(out/'status.json',result)
    return result


def run(args):
    if not args.execute:raise RuntimeError('Explicit --execute required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    old_protocol=json.loads((args.parent/'protocol.json').read_text(encoding='utf-8'))
    assert old_protocol['source_hashes']==core.sources()
    for name,h in old_protocol['new_code_hashes'].items():
        assert core.sha(EXPERIMENT/'priority-20261101'/name)==h
    manifest=args.pilot/'data/dataset.json'
    assert old_protocol['input_hashes'][str(manifest)]==core.sha(manifest)
    cases={};items=json.loads(manifest.read_text(encoding='utf-8'))['cases']
    for item in items:
        p=Path(item['path']);assert core.sha(p)==item['sha256']
        cases.setdefault(item['split'],[]).append((p.name,np.load(p,mmap_mode='r',allow_pickle=False)))
    parent_hashes={}
    for a in ['hasac','hatrpo']:
        m=json.loads((args.parent/a/'latest.json').read_text(encoding='utf-8'))
        assert core.sha(Path(m['path']))==m['sha256'];parent_hashes[a]=m
    protocol=dict(date='2026-11-01',request='Continue HASAC448 and HATRPO704 to total1024; no performance early stop',
        algorithms=['hasac','hatrpo'],starting_episodes={'hasac':448,'hatrpo':704},target=1024,
        added_episodes={'hasac':576,'hatrpo':320},parent=str(args.parent),parent_hashes=parent_hashes,
        parent_protocol_hash=core.sha(args.parent/'protocol.json'),original_module_hashes=core.sources(),
        orchestration_hashes={p.name:core.sha(p) for p in Path(__file__).parent.glob('*.py')},
        unchanged=['learning algorithms','optimizers and hyperparameters','rewards','service and RetryGuard settings',
                   'observations and physics','training layout sequence','validation layouts and action seeds','best-model adoption criterion'],
        resume='full learner/target/optimizer/RNG/replay and counters; no reset from best point',
        validation='each64: same2 parents x4 layouts, sampled3101/3102 and greedy separately; reuse development set',
        best_selection='carry previous accepted best, apply unchanged priority_train.significant_improvement; not necessarily maximum RCR',
        checkpoints='independent full state each32; prior files retained',
        performance_early_stopping=False,walltime_early_stopping=False,
        stop='numerical/runtime exception, or user STOP_AFTER_CHECKPOINT; no automatic retry',
        cpu_threads=2,concurrent_training_jobs=1,sealed50_test=False)
    core.write(args.output/'protocol.json',protocol);ph=core.sha(args.output/'protocol.json')
    begin=time.perf_counter();results={}
    for a in protocol['algorithms']:
        if (args.output/'STOP_AFTER_CHECKPOINT').exists():break
        core.write(args.output/'queue-status.json',dict(status='training',algorithm=a,pid=os.getpid(),
            completed=list(results),target=1024,performance_early_stopping=False))
        results[a]=train(a,args.parent,cases,args.output/a,ph)
        core.write(args.output/'results.json',results);gc.collect()
        if results[a]['status']!='completed':break
    completed=len(results)==2 and all(r['final_episode']==1024 for r in results.values())
    core.write(args.output/'queue-status.json',dict(status='completed' if completed else 'stopped',
        completed={a:r['final_episode'] for a,r in results.items()},elapsed_s=time.perf_counter()-begin,
        performance_early_stopping=False,formal_test_executed=False))


def finish(out,repo,parent):
    results=json.loads((out/'results.json').read_text(encoding='utf-8')) if (out/'results.json').exists() else {}
    before=json.loads((parent/'results.json').read_text(encoding='utf-8'))
    lines=['# HASAC/HATRPO完整1024回合续训（2026-11-01）','',
           '从上一队列最新448/704点继续，恢复模型、目标网络、优化器、随机状态、回放与步数。仅取消表现与耗时提前停止；算法、奖励、输入和验证选模规则不变。','',
           '|模型|本次新增|实际累计|保留版本|续训前较好覆盖/获救|当前保留覆盖/获救|',
           '|---|---|---|---|---|---|']
    def pair(s):return f"{s['coverage']*100:.2f}% / {s['RCR']*100:.2f}%"
    audit={}
    for a,r in results.items():
        logs=[json.loads(x) for x in (out/a/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [x['total_episode'] for x in logs]==list(range(r['start_episode']+1,r['final_episode']+1))
        assert len(logs)==r['added_episodes']
        audit[a]=dict(logged_episodes=len(logs),contiguous=True,checkpoint_files=len(list((out/a).glob('checkpoint-*.pt'))))
        lines.append(f"|{a.upper()}|{r['added_episodes']}|{r['final_episode']}|{r['best_episode']}|{pair(before[a]['best_validation']['summary'])}|{pair(r['best_validation']['summary'])}|")
    lines+=['','## 所有本次阶段验证（随机执行）','','|模型|回合|覆盖|人员获救|潜在冲突/步|全任务完成比例|是否替换保留版本|','|---|---|---|---|---|---|---|']
    for a,r in results.items():
        for v in r['validation_history']:
            s=v['sample'];lines.append(f"|{a.upper()}|{v['episode']}|{s['coverage']*100:.2f}%|{s['RCR']*100:.2f}%|{s['ACC']:.5f}|{(1-s['timeout_fraction'])*100:.2f}%|{'是' if v['adopted'] else '否'}|")
    lines+=['','## 解释与回归','',
            '- 保留版本不等于最后1024版本，也不保证在任意洪水中最优；仅按既定开发验证规则选择。所有阶段（含更高获救但冲突增加的版本）均保留，不事后改阈值。',
            '- 旧256/448/704结果、原最佳HASAC256与HATRPO512均保留。随机执行、固定执行、规则控制收益不混算。新增checkpoint每32保存，只有每64的固定验证点参与本轮选模。',
            '- 训练样本来自4个既有合成水场；验证2水场×4布局×2动作种子，不是16个独立洪水场景，也不是独立训练种子。封存50场测试尚未进行。',
            '- Resume测试对HASAC与HATRPO分别核对中断保存/恢复后真实后续回合、模型权重、优化器、随机状态与步数一致，详见preflight；未仅凭成功读取文件认定续训正确。',
            '- 无新环境、奖励或学习器修复，不因此声称洪水精度提高。停止条件只保留人工停止与数值/运行异常；失败不自动反复启动。',
            '- 新增代码独立continue1024-20261101目录，可回到旧入口和旧完整检查点；不覆盖源代码或旧结果。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8');core.write(out/'audit.json',audit)
    index=[]
    for p in out.rglob('*'):
        if p.is_file() and p.name!='artifact-index.json' and p.suffix!='.partial':
            index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=core.sha(p)))
    core.write(out/'artifact-index.json',index)
    docs=repo/'uav-workbench/docs/full1024-20261101';docs.mkdir(parents=True,exist_ok=True)
    for p in out.rglob('*'):
        if p.is_file() and p.suffix not in ('.pt','.partial'):
            q=docs/p.relative_to(out);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            assert core.sha(q)==core.sha(p)
    state=json.loads((out/'queue-status.json').read_text(encoding='utf-8'))['status']
    note='## 2026-11-01 两种模型1024续训队列结束\n\n'+f'实际状态{state}。'
    for a,r in results.items():note+=f"{a.upper()}实际累计{r['final_episode']}、本次新增{r['added_episodes']}、保留{r['best_episode']}版本，覆盖/人员获救{pair(r['best_validation']['summary'])}。"
    note+='本轮取消性能早停，其他学习/物理/选模规则不变；全阶段及固定/随机对照保留，独立50场测试未运行。双份outputs/uav6-full1024-20261101与docs/full1024-20261101，完整模型留outputs。未推送PR。\n\n'
    for folder in [out.parent,repo/'terrain-workbench/docs']:
        for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
            p=folder/name;old=p.read_text(encoding='utf-8-sig')
            if note.splitlines()[0] not in old:p.write_text(note+old,encoding='utf-8')
    print('Reports and full results archived',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    try:run(args)
    except Exception:
        if args.output.exists():
            core.write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
            core.write(args.output/'queue-status.json',dict(status='failed',pid=os.getpid()))
        raise
