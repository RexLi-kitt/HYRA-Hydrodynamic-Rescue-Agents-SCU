"""Archive completed local priority experiment without copying large weights."""
import hashlib,json,shutil,sys
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finish(out,repo):
    docs=repo/'uav-workbench/docs/priority-20261101';docs.mkdir(parents=True,exist_ok=True)
    comp=json.loads((out/'comparison.json').read_text(encoding='utf-8')) if (out/'comparison.json').exists() else {}
    select=json.loads((out/'selection.json').read_text(encoding='utf-8')) if (out/'selection.json').exists() else {}
    results=json.loads((out/'results.json').read_text(encoding='utf-8')) if (out/'results.json').exists() else {}
    status=json.loads((out/'queue-status.json').read_text(encoding='utf-8'))
    old=out.parent/'uav6-total256-20261101'
    lines=['# 无人机改进与优先训练（2026-11-01）','',
           '用户要求先改进，再优先训练更好的模型。比较保持原任务、水情、奖励和学习器不变；增加可选的重复受阻动作约束，并对六种模型统一进行固定动作及两种子随机执行的验证。自动施救开启，覆盖补漏规划关闭。',
           '',f"队列状态：{status['status']}。",'',
           '## 256回合原模型的执行改动对照','',
           '表中为覆盖率/人员获救比例；均为同一开发验证集，不是最终50场测试或独立训练种子复验。随机执行与旧固定执行分开列出。',
           '', '|算法|原固定执行|随机执行，原控制|随机执行，加防受阻|是否采用防受阻|',
           '|---|---|---|---|---|']
    def pair(s):return f"{100*s['coverage']:.2f}% / {100*s['RCR']:.2f}%"
    for a,c in comp.items():
        original=json.loads((old/a/'validation-0256-service.json').read_text(encoding='utf-8'))['summary']
        lines.append(f"|{a.upper()}|{pair(original)}|{pair(c['original']['summary'])}|{pair(c['guarded']['summary'])}|{'是' if c['guard_accepted'] else '否'}|")
    lines += ['', '采用规则在结果前冻结：防受阻版人员获救、覆盖均不下降，潜在冲突不增加，且至少一项有严格改善；不满足即保留该模型原控制。该改动是学习模型与控制规则组合的改变，不声称神经网络权重得到提升。',
              '', '## 优先训练结果','',
              '优先名次：'+ '、'.join(x['algorithm'].upper() for x in select.get('ranking',[])),
              '入选：'+ '、'.join(x.upper() for x in select.get('selected',[])),
              '', '|模型|新增回合|实际累计回合|保留最佳回合|训练前覆盖/获救|最佳覆盖/获救|停止原因|',
              '|---|---|---|---|---|---|---|']
    for a,r in results.items():
        base=next(x['validation']['summary'] for x in select['ranking'] if x['algorithm']==a)
        lines.append(f"|{a.upper()}|{r['added_episodes']}|{r['final_episode']}|{r['best_episode']}|{pair(base)}|{pair(r['best_validation']['summary'])}|{r['reason']}|")
    lines += ['', '最佳模型只按预先登记的随机执行验证规则选择，原固定动作成绩及后续固定动作成绩仍全部保存。最高1024回合是上限，连续三次验证无改善或到一小时成本上限时停止，不能把早停写成已完成1024。选用随机执行是单列方法配置，不追溯改写原六种基线评分。',
              '', '## 更改、依据、回退与限制','',
              '- RetryGuard只读取公共观测中的已发生位移和自身既往动作；连续3次移动未发生位移时，暂禁失败方向5次决策，并在有其他合法移动时暂不悬停。正常主动悬停本身不触发，服务掩码优先。物理避碰继续原样执行，潜在冲突照常计分。',
              '- 对照固定了六种256模型、2个验证水场、4种人员布局及两个动作采样种子3101/3102。此阶段未训练参数；同水场下多布局不能算新的独立洪水地图。',
              '- 排名优先考虑较差动作种子的平均获救比例，再考虑总体获救、覆盖和潜在冲突。取前两名串行续训，其他四种本轮未扩训；不是删除失败算法。',
              '- 续训恢复原256检查点的模型、优化器、目标网络、随机状态与有效回放；学习器和奖励未改。每32回合独立保存完整检查点，每64回合做随机/固定验证；过程保存自由悬停占比、采样动作惊讶度、实际损失与梯度范数。惊讶度不是逐状态精确策略熵。',
              '- 8项新增检查通过，包括服务优先、正常悬停、不扩展动作权限、真实动作/下一掩码一致性，以及增加日志前后MAAC更新权重逐值相等。实际MAT更新也通过有限性检查。',
              '- 原源代码和全部256检查点保持不变；新代码在独立priority-20261101目录，关闭RetryGuard/使用旧训练入口即可回退。大检查点只保留outputs；本目录manifest记录其路径/大小/SHA256。',
              '- 未运行正式封存50测试，未改洪水模型，未推送PR。仅单个训练种子延续及小规模复用验证，不能声称泛化可靠性或总体准确率。',
              '- 如未达到改善条件，原256仍是保留最佳点；下一步应检查失败行为或增加独立训练种子，不能仅以回合数增加声称成功。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    index=[]
    for p in out.rglob('*'):
        if p.is_file() and p.name not in ('artifact-index.json',) and '.partial' not in p.suffix:
            index.append(dict(path=str(p),relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
    (out/'artifact-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in out.rglob('*'):
        if p.is_file() and p.suffix not in ('.pt','.partial'):
            q=docs/p.relative_to(out);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            assert sha(q)==sha(p)
    msg='## 2026-11-01 无人机优先训练本次队列结束\n\n'
    msg+=f"状态{status['status']}；入选{','.join(select.get('selected',[]))}。"
    for a,r in results.items():msg+=f"{a.upper()}实际累计{r['final_episode']}、保留最佳{r['best_episode']}回合，最佳覆盖/获救{pair(r['best_validation']['summary'])}；"
    msg+='固定动作/随机执行/控制规则收益分列，原256与全部退化保留。protocol、逐场数据、模型索引及报告双份outputs/uav6-priority-20261101与uav-workbench/docs/priority-20261101。未运行封存50测试或推送PR。\n\n'
    for folder in [out.parent,repo/'terrain-workbench/docs']:
        for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
            p=folder/name;previous=p.read_text(encoding='utf-8-sig')
            if msg.splitlines()[0] not in previous:p.write_text(msg+previous,encoding='utf-8')
    print('Report archived:',str(out/'report.md'),flush=True)

if __name__=='__main__':finish(Path(sys.argv[1]),Path(sys.argv[2]))
