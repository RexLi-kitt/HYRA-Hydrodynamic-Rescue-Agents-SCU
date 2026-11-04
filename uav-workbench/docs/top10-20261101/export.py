"""Extract validation-selected, test-frozen checkpoints without changing any model."""
import collections
import datetime
import gc
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import sys

BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
EXP=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    dest=BASE/'package';dest.mkdir(exist_ok=False);(dest/'checkpoints').mkdir()
    strict=ROOT/'uav-strict-eval-20261101/run01';singles=ROOT/'uav-singles1024-20261101/run01';pairs=ROOT/'uav-all-pairs-20261101/run01'
    frozen=read(strict/'frozen-candidates.json');parents=read(singles/'protocol.json')['parents']
    ordered=sorted(frozen['candidates'],key=lambda c:(tuple(c['validation_quality']),c['id']),reverse=True)
    selection=dict(created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),
      reason='Top10 by existing frozen validation_quality, ID descending tie-break; NOT selected using test scores',
      frozen_source_sha256=sha(strict/'frozen-candidates.json'),candidates=ordered[:10],all21_order=[c['id'] for c in ordered])
    write(dest/'selection.json',selection)
    # Read test data only after the validation-only selection has been saved.
    raw=(strict/'results.jsonl').read_bytes();(dest/'test-snapshot.jsonl').write_bytes(raw.rsplit(b'\n',1)[0]+b'\n')
    tests=[json.loads(line) for line in raw.split(b'\n')[:-1] if line.strip()]
    sys.path.insert(0,str(EXP))
    import torch
    from flood1km.learners import Learner
    torch.set_num_threads(1)
    records=[]
    for rank,c in enumerate(ordered[:10],1):
        src=Path(c['path']);assert sha(src)==c['sha256'],src
        ck=torch.load(src,map_location='cpu',weights_only=False);models=[];parts=[]
        for state in ck['models']:
            m=Learner(state['algorithm'],state['dim']);m.load_state_dict(state['model'],strict=True)
            params=dict(total=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
            parts.append(dict(algorithm=state['algorithm'],input_dim=state['dim'],action_count=9,
              parameters=params,parent_episodes=parents[state['algorithm']]['pretrained_episodes'],
              updates_at_checkpoint=state['updates'],gamma=state['gamma'],alpha=state['alpha'],
              optimizer_group_lrs=[[g['lr'] for g in opt['param_groups']] for opt in state['optimizers']],
              learning_rate_note='HATRPO policy uses trust-region update; stored Adam policy groups do not imply they are used.'))
            del m
        source_run=singles if c['family']=='single' else pairs
        status=read(source_run/c['id']/'status.json');val=read(source_run/c['id']/f'eval-{c["episode"]:04d}.json')
        summaries=[]
        for stratum,expected in [('iid50',50),('orientation_cluster20',20)]:
            for variant in ('full','no-cover'):
                rs=[r for r in tests if r['candidate']==c['id'] and r['stratum']==stratum and r['variant']==variant and r['mode']=='fixed']
                roles=1 if c['family']=='single' else 2; by_case=collections.defaultdict(list)
                for r in rs:by_case[r['case']].append(r)
                complete=len(by_case)==expected and len(rs)==expected*roles and all(len(v)==roles for v in by_case.values())
                means={key:statistics.mean(statistics.mean(float(x[key]) for x in vs) for vs in by_case.values()) for key in ('task_complete','RCR','coverage','ACC','ST')} if rs else None
                summaries.append(dict(stratum=stratum,variant=variant,complete=complete,observed_parents=len(by_case),expected_parents=expected,rollouts=len(rs),expected_rollouts=expected*roles,
                  completed_rollouts=sum(r['task_complete'] for r in rs),means=means,max_conflicts=max((r['conflicts'] for r in rs),default=None)))
        name=f'{c["id"]}-ep{c["episode"]:04d}.pt';target=dest/'checkpoints'/name;shutil.copy2(src,target);assert sha(target)==c['sha256']
        records.append(dict(rank=rank,id=c['id'],family=c['family'],best_episode=c['episode'],trained_episodes=status['joint_episode'],
          parts=parts,total_parameters=sum(x['parameters']['total'] for x in parts),trainable_parameters=sum(x['parameters']['trainable'] for x in parts),
          checkpoint_mib=src.stat().st_size/1024**2,checkpoint=name,source_path=str(src),sha256=c['sha256'],validation=val['summary'],strict_test=summaries))
        del ck;gc.collect()
    data=dict(snapshot=selection['created'],test_records=len(tests),test_snapshot_sha256=sha(dest/'test-snapshot.jsonl'),
      scope='Development shortlist only, not final test ranking; full controllers include coverage and service rules',records=records)
    write(dest/'statistics.json',data)
    lines=['# 10个候选最佳版本与参数统计','',f'快照：{data["snapshot"]}。严格测试已有{len(tests)}/5630条回放，仍在运行。',
      '', '名单只按测试前已保存的开发验证排序选择。不是新测试最终排名；不修改全21方案的正式测试范围。全部检查点与测试前锁定SHA256一致。',
      '', '| 开发排序 | 模型/组合 | 已训练轮次 | 保留版本轮次 | 预训练轮次 | 可训练参数 | 总参数含目标网络 | 完整检查点MiB |',
      '|---|---|---:|---:|---|---:|---:|---:|']
    for r in records:
        prior=' / '.join(str(p['parent_episodes']) for p in r['parts'])
        lines.append(f'| {r["rank"]} | {r["id"].upper().replace("_","＋")} | {r["trained_episodes"]} | {r["best_episode"]} | {prior} | {r["trainable_parameters"]:,} | {r["total_parameters"]:,} | {r["checkpoint_mib"]:.2f} |')
    lines+=['','预训练轮次与本次轮次分列，不相加当作同条件从零训练。保留0轮表示选用本次训练前的预训练版本。参数量是全体4架无人机策略与训练价值网络的实际参数，组合按两个学习器求和；总参数另包含冻结目标网络。完整检查点还含优化器、随机状态和经验池，因此文件大小不等于参数内存，也不等于部署推理内存。',
      '', '## 同一批新地图的测试进度与指标','',
      '| 候选 | 主测试完整任务 | 人员服务比例 | 平均覆盖 | 潜在冲突/步 | 单场最多冲突 | 压力测试 | 关闭覆盖主测试完整任务 |',
      '|---|---|---|---|---|---|---|---|']
    def fraction(t):
        if not t['complete']:return f'未完成 {t["rollouts"]}/{t["expected_rollouts"]}'
        return f'{t["completed_rollouts"]}/{t["rollouts"]}（{t["means"]["task_complete"]:.1%}）'
    for r in records:
        ts=r['strict_test'];a=ts[0];n=ts[1];b=ts[2];m=a['means'] if a['complete'] else None
        cells=[r['id'].upper().replace('_','＋'),fraction(a),f'{m["RCR"]:.2%}' if m else '—',f'{m["coverage"]:.2%}' if m else '—',f'{m["ACC"]:.5f}' if m else '—',str(a['max_conflicts']) if m else '—',fraction(b),fraction(n)]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','未完成项不展示可比较得分；statistics.json保留原进度和部分结果。完整任务要求全区域被观察且20人服务完成，冲突另列而未合入完成判定；高完成比例不代表飞行安全达标。主测试50个独立合成父场、压力测试20个独立父场；组合两种角色分工先场内平均，不能当100张独立地图。仍只有一个训练种子，完整方法包含覆盖规划与服务规则。',
      '', '## 文件与使用','',
      '`checkpoints`保存10份未经修改的完整原检查点，可供原研究环境继续读取。`selection.json`保存排序依据与冻结文件地址。`statistics.json`保存逐学习器参数、折扣因子、保存的优化器学习率、更新次数和全部分层指标。`manifest.json`保存本包每个文件的SHA256。原训练与所有不利结果均保留。',
      '', '本包为上述时间快照，不自动刷新。最终测试完成后再提供全量比较；此名单不授权据测试结果调参或重选版本。']
    (dest/'参数统计.md').write_text('\n'.join(lines),encoding='utf8')
    body=[];in_table=False
    for line in lines:
        if line.startswith('|---'):continue
        if line.startswith('|'):
            if not in_table:body.append('<table>');in_table=True
            cells=line.strip('|').split('|');body.append('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in cells)+'</tr>')
        else:
            if in_table:body.append('</table>');in_table=False
            if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line:body.append('<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</table>')
    (dest/'参数统计.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HYRA模型参数统计</title><style>body{font:16px/1.65 Microsoft YaHei,sans-serif;margin:32px;color:#182b42}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #ccd5df;padding:8px}tr:first-child{background:#16324f;color:white}tr:nth-child(even){background:#f1f5fa}h2{margin-top:32px}p{max-width:1150px}</style>'+''.join(body)+'</html>',encoding='utf8')
    files=[dict(file=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    write(dest/'manifest.json',dict(files=files))
    print(json.dumps([dict(id=r['id'],episode=r['best_episode'],trainable=r['trainable_parameters'],total=r['total_parameters'],mib=round(r['checkpoint_mib'],2)) for r in records],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
