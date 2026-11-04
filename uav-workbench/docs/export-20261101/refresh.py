"""Read-only extraction of all available single/pair experiment records."""
import argparse, hashlib, itertools, json, math, os, re, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
DOC=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\docs\singles1024-20261101')
ARCHIVE=OUT/'uav-singles1024-20261101'
ALGOS=('happo','hatrpo','hasac','maac','mat','coma')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def mean(rows,k):return sum(r[k] for r in rows)/len(rows)
def extract():
 sources=[];source_map={}
 def source(p,data=None):
  p=Path(p).resolve();key=str(p)
  if key not in source_map:
   b=p.read_bytes() if data is None else data;ident=f'S{len(sources)+1:04d}';source_map[key]=ident
   sources.append(dict(id=ident,path=key,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
  return source_map[key]
 d=dict(timestamp=datetime.now().astimezone().isoformat(),overview=[],evaluations=[],details=[],training=[],checkpoints=[],sources=sources,queues={})
 pairrun=ROOT/'uav-all-pairs-20261101/run01';singlerun=BASE/'run01'
 ids=list(ALGOS);pairs=['_'.join(x) for x in itertools.combinations(ALGOS,2)]
 def append_eval(p,experiment,candidate,episode,control,rows,nparents,summary=None):
  if not rows:return None
  sid=source(p);n=len(rows);s=dict(n=n,completed=sum(bool(r['task_complete']) for r in rows),**{k:mean(rows,k) for k in ('RCR','coverage','ACC','ST')},max_conflicts=max(r['conflicts'] for r in rows))
  s['completion']=s['completed']/n
  if summary:
   assert n==summary['n'] and s['completed']==summary['completed'],str(p)
   for k in ('completion','RCR','coverage','ACC','ST'):assert abs(s[k]-summary[k])<1e-9,(p,k)
  vid=f'V{len(d["evaluations"])+1:04d}'
  d['evaluations'].append(dict(id=vid,experiment=experiment,candidate=candidate,episode=episode,control=control,parents=nparents,source=sid,**s))
  for i,r in enumerate(rows):d['details'].append({**r,'evaluation_id':vid,'experiment':experiment,'candidate':candidate,'episode':episode,'control':control,'row':i+1,'source':sid})
  return vid
 for exp,run,candidates in [('当前单模型',singlerun,ids),('当前双模型',pairrun,pairs)]:
  queue=read(run/'queue-status.json') if (run/'queue-status.json').exists() else {'status':'not_started'};d['queues'][exp]=queue
  extension=read(run/'extension-1024.json') if (run/'extension-1024.json').exists() else {}
  for cid in candidates:
   folder=run/cid;status=read(folder/'status.json') if (folder/'status.json').exists() else {};actual=status.get('joint_episode',0)
   best=read(folder/'best.json') if (folder/'best.json').exists() else {};eval_ids={};latest_ep=None
   for p in sorted(folder.glob('eval-*.json')):
    value=read(p);ep=value['joint_episode'];ctrl='关闭覆盖规划' if 'no-cover' in p.name else '服务+重试+覆盖规划'
    vid=append_eval(p,exp,cid,ep,ctrl,value['episodes'],6,value['summary']);eval_ids[p.name]=vid
    if 'no-cover' not in p.name:latest_ep=ep if latest_ep is None else max(latest_ep,ep)
   if (folder/'episodes.jsonl').exists():
    p=folder/'episodes.jsonl';raw=p.read_bytes();sid=source(p,raw)
    lines=raw.splitlines();last=0
    for i,line in enumerate(lines):
     try:r=json.loads(line)
     except json.JSONDecodeError:
      if i==len(lines)-1 and not raw.endswith(b'\n'):break
      raise
     assert r['joint_episode']==last+1,(cid,last,r['joint_episode']);last=r['joint_episode']
     d['training'].append(dict(**r,experiment=exp,candidate=cid,source=sid,source_line=i+1))
    actual=max(actual,last)
   decision=extension.get(cid);target=1024 if exp=='当前单模型' else status.get('target',64)
   if decision:ext='晋级2048' if decision['eligible'] else '未满足2048条件'
   elif target>=1024:ext='待1024后评定'
   else:ext='未入1024名单'
   fallback=queue['status'] if queue.get('current')==cid or queue['status'] in ('waiting_for_pairs','preflight') else 'queued'
   d['overview'].append(dict(experiment=exp,candidate=cid,actual=actual,target=target,status=status.get('status',fallback),parent_episodes='+'.join('512' if a=='hatrpo' else '256' for a in cid.split('_')),best_episode=best.get('joint_episode'),best_id=eval_ids.get(best.get('evaluation')),latest_evaluation=latest_ep,latest_id=eval_ids.get(f'eval-{latest_ep:04d}.json') if latest_ep is not None else None,extension=ext,optimizer_updates=status.get('optimizer_updates'),decision=decision))
   for p in sorted(folder.glob('checkpoint-*.pt')):
    d['checkpoints'].append(dict(experiment=exp,candidate=cid,episode=int(p.stem.split('-')[1]),bytes=p.stat().st_size,path=str(p)))
 # Historical single model source export preserves all stages, including failures.
 hp=ROOT/'uav-tables-20261101/data.json';hist=read(hp);source(hp)
 histmap={}
 for r in hist['sources']:
  p=OUT/r['file'];sid=source(p);assert next(s for s in sources if s['id']==sid)['sha256']==r['sha256'],str(p);histmap[r['id']]=sid
 hvid={}
 for r in hist['checkpoints']:
  d['checkpoints'].append(dict(experiment='历史单模型/'+r['stage'],candidate=r['algorithm'].lower(),episode=r['episode'],bytes=r['bytes'],path=str(OUT/r['file'])))
 for v in hist['evaluations']:
  rows=[r for r in hist['details'] if r['validation_id']==v['id']]
  assert len(rows)==v['n'],v['id']
  vid=append_eval(OUT/v['file'],'历史单模型',v['algorithm'].lower(),v['episode'],v['stage']+' / '+v['control']+' / '+v['mode'],rows,v['water_parents'])
  hvid[v['id']]=vid
 for r in hist['training']:
  d['training'].append({**r,'joint_episode':r['total_episode'],'candidate':r['algorithm'].lower(),'experiment':'历史单模型','source':histmap[r['source']]})
 # Old pair training, fixed-weight combinations and the208 prescreen episodes are separate experiments.
 old=OUT/'uav6-joint-training-20261101'
 for p in sorted(old.glob('*/validation-*.json')):
  if not re.fullmatch(r'validation-\d+(?:-greedy)?\.json',p.name):continue
  v=read(p);rows=v.get('episodes',[])
  append_eval(p,'历史双模型学习',p.parent.name,int(re.search(r'\d+',p.name).group()),'旧设置/'+('fixed' if 'greedy' in p.name else 'sample'),rows,2)
 for p in sorted(old.glob('*/episodes.jsonl')):
  sid=source(p)
  for i,line in enumerate(p.read_text(encoding='utf8').splitlines()):
   r=json.loads(line);d['training'].append({**r,'joint_episode':r.get('joint_episode',r.get('episode')),'candidate':p.parent.name,'experiment':'历史双模型学习','source':sid,'source_line':i+1})
 fixed=OUT/'uav6-combinations-20261101'
 for p in sorted(fixed.glob('*.json')):
  if not(re.fullmatch(r'P\d+-allocation[01]\.json',p.name) or re.fullmatch(r'(HASAC|HATRPO|MAT)\d+\.json',p.name)):continue
  v=read(p);append_eval(p,'冻结权重组合对照',p.stem,0,'只评估未联合学习',v.get('episodes',[]),2)
 p=OUT/'uav-plan-preselection-20261101/results.json'
 for cid,rows in read(p).items():append_eval(p,'冻结权重规划预选',cid,0,'仅预选未训练',rows,4)
 # Prior linked3 run includes a non-selected256 branch; retain its distinct record, no additive count claim.
 old=ROOT/'uav-linked-training-20261101/run01'
 for folder in sorted(old.iterdir()):
  if not folder.is_dir():continue
  for p in sorted(folder.glob('eval-*.json')):
   v=read(p);append_eval(p,'旧三组合队列(部分复用)',folder.name,v['joint_episode'],'关闭覆盖规划' if 'no-cover' in p.name else '服务+重试+覆盖规划',v['episodes'],6,v['summary'])
 strict=ROOT/'uav-strict-eval-20261101/run01'
 d['strict_status']=read(strict/'status.json') if (strict/'status.json').exists() else None
 d['strict_summary']=read(strict/'strict-summary.json') if (strict/'strict-summary.json').exists() else None
 if (strict/'results.jsonl').exists():
  p=strict/'results.jsonl';raw=p.read_bytes();source(p,raw);groups={}
  for line in raw.splitlines():
   r=json.loads(line);key=(r['candidate'],r['variant'],r['mode'],r['stratum']);groups.setdefault(key,[]).append(r)
  for (cid,variant,mode,stratum),rows in sorted(groups.items()):append_eval(p,'严格测试/'+stratum,cid,rows[0]['checkpoint_episode'],variant+'/'+mode,rows,len({r['case'] for r in rows}))
 if d['strict_summary']:source(strict/'strict-summary.json')
 d['audit']=dict(unique_overview=len(d['overview'])==21,historical_training_count=len(hist['training']),summary_recomputed=True,
  evaluations=len(d['evaluations']),validation_rows=len(d['details']),training_rows=len(d['training']),historical_reuse_note='旧三组合的兼容前缀已在15组合中复用；不将两表轮次相加。历史阶段单独列出，不与当前环境混排。')
 assert len(d['overview'])==21 and len(hist['training'])==3072
 write(BASE/'data.json',d)
 lines=['# 单模型与双模型训练全量数据', '', '快照时间：'+d['timestamp'],'','|类型|模型|已训练回合|阶段目标|最佳回合|最新评估|2048|','|---|---|---:|---:|---:|---:|---|']
 for r in d['overview']:lines.append(f"|{r['experiment']}|{r['candidate']}|{r['actual']}|{r['target']}|{r['best_episode'] if r['best_episode'] is not None else '未评估'}|{r['latest_evaluation'] if r['latest_evaluation'] is not None else '未评估'}|{r['extension']}|")
 lines+=['','开发验证：单模型36回放，双模型72回放，共6个合成验证父水场；不按原始完成次数直接比较。最佳0表示未证明新增学习收益。历史4训练/2验证结果独立保留。完整任务=覆盖全区域且20人服务完成；人员服务比例不等于完整任务比例。','',f"本快照含{len(d['evaluations'])}份评估、{len(d['details'])}条逐场验证、{len(d['training'])}条逐回合训练记录。来源文件与哈希见工作簿及data.json。",'', '严格测试另行预登记50个独立主场景+20个方向/布局压力场景，权重锁定后才生成；包括纯规划和随机探索对照。其状态：'+json.dumps(d['strict_status'],ensure_ascii=False),'', '训练及严格测试结束自动刷新全量工作簿；Excel是注明时间的快照。当前进程见live-run.json。']
 (BASE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 return d
def publish():
 for root in (ARCHIVE,DOC):
  root.mkdir(parents=True,exist_ok=True)
  for name in ('data.json','report.md','refresh.py','build.mjs','verification.json','单双模型训练全量数据-20261101.xlsx'):
   p=BASE/name
   if p.exists():shutil.copy2(p,root/name)
 strict=ROOT/'uav-strict-eval-20261101/run01'
 if (strict/'strict-summary.json').exists():
  result=read(strict/'strict-summary.json');message='# 严格检验结果\n\n状态：'+result['status']+'。总回放：'+str(result['rows'])+'。逐场结果及父场景区间见strict-summary.json、strict-summary.csv和总工作簿严格测试页。测试没有用于训练、选检查点或晋级2048。仅一个独立训练种子；合成任务结果不能解释为真实搜救成功率。准备阶段审查记录见information-audit.json和smoke-audit.json。\n'
  for root in (strict.parent,OUT/'uav-strict-eval-20261101',DOC.parent/'strict-eval-20261101'):
   (root/'report.md').write_text(message,encoding='utf8')
def main():
 args=argparse.ArgumentParser();args.add_argument('--workbook',action='store_true');opt=args.parse_args();d=extract()
 if opt.workbook:
  node=Path(r'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
  subprocess.run([str(node),'--expose-gc','--max-old-space-size=6144',str(BASE/'build.mjs')],check=True,cwd=BASE)
 publish();print(json.dumps(d['audit'],ensure_ascii=False))
if __name__=='__main__':main()
