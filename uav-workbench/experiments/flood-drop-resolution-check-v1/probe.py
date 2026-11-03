"""Shared step1 check on all frozen new-map confirmation scenes; no tuning."""
from pathlib import Path
import sys,importlib.util,copy,json,time,traceback,argparse
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('flow_fog',HERE.parent/'flood-drop-flow-fog-v1/experiment.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def run(out,previous,weights):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();done=0;rows=[];torch.set_num_threads(4)
    methods=['coverage100','flow_coverage','gru-16','gru-32_flowblend'];params=['view60-w360']
    cases=copy.deepcopy(f.read(previous/'scenarios.json'))
    for c in cases:c['parameter']='view60-w360'
    for c in cases:c['scenario']['config']['step']=1
    source={str(Path(__file__)):f.sha(__file__),**f.read(previous/'protocol.json')['source_hashes']}
    protocol=dict(version='shared-step1-frozen-newmap-check-v1',methods=methods,parameters=params,old_step_s=5,new_step_s=1,old_results_sha=f.sha(previous/'results.jsonl'),
        change='Same flight speed10m/s but observe and make decisions each1s instead5s for allmethods. Both sampling and control change; do not attribute difference solely to sensors. Original5s physical drift/servicing already per1s. No model weight training.',
        reason='120s trace on first TRAINcase at20m showed50m jumps overshooting nearby goals, repeated reversals,378observed cells vs1744 at1s. Full shared-baseline diagnostic needed before interpreting learning advantage under narrowvisibility.',
        acceptance='All12 previously frozen newmap scenes and both frozen candidates;48rollouts. No parameter/model selection. Check whether60m observed advantage survives shared step1. Since previous results have been examined, this is a resolution sensitivity check, not a newly blind cohort.',
        source_hashes=source,weights={str(weights/(n+'.pt')):f.sha(weights/(n+'.pt')) for n in ('gru-16','gru-32')},budget='48serialCPUrollouts,15min cap,STOPfile;0epochs')
    f.write(out/'protocol.json',protocol);f.write(out/'scenarios.json',cases)
    def status(s,**kw):f.write(out/'status.json',dict(status=s,elapsed_s=time.monotonic()-start,rollouts=done,**kw))
    models={}
    for n in ('gru-16','gru-32'):
        ck=torch.load(weights/(n+'.pt'),weights_only=True,map_location='cpu');m=f.Model(ck['kind'],ck['mean']);m.load_state_dict(ck['state']);m.eval();mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
        def predict(x,m=m,mu=mu,sd=sd):
            with torch.no_grad():return m(torch.tensor((x-mu)/sd)).numpy()
        models[n]=predict
    try:
        for method in methods:
            for c in cases:
                if (out/'STOP').exists():raise InterruptedError('User STOP')
                if time.monotonic()-start>900:raise TimeoutError('15min ceiling')
                status('evaluating',method=method,case=c['id'])
                e=f.FlowFogEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],method,models.get(method.removesuffix('_flowblend')))
                while True:
                    _,_,finished,_=e.step(e.choices()[1])
                    if finished:break
                persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(40)]
                supplied=int((e.status==2).sum());expired=int((e.status==3).sum());outside=int((e.status==4).sum());assert supplied+expired+outside==40
                for p in persons:
                    if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                assert np.isfinite(e.pos).all() and np.isfinite(e.person).all()
                r=dict(method=method,parameter=c['parameter'],case=c['id'],people=40,supplied=supplied,expired=expired,outside=outside,persons=persons,
                    unseen_expired=sum(p['status']==3 and (p['first_seen']<0 or p['first_seen']>p['deadline']) for p in persons),refloat_events=int(e.refloat_counts.sum()),coverage_cells=int(e.known.sum()),conflicts=int(e.conflicts))
                rows.append(r);done+=1
                with (out/'results.jsonl').open('a',encoding='utf8') as fp:fp.write(json.dumps(r,allow_nan=False)+'\n')
        old=f.read(previous/'summary.json')['results'];results={p:{m:dict(step5=old[m],step1=f.aggregate([r for r in rows if r['method']==m and r['parameter']==p])) for m in methods} for p in params}
        for p,h in {**source,**protocol['weights']}.items():assert f.sha(p)==h,p
        f.write(out/'summary.json',dict(status='completed',results=results,rollouts=done,epochs=0,only_resolution_sensitivity=True));status('completed')
    except BaseException as ex:
        f.write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.previous,a.weights)
