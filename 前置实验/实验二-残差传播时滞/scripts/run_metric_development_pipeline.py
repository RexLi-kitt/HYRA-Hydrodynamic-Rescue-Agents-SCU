"""Continue the authorized development workflow when audited families become ready."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hyra_validation.artifacts import new_run,write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=read(ROOT/'configs/metric_models.development.v1.json')
    data=Path(read(ROOT/'results/latest_metric_development_data.json')['run']);cache=Path(read(ROOT/'results/latest_metric_feature_cache.json')['folder'])
    run=new_run('metric_development_pipeline',dict(protocol=protocol,data_run=str(data),feature_cache=str(cache),pilot_or_confirmation_allowed=False))
    sources=read(run/'provenance.json')['files']
    for rel in sources:
        p=run/'frozen_source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    write_json(ROOT/'results/latest_metric_development_pipeline.json',dict(run=str(run),completed=False))
    print('PIPELINE '+str(run),flush=True)
    def verify():
        for rel,checksum in sources.items():assert digest(ROOT/rel)==checksum,rel
    def wait_for(split,count):
        previous=None
        while True:
            verify();ready=[]
            for i in range(count):
                family=f'metric_v2_{split}_{i:03}'
                receipts=[data/'data_audit'/(family+'.json'),cache/(family+'.json'),cache/'input_audit'/(family+'.json')]
                if all(p.exists() for p in receipts):
                    raw,features,inputs=map(read,receipts)
                    assert raw['passed'] and inputs['passed'] and features['cache_sha256']==inputs['cache_sha256']
                    ready.append(family)
            state=dict(stage=f'waiting_audited_{split}',ready=len(ready),required=count,ready_families=ready)
            write_json(run/'progress.json',state)
            if previous!=len(ready):print('READY '+json.dumps({k:v for k,v in state.items() if k!='ready_families'}),flush=True);previous=len(ready)
            if len(ready)==count:return
            time.sleep(20)
    def execute(name,args):
        verify();write_json(run/'progress.json',dict(stage=name,command=args))
        with (run/(name+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        if code:raise RuntimeError(f'{name} exited {code}; original log retained; no automatic restart')
    wait_for('train',24)
    execute('training',['scripts/train_metric_models.py','--train-count','24','--epochs','120'])
    models=read(ROOT/'results/latest_metric_model_training.json');modelrun=Path(models['run']);assert models['completed']
    registry=read(modelrun/'registry.json');assert len(registry)==79
    write_json(run/'trained_models.json',models)
    wait_for('validation',12)
    execute('global_validation',['scripts/evaluate_metric_models.py','--models',str(modelrun),'--split','validation','--start','0','--count','12'])
    selection=read(ROOT/'results/latest_metric_development_selection.json');selected_run=Path(selection['run']);assert not selection['training_diagnostic']
    assert len(read(selected_run/'candidates.json'))==protocol['total_candidates']
    execute('paired_contrasts',['scripts/summarize_metric_development.py','--run',str(selected_run)])
    execute('information_action',['scripts/check_metric_information_action.py','--selection',str(selected_run)])
    write_json(run/'completion.json',dict(completed=True,model_run=str(modelrun),selection_run=str(selected_run),pilot_or_confirmation_opened=False,
        remaining='general population, delayed inputs, wrong-link controls, freeze, pilot planning and fresh confirmation; objective not achieved'))
    write_json(ROOT/'results/latest_metric_development_pipeline.json',dict(run=str(run),completed=True,model_run=str(modelrun),selection_run=str(selected_run)))
    print('DEVELOPMENT_PIPELINE_DONE '+str(run),flush=True)


if __name__=='__main__':main()
