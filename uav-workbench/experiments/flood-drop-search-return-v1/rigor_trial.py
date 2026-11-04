"""Fixed-size confirmation, simple baselines, structural and refloat coverage."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,time,traceback,io,contextlib,unittest,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from torch import nn
from rigor_support import *
from memory import augment
from window_adapter import normalized

def aggregate(rr):return dict(scenes=len(rr),people=sum(r['people'] for r in rr),supplied=sum(r['supplied'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr),all_scenes=sum(r['all_scene'] for r in rr),refloats=sum(r['refloats'] for r in rr))

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4);counts=dict(fields=0,cal_collections=0,linear_epochs=0,evaluations=0)
    prior=ROOT/'uav-count-confirmation-20261104/run01';cfg=Config(**read(prior/'protocol.json')['config']);oldsc=read(OLD/'scenarios.json')
    inputs={str(prior/n):sha(prior/n) for n in ['protocol.json','summary.json']}
    inputs.update({str(OLD/n):sha(OLD/n) for n in ['scenarios.json','normalization.json','fine-s202611142-e16.pt']})
    for i in range(24):inputs[str(OLD/f'fit-base-{i:02d}.npz')]=sha(OLD/f'fit-base-{i:02d}.npz')
    for c in oldsc['calibration']:inputs[c['field']['path']]=sha(c['field']['path'])
    sources={**read(prior/'protocol.json')['source_sha256'],**{str(HERE/n):sha(HERE/n) for n in ['rigor_support.py','rigor_trial.py']}}
    protocol=dict(version='frozen-rigor-v1',config=cfg.__dict__,source_sha256=sources,input_sha256=inputs,
        primary='40 NEW independent same-generator parents1105501..1105540,scenarios2026112700..39. FixedsampleN40 before outcomes; no interimselection or earlysuccessstop. Earlier failed andpassed20-parent tests retained separately,NOT pooled toinflateN.',
        structural='4two-valley parents1105601..04 plus4diagonal-valley parents1105605..08,scenarios2026112800..07. SameSWEsolver/totalinflowrange/40people/660window/20msensor; family-specificresults separate. These are new synthetic geometries,notrealurbanvalidation.',
        mechanisms='9existingrefloatmechanismchecks:drynonmovement,2m/1mpsboundaries,randomhazard,obstaclecapture,deadline,repeatrefloat,etc. Additional4layoutsonONE manufactureduniformrewettingfield with40forced-depthfalls,threewet/drycycles. Fullmissions eachmethod butNOTindependentfloodmapsorSWEvalidation; do not poolwithprimaryorOOD. Reportactualrefloatcounts,notjustenabledflag.',
        models='Frozenold142-e16 GRU+windowadapter+CountCoupledEnv. Intactcoverage100/flowcoverage sharejointavoidance. Twoextra SAMEinterface controls:statistic(.1+age)*(.25+2*observedwet+1-known), globalPoisson scaleFITonly; linearsoftplus256->1 on same4x64 publichistory. Samecache/transport/countcoefficient8 forallpredictors. NoforbiddenRLalgorithms.',
        simple_fit='Same24FITparents/48framesperparent/1152maplabels usedbyoldGRU. SameFITnormalizer andphysicalwindowadapter. Linearzero-inithead+logmeanbias;Adam.001,weight_decay1e-4,batch32,D4augmentation,clip5,fixedseed2026112901,64supervisedepochs. Choose8/16/32/64 bymeanparent CALPoissonloss on8existingCALparents (newshared-fixedflowrolloutlabels). NoTESTdataforfitting/normalizing/selection. Statisticalscale=sumcounts/sumrawscore onFIT.',
        freeze='Allpredictors andsimplecheckpoint selected/hashlocked BEFORE generatingallnewTESTfields. MainGRUneverretrainedorselectedagain. No retuning difficulty/score/threshold onresults. Failed panelsretained.',
        inference='Currentknownobservations only; hidden sector_counts raises for everyevaluateddeployablemethod. No future water,falltime,or true deadlines as policyinputs. Physics retains hidden truth for environment only.',
        decision='Reportpeoplepriority; original85% and>=bothplanners plusp95scene+15pp as pointtarget. Paired20000parentbootstrap95%interval; stable>=15pp claim onlyiflowerlimit>=15againstbothplanners. CheckGRUvsstatistic/linear likewise. OOD/mechanism failures cannot be hidden bypooling. Do not addGRUepochs automatically: iflinearties/beatsGRU,deep-specificbenefitunproven; ifOODloss,needdiverseFITdata beforemoreepochs. Finaltest not reused forcheckpointselection.',
        resource='ONE managedorchestrator with4CPUworkers eachtorch1thread,read-onlymmapwater; linearfit4threads occursseparately. 8CALcollections+260newmissionrollouts,48SWEfields+1manufacturedfield. 7200sbudget/STOP. NoCUDAbenefitassumed,noenvironmentinstall,productionchange,push.',
        limitations='Allwater synthetic; knownfixedmissionN40 and eventlawthreebursts. Manufacturedrewetvalidatesimplementation/stressonly. FourOODmapsperfamilyandfourlayoutsnotpopulationestimates. Preregisteredlocaltimestamp/hash,notexternalregistration.')
    write(OUT/'protocol.json',protocol)
    def status(state,**kw):write(OUT/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,**counts,**kw))
    def check():
        guard()
        if time.monotonic()-start>7200:raise TimeoutError('7200s budget')
    def poolmap(fn,jobs,phase):
        result=[]
        with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker) as pool:
            futures={pool.submit(fn,j):i for i,j in enumerate(jobs)}
            for f in as_completed(futures):
                check();r=f.result();result.append((futures[f],r))
                if phase=='calibration':counts['cal_collections']+=1
                elif phase=='fields':counts['fields']+=1
                status(phase)
                print(json.dumps(dict(phase=phase,completed=len(result),total=len(jobs))),flush=True)
        return [r for _,r in sorted(result)]
    try:
        # Unit-level manufactured observations are explicitly separate from mission performance.
        file=HERE.parent/'flood-drop-convlstm20-v1/test_refloat.py';spec=importlib.util.spec_from_file_location('refloat_contracts',file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
        (OUT/'mechanism-unit-tests.txt').write_text(stream.getvalue(),encoding='utf-8');assert result.wasSuccessful();write(OUT/'mechanism-unit-tests.json',dict(tests=result.testsRun,passed=True,source_sha256=sha(file)))
        cal=poolmap(episode,[(c,'flow_coverage',True) for c in oldsc['calibration']],'calibration');write(OUT/'calibration-collection.json',cal)
        xx=[];yy=[]
        for i in range(24):
            with np.load(OLD/f'fit-base-{i:02d}.npz') as data:
                ids=np.linspace(0,len(data['x'])-1,48).round().astype(int);xx.append(data['x'][ids]);yy.append(data['y'][ids])
        x=np.concatenate(xx);y=np.concatenate(yy);norm=read(OLD/'normalization.json');mu=np.array(norm['mu'],np.float32);sd=np.array(norm['sd'],np.float32);mean=float(y.mean())
        scale=float(y.sum()/statistic_shape(x).sum());write(OUT/'statistic.json',dict(scale=scale,fit_only=True))
        cx=[];cy=[]
        for row in cal:
            with np.load(row['data']) as d:
                ids=np.linspace(0,len(d['x'])-1,48).round().astype(int);cx.append(d['x'][ids]);cy.append(d['y'][ids])
        cx=np.stack(cx);cy=np.stack(cy)
        xt=torch.from_numpy(x);yt=torch.from_numpy(y);mt=torch.from_numpy(mu)[None,None,:,None,None];st=torch.from_numpy(sd)[None,None,:,None,None]
        torch.manual_seed(2026112901);rng=np.random.default_rng(2026112901);m=LinearCount(mean);opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-4);trainlog=[];choices={}
        for ep in range(1,65):
            check();m.train();total=0
            for ids in torch.randperm(len(x)).split(32):
                z,q=augment(xt[ids],yt[ids],int(rng.integers(4)),bool(rng.integers(2)));v=m((z-mt)/st);loss=nn.functional.poisson_nll_loss(v,q,log_input=False)
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ids)
            trainlog.append(dict(epoch=ep,fit_poisson=total/len(x)));counts['linear_epochs']=ep;status('linear_training',epoch=ep)
            if ep in (8,16,32,64):
                m.eval();losses=[]
                with torch.inference_mode():
                    for a,b in zip(cx,cy):losses.append(float(nn.functional.poisson_nll_loss(m((torch.from_numpy(a)-mt)/st),torch.from_numpy(b),log_input=False)))
                choices[ep]=dict(parent_losses=losses,mean=float(np.mean(losses)))
                torch.save(dict(state=m.state_dict(),mean=mean,mu=mu.tolist(),sd=sd.tolist(),epoch=ep,seed=2026112901),OUT/f'linear-e{ep}.pt')
        selected=min(choices,key=lambda ep:(choices[ep]['mean'],ep));import shutil;shutil.copy2(OUT/f'linear-e{selected}.pt',OUT/'linear.pt')
        write(OUT/'linear-training.json',trainlog);write(OUT/'linear-selection.json',dict(epoch=selected,calibration=choices,before_new_test=True))
        write(OUT/'predictor-lock.json',{str(p):sha(p) for p in [OUT/'linear.pt',OUT/'statistic.json',OLD/'fine-s202611142-e16.pt']})
        jobs=[('original',1105501+i,2026112700+i,cfg.__dict__) for i in range(40)]+[(('two_valleys' if i<4 else 'diagonal'),1105601+i,2026112800+i,cfg.__dict__) for i in range(8)]
        cases=poolmap(generate_case,jobs,'fields');cases+=mechanism_cases(cfg)
        oldparents={c['field']['parent'] for group in oldsc.values() for c in group};assert not oldparents&{c['field']['parent'] for c in cases}
        write(OUT/'scenarios.json',cases);waterhash={c['field']['path']:sha(c['field']['path']) for c in cases};write(OUT/'input-lock.json',dict(scenarios_sha256=sha(OUT/'scenarios.json'),water_sha256=waterhash))
        rows=[]
        with ProcessPoolExecutor(max_workers=4,initializer=initialize_worker) as pool:
            futures=[pool.submit(episode,(c,n,False)) for c in cases for n in METHODS]
            for f in as_completed(futures):
                check();r=f.result();rows.append(r);counts['evaluations']+=1
                with (OUT/'results.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(r)+'\n')
                status('evaluating',case=r['case'],method=r['method']);print(json.dumps({k:v for k,v in r.items() if k!='persons'}),flush=True)
        results={family:{n:aggregate([r for r in rows if r['family']==family and r['method']==n]) for n in METHODS} for family in ['original','two_valleys','diagonal','manufactured_rewet']}
        a=results['original'];point=a['gru-frozen']['supplied']/1600>=.85 and all(a['gru-frozen']['supplied']>=a[n]['supplied'] and a['gru-frozen']['p95_scenes']-a[n]['p95_scenes']>=6 for n in METHODS[:2])
        paired={};rng=np.random.default_rng(2026112902);idx=rng.integers(0,40,(20000,40));gr={r['case']:r for r in rows if r['family']=='original' and r['method']=='gru-frozen'};keys=sorted(gr)
        for n in METHODS[:-1]:
            br={r['case']:r for r in rows if r['family']=='original' and r['method']==n};p=np.array([(gr[k]['supplied']-br[k]['supplied'])/40*100 for k in keys]);q=np.array([(int(gr[k]['p95_scene'])-int(br[k]['p95_scene']))*100 for k in keys])
            paired[n]=dict(people_pp=float(p.mean()),scene_pp=float(q.mean()),people_95=np.quantile(p[idx].mean(1),[.025,.975]).tolist(),scene_95=np.quantile(q[idx].mean(1),[.025,.975]).tolist(),wins=int((p>0).sum()),losses=int((p<0).sum()))
        for p,h in {**sources,**inputs,**waterhash,**read(OUT/'predictor-lock.json')}.items():assert sha(p)==h,p
        stable=point and all(paired[n]['scene_95'][0]>=15 for n in METHODS[:2])
        write(OUT/'summary.json',dict(results=results,point_target=point,stable_15pp_lower_bound=stable,paired_primary=paired,counts=counts,elapsed_s=time.monotonic()-start,source_inputs_unchanged=True,gru_training_epochs=0))
        status('completed')
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':main()
