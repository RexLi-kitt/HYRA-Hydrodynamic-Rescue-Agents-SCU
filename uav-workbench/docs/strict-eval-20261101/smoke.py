"""Exercise the isolated evaluator on existing TRAIN fields; never generates tests."""
import json, pathlib, tempfile
import strict_eval as v
v.bootstrap();v.torch.set_num_threads(1)
v.RUN=v.BASE/'smoke';v.RUN.mkdir(exist_ok=False);(v.RUN/'traces').mkdir()
# Prevent smoke status from masquerading as the real queued test.
v.emit=lambda *args,**kwargs:None
e=v.e;row=e.read(v.SINGLE/'run01/dataset.json')['cases'][0];water=v.np.load(row['path'],mmap_mode='r',allow_pickle=False)
people=e.Env(water,778899).people
case=dict(id='training-only-smoke',stratum='iid50',path=row['path'],layout_seed=778899,people=people.tolist())
records=[]
for names in [('happo',),('mat',),('hatrpo','hasac'),('hasac','mat')]:
 models=[e.adapted(a)[0] for a in names];candidate=dict(id='_'.join(names),family='single' if len(names)==1 else 'pair',episode=0)
 roles=(0,0,0,0) if len(names)==1 else (0,1,1,0)
 for variant in ('full','no-cover'):records.append(v.trial(candidate,models,case,roles,0,variant,'fixed',510001,len(records)))
for cid,variant in [('pure_planner','planner'),('uniform_service','no-cover')]:
 models=None
 if cid=='uniform_service':
  m=e.Learner('happo',e.DIM)
  with v.torch.no_grad():
   for p in m.parameters():p.zero_()
  m.eval();models=[m]
 records.append(v.trial(dict(id=cid,family='rule'),models,case,(0,0,0,0),0,variant,'fixed' if models is None else 'sample',510001,len(records)))
frozen=dict(validation_chosen_leaders={'single':'happo','pair':'hasac_mat'})
v.aggregate(records,frozen);out=v.read(v.RUN/'strict-summary.json')
for r in out['summary']:
 assert r['independent_parents']==1
 low,high=r['all_assignments_complete']['wilson_ci95'];assert 0<=low<=high<=1
 if r['all_assignments_complete']['completed_parents']==1:assert low<1
for seed in (123,321):
 p=v.clustered_people(seed);assert len(v.np.unique(p,axis=0))==20;assert (p>0).all() and (p<1000).all()
v.write(v.BASE/'smoke-audit.json',dict(status='passed',existing_train_field_only=True,test_fields_generated=False,rollouts=len(records),checks=['Single/pair legacy and causal MAT adapters','Explicit held-out people injection','Full/no-cover/planner/uniform controls','Trace and metric reconciliation','Weight hash unchanged during each trial','Scene grouping and nondegenerate Wilson endpoint interval','Clustered unique20-person layouts']))
print(json.dumps(dict(status='passed',smoke_rollouts=len(records),test_fields_generated=False)))
