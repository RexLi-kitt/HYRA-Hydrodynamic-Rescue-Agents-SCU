from pathlib import Path
import json,shutil,hashlib,os,subprocess,time,struct
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');R=O/'round91';R.mkdir(exist_ok=False);C=R/'full_event';C.mkdir()
assert json.loads((O/'round90/parallel-status.json').read_text())['use_four_for_reference']
B=O/'round90/cost7200_omp4';E=O/'round86/build-isotropic-cpu64/Release/lisflood.exe'
for n in ['Carlisle_5m.asc','carlisle.bci','carlisle_run7.bdy']:shutil.copy2(B/n,C/n)
p=(B/'case.par').read_text().replace('sim_time 7200','sim_time 245700').replace('saveint 1800','saveint 21600')
(C/'case.par').write_text(p)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol={'case_id':'Carlisle2005-Kabir2020-5m-independent-input-candidate','purpose':'first complete-window conditional reference simulation with pre-registered real maximum-outline comparison','target_s':245700,'wall_limit_s':3600,'threads':4,'source_revision':'eadfc848a6d37cb3476da2fe6ec7087262d9ef54','exe_sha256':sha(E),'configuration_hashes':{p.name:sha(p) for p in C.iterdir() if p.is_file()},'observation_protocol_sha256':sha(O/'round90/observation-protocol.json'),'observation_mask_sha256':sha(O/'round90/observation-footprint-mask.npz'),'all_observations':'all5 polygons/all263 marks retained; only independently preregistered covered cells eligible for positive wet check; outside and edge remain unknown','input_source':'Run7.bdy independently verified equal to 2005 Run1.csv with original23 source widths, no fit','not_exact_original_software':'v8 fast backend, author v6.3.1 example; no score inheritance from DG2 or production','limits':['dry initial state as archive configuration, not observed state','no proven restart; one bounded run, do not relaunch automatically','max extent only, no time-of-day/road/person claims','no known-dry coverage so no overall accuracy/false alarm estimate','initial wall cost extrapolation ~8min is not a duration promise'],'failure_gate':'exit nonzero, wall3600s, nonfinite latest mass or abs per-step Verror >max(1e-4,1e-8*abs(volume)); retain all failures','physical_defaults_changed':False,'full_event_no_prior_runs':True}
(R/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copy2(Path(__file__),R/'run_full91.py')
A=D/'round91-artifacts';A.mkdir(exist_ok=False);shutil.copy2(R/'protocol.json',A/'protocol.json');shutil.copy2(R/'run_full91.py',A/'run_full91.py')
entry='''## 2026-10-31 第91轮：5米完整事件一次性条件运行已登记

继第90轮入口与4线程一致性关口，使用相同独立SGC快速后端，原5m地形、23点/原2005曲线、n=.055、FREE .0006和干初态，目标245700s。一次运行墙钟上限3600s，无可信checkpoint；失败保留、不自动重启。outputs/round91/status.json和full_event/run.log、results/check.mass为当前状态，以实际进程为准，禁止重复启动。生产、网页、PR不变。

观测已冻结在round90/observation-protocol.json及footprint-mask；整场完成后比较最大范围，不把早期局部结果或软件一致率当现实准确率。外部/边界未知、263点全保留。启动不算现场通过；报告待完整结果。完整原始结果完成后须双份归档。当前run_full91.py监控退出与重大质量异常，读取PID/时间先核对再续作。

'''
for root in [O,D]:
 for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  f=root/n;f.write_text(entry+f.read_text(encoding='utf8'),encoding='utf8')
env=os.environ.copy();env['OMP_NUM_THREADS']='4';start=time.perf_counter();reason=None
with (C/'run.log').open('w') as f:
 proc=subprocess.Popen([str(E),'-v','case.par'],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT)
 status={'state':'running','pid':proc.pid,'wrapper_pid':os.getpid(),'target_s':245700,'wall_limit_s':3600,'threads':4,'field_passes_added':0}
 def save():
  status['wall_s']=time.perf_counter()-start;(R/'status.json').write_text(json.dumps(status,indent=2))
 save();print(json.dumps(status),flush=True)
 while proc.poll() is None:
  time.sleep(10)
  mass=C/'results/check.mass'
  if mass.exists():
   with mass.open('rb') as mf:mf.seek(max(0,mass.stat().st_size-4096));tail=mf.read().decode(errors='replace').splitlines()
   for line in reversed(tail):
    try:vals=[float(v) for v in line.split()]
    except ValueError:continue
    if len(vals)!=12:continue
    status.update(simulated_s=vals[0],last_step_s=vals[1],steps=int(vals[3]),last_volume_m3=vals[5],last_Verror_m3=vals[10]);
    if not np.isfinite(vals).all() or abs(vals[10])>max(1e-4,1e-8*abs(vals[5])):reason='mass_guard'
    break
  if time.perf_counter()-start>3600:reason='wall_limit'
  if reason:proc.terminate();proc.wait(timeout=30)
  save()
 code=proc.wait()
status.update(state='completed' if code==0 and reason is None else 'failed',exit_code=code,stop_reason=reason);save()
shutil.copy2(R/'status.json',A/'status.json')
print(json.dumps(status),flush=True)
