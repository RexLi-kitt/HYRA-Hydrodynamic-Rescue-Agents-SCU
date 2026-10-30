import sys,unittest,json,hashlib,time,os
from pathlib import Path
os.environ['HYRA_FLOOD_BACKEND']='numpy'
root=Path('C:/Users/Lenovo/Desktop/无人机大创/HYRA/terrain-workbench/terrain-lab')
sys.path[:0]=[str(root),str(root/'tests')]
import test_refinement,test_refined_mode_support
out=Path('outputs/round12');out.mkdir(exist_ok=True)
skip='test_legacy_refinement_request_runs_and_exports_native_grid'
names=[n for n in unittest.defaultTestLoader.getTestCaseNames(test_refinement.RefinementTests) if n!=skip]
plan=dict(category='existing invariant regression on current code',excluded={skip:'Prior integration invocation stalled over one hour; interrupted, not passed'},tests=names,sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['refined_solver.py','refined_mesh.py','flood_solver.py']})
(out/'plan.json').write_text(json.dumps(plan,indent=2))
suite=unittest.TestSuite([test_refinement.RefinementTests(n) for n in names])
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_refined_mode_support))
with (out/'regression.log').open('w',encoding='utf-8') as log:
 start=time.perf_counter();r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 summary=dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),seconds=time.perf_counter()-start)
 (out/'results.json').write_text(json.dumps(summary,indent=2));print(summary)
sys.exit(not r.wasSuccessful())
