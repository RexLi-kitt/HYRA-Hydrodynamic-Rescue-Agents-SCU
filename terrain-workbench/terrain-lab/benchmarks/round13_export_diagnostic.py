import sys,faulthandler,unittest
from pathlib import Path
root=Path('C:/Users/Lenovo/Desktop/无人机大创/HYRA/terrain-workbench/terrain-lab')
sys.path[:0]=[str(root),str(root/'tests')]
out=Path('outputs/round13');out.mkdir(exist_ok=True)
with (out/'export-diagnostic.log').open('w',encoding='utf-8') as log:
 faulthandler.enable(file=log)
 faulthandler.dump_traceback_later(20,exit=True,file=log)
 import test_refinement
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite([test_refinement.RefinementTests('test_legacy_refinement_request_runs_and_exports_native_grid')]))
 faulthandler.cancel_dump_traceback_later()
 sys.exit(not result.wasSuccessful())
