import json,subprocess,sys,shutil
from pathlib import Path
from continue_full1024 import finish

root=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
out=root/'uav6-full1024-20261101';parent=root/'uav6-priority-20261101';here=Path(__file__).parent
preflight=json.loads((here/'preflight.json').read_text(encoding='utf-8'))
if not preflight.get('passed'):raise RuntimeError('Resume checks must pass before launching')
if out.exists():raise RuntimeError('Output exists; refuse duplicate launch or overwrite')
with (root/'uav6-full1024-20261101.log').open('x',encoding='utf-8') as log:
    code=subprocess.call([sys.executable,str(here/'continue_full1024.py'),'--parent',str(parent),
        '--pilot',str(root/'uav6-pilot-20261101'),'--output',str(out),'--execute'],stdout=log,stderr=subprocess.STDOUT,cwd=str(here))
if out.exists():
    for p in here.iterdir():
        if p.suffix in ('.py','.json','.log'):shutil.copy2(p,out/p.name)
    (out/'queue-returncode.json').write_text(json.dumps({'returncode':code}),encoding='utf-8')
    finish(out,repo,parent)
sys.exit(code)
