import json,subprocess,sys
from pathlib import Path
from finish_report import finish

here=Path(__file__).parent
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
outputs=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
out=outputs/'uav6-priority-20261101'
if out.exists():raise RuntimeError('Preserve existing output; no duplicate launch')
log=outputs/'uav6-priority-20261101.log'
with log.open('x',encoding='utf-8') as f:
    rc=subprocess.call([sys.executable,str(here/'priority_train.py'),'--pilot',str(outputs/'uav6-pilot-20261101'),
                        '--baseline',str(outputs/'uav6-total256-20261101'),'--output',str(out),'--execute'],
                       cwd=str(here),stdout=f,stderr=subprocess.STDOUT)
if out.exists():
    for p in here.glob('*.py'):
        (out/p.name).write_bytes(p.read_bytes())
    (out/'queue-returncode.json').write_text(json.dumps({'returncode':rc}),encoding='utf-8')
    finish(out,repo)
sys.exit(rc)
