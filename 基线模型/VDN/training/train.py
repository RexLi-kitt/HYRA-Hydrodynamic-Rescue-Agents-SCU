import sys
from pathlib import Path
import runpy
HERE=Path(__file__).resolve().parent
name='VDN'
sys.argv[1:1]=['--model',name]
runpy.run_path(str(HERE.parent.parent/'training_runtime'/'run.py'),run_name='__main__')

