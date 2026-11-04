from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-search-probe-v1'))
from search_env import SearchEnv
from refloat_dispatch import DispatchEnv
class RefloatSearch(DispatchEnv):
    sector_counts=SearchEnv.sector_counts
    search_score=SearchEnv.search_score
    choices=SearchEnv.choices
    def __init__(self,water,scenario,mode='coverage100'):
        self.mode=mode
        super().__init__(water,scenario)
