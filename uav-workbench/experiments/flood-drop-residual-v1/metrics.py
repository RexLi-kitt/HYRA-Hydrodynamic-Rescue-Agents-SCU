"""User amendment: people supplied is primary; report19/20 and20/20 separately."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-v2'))
from pilot import summary as original_summary

def summary(rows):
    s=original_summary(rows);n=len(rows)
    s.update(people_total=20*n,people_supplied=sum(r['supplied'] for r in rows),
        scenes_at_least_19=sum(r['supplied']>=19 for r in rows))
    s['people_completion']=s['people_supplied']/s['people_total']
    s['scene_at_least_19_rate']=s['scenes_at_least_19']/n
    s['people_missed']=s['people_total']-s['people_supplied']
    return s

def quality(result):
    # User: people first, scene advantage second. Secondary improvements may not
    # compensate for even one fewer delivered person. Exact ties keep earlier.
    rows=result['rows']
    return (sum(r['supplied'] for r in rows),sum(r['supplied']==20 for r in rows),sum(r['supplied']>=19 for r in rows))
