"""Rebuild the study's engineering prerequisites in a fresh reproduction checkout.

This does not fit a predictive model, generate the new holdout or consume a new
mechanism round. Existing registered research is never reset.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.experiments import write_json


def inspect():
    paths=[ROOT/'generated/p2-formal-20261025/decision.json',
           ROOT/'generated/p3-weighted-scale-preflight-20261025/decision.json']
    return [dict(path=str(p.relative_to(ROOT)),exists=p.exists()) for p in paths]


def main(build=False):
    if not build:
        print(json.dumps(dict(prerequisites=inspect(),registered=(ROOT/'research/rounds.json').exists())));return
    if (ROOT/'research/rounds.json').exists():raise ValueError('已有研究登记，禁止重置；复现请使用独立克隆')
    from hyra_uav.scales import formal_scales
    from hyra_uav.diagnostic import generate_scene
    from weighted_preflight import run
    formal=ROOT/'generated/p2-formal-20261025'
    if not (formal/'decision.json').exists():formal_scales(formal)
    oldfit=ROOT/'generated/p3-20261025'
    for mechanism in ('slow','rapid','async'):
        folder=oldfit/'scenes'/('fit-'+mechanism)
        if not (folder/'scene.json').exists():generate_scene(folder,mechanism,'fit')
    output=ROOT/'generated/p3-weighted-scale-preflight-20261025'
    if not (output/'decision.json').exists():run(oldfit,output,(2.5,5,10))
    print(json.dumps(dict(prerequisites=inspect(),next='python -m hyra_uav.strong_study prepare generated/p3-nonlinear-round1 --round 1')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');main(parser.parse_args().build)
