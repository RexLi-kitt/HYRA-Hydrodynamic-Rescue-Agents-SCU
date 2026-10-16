"""Offline fixed-state action sensitivity to evidence, not a trained baseline."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.agents import SearchMAC
from hyra_validation.data import load_events
from hyra_validation.envs import RescueEnv


if __name__=="__main__":
    training=Path(sys.argv[1]).resolve()
    config=json.loads((training/"config.json").read_text(encoding="utf-8"))
    graph=json.loads((training/"frozen_structure.json").read_text(encoding="utf-8"))
    events=load_events(ROOT/config["manifest"])
    run=new_run("evidence_action_diagnostic",dict(training_run=str(training),
        purpose="fixed-state evidence-value perturbation; no performance baseline; no training"))
    torch.set_num_threads(1)
    rows=[]
    for seed in config["training"]["seeds"]:
        source=training/f"B2_structure_seed{seed}"
        selection=json.loads((source/"validation_selection.json").read_text(encoding="utf-8"))
        shape=json.loads((source/"capacity.json").read_text(encoding="utf-8"))["observation_shape"]
        mac=SearchMAC(shape,4,config["training"],"cuda")
        mac.agent.load_state_dict(torch.load(source/selection["selected_checkpoint"]/"agent.th",map_location="cuda",weights_only=True))
        mac.agent.eval()
        for index,event in enumerate(events["test"]):
            for layout in (0,1):
                scene_seed=9000+100*index+layout
                env=RescueEnv(event,config,graph,"B2_structure",scene_seed)
                previous=np.zeros((4,8),dtype=np.float32)
                hidden=torch.zeros(4,config["training"]["hidden_dim"],device="cuda")
                trace=json.loads((training/f"B2_structure_seed{seed}_{event['metadata']['event_id']}_layout{layout}_trace.json").read_text(encoding="utf-8"))
                recorded=[r for r in trace["actions_and_services"] if r["type"]=="decision"]
                decisions=[]
                for t in range(len(recorded)):
                    obs=env.get_obs()
                    changed=obs.copy()
                    evidence_channel=4+(4*config["representation"]["history_length"] if config["representation"].get("preserve_raw_history") else 0)
                    pooled_offset=(obs.shape[1]-3)//2
                    changed[:,evidence_channel]=0;changed[:,pooled_offset+evidence_channel]=0
                    with torch.no_grad():
                        q,_=mac.agent(torch.as_tensor(changed,device="cuda"),torch.as_tensor(previous,device="cuda"),t,hidden,enhance=False)
                    actions,next_hidden=mac.act(obs,previous,t,hidden,0,np.random.default_rng(0))
                    if actions.tolist()!=recorded[t]["actions"]:
                        raise AssertionError("Saved policy trace cannot be reproduced")
                    supported=bool(np.any(obs[:,evidence_channel+1]))
                    free=np.array([b is None for b in env.busy])
                    difference=actions!=q.argmax(dim=-1).cpu().numpy()
                    decisions.append(dict(time_s=env.now,support=supported,
                        actual_actions=actions.tolist(),zero_evidence_actions=q.argmax(dim=-1).cpu().tolist(),
                        changed_agents=int(np.count_nonzero(difference)),
                        free_agents=int(free.sum()),changed_free_agents=int(np.count_nonzero(difference&free))))
                    env.step(actions);hidden=next_hidden;previous=np.eye(8,dtype=np.float32)[actions]
                rows.append(dict(seed=seed,event_id=event["metadata"]["event_id"],layout=layout,
                    replay_matches_saved_trace=True,decisions=decisions,
                    supported_decisions=sum(d["support"] for d in decisions),
                    supported_changed_agent_actions=sum(d["changed_agents"] for d in decisions if d["support"]),
                    supported_free_agent_actions=sum(d["free_agents"] for d in decisions if d["support"]),
                    supported_changed_free_agent_actions=sum(d["changed_free_agents"] for d in decisions if d["support"])))
    write_json(run/"action_sensitivity.json",rows)
    write_json(run/"completion.json",dict(completed=True,interpretation="action sensitivity only; zero-evidence evaluation is not an independently trained baseline"))
    print(run,flush=True)
