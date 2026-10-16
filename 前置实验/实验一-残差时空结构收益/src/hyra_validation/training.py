"""GPU episode replay through the verified PyMARL learner and target networks."""
import copy
import logging
import time
from pathlib import Path
import numpy as np
import torch
from .artifacts import write_json
from .adapters.pymarl import build_learner,components
from .envs import RescueEnv,interval_valid
from .evaluation import compare_paired_events


class RunLogger:
    def __init__(self):
        self.console_logger=logging.getLogger("hyra_training")
        self.stats=[]

    def log_stat(self,key,value,t):
        if isinstance(value,torch.Tensor):
            value=value.detach().item()
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite {key}")
        self.stats.append(dict(key=key,value=float(value),env_step=int(t)))


def rollout(mac,env,scheme,groups,epsilon,rng,collect=True):
    EpisodeBatch,_,_=components()
    batch=EpisodeBatch(scheme,groups,1,env.ec["episode_limit"]+1,device="cpu") if collect else None
    previous=np.zeros((env.n_agents,8),dtype=np.float32)
    hidden=torch.zeros(env.n_agents,mac.agent.hidden_dim,device=mac.device)
    done=False; total=0.
    while not done:
        t=env.step_count
        obs=env.get_obs()
        if collect:
            batch.update(dict(obs=obs[None],state=env.get_state()[None],
                avail_actions=np.ones((1,env.n_agents,8),dtype=int)),ts=t)
        actions,hidden=mac.act(obs,previous,t,hidden,epsilon,rng)
        reward,done,_=env.step(actions)
        if collect:
            batch.update(dict(actions=actions[None,:,None],reward=[[reward]],terminated=[[done]]),ts=t)
        previous=np.eye(8,dtype=np.float32)[actions]
        total+=reward
    if collect:
        batch.update(dict(obs=env.get_obs()[None],state=env.get_state()[None],
            avail_actions=np.ones((1,env.n_agents,8),dtype=int)),ts=env.step_count)
    return batch,dict(env.metrics(),episode_return=total,steps=env.step_count)


def evaluation_rollouts(mac,events,config,graph,method,layouts=(0,1),static=False):
    rows=[]
    for index,event in enumerate(events):
        for layout in layouts:
            # Identical scene seeds across methods AND training seeds.
            scene_seed=9000+100*event["metadata"].get("paired_layout_index",index)+layout
            env=RescueEnv(event,config,graph,method,scene_seed,static=static)
            _,metrics=rollout(mac,env,None,None,0,np.random.default_rng(0),collect=False)
            rows.append((event,env,dict(metrics,scene_seed=scene_seed,layout=layout)))
    return rows


def train_one(events,config,graphs,method,seed,path,static=False,enhancement=True,episodes=None):
    path.mkdir(parents=True,exist_ok=False)
    device=config["training"].get("device","cuda")
    rng=np.random.default_rng(seed)
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    graph=graphs[1] if method=="B1_distance" else graphs[0]
    env=RescueEnv(events["train"][0],config,graph,method,seed,static=static)
    logger=RunLogger()
    mac,learner,buffer,scheme,groups=build_learner(env,config,logger,device,enhancement)
    capacity=sum(p.numel() for p in mac.agent.parameters())
    mixing_capacity=sum(p.numel() for p in learner.mixer.parameters())
    write_json(path/"capacity.json",dict(actor_parameters=capacity,mixer_parameters=mixing_capacity,
        observation_shape=mac.agent.obs_shape,device=device,enhancement_train_only=enhancement))
    episodes=episodes or config["training"]["episodes"]
    history=[];validation=[];best_score=-float("inf");best_checkpoint=None;total_steps=0
    start=time.perf_counter()
    for episode in range(1,episodes+1):
        event=events["train"][(episode-1)%len(events["train"])]
        env=RescueEnv(event,config,graph,method,int(rng.integers(0,2**31)),static=static)
        epsilon=max(.1,1-.9*(episode-1)/max(episodes-1,1))
        batch,row=rollout(mac,env,scheme,groups,epsilon,rng)
        buffer.insert_episode_batch(batch)
        total_steps+=env.step_count
        if buffer.can_sample(config["training"]["batch_size"]):
            sample=buffer.sample(config["training"]["batch_size"])
            sample.to(device)
            mac.agent.train()
            learner.train(sample,total_steps,episode)
        history.append(dict(episode=episode,epsilon=epsilon,event_id=event["metadata"]["event_id"],**row))
        if episode%20==0 or episode==episodes:
            print(f"{path.name} episode={episode}/{episodes} steps={total_steps} elapsed_s={time.perf_counter()-start:.1f}",flush=True)
            write_json(path/"training_progress.json",dict(episode=episode,episodes=episodes,env_steps=total_steps,elapsed_s=time.perf_counter()-start))
        if episode%config["training"]["validation_interval"]==0 or episode==episodes:
            checkpoint=path/f"checkpoint_{episode:05}"
            checkpoint.mkdir()
            learner.save_models(str(checkpoint))
            rows=evaluation_rollouts(mac,events["validation"],config,graph,method,static=static)
            score=float(np.mean([r[2]["weighted_service_yield"] for r in rows]))
            validation.append(dict(episode=episode,score=score,rows=[dict(event_id=e["metadata"]["event_id"],**r) for e,_,r in rows]))
            if score>best_score:
                best_score,best_checkpoint=score,checkpoint
    write_json(path/"training_history.json",history)
    write_json(path/"learner_stats.json",logger.stats)
    write_json(path/"validation_selection.json",dict(rule="highest mean completed service yield; earliest checkpoint on tie",validation=validation,
        selected_checkpoint=best_checkpoint.name,selected_score=best_score,test_used=False))
    mac.agent.load_state_dict(torch.load(best_checkpoint/"agent.th",map_location=device,weights_only=True))
    mac.agent.eval()
    return mac,dict(capacity=capacity,mixing_capacity=mixing_capacity,steps=total_steps,elapsed_s=time.perf_counter()-start,
                    selected_checkpoint=str(best_checkpoint),loss_updates=sum(s["key"]=="loss" for s in logger.stats))


def save_test_rollouts(mac,events,config,graph,method,seed,path):
    rows=[]
    for event,env,metrics in evaluation_rollouts(mac,events,config,graph,method):
        event_id=event["metadata"]["event_id"]
        stem=f"{method}_seed{seed}_{event_id}_layout{metrics['layout']}"
        write_json(path/f"{stem}_trace.json",dict(event_id=event_id,method=method,seed=seed,
            targets=env.targets,final_status=env.status,actions_and_services=env.log,observations=env.records,
            enhancement_execution=False,training_state_read_by_actor=False))
        # Offline last feasible service starts, using true simulation only for explanation.
        last_starts={}
        for r in env.targets:
            feasible=[float(t) for t in np.arange(0,301,env.ec["step_s"])
                if interval_valid(event,int(r),t,t+env.ec["service_s"],env.ec["depth_limit_m"])]
            last_starts[str(r)]=max(feasible) if feasible else None
        rows.append(dict(event_id=event_id,event_family_id=event["metadata"]["event_family_id"],method=method,
            training_seed=seed,**metrics,last_feasible_start_s=last_starts))
    return rows


def closed_loop(events,config,graphs,path):
    scores=[];summaries=[]
    for method in config["methods"]:
        for seed in config["training"]["seeds"]:
            run=path/f"{method}_seed{seed}"
            mac,summary=train_one(events,config,graphs,method,seed,run)
            graph=graphs[1] if method=="B1_distance" else graphs[0]
            scores.extend(save_test_rollouts(mac,events["test"],config,graph,method,seed,path))
            summaries.append(dict(method=method,seed=seed,**summary))
            write_json(path/"closed_loop_scores.json",scores)
            write_json(path/"training_summary.json",summaries)
    write_json(path/"closed_loop_paired.json",compare_paired_events(scores,dict(metric="weighted_service_yield",seed=config["evaluation"]["bootstrap_seed"])))
    return scores
