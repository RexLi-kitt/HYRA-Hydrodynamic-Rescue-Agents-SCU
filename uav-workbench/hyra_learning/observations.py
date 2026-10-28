"""Actor encoding from TeamView only: local map, pooled shared map, pose."""
import math
import numpy as np


def pool_mean(values, size):
    nz, nx = values.shape[-2:]
    result = np.empty((*values.shape[:-2], size, size), dtype=np.float32)
    for z in range(size):
        za, zb = math.floor(z*nz/size), math.ceil((z+1)*nz/size)
        for x in range(size):
            xa, xb = math.floor(x*nx/size), math.ceil((x+1)*nx/size)
            result[..., z, x] = values[..., za:zb, xa:xb].mean(axis=(-2, -1))
    return result


def encode(view, patch_size=7, max_age_s=600.):
    if patch_size < 3 or patch_size % 2 != 1 or max_age_s <= 0:
        raise ValueError('Odd patch size >=3 and positive age scale required')
    agents, bed = view.agents, view.bed
    images, poses = [], []
    nz, nx = bed.shape
    static = np.clip((bed-bed.mean())/max(float(bed.std()), .05), -5, 5)/5
    for i, agent in enumerate(agents):
        known = agent.labels >= 0
        age = np.where(known, np.clip((agent.time_s-agent.observed_s)/max_age_s, 0, 1), 0)
        own, mates = np.zeros((nz, nx)), np.zeros((nz, nx))
        for j, other in enumerate(agents):
            x = int(np.clip(math.floor(other.pose.x_m/agent.dx+.5), 0, nx-1))
            z = int(np.clip(math.floor(other.pose.z_m/agent.dx+.5), 0, nz-1))
            (own if j == i else mates)[z, x] = 1.
        maps = np.stack((np.where(known, agent.labels, 0), known, age, agent.quality,
                         static, own, mates, np.ones((nz, nx)))).astype(np.float32)
        x = int(np.clip(math.floor(agent.pose.x_m/agent.dx+.5), 0, nx-1))
        z = int(np.clip(math.floor(agent.pose.z_m/agent.dx+.5), 0, nz-1))
        radius = patch_size//2
        padded = np.pad(maps, ((0, 0), (radius, radius), (radius, radius)))
        local = padded[:, z:z+patch_size, x:x+patch_size]
        images.append(np.concatenate((local, pool_mean(maps, patch_size)), axis=0))
        others = [other.pose for j, other in enumerate(agents) if j != i]
        relative = [np.mean([(p.x_m-agent.pose.x_m)/(nx*agent.dx) for p in others]),
                    np.mean([(p.z_m-agent.pose.z_m)/(nz*agent.dx) for p in others])]
        angle = math.radians(agent.pose.heading_deg)
        poses.append([math.sin(angle), math.cos(angle),
                      (agent.pose.x_m+agent.dx/2)/(nx*agent.dx),
                      (agent.pose.z_m+agent.dx/2)/(nz*agent.dx), *relative])
    return dict(images=np.stack(images), poses=np.asarray(poses, dtype=np.float32),
                time_s=float(agents[0].time_s))
