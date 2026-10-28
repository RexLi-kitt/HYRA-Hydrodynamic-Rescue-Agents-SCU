"""Static weighted relations and an exact-distance matched weight ablation.

The raw evidence pool is shared by every model. The wrong relation permutes
topographic factors only among positive edges at the exact same grid distance;
support, all node degrees, and per-target weighted mass by distance stay fixed.
"""
import numpy as np
from .diagnostic import static_graph


def weighted_relations(process, pool_size=64, terrain_cost_scale_m=5.0, seed=20261025):
    # Rewiring is not used here; only the verified static path computation.
    graph=static_graph(process,pool_size=pool_size,degree=12,
                       rewire_fn=lambda pool, adjacency, distances:(adjacency.copy(),0))
    pool=graph["pool"];nz,nx=process.shape
    target=np.arange(len(pool))[:,None]
    manhattan=np.abs(pool//nx-target//nx)+np.abs(pool%nx-target%nx)
    strength=graph["strength"]
    costs=-25*np.log(np.maximum(strength,1e-30))
    excess=np.maximum(0,costs-process.dx*manhattan)
    factor=np.where(strength>0,np.exp(-excess/terrain_cost_scale_m),0.)
    geometric=np.exp(-graph["distance"]*process.dx/15)
    weights=geometric*factor
    wrong_factor=factor.copy();rng=np.random.default_rng(seed)
    rings=np.rint(graph["distance"]**2).astype(int)
    for i in range(len(pool)):
        for ring in np.unique(rings[i]):
            positions=np.flatnonzero((rings[i]==ring)&(factor[i]>0))
            if len(positions)>1:
                wrong_factor[i,positions]=rng.permutation(factor[i,positions])
    wrong=geometric*wrong_factor
    before=weights/np.maximum(weights.sum(1,keepdims=True),1e-20)
    after=wrong/np.maximum(wrong.sum(1,keepdims=True),1e-20)
    changed=(np.abs(weights-wrong)>1e-8)&(weights>0)
    audit=dict(support_exact=bool(np.array_equal(weights>0,wrong>0)),
               out_strength_exact=bool(np.allclose(weights.sum(1),wrong.sum(1),rtol=1e-10,atol=1e-12)),
               distance_weighted_mass_exact=all(np.allclose([weights[i,rings[i]==r].sum() for r in np.unique(rings[i])],
                                                          [wrong[i,rings[i]==r].sum() for r in np.unique(rings[i])],rtol=1e-10,atol=1e-12)
                                                for i in range(len(pool))),
               changed_positive_weight_fraction=float(changed.sum()/max((weights>0).sum(),1)),
               mean_operator_total_variation=float(np.abs(before-after).sum(1).mean()/2),
               incoming_weight_relative_l1=float(np.abs(np.bincount(pool.ravel(),weights=weights.ravel(),minlength=len(pool))-
                                                       np.bincount(pool.ravel(),weights=wrong.ravel(),minlength=len(pool))).sum()/max(weights.sum(),1e-20)))
    graph.update(B3_weight=weights,A2_weight=wrong,B2_weight=geometric,terrain_factor=factor,audit=audit)
    return graph
