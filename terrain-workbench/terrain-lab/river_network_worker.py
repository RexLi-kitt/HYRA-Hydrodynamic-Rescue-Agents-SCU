"""Isolated RivGraph process. Receives arrays, writes JSON, never edits terrain."""
import json
import importlib.metadata
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage
from rivgraph import mask_to_graph
import rivgraph


def analyze(mask, bed, water, step):
    four, count4 = ndimage.label(mask)
    _, count8 = ndimage.label(mask, structure=np.ones((3, 3)))
    warnings = []
    if count4 > 1:
        warnings.append(f'原始河道有 {count4} 个四邻域连通分区；可能为独立河流、裁剪或断河，需结合边界判断')
    if count4 > count8:
        warnings.append('存在仅对角相接的河道；RivGraph 图连接不代表求解器可直接过水')
    nodes_out = []; links_out = []; unresolved = []; isolated = 0
    # Analyze each ORIGINAL component independently: skeleton cleanup must never
    # silently turn distinct source components into a connected source network.
    for component, slices in enumerate(ndimage.find_objects(four), 1):
        if slices is None:
            continue
        local = four[slices] == component
        if local.sum() == 1:
            isolated += 1
            continue
        padded = np.pad(local, 2)
        skeleton = mask_to_graph.skeletonize_mask(padded)
        def position(index):
            z, x = np.unravel_index(int(index), skeleton.shape)
            return [int(z + slices[0].start - 2), int(x + slices[1].start - 2)]
        degree = ndimage.convolve(skeleton.astype(np.int16), np.ones((3, 3), np.int16), mode='constant') - skeleton
        if not np.any(skeleton & (degree == 1)):
            # The pinned walker can fail to terminate on endpoint-free loops.
            # Preserve them explicitly rather than pruning valid loop geometry.
            unresolved.append(dict(component=component, reason='endpoint_free_skeleton',
                                   cells=[position(p) for p in np.flatnonzero(skeleton)]))
            continue
        links, nodes = mask_to_graph.skel_to_graph(skeleton.copy())
        for node_id, idx, conn in zip(nodes['id'], nodes['idx'], nodes['conn']):
            nodes_out.append(dict(id=f'{component}:{node_id}', cell=position(idx), degree=len(conn)))
        for link_id, pixels, conn in zip(links['id'], links['idx'], links['conn']):
            cells = [position(p) for p in pixels]
            length = sum(np.hypot(a[0]-b[0], a[1]-b[1]) * step for a, b in zip(cells, cells[1:]))
            links_out.append(dict(id=f'{component}:{link_id}', component=component,
                                  nodes=[f'{component}:{n}' for n in conn], cells=cells, length_blocks=float(length)))
    if isolated:
        warnings.append(f'{isolated} 个单格孤立河道未进行骨架建图')
    if unresolved:
        warnings.append(f'{len(unresolved)} 个闭环或无端点分区保留骨架待核查，未强制进行河段建图')
    dry = int(np.count_nonzero(mask & (~np.isfinite(water) | (water <= bed + .01))))
    if dry:
        warnings.append(f'{dry} 个河道分类格没有有效初始水深；不直接判为断河')
    sizes = np.bincount(four.ravel())[1:]
    origin = json.loads(importlib.metadata.distribution('rivgraph').read_text('direct_url.json') or '{}')
    return dict(status='review' if warnings else 'analyzed', engine='RivGraph',
                upstream_revision=origin.get('vcs_info', {}).get('commit_id', 'unknown'),
                engine_version=getattr(rivgraph, '__version__', 'unknown'),
                river_cells=int(mask.sum()), components_four=int(count4), components_eight=int(count8),
                largest_component_fraction=float(sizes.max() / mask.sum()),
                nodes=len(nodes_out), links=len(links_out), dry_channel_cells=dry,
                warnings=warnings, message='河网拓扑已检查；不等于河床、河岸或水动力已通过验证',
                graph=dict(shape=list(mask.shape), axis_order=['z', 'x'], step_blocks=step,
                           directed=False, nodes=nodes_out, links=links_out, unresolved_components=unresolved,
                           note='RivGraph skeleton geometry may differ from source pixels; directions are not inferred'))


if __name__ == '__main__':
    with np.load(sys.argv[1], allow_pickle=False) as arrays:
        report = analyze(arrays['mask'], arrays['bed'], arrays['water'], float(sys.argv[3]))
    Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
