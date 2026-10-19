"""Static land-cover Manning n from existing native map layers.

These are documented initial estimates, not calibrated measurements. Material
and observed cover take precedence over biome proxies; water classes are last.
"""
from pathlib import Path
import numpy as np

VERSION = 'hyra-static-landcover-v1'
SOURCE = 'https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.0/developing-a-terrain-model-and-geospatial-layers/creating-land-cover-mannings-n-values-and-impervious-layers'
# Project initial estimates. Values are deliberately few and interpretable.
CLASSES = {
    'fallback': ('未知地表', None), 'channel': ('天然河槽', .035),
    'water': ('湖泊与海域底床', .030), 'soil': ('裸土与黏土', .025),
    'sand': ('砂质地表', .025), 'gravel': ('砂砾', .035),
    'rock': ('裸岩', .030), 'grass': ('草本地表', .035),
    'shrub': ('灌丛与高秆植被', .060), 'forest': ('树木覆盖', .100),
    'wetland': ('草本湿地', .070), 'woody_wetland': ('木本湿地', .120),
    'paved': ('硬化地表', .015), 'snow': ('雪冰覆盖地表', .030),
}
KEYS = list(CLASSES)


def biome_class(name):
    name = name.split(':')[-1]
    if 'ocean' in name or name in ('river', 'frozen_river'):
        return 'fallback'  # Water identity is assigned separately, not from climate alone.
    if name in ('swamp','mangrove_swamp'): return 'woody_wetland'
    if any(s in name for s in ('forest','jungle','taiga','grove')): return 'forest'
    if 'savanna' in name: return 'shrub'
    if 'stony' in name or 'peaks' in name: return 'rock'
    if 'snow' in name or 'frozen' in name or 'ice' in name: return 'snow'
    if 'desert' in name or name=='beach': return 'sand'
    if 'badlands' in name: return 'soil'
    if name in ('plains','sunflower_plains','meadow'): return 'grass'
    return 'fallback'


def material_class(name):
    name = name.split(':')[-1]
    if name in ('grass_block','moss_block'): return 'grass'
    if name in ('dirt','rooted_dirt','podzol','coarse_dirt','mud','clay','mycelium'): return 'soil'
    if name in ('sand','red_sand'): return 'sand'
    if name=='gravel': return 'gravel'
    if name in ('snow_block','ice','packed_ice','blue_ice'): return 'snow'
    if name in ('stone','granite','diorite','andesite','tuff','smooth_basalt','deepslate','sandstone','red_sandstone') or name.endswith('_ore'): return 'rock'
    if name in ('smooth_stone','stone_bricks','bricks','dirt_path') or name.endswith('_concrete'): return 'paved'
    return None


def classify(types, terrain_ids, fallback, *, biomes=None, biome_ids=None,
             materials=None, block_ids=None, cover=None):
    types = np.asarray(types)
    classes = np.zeros(types.shape, np.uint8)
    def assign(mask, key): classes[mask] = KEYS.index(key)
    for code, name in terrain_ids.items():
        # Geomorphological mountain/hill names alone do not imply bare rock.
        if 'badlands' in name: assign(types==int(code),'soil')
        elif name=='plains': assign(types==int(code),'grass')
        elif name=='wetland': assign(types==int(code),'wetland')
    if biomes is not None:
        for name, code in (biome_ids or {}).items():
            key=biome_class(name)
            if key!='fallback': assign(biomes==int(code),key)
    if materials is not None:
        for name, code in (block_ids or {}).items():
            key=material_class(name)
            if key is None: continue
            selected=materials==int(code)
            # Without cover flags retain biome vegetation proxies on soil/grass.
            if cover is None and key in ('soil','grass'):
                selected &= ~np.isin(classes,[KEYS.index(k) for k in ('forest','shrub','woody_wetland','wetland')])
            assign(selected,key)
    if cover is not None:
        cover=np.asarray(cover)
        if cover.shape!=types.shape or not np.isfinite(cover).all() or (cover!=np.floor(cover)).any() or (cover<0).any() or (cover>15).any():
            raise ValueError('植被覆盖标记无效')
        cover=cover.astype(np.uint8)
        other=(cover&2)!=0
        assign(other,'grass')
        if biomes is not None:
            for name,code in (biome_ids or {}).items():
                if 'bamboo' in name or 'desert' in name: assign(other&(biomes==int(code)),'shrub')
                if name.split(':')[-1] in ('swamp','mangrove_swamp'): assign(other&(biomes==int(code)),'wetland')
        trees=(cover&1)!=0
        assign(trees,'forest')
        if biomes is not None:
            for name,code in (biome_ids or {}).items():
                if name.split(':')[-1] in ('swamp','mangrove_swamp'): assign(trees&(biomes==int(code)),'woody_wetland')
        assign(((cover&12)!=0)&~trees&~other,'snow')
    for code,name in terrain_ids.items():
        if name=='river': assign(types==int(code),'channel')
        elif name=='lake' or 'ocean' in name: assign(types==int(code),'water')
    lookup=np.array([fallback if value is None else value for _,value in CLASSES.values()])
    n=lookup[classes]
    counts=np.bincount(classes.ravel(),minlength=len(KEYS))
    return n, dict(profile=VERSION, mode='landcover', calibrated=False, reference=SOURCE,
        priority='terrain proxy < biome < material/cover < channel/lake/ocean; explicit n and authored patches override',
        inputs=dict(biome=biomes is not None, material=materials is not None, cover_flags=cover is not None),
        native_classes=[dict(key=key,label=CLASSES[key][0],n=float(lookup[i]),cells=int(counts[i]))
                        for i,key in enumerate(KEYS) if counts[i]],
        fallback_cells=int(counts[0]),fallback_n=float(fallback),
        aggregation='native n area mean; approximate, not a subgrid conveyance model')


def from_terrain(root, terrain, fallback):
    root=Path(root).resolve(); fields=terrain['fields']; shape=tuple(terrain['shape'])
    used=[]
    def read(name):
        if name not in fields:return None
        field=fields[name];path=(root/field['url'].lstrip('/')).resolve()
        if root not in path.parents:raise ValueError('糙率源图层路径不正确')
        array=np.fromfile(path,dtype=field['dtype'])
        if array.size!=np.prod(shape) or not np.isfinite(array).all():raise ValueError(f'{name} 糙率源图层不完整')
        used.append(name)
        return array.reshape(shape)
    n,details=classify(read('terrain'),terrain['terrain_ids'],fallback,
        biomes=read('biome_id'),biome_ids=terrain.get('biome_ids'),
        materials=read('ground_block_id'),block_ids=terrain.get('block_ids'),cover=read('cover_flags'))
    details.update(source_layers=used,source_kind='actual_surface' if terrain['full'] else 'terrain_preview_proxy')
    return n,details
