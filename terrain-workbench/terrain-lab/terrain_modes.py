"""Preset controls and measurable acceptance of native terrain, without editing heights."""
import json
from pathlib import Path

VERSION = 'hyra-terrain-modes-v2'
PROFILES = {
    'water_network': dict(label='稠密水网', parameters=dict(vertical_scale=.55,
        continent_scale=4000, region_size=800, plains_weight=10, hills_weight=.5,
        mountains_weight=0, river_count=30, main_width=16, branch_width=12,
        main_bank_width=32, branch_bank_width=24, river_depth=7, branch_depth=5,
        lake_chance=.8, wetland_chance=.85, lake_distance=.6)),
    'mountains': dict(label='高山', parameters=dict(vertical_scale=1.6,
        continent_scale=2400, region_size=800, plains_weight=0, hills_weight=.5,
        mountains_weight=10, river_count=12, main_width=8, branch_width=5,
        main_bank_width=20, branch_bank_width=14, river_depth=5, branch_depth=4,
        lake_chance=.25, wetland_chance=.15, lake_distance=.15)),
    'plains': dict(label='平原', parameters=dict(vertical_scale=.35,
        continent_scale=4000, region_size=800, plains_weight=10, hills_weight=0,
        mountains_weight=0, river_count=12, main_width=10, branch_width=6,
        main_bank_width=24, branch_bank_width=16, river_depth=5, branch_depth=3,
        lake_chance=.35, wetland_chance=.4, lake_distance=.25)),
}


def configure_preset(preset, mode):
    if mode == 'custom': return
    # UPLIFT adds continental relief independently of the terrain-family weights.
    if mode in ('plains', 'water_network'):
        preset['world']['continent']['continentType'] = 'MULTI'
        # Native mountain chains blend in separately, even when mountains.weight is zero.
        # Align base elevations and reduce that independent chain amplitude through settings.
        for name, values in preset['terrain'].items():
            if name != 'general': values['baseScale'] = 1.1
        preset['terrain']['mountains']['verticalScale'] = .04
        preset['terrain']['hills']['verticalScale'] = .3
    # Other native terrain families otherwise retain nonzero defaults and dilute the target.
    for name, values in preset['terrain'].items():
        if name not in ('general', 'plains', 'hills', 'mountains') and 'weight' in values:
            values['weight'] = 0


def candidate_origins(config):
    """Search a fixed, reproducible neighborhood of the requested crop; keep the seed."""
    x, z = config['origin_x'], config['origin_z']
    if config.get('terrain_mode', 'custom') == 'custom': return [(x, z)]
    stride = max(config['span'], config['continent_scale'])
    offsets = [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
    return [(x+dx*stride, z+dz*stride) for dx,dz in offsets
            if abs(x+dx*stride) <= 30000000 and abs(z+dz*stride) <= 30000000]


def river_length(mask, step):
    """Estimate channel length from a one-cell skeleton, rather than channel width."""
    import numpy as np
    line = np.asarray(mask, bool).copy()
    changed = True
    while changed:
        changed = False
        for phase in (0,1):
            p = np.pad(line,1)
            neighbors = [p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],
                         p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            count = np.stack(neighbors).sum(axis=0)
            transitions = sum((~neighbors[i] & neighbors[(i+1)%8]).astype('u1') for i in range(8))
            n,e,s,w = [neighbors[i] for i in (0,2,4,6)]
            guard = (~(n&e&s) & ~(e&s&w)) if phase == 0 else (~(n&e&w) & ~(n&s&w))
            remove = line & (count>=2) & (count<=6) & (transitions==1) & guard
            if remove.any(): line[remove] = False; changed = True
    cardinal = (line[:,:-1]&line[:,1:]).sum() + (line[:-1,:]&line[1:,:]).sum()
    # Count a diagonal only when the two cells have no shorter cardinal connection.
    diagonal = ((line[:-1,:-1]&line[1:,1:]) & ~(line[:-1,1:]|line[1:,:-1])).sum()
    diagonal += ((line[:-1,1:]&line[1:,:-1]) & ~(line[:-1,:-1]|line[1:,1:])).sum()
    return float(step * (cardinal + np.sqrt(2)*diagonal))


def assess_arrays(bed, types, water, terrain_ids, sea_level, step):
    import numpy as np
    from river_inlets import detect
    bed = np.asarray(bed, float); types = np.asarray(types); water = np.asarray(water, float)
    if bed.shape != types.shape or water.shape != bed.shape or bed.ndim != 2 or not np.isfinite(bed).all():
        raise ValueError('地形检查需要完整、有限的二维高程')
    ocean = np.isin(types, [int(k) for k,v in terrain_ids.items() if 'ocean' in v])
    land = ~ocean & (bed >= sea_level)
    inland = ~ocean & np.isfinite(water) & (water > bed + .01)
    river = inland & np.isin(types, [int(k) for k,v in terrain_ids.items() if v == 'river'])
    mountain = np.isin(types, [int(k) for k,v in terrain_ids.items() if 'mountain' in v or v == 'torridonian'])
    slopes = np.hypot(*np.gradient(bed, step))
    covered = sum(bool(part.any()) for row in np.array_split(river,4,axis=0)
                  for part in np.array_split(row,4,axis=1)) / 16
    elevations = bed[land]
    length_km = river_length(river,step)/1000
    inland_area_km2 = float((land|inland).sum())*step**2/1e6
    return dict(land_fraction=float(land.mean()), inland_water_fraction=float(inland.mean()),
        wet_river_fraction=float(river.mean()), river_tile_coverage=covered,
        river_length_km=length_km,
        river_length_density_km_per_km2=length_km/inland_area_km2 if inland_area_km2 else 0,
        river_boundary_crossings=len(detect(river,bed,water,step=step)),
        land_relief_m=float(np.percentile(elevations,95)-np.percentile(elevations,5)) if elevations.size else 0,
        land_p90_height_m=float(np.percentile(elevations,90)) if elevations.size else 0,
        land_p90_slope=float(np.percentile(slopes[land],90)) if elevations.size else 0,
        mountain_land_fraction=float(mountain[land].mean()) if elevations.size else 0)


def verdict(mode, metrics):
    if mode == 'custom': return True
    m = metrics
    if mode == 'water_network':
        return (m['land_fraction'] >= .55 and m['river_length_density_km_per_km2'] >= .8
                and m['river_tile_coverage'] >= .625 and m['river_boundary_crossings'] >= 4
                and m['land_relief_m'] <= 80)
    if mode == 'mountains':
        return (m['land_fraction'] >= .7 and m['mountain_land_fraction'] >= .25
                and m['land_relief_m'] >= 80 and m['land_p90_height_m'] >= 140)
    if mode == 'plains':
        return m['land_fraction'] >= .7 and m['land_relief_m'] <= 45 and m['land_p90_slope'] <= .15
    raise ValueError('请选择地形类型')


def assess(folder, mode):
    import numpy as np
    from river_inlets import water_types
    folder = Path(folder); grid = json.loads((folder/'grid.json').read_text())
    fields = grid['fields']; shape = grid['shape']; scale = grid['terrain_scale_factor']
    def read(name): return np.fromfile(folder/fields[name]['file'], dtype=fields[name]['dtype']).reshape(shape)
    bed = read('height')*scale; water = read('riverWaterLevel')*scale
    surface = folder/'surface/surface.json'
    if surface.is_file():
        sf = json.loads(surface.read_text())['fields']
        bed = np.fromfile(surface.parent/sf['ground_height']['file'], dtype=sf['ground_height']['dtype']).reshape(shape)
        water = np.fromfile(surface.parent/sf['water_surface']['file'], dtype=sf['water_surface']['dtype']).reshape(shape)
    ids = fields['terrain']['values']
    types, _ = water_types(read('terrain'),bed,water,read('riverZone'),ids,
                          fields['riverZone']['values'],grid['sea_level_blocks'])
    metrics = assess_arrays(bed,types,water,ids,grid['sea_level_blocks'],grid['step_blocks'])
    return dict(schema=VERSION, terrain_mode=mode, label=PROFILES[mode]['label'],
                accepted=bool(verdict(mode,metrics)), full_surface=surface.is_file(), metrics=metrics)
