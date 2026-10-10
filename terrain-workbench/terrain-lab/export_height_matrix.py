"""Export the same terrain elevations shown by the local map viewer."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def read_height_matrix(directory: Path):
    grid = json.loads((directory / 'grid.json').read_text())
    surface_file = directory / 'surface' / 'surface.json'
    if surface_file.is_file():
        source = json.loads(surface_file.read_text())
        field = source['fields']['ground_height']
        path = directory / 'surface' / field['file']
        scale = 1
        definition = 'Actual ground block top face; excludes vegetation and water.'
    else:
        source = grid
        field = source['fields']['height']
        path = directory / field['file']
        scale = grid['terrain_scale_factor']
        definition = 'Continuous terrain elevation, including river/lake beds; not water surface.'
    rows, columns = source['shape']
    data = path.read_bytes()
    if field['dtype'] not in ('<f4', '<i4') or len(data) != rows * columns * 4:
        raise ValueError('Height field has an unsupported type or incomplete data')
    fmt = '<f' if field['dtype'] == '<f4' else '<i'
    values = [value[0] * scale for value in struct.iter_unpack(fmt, data)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Non-finite terrain elevation')
    matrix = [values[row * columns:(row + 1) * columns] for row in range(rows)]
    metadata = {
        'seed': grid['seed'], 'shape': [rows, columns], 'axis_order': ['z_row', 'x_column'],
        'origin_blocks': source['origin_blocks'], 'step_blocks': source['step_blocks'],
        'units': 'terrain block units; not calibrated metres', 'height_definition': definition,
        'index_base': 0, 'minimum': min(values), 'maximum': max(values),
    }
    return matrix, metadata


def current_directory():
    pointer = ROOT / 'current.json'
    relative = json.loads(pointer.read_text())['directory'] if pointer.is_file() else 'example/raw'
    return ROOT / relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, help='Raw grid directory; defaults to the latest map')
    parser.add_argument('--output', type=Path, default=ROOT / 'exports')
    parser.add_argument('--npy', action='store_true', help='Also write a NumPy matrix; requires NumPy')
    args = parser.parse_args()
    matrix, metadata = read_height_matrix(args.directory or current_directory())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'height-matrix.json').write_text(json.dumps(matrix, allow_nan=False) + '\n')
    (args.output / 'height-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if args.npy:
        import numpy as np
        np.save(args.output / 'height-matrix.npy', np.asarray(matrix, dtype=np.float64), allow_pickle=False)
    print(json.dumps({'output': str(args.output), **metadata}, ensure_ascii=False))


if __name__ == '__main__':
    main()
