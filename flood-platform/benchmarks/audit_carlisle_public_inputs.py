"""Read-only source comparisons; never align observations or build a solver case.

Requires numpy and tifffile. --reader-libs optionally supplies an isolated install.
The output distinguishes public input availability from field validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bdy(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    curves, i = {}, 1  # First line is the free-text description.
    factors = {"seconds": 1., "minutes": 60., "hours": 3600.}
    while i < len(lines):
        name = lines[i]
        count_text, unit = lines[i + 1].split()
        count = int(count_text)
        if name in curves or unit.lower() not in factors or count < 2:
            raise ValueError("duplicate name, unsupported time unit, or short series")
        values = np.asarray([[float(v) for v in row.split()]
                             for row in lines[i + 2:i + 2 + count]])
        if values.shape != (count, 2) or not np.isfinite(values).all():
            raise ValueError("incomplete/nonfinite boundary series")
        t = values[:, 1] * factors[unit.lower()]
        if not np.all(np.diff(t) > 0):
            raise ValueError("boundary time must strictly increase")
        curves[name] = {"q_as_stored": values[:, 0], "time_s": t, "source_time_unit": unit}
        i += count + 2
    return curves


def fingerprint_correspondence(old, new, tolerance=1e-5):
    """Identify shared input value sequences only; no observations enter this step.

    Tolerance is for the decimal export of q*5*point_count, not a fit criterion.
    A detected offset is diagnostic metadata and is never applied to any series.
    """
    candidates = []
    for offset in range(max(1, len(old) - 100)):
        n = min(len(old) - offset, len(new))
        delta = old[offset:offset + n] - new[:n]
        good = np.abs(delta) <= tolerance
        candidates.append((int(good.sum()), n, offset, delta))
    matched, count, offset, delta = max(candidates, key=lambda row: (row[0], row[1]))
    return {"matched_values": matched, "overlap_values": count,
            "new_row_zero_corresponds_to_old_row": offset,
            "not_applied_to_input_or_observation": True,
            "tolerance_m3_s": tolerance,
            "mismatches": [{"new_row": int(i), "old_minus_new_m3_s": float(delta[i])}
                           for i in np.flatnonzero(np.abs(delta) > tolerance)],
            "new_unmatched_tail": new[count:].tolist()}


def audit(outputs, destination):
    import tifffile
    outputs, destination = Path(outputs), Path(destination)
    r72 = outputs / "round72"
    source = r72 / "flooda5-inputs"
    old_contract = json.loads((outputs / "round71/carlisle/source-contract.json").read_text())
    curves = read_bdy(source / "carlisle.bdy")
    tracer = r72 / "tracer-inputs"
    checks = {}
    comparisons = []
    mapping = [("main", "main1.csv", "upstream1", "Eden"),
               ("cummersdale", "cummersdale1.csv", "upstream3", "Caldew"),
               ("harraby", "harraby1.csv", "upstream2", "Petteril")]
    for name, filename, old_name, river in mapping:
        csv = np.loadtxt(tracer / filename)
        new = curves[name]
        checks[name + "_same_values_as_zenodo"] = bool(np.array_equal(csv[:, 1], new["q_as_stored"]))
        checks[name + "_241_rows"] = len(csv) == len(new["time_s"]) == 241
        checks[name + "_half_hour_explicit_bdy"] = bool(np.all(np.diff(new["time_s"]) == 1800))
        old = next(h for h in old_contract["hydrographs"] if h["name"] == old_name)
        old_q = np.array(old["unit_width_values_m2_s"]) * old["point_count"] * old["original_grid_dx_m"]
        old_t = np.asarray(old["time_since_model_start_s"])
        if old_t.shape != old_q.shape or not np.all(np.diff(old_t) == 900):
            raise ValueError("original T07 sampling contract changed")
        old_step, new_step = float(np.diff(old_t)[0]), float(np.diff(new["time_s"])[0])
        match = fingerprint_correspondence(old_q, new["q_as_stored"])
        comparisons.append({"river": river, "new_name": name, "old_name": old_name,
            "old_step_s": old_step, "new_step_s": new_step, "duration_scale_for_matching_values": new_step / old_step,
            "old_count": len(old_q), "new_count": len(csv),
            "old_peak_m3_s": float(old_q.max()), "old_peak_relative_h": float(old_t[old_q.argmax()] / 3600),
            "new_peak_as_stored": float(new["q_as_stored"].max()),
            "new_peak_relative_h": float(new["time_s"][new["q_as_stored"].argmax()] / 3600),
            "input_value_fingerprint": match})
    with tifffile.TiffFile(source / "Carlisle_LiDAR_5m_mean.tif") as tf:
        page = tf.pages[0]
        terrain = page.asarray()
        geo = tf.geotiff_metadata
        nodata = page.nodata
    checks["geotiff_float64_612x951"] = terrain.dtype == np.float64 and terrain.shape == (612, 951)
    checks["geotiff_finite"] = bool(np.isfinite(terrain).all())
    checks["geotiff_5m"] = list(geo["ModelPixelScale"][:2]) == [5., 5.]
    with (tracer / "carlisle15m.txt").open() as f:
        header = dict((key.lower(), float(value)) for key, value in [next(f).split() for _ in range(6)])
        terrain15 = np.loadtxt(f)
    checks["ascii_dimensions"] = terrain15.shape == (int(header["nrows"]), int(header["ncols"]))
    extent = [338495., 554700., 343250., 557760.]
    checks["same_raster_extent"] = (list(geo["ModelTiepoint"][3:5]) == [extent[0], extent[3]] and
        header["xllcorner"] == extent[0] and header["yllcorner"] == extent[1] and
        header["ncols"] * header["cellsize"] == extent[2] - extent[0] and
        header["nrows"] * header["cellsize"] == extent[3] - extent[1])
    xml = ET.parse(tracer / "carlisle15m load file.xml").getroot()
    source_cells = [{"file": n.findtext("Filename"), "x_cell": int(n.findtext("X")),
                     "y_cell": int(n.findtext("Y"))}
                    for n in xml.findall("./File-Parms/Sources") if n.findtext("input") == "true"]
    source_samples = []
    for line in (source / "carlisle.bci").read_text().splitlines():
        row = line.split()
        if row and row[0] == "P":
            x, y = map(float, row[1:3])
            col, iy = int(np.floor((x - extent[0]) / 5)), int(np.floor((extent[3] - y) / 5))
            if not (0 <= col < 951 and 0 <= iy < 612):
                raise ValueError("point outside downloaded DEM")
            source_samples.append({"name": row[4], "x_m": x, "y_m": y,
                "row0": iy, "col0": col, "raw_dem_value": float(terrain[iy, col])})
    mean15 = terrain.reshape(204, 3, 317, 3).mean(axis=(1, 3))
    valid15 = np.isfinite(terrain15) & (terrain15 != header["nodata_value"])
    delta = terrain15[valid15] - mean15[valid15]
    gauge = json.loads((outputs / "round68/prepared/gauge-context.json").read_text())
    maximum = max(float(row["Value[m]"]) for row in gauge)
    peak_rows = [row for row in gauge if float(row["Value[m]"]) == maximum]
    summary = {"evidence_class": "public model inputs; no new field simulation or validation",
        "check_results": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "source_hydrograph_comparisons": comparisons,
        "terrain": {"array_shape": list(terrain.shape), "dtype": str(terrain.dtype),
            "geo_metadata": geo, "nodata": nodata, "nodata_count": int(np.sum(terrain == nodata)),
            "value_min": float(terrain.min()), "value_max": float(terrain.max()),
            "extent_m": extent, "source_point_samples": source_samples,
            "original_T07_identity_proven": False, "surveyed_channel_bed_proven": False,
            "vintage_proven": False},
        "tracer15": {"header": header, "source_cells_as_reported": source_cells,
            "headeroverride": xml.findtext("./General-Parms/headeroverride"),
            "unused_xml_cellsize": xml.findtext("./General-Parms/cellsize"),
            "inputtimestep_xml": xml.findtext("./File-Parms/inputtimestep"),
            "zero_or_one_based_cells_resolved": False,
            "difference_from_3x3_means_of_downloaded_5m": {
                "samples": int(valid15.sum()), "min": float(delta.min()), "max": float(delta.max()),
                "median": float(np.median(delta)), "max_abs": float(np.max(np.abs(delta))) }},
        "field_time_context": {"station": "Sheepmount", "maximum_rows_preserved": peak_rows,
            "observation_time_modified": False, "model_epoch_assigned": False,
            "note": "G;Ed flag retained. Publication tracer epoch differs; not transferred to T07."},
        "decisions": {"downloaded_5m_available": True, "original_5m_T07_reproduced": False,
            "tracer_hydrograph_direct_substitution_allowed": False,
            "new_field_items_passed": 0,
            "reason": "Matching input values use 900s versus 1800s and differing series starts. No date/time remapping justified. QVAR unit contract differs between solvers."},
        "source_files_sha256": {str(f.relative_to(outputs)): sha256(f)
            for root in [source, tracer] for f in root.iterdir() if f.is_file()}}
    destination.mkdir(exist_ok=True, parents=True)
    (destination / "public-input-audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(destination / "terrain-audit.npz", terrain5=terrain, terrain15=terrain15, means5_to15=mean15)
    if not all(checks.values()):
        raise ValueError("Input integrity check failed; see saved audit")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reader-libs")
    args = parser.parse_args()
    if args.reader_libs:
        sys.path.insert(0, args.reader_libs)
    result = audit(args.outputs, args.out)
    print(json.dumps({k: result[k] for k in ["checks_passed", "checks_total", "source_hydrograph_comparisons", "terrain", "tracer15", "decisions"]}, ensure_ascii=False, indent=2, default=str))
