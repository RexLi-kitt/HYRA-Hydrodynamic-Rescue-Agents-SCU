"""Read the McClean2020 field-data subset; never score software output as truth.

Run with the existing reference Python and --extra-libs for isolated Shapely.
The downloaded author scripts are provenance only and are not executed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import struct
import sys
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def decode_geometry(blob, expected_srid):
    from shapely import from_wkb
    if blob[:2] != b"GP" or blob[2] != 0:
        raise ValueError("unsupported GeoPackage geometry header")
    flags = blob[3]
    srid = struct.unpack_from("<i" if flags & 1 else ">i", blob, 4)[0]
    if srid != expected_srid:
        raise ValueError("geometry/header CRS mismatch")
    envelope = (flags >> 1) & 7
    if envelope not in (0, 1, 2, 3, 4) or flags & 0x30:
        raise ValueError("empty/extended/unsupported geometry")
    n = (0, 4, 6, 6, 8)[envelope]
    geometry = from_wkb(blob[8 + 8*n:])
    if geometry.is_empty or not geometry.is_valid or not all(map(math.isfinite, geometry.bounds)):
        raise ValueError("empty, invalid or nonfinite geometry; do not auto-repair observations")
    return geometry


def read_gpkg(path):
    from pyproj import Transformer
    from shapely.ops import transform
    quote = lambda v: '"' + v.replace('"', '""') + '"'
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        specs = db.execute("select * from gpkg_geometry_columns").fetchall()
        if len(specs) != 1:
            raise ValueError("expected one feature layer")
        spec = dict(specs[0]); srid = spec["srs_id"]
        if srid not in (27700, 3035):
            raise ValueError("unregistered CRS")
        transformer = Transformer.from_crs(srid, 3035, always_xy=True, allow_ballpark=False)
        rows = []
        for row in db.execute("select * from " + quote(spec["table_name"])):
            row = dict(row); blob = row.pop(spec["column_name"])
            geom = decode_geometry(blob, srid)
            projected = transform(transformer.transform, geom)
            if not projected.is_valid:
                raise ValueError("invalid transformed geometry")
            rows.append({"properties": row, "geometry": projected, "source_geometry": geom})
        transform_record = {"from_epsg": srid, "to_epsg": 3035,
                            "always_xy": True, "allow_ballpark": False,
                            "reported_accuracy_m": transformer.accuracy,
                            "description": transformer.description}
        if srid != 3035:
            used = transformer.get_last_used_operation()
            transform_record["reported_accuracy_m"] = used.accuracy
            transform_record["description"] = used.description
    return rows, transform_record


def read_gauge(path, start, end):
    metadata = {}
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for _ in range(18):
            k, v = next(csv.reader([next(f)])); metadata[k] = v
        record_range = next(f).strip(); quality_legend = next(f).strip()
        if metadata["Time series unit"] != "m" or metadata["Time series value distance"] != "15 minute":
            raise ValueError("unregistered gauge unit or cadence")
        for line_no, row in enumerate(csv.DictReader(f), 22):
            stamp = datetime.strptime(row["Time stamp"], "%d/%m/%Y %H:%M:%S")
            if start <= stamp <= end:
                for key in ("Value[m]", "Absolute value[mAOD]"):
                    if not Decimal(row[key]).is_finite():
                        raise ValueError("nonfinite gauge value in selected period")
                rows.append({"time_as_reported": stamp.isoformat(), "source_line": line_no, **row})
    if len({r["time_as_reported"] for r in rows}) != len(rows):
        raise ValueError("duplicate gauge time")
    times = [datetime.fromisoformat(r["time_as_reported"]) for r in rows]
    gaps = [{"before":a.isoformat(), "after":b.isoformat(), "seconds":(b-a).total_seconds()}
            for a,b in zip(times,times[1:]) if b-a != timedelta(minutes=15)]
    if times != sorted(times):
        raise ValueError("out-of-order source series")
    return metadata, record_range, quality_legend, rows, gaps


def net_phase(delta, band):
    return "net_rising" if delta > band else "net_falling" if delta < -band else "unresolved_at_record_rounding"


def prepare(root, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from pyproj import Transformer
    from shapely.geometry import mapping, Point
    from shapely.ops import transform, unary_union

    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root/"observation-protocol.json").read_text(encoding="utf-8"))
    if protocol["schema"] != "hyra.carlisle.qualitative.v1":
        raise ValueError("unknown protocol")
    start = datetime.fromisoformat(protocol["phase_rules"]["start_as_reported"])
    end = datetime.fromisoformat(protocol["phase_rules"]["end_as_reported"])
    band = Decimal(protocol["phase_rules"]["rounding_band_m"])
    data = root/"data"
    layers = {}; transforms = {}
    for name, filename in [("settlement", "carlisle_boundary_epsg3035.gpkg"),
                           ("outlines", "carlisle_recorded_flood_outlines.gpkg"),
                           ("marks", "carlisle_sample_points.gpkg")]:
        layers[name], transforms[name] = read_gpkg(data/filename)
    if len(layers["outlines"]) != 5 or len(layers["marks"]) != 263:
        raise ValueError("source count differs from preregistered author inventory; investigate")
    to_wgs = Transformer.from_crs(3035,4326,always_xy=True,allow_ballpark=False)
    for name, rows in layers.items():
        features=[]
        for r in rows:
            features.append({"type":"Feature", "properties": {**r["properties"], "source_layer":name},
                             "geometry":mapping(transform(to_wgs.transform,r["geometry"]))})
        dump(output/(name+"-wgs84.geojson"), {"type":"FeatureCollection", "features":features})
    outlines = unary_union([r["geometry"] for r in layers["outlines"]])
    settlement = unary_union([r["geometry"] for r in layers["settlement"]])
    spatial=[]
    for r in layers["outlines"]:
        p = r["properties"]
        if (p["start_date"],p["end_date"]) != ("2005-01-07","2005-01-08"):
            raise ValueError("wrong event outline")
        spatial.append({"object_id":p["outline_co"], "name":p["name"],
                        "source_attributes":p, "area_m2_in_epsg3035":r["geometry"].area,
                        "observation_class":"event_maximum_reported_inundation",
                        "outside_class":"unknown_without_dry_or_coverage_evidence",
                        "model_comparison":"unknown_no_matching_simulation"})
    marks=[]; en_to_3035=Transformer.from_crs(27700,3035,always_xy=True,allow_ballpark=False)
    for r in layers["marks"]:
        p=r["properties"]
        ex,ny=en_to_3035.transform(p["E"],p["N"])
        marks.append({"source_attributes":p, "x_epsg3035":r["geometry"].x, "y_epsg3035":r["geometry"].y,
                      "original_EN_to_geometry_difference_m":math.hypot(ex-r["geometry"].x,ny-r["geometry"].y),
                      "Z_semantics":"surveyed_peak_water_surface_elevation_MASL_not_ground_depth",
                      "inside_outline":bool(outlines.covers(r["geometry"])),
                      "inside_settlement":bool(settlement.covers(r["geometry"])),
                      "distance_to_outline_m":r["geometry"].distance(outlines),
                      "model_comparison":"unknown_no_matching_simulation"})
    metadata, record_range, legend, gauge, gaps=read_gauge(data/"PORecData/Sheepmount.SG[Stage].15min.csv",start,end)
    lookup={datetime.fromisoformat(r["time_as_reported"]):r for r in gauge}
    phases=[]; t=start
    while t+timedelta(hours=6)<=end:
        stop=t+timedelta(hours=6)
        expected=[t+timedelta(minutes=15*i) for i in range(25)]
        section=[lookup[x] for x in expected if x in lookup]
        complete=len(section)==25
        good=complete and all(r["State of value"]=="G" for r in section)
        values=[Decimal(r["Value[m]"]) for r in section]
        delta=values[-1]-values[0] if complete else None
        differences=[b-a for a,b in zip(values,values[1:])]
        phases.append({"start_as_reported":t.isoformat(),"end_as_reported":stop.isoformat(),
                       "sample_count":len(section), "expected_inclusive_count":25,
                       "source_quality_counts":dict(Counter(r["State of value"] for r in section)),
                       "net_delta_m":float(delta) if delta is not None else None,
                       "phase_from_complete_record":net_phase(delta,band) if complete else "unknown_missing_samples",
                       "observed_net_phase":net_phase(delta,band) if good else "withheld_requires_quality_review",
                       "sensitivity_2mm":net_phase(delta,band*2) if good else "withheld_requires_quality_review",
                       "quality_rule_note":"G;Ed means good and edited, not bad; preserve values and withhold automatic acceptance pending source-review",
                       "internal_reversals":any(d>0 for d in differences) and any(d<0 for d in differences),
                       "model_comparison":"unknown_no_matching_simulation"})
        t=stop
    maximum=max(Decimal(r["Value[m]"]) for r in gauge)
    peak=[r for r in gauge if Decimal(r["Value[m]"])==maximum]
    offsets=Counter(str(Decimal(r["Absolute value[mAOD]"])-Decimal(r["Value[m]"])) for r in gauge)
    with Image.open(data/"dem/os.tif") as im:
        dem={"size_pixels":im.size,"pixel_scale":im.tag_v2[33550],"tie_point":im.tag_v2[33922],
             "geokey_directory":im.tag_v2[34735], "crs_description":im.tag_v2[34737], "nodata":im.tag_v2[42113],
             "source_product":"OS Terrain 50; not the missing 5m T07 DEM", "used_in_simulation":False}
    summary={"source_doi":protocol["source_doi"],"evidence_category":"field observation republished in author archive",
             "license_record":"CC BY 4.0; retain original EA/OS/Neal provenance and check any underlying terms before redistribution",
             "outline_count":len(spatial),"outline_union_area_m2":outlines.area,
             "outline_in_settlement_area_m2":outlines.intersection(settlement).area,
             "mark_count":len(marks),"marks_inside_outline":sum(r["inside_outline"] for r in marks),
             "marks_outside_outline":sum(not r["inside_outline"] for r in marks),
             "max_original_EN_to_geometry_difference_m":max(r["original_EN_to_geometry_difference_m"] for r in marks),
             "gauge_metadata":metadata,"source_record_period":record_range,"quality_legend":legend,
             "context_record_count":len(gauge),"context_gaps":gaps,
             "gauge_quality_counts":dict(Counter(r["State of value"] for r in gauge)),
             "timezone":"not stated in original CSV; preserve as-reported times; no UTC assertion",
             "peak_stage_m":float(maximum),"peak_records":peak,
             "absolute_minus_relative_m_counts":dict(offsets),
             "phase_counts":dict(Counter(r["observed_net_phase"] for r in phases)),
             "phase_sensitivity_changed_count":sum(r["observed_net_phase"]!=r["sensitivity_2mm"] for r in phases),
             "transformations":transforms,"dem":dem,
             "model_passes":0,"model_status":"not_run_no_matching_input_package",
             "blocked_items":["5m event-compatible DEM/channel/boundary link not closed; 50m data cannot be relabelled 5m",
                              "outline is maximum extent; no independent dry-coverage mask or horizontal uncertainty supplied",
                              "survey marks outside outline retained; point/polygon discrepancy requires provenance review",
                              "no observed route availability, velocity vectors or people distributions",
                              "point-source timing in experimental DG2 event patch not yet validated"]}
    dump(output/"summary.json",summary)
    dump(output/"spatial-labels.json",spatial)
    dump(output/"all-survey-marks.json",marks)
    dump(output/"gauge-context.json",gauge)
    dump(output/"gauge-phase-labels.json",phases)
    fig,ax=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    def draw(geom,axis,color,lw=0.8):
        parts=list(geom.geoms) if geom.geom_type=="MultiPolygon" else [geom]
        for p in parts:
            x,y=p.exterior.xy; axis.plot(x,y,color=color,lw=lw)
            for ring in p.interiors:
                x,y=ring.xy;axis.plot(x,y,color=color,lw=lw)
    draw(settlement,ax[0],"0.5");draw(outlines,ax[0],"#086eae")
    for inside,color in [(True,"#207d37"),(False,"#ce461f")]:
        rows=[r for r in marks if r["inside_outline"]==inside]
        ax[0].scatter([r["x_epsg3035"] for r in rows],[r["y_epsg3035"] for r in rows],s=7,c=color,
                      label=f'Survey points {"inside" if inside else "outside"} outline: {len(rows)}')
    ax[0].set(aspect="equal",xlabel="Easting (EPSG:3035, m)",ylabel="Northing (m)",title="Recorded maximum extent and all survey points")
    ax[0].legend(fontsize=8)
    ax[1].plot([datetime.fromisoformat(r["time_as_reported"]) for r in gauge],
               [float(r["Value[m]"]) for r in gauge],color="#086eae",label="Original records (all retained)")
    edited=[r for r in gauge if "Ed" in r["State of value"].split(";")]
    ax[1].scatter([datetime.fromisoformat(r["time_as_reported"]) for r in edited],
                  [float(r["Value[m]"]) for r in edited],s=4,color="#d07b1c",label="Good; edited (source flag)")
    for r in peak: ax[1].plot(datetime.fromisoformat(r["time_as_reported"]),float(r["Value[m]"]),"o",color="#ce461f",ms=4)
    ax[1].set(ylabel="Gauge-relative stage (m); not ground depth",xlabel="January 2005, time as reported",title="Sheepmount original 15-minute series")
    ax[1].grid(alpha=.2);ax[1].tick_params(axis="x",rotation=25)
    ax[1].legend(fontsize=8)
    fig.suptitle("Carlisle field evidence only — no HYRA comparison yet")
    fig.savefig(output/"field-evidence.png",dpi=160);plt.close(fig)
    sources=[root/"observation-protocol.json",root/"carlisle-author-metadata.json"]+list(data.rglob("*"))
    dump(output/"manifest.json",{"script_sha256":sha(__file__),"inputs":[{"path":str(f),"sha256":sha(f),"bytes":f.stat().st_size} for f in sources if f.is_file()],
                                "outputs":[{"path":str(f),"sha256":sha(f)} for f in output.iterdir() if f.is_file() and f.name!="manifest.json"]})
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--extra-libs",type=Path)
    a=p.parse_args()
    if a.extra_libs:sys.path.insert(0,str(a.extra_libs))
    result=prepare(a.input,a.output)
    print(json.dumps({k:result[k] for k in ["outline_count","mark_count","marks_outside_outline","context_record_count","gauge_quality_counts","peak_stage_m","phase_counts","phase_sensitivity_changed_count","model_status"]},ensure_ascii=False,indent=2))
