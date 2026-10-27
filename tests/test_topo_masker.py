"""Candidates within 40 m of a building or 10 m of a car road or railway are masked.
Their distance to a building, up to 100 m, nudges the score down.
So are candidates in open water or built land, or within 1 m of their outlines."""

import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.ingest.topo_masker import LAYERS, WATER, distance, masked
from pipeline.run_unit import BUILDING_REACH, WATER_REACH
from scoring.heuristic_scorer import score

POINT, LINE, POLYGON = 1, 2, 3


def geom(kind: int, coords: list[tuple[float, float]]) -> bytes:
    """GeoPackage binary: an 8-byte header (no envelope, EPSG:3006), then little-endian WKB."""
    flat = [v for xy in coords for v in xy]
    if kind == POINT:
        wkb = struct.pack("<BI2d", 1, POINT, *flat)
    elif kind == LINE:  # stored as MultiLineString, as Topografi 50 does
        wkb = struct.pack("<BII", 1, 5, 1) + struct.pack(
            f"<BII{len(flat)}d", 1, LINE, len(coords), *flat
        )
    else:  # one-ring MultiPolygon
        ring = struct.pack(f"<I{len(flat)}d", len(coords), *flat)
        wkb = struct.pack("<BII", 1, 6, 1) + struct.pack("<BII", 1, POLYGON, 1) + ring
    return b"GP\x00\x01" + struct.pack("<i", 3006) + wkb


def write_gpkg(path: Path, tables: dict[str, list[tuple[Any, ...]]]) -> None:
    """Features are (kind, coords) or (kind, coords, objekttyp)."""
    db = sqlite3.connect(path)
    for table, features in tables.items():
        db.execute(f'CREATE TABLE "{table}" (fid INTEGER PRIMARY KEY, geom BLOB, objekttyp TEXT)')
        db.execute(
            f'CREATE VIRTUAL TABLE "rtree_{table}_geom" USING rtree(id, minx, maxx, miny, maxy)'
        )
        for fid, (kind, coords, *objekttyp) in enumerate(features, 1):
            xs, ys = [x for x, _ in coords], [y for _, y in coords]
            db.execute(
                f'INSERT INTO "{table}" VALUES (?, ?, ?)',
                (fid, geom(kind, coords), *(objekttyp or [None])),
            )
            db.execute(
                f'INSERT INTO "rtree_{table}_geom" VALUES (?, ?, ?, ?, ?)',
                (fid, min(xs), max(xs), min(ys), max(ys)),
            )
    db.commit()
    db.close()


def topo(tmp_path: Path) -> Path:
    write_gpkg(
        tmp_path / "byggnadsverk_ln22.gpkg",
        {
            "byggnadspunkt": [(POINT, [(1000.0, 1000.0)])],
            "byggnad": [
                (
                    POLYGON,
                    [
                        (3000.0, 1000.0),
                        (3020.0, 1000.0),
                        (3020.0, 1020.0),
                        (3000.0, 1020.0),
                        (3000.0, 1000.0),
                    ],
                )
            ],
        },
    )
    write_gpkg(
        tmp_path / "kommunikation_ln22.gpkg",
        {
            "vaglinje": [(LINE, [(0.0, 2000.0), (1000.0, 2000.0), (2000.0, 2500.0)])],
            "ralstrafik": [(LINE, [(5000.0, 0.0), (5000.0, 3000.0)])],
            "ovrig_vag": [(LINE, [(0.0, 4000.0), (2000.0, 4000.0)])],  # a tractor road: kept
        },
    )
    lake = [(7000.0, 0.0), (7100.0, 0.0), (7100.0, 100.0), (7000.0, 100.0), (7000.0, 0.0)]
    forest = [(7100.0, 0.0), (7300.0, 0.0), (7300.0, 100.0), (7100.0, 100.0), (7100.0, 0.0)]
    village = [(7300.0, 0.0), (7400.0, 0.0), (7400.0, 100.0), (7300.0, 100.0), (7300.0, 0.0)]
    write_gpkg(
        tmp_path / "mark_ln22.gpkg",
        {
            "mark": [
                (POLYGON, lake, "Sjö"),
                (POLYGON, forest, "Barr- och blandskog"),
                (POLYGON, village, "Låg bebyggelse"),
            ]
        },
    )
    return tmp_path


# (x, y, masked): just inside and just outside each buffer.
CASES = [
    (1039.0, 1000.0, True), (1000.0, 1041.0, False),  # building point, 40 m
    (3010.0, 1010.0, True), (3059.0, 1010.0, True), (3061.0, 1010.0, False),  # building polygon
    (500.0, 2009.0, True), (500.0, 2011.0, False),  # car road, 10 m
    (-9.0, 2000.0, True), (-11.0, 2000.0, False),  # beyond the road's end
    (5009.0, 1500.0, True), (5011.0, 1500.0, False),  # railway
    (1000.0, 4000.0, False),  # on the tractor road
    (7050.0, 50.0, True), (7101.5, 50.0, False),  # lake, 1 m
    (7200.0, 50.0, False),  # forest
    (7350.0, 50.0, True), (7298.5, 50.0, False),  # built land, 1 m (from its west side)
]  # fmt: skip
# Within the 1 m of a lake or built land: masked too.
SHORE = [(7100.5, 50.0), (7299.5, 50.0)]


def test_candidates_inside_each_buffer_are_masked(tmp_path: Path) -> None:
    x, y, expected = (np.array(v) for v in zip(*CASES, strict=True))
    sx, sy = np.array(SHORE).T
    hit = masked(np.r_[x, sx], np.r_[y, sy], topo(tmp_path))
    assert list(hit) == [*expected, True, True]


def test_building_distance_nudges_the_score(tmp_path: Path) -> None:
    # Inside the building polygon, 70 m east of it, and beyond the nudge's 100 m.
    x, y = np.array([3010.0, 3090.0, 3121.0]), np.full(3, 1010.0)
    theme, tables, *_ = LAYERS[0]  # as run_unit sets dist_building
    topo_dir = topo(tmp_path)
    near = distance(x, y, topo_dir, theme, tables, BUILDING_REACH)
    assert list(near) == [0.0, 70.0, np.inf]
    assert list(masked(x, y, topo_dir)) == [True, False, False]
    # No other feature: only the nudge.
    assert [score({"dist_building": d}) for d in near] == [-2.0, -1.0, 0.0]


def test_water_distance_counts_open_water_only(tmp_path: Path) -> None:
    # In the lake, 3 m from its shore inside the forest, 30 m into the forest (beyond
    # the 20 m reach); the forest polygon in the same table is not water.
    x, y = np.array([7050.0, 7103.0, 7130.0]), np.full(3, 50.0)
    water = distance(x, y, topo(tmp_path), *WATER[:2], WATER_REACH, WATER[2])  # as run_unit
    assert list(water) == [0.0, 3.0, np.inf]
