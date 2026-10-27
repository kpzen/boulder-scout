"""Building, road, water and built-land exclusion from Topografi 50.

Topografi 50 Nedladdning, vektor comes as one GeoPackage per theme for a län or all of Sweden,
e.g. `byggnadsverk_sverige.gpkg`, in EPSG:3006 (geometries stored easting first). Python's
sqlite3 reads it: each table has a `geom` column of GeoPackage binary (a small header, then
WKB) and an R-tree `rtree_{table}_geom` over the features' bounding boxes. Applied after the
proposer, to candidate centroids, so shapes, scores and label cut-outs do not change.
"""

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pipeline.area import Bbox, inside

# Open water in `mark`, for dist_water (the score's water-edge gate).
WATER = ("mark", ("mark",), ("Sjö", "Vattendragsyta", "Hav", "Anlagt vatten"))
BUILT_LAND = (
    "Låg bebyggelse",
    "Hög bebyggelse",
    "Sluten bebyggelse",
    "Industri- och handelsbebyggelse",
)
# (theme, tables, buffer m, objekttyp kinds), agreed with the author on the
# area labels: no labelled boulder lies within 48 m of a building or 16 m of a car road.
# Buildings get 40 m: Topografi 50 leaves out many small buildings, which sit near the
# ones it keeps. Paths and tractor roads (ovrig_vag) are left out: boulders sit along them.
# Open water and built land too, 1 m past their outlines; a boulder at a shore may go.
LAYERS = [
    # byggnad: polygons, byggnadspunkt: points
    ("byggnadsverk", ("byggnad", "byggnadspunkt"), 40.0, ()),
    ("kommunikation", ("vaglinje", "ralstrafik"), 10.0, ()),  # car roads and rail
    ("mark", ("mark",), 1.0, WATER[2]),
    ("mark", ("mark",), 1.0, BUILT_LAND),
]
ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}  # by the GeoPackage header's envelope code


@dataclass
class Topo:
    """One buffer's features, cut to a bounding box."""

    points: np.ndarray  # (n, 2)
    segments: np.ndarray  # (m, 4) x0, y0, x1, y1: lines and polygon outlines
    polygons: list[list[np.ndarray]]  # each polygon's rings, holes included


def _wkb(b: bytes, o: int, out: list[tuple[int, np.ndarray]]) -> int:
    """Append (type, (n, 2) coordinates) for each point (1), line (2) or ring (3 for a polygon's
    outer ring, -3 for its holes); return the end offset."""
    bo = "<" if b[o] == 1 else ">"
    kind = struct.unpack_from(bo + "I", b, o + 1)[0]
    o += 5
    if kind == 1:
        out.append((1, np.array([struct.unpack_from(bo + "2d", b, o)])))
        return o + 16
    (n,) = struct.unpack_from(bo + "I", b, o)
    o += 4
    if kind in (2, 3):
        for i in range(n if kind == 3 else 1):
            if kind == 3:
                m = struct.unpack_from(bo + "I", b, o)[0]
                o += 4
            else:
                m = n
            ring = np.frombuffer(b, bo + "f8", 2 * m, o).reshape(m, 2)
            out.append((-3 if i else kind, ring))
            o += 16 * m
        return o
    if kind in (4, 5, 6):
        for _ in range(n):
            o = _wkb(b, o, out)
        return o
    raise ValueError(f"unsupported WKB geometry type {kind} (only 2D points, lines, polygons)")


def read_topo(
    topo_dir: Path, theme: str, tables: tuple[str, ...], bbox: Bbox, kinds: tuple[str, ...] = ()
) -> Topo:
    """The theme's features in `tables` whose parts reach into bbox, from every län's file;
    with `kinds`, only those whose `objekttyp` is one of them."""
    x0, y0, x1, y1 = bbox
    points, segments, polygons = [], [], []
    paths = sorted(topo_dir.glob(f"{theme}_*.gpkg"))
    if not paths:
        raise FileNotFoundError(f"no {theme}_*.gpkg in {topo_dir} (unzip Topografi 50 there)")
    for path in paths:
        db = sqlite3.connect(path)
        for table in tables:
            only = f" AND objekttyp IN ({','.join('?' * len(kinds))})" if kinds else ""
            rows = db.execute(
                f'SELECT geom FROM "{table}" WHERE fid IN (SELECT id FROM "rtree_{table}_geom" '
                "WHERE maxx >= ? AND minx <= ? AND maxy >= ? AND miny <= ?)" + only,
                (x0, x1, y0, y1, *kinds),
            )
            for (g,) in rows:
                parts: list[tuple[int, np.ndarray]] = []
                _wkb(g, 8 + ENVELOPE_BYTES[(g[3] >> 1) & 7], parts)
                for kind, a in parts:
                    if kind == 1:
                        points.append(a)
                        continue
                    segments.append(np.hstack([a[:-1], a[1:]]))
                    if kind == 3:
                        polygons.append([a])
                    elif kind == -3:
                        polygons[-1].append(a)
        db.close()
    pts = np.concatenate(points) if points else np.zeros((0, 2))
    seg = np.concatenate(segments) if segments else np.zeros((0, 4))
    keep = (
        (np.maximum(seg[:, 0], seg[:, 2]) >= x0) & (np.minimum(seg[:, 0], seg[:, 2]) <= x1)
        & (np.maximum(seg[:, 1], seg[:, 3]) >= y0) & (np.minimum(seg[:, 1], seg[:, 3]) <= y1)
    )  # fmt: skip
    in_box = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return Topo(pts[in_box], seg[keep], polygons)


def distance(
    x: np.ndarray,
    y: np.ndarray,
    topo_dir: Path,
    theme: str,
    tables: tuple[str, ...],
    reach: float,
    kinds: tuple[str, ...] = (),
) -> np.ndarray:
    """Each point's distance to the nearest feature in `tables` (of `kinds`, see read_topo; 0
    inside a polygon), inf when none lies within `reach`."""
    d = np.full(len(x), np.inf)
    if not len(x):
        return d
    tree = cKDTree(np.column_stack([x, y]))
    r = reach
    box = (x.min() - r, y.min() - r, x.max() + r, y.max() + r)
    topo = read_topo(topo_dir, theme, tables, box, kinds)
    for (px, py), idx in zip(
        topo.points, tree.query_ball_point(topo.points, r) if len(topo.points) else [], strict=True
    ):
        d[idx] = np.minimum(d[idx], np.hypot(x[idx] - px, y[idx] - py))
    ax, ay, bx, by = topo.segments.T
    half = np.hypot(bx - ax, by - ay) / 2
    near = tree.query_ball_point(np.column_stack([(ax + bx) / 2, (ay + by) / 2]), half + r)
    for (sx, sy, ex, ey), idx in zip(topo.segments, near, strict=True):
        if not idx:
            continue
        px, py = x[idx], y[idx]
        vx, vy = ex - sx, ey - sy
        t = np.clip(((px - sx) * vx + (py - sy) * vy) / max(vx * vx + vy * vy, 1e-12), 0, 1)
        d[idx] = np.minimum(d[idx], np.hypot(px - sx - t * vx, py - sy - t * vy))
    for rings in topo.polygons:
        d[inside(rings, x, y)] = 0.0
    d[d > r] = np.inf
    return d


def masked(x: np.ndarray, y: np.ndarray, topo_dir: Path) -> np.ndarray:
    """Which points lie within LAYERS' buffer of a building, road, open water or built land
    (or inside a building, water or built land polygon)."""
    hit = np.zeros(len(x), bool)
    for theme, tables, r, kinds in LAYERS:
        hit |= distance(x, y, topo_dir, theme, tables, r, kinds) <= r
    return hit
