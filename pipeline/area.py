"""The 2.5 km work units of a run's extent and the file that serves each.

The STAC search returns every file whose bbox touches the extent, in both packagings. The
rule: a work unit is a whole nominal 2.5 km tile of the extent, and its file is the one whose
own nominal tile (`proj:bbox`) that is, read clipped to the tile. So a 10 km `m` file never
qualifies, nor does a neighbour that only spills a few metres over the edge. A tile with no
file (sea, the border) has no work unit.
"""

import math
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

TILE_M = 2500.0

Bbox = tuple[float, float, float, float]

TO_SWEREF = Transformer.from_crs(4326, 3006, always_xy=True)


@dataclass(frozen=True)
class WorkUnit:
    """One tile of the run: the ground it covers and the file that holds it."""

    tile_id: str  # the LAZ stem, which is how run_unit names a tile
    href: str
    bbox: Bbox  # the nominal tile, EPSG:3006, that the file is read clipped to

    @property
    def name(self) -> str:
        """The file's name, as at the source and in the raw directory."""
        return Path(urllib.parse.urlsplit(self.href).path).name


def check_extent(area: Bbox) -> Bbox:
    """A run's extent from the command line: whole 2.5 km tiles of the grid, else ValueError."""
    x0, y0, x1, y1 = area
    if not (x0 < x1 and y0 < y1) or any(v % TILE_M for v in area):
        raise ValueError(f"the extent must be whole {TILE_M:.0f} m tiles (EPSG:3006), got {area}")
    return area


def nominal_tiles(area: Bbox) -> list[Bbox]:
    """The whole 2.5 km tiles tiling the extent, west to east then south to north."""
    x0, y0, x1, y1 = area
    return [
        (x0 + j * TILE_M, y0 + i * TILE_M, x0 + (j + 1) * TILE_M, y0 + (i + 1) * TILE_M)
        for i in range(round((y1 - y0) / TILE_M))
        for j in range(round((x1 - x0) / TILE_M))
    ]


def polygon_tiles(geojson: dict[str, Any]) -> list[Bbox]:
    """The 2.5 km tiles whose centre lies inside a GeoJSON Polygon or MultiPolygon in WGS84 (as
    the area page or geojson.io saves it, also as a Feature or FeatureCollection), in
    `nominal_tiles` order: the union of its polygons, each with its holes. Raises ValueError
    when it holds no polygon or no tile centre lies inside."""
    if geojson.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") or {} for f in geojson.get("features", [])]
    else:
        geometries = [geojson.get("geometry") or geojson]
    polygons: list[list[np.ndarray]] = []  # each a polygon's rings, outline then holes
    for g in geometries:
        coordinates: list[Any] = g.get("coordinates") or []
        by_type = {"Polygon": [coordinates], "MultiPolygon": coordinates}
        for polygon in by_type.get(g.get("type", ""), []):
            rings = []
            for ring in polygon:
                x, y = TO_SWEREF.transform(*np.array(ring, float)[:, :2].T)
                rings.append(np.column_stack([x, y]))
            polygons.append(rings)
    if not polygons:
        raise ValueError("the GeoJSON holds no Polygon or MultiPolygon")
    xy = np.concatenate([r for rings in polygons for r in rings])
    lo = [math.floor(v / TILE_M) * TILE_M for v in xy.min(axis=0)]
    hi = [math.ceil(v / TILE_M) * TILE_M for v in xy.max(axis=0)]
    tiles = nominal_tiles((lo[0], lo[1], hi[0], hi[1]))
    centres = np.array([((t[0] + t[2]) / 2, (t[1] + t[3]) / 2) for t in tiles])
    hits = np.any([inside(rings, centres[:, 0], centres[:, 1]) for rings in polygons], axis=0)
    out = [t for t, hit in zip(tiles, hits, strict=True) if hit]
    if not out:
        raise ValueError("no 2.5 km tile has its centre inside the polygon")
    return out


def _laz_asset(item: dict[str, Any]) -> dict[str, Any] | None:
    assets: dict[str, dict[str, Any]] = item["assets"]
    for a in assets.values():
        if a["href"].lower().split("?")[0].endswith(".laz") or "laz" in a.get("type", "").lower():
            return a
    return None


def _corner(item: dict[str, Any]) -> tuple[int, int] | None:
    """The south-west corner of the item's nominal tile, if it is a whole 2.5 km tile."""
    bbox = item["properties"].get("proj:bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x0, y0, x1, y1 = (round(v) for v in bbox)
    if (x1 - x0, y1 - y0) != (round(TILE_M), round(TILE_M)):
        return None
    return x0, y0


def select_work_units(items: list[dict[str, Any]], area: Bbox) -> list[WorkUnit]:
    """The selection rule over a STAC search response, in `nominal_tiles` order. A tile with
    no file has no unit.

    Raises ValueError when a tile has more than one file: the rule cannot decide the run on
    its own and the choice is the author's.
    """
    tiles = nominal_tiles(area)
    wanted = {(round(t[0]), round(t[1])): t for t in tiles}
    found: dict[tuple[int, int], list[WorkUnit]] = {}
    for item in items:
        corner = _corner(item)
        asset = _laz_asset(item)
        if corner is None or corner not in wanted or asset is None:
            continue
        name = Path(urllib.parse.urlsplit(asset["href"]).path).name
        found.setdefault(corner, []).append(
            WorkUnit(
                tile_id=name.split(".")[0],
                href=asset["href"],
                bbox=wanted[corner],
            )
        )
    several = {c: [u.tile_id for u in us] for c, us in found.items() if len(us) > 1}
    if several:
        raise ValueError(f"the rule gives several files on {len(several)} tiles: {several}")
    return [found[c][0] for t in tiles if (c := (round(t[0]), round(t[1]))) in found]


def with_neighbours(
    files: list[tuple[Path, Bbox]],
) -> list[tuple[Path, Bbox, list[tuple[Path, Bbox]]]]:
    """The batch run's plan: each (file, nominal tile) with its neighbours (file, nominal
    tile), the tiles that share an edge or a corner with it."""
    return [
        (
            path,
            t,
            [
                (p, n)
                for p, n in files
                if n != t and abs(n[0] - t[0]) <= TILE_M and abs(n[1] - t[1]) <= TILE_M
            ],
        )
        for path, t in files
    ]


def units_on_disk(raw_dir: Path, area: Bbox) -> list[tuple[Path, Bbox, list[tuple[Path, Bbox]]]]:
    """`with_neighbours` over the files in `raw_dir`; a tile with no file there has no unit.

    Offline: the fetch CLI names files as at the source, and the stem encodes the nominal tile's
    south-west corner in 100 m (y20e018-69175_4900_25 is N 6917500, E 490000). Raises
    ValueError when a tile has several files.
    """
    found: dict[tuple[int, int], list[Path]] = {}
    for path in sorted(raw_dir.glob("y*.copc.laz")):
        found.setdefault(tile_corner(path.name.split(".")[0]), []).append(path)
    files = []
    for t in nominal_tiles(area):
        paths = found.get((round(t[0]), round(t[1])), [])
        if len(paths) > 1:
            raise ValueError(
                f"tile E {t[0]:.0f}, N {t[1]:.0f} has several files in {raw_dir}: "
                f"{[p.name for p in paths]}"
            )
        files += [(p, t) for p in paths]
    return with_neighbours(files)


def tile_corner(stem: str) -> tuple[int, int]:
    """The nominal tile's south-west corner (E, N) from a file's stem, which encodes it in 100 m
    (y20e018-69175_4900_25 is E 490000, N 6917500)."""
    northing, easting = stem.split("-")[1].split("_")[:2]
    return int(easting) * 100, int(northing) * 100


def files_to_fetch(
    items: list[dict[str, Any]], raw_dir: Path, area: Bbox
) -> list[tuple[WorkUnit, Path]]:
    """The fetch CLI's plan: each work unit whose file is not in `raw_dir`, and where it goes.

    A file is named as at the source, so a run already holding it skips it; an interrupted
    download leaves only a `.part`, which does not count as present.
    """
    plan = []
    for unit in select_work_units(items, area):
        dest = raw_dir / unit.name
        if not dest.exists():
            plan.append((unit, dest))
    return plan


def overlapping(dtm: list[Path], box: Bbox) -> list[Path]:
    """The Markhöjdmodell files whose 10 km square overlaps `box`. The name gives the square's
    south-west corner in 10 km: m691_48 is N 6910000, E 480000."""
    out = []
    for path in dtm:
        northing, easting = path.stem[1:].split("_")
        e, n = int(easting) * 10000, int(northing) * 10000
        if e < box[2] and box[0] < e + 10000 and n < box[3] and box[1] < n + 10000:
            out.append(path)
    return out


def inside(rings: list[np.ndarray], px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Even-odd ray crossing: which of the query points fall inside the rings."""
    hit = np.zeros(px.shape, bool)
    for a in rings:
        bx, by = np.roll(a[:, 0], -1), np.roll(a[:, 1], -1)
        for ax, ay, cx, cy in zip(a[:, 0], a[:, 1], bx, by, strict=True):
            if ay == cy:
                continue
            crosses = (ay > py) != (cy > py)
            hit ^= crosses & (px < ax + (py - ay) * (cx - ax) / (cy - ay))
    return hit
