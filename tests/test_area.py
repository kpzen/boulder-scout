"""The selection rule, and the fetch plan, over the recorded STAC search."""

import copy
import json
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pyproj import Transformer

from pipeline.area import (
    TILE_M,
    check_extent,
    files_to_fetch,
    nominal_tiles,
    overlapping,
    polygon_tiles,
    select_work_units,
    units_on_disk,
)

# The real search response (57 items, 2026-10-17), trimmed of the bulky per-item statistics and
# schemas: tests need no network.
ITEMS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "area_stac_items.json").read_text(encoding="utf-8")
)["features"]
# The recorded search's extent: 5 x 5 whole tiles around Överturingen.
AREA = (490000.0, 6917500.0, 502500.0, 6930000.0)

# The 25 files of the recorded search, in nominal_tiles order. The stem encodes the
# tile's south-west corner: y20e018-69175_4900_25 is N 6917500, E 490000.
EXPECTED = [
    "y20e018-69175_4900_25",
    "y20e018-69175_4925_25",
    "y20e018-69175_4950_25",
    "y20e018-69175_4975_25",
    "y20e019-69175_5000_25",
    "y20e018-69200_4900_25",
    "y20e018-69200_4925_25",
    "y20e018-69200_4950_25",
    "y20e018-69200_4975_25",
    "y20e019-69200_5000_25",
    "y20e018-69225_4900_25",
    "y20e018-69225_4925_25",
    "y20e018-69225_4950_25",
    "y20e018-69225_4975_25",
    "y20e019-69225_5000_25",
    "y20e029-69250_4900_25",
    "y20e029-69250_4925_25",
    "y20e029-69250_4950_25",
    "y20e029-69250_4975_25",
    "y20e030-69250_5000_25",
    "y20e029-69275_4900_25",
    "y20e029-69275_4925_25",
    "y20e029-69275_4950_25",
    "y20e029-69275_4975_25",
    "y20e030-69275_5000_25",
]


def test_rule_picks_the_expected_files() -> None:
    units = select_work_units(ITEMS, AREA)
    assert [u.tile_id for u in units] == EXPECTED
    for u in units:
        northing, easting = u.tile_id.split("-")[1].split("_")[:2]
        assert (u.bbox[0], u.bbox[1]) == (float(easting) * 100, float(northing) * 100)
        assert u.href.endswith(f"{u.tile_id}.copc.laz")


def test_the_10_km_files_inside_the_extent_are_not_picked() -> None:
    """The rule has something to reject: two `m` files also lie wholly inside the extent."""
    inside = [
        it
        for it in ITEMS
        if (b := it["properties"]["proj:bbox"])
        and b[0] >= AREA[0]
        and b[1] >= AREA[1]
        and b[2] <= AREA[2]
        and b[3] <= AREA[3]
    ]
    ten_km = [it for it in inside if (b := it["properties"]["proj:bbox"])[2] - b[0] > TILE_M]
    assert len(ten_km) == 2
    picked = {u.href for u in select_work_units(ITEMS, AREA)}
    assert picked.isdisjoint({a["href"] for it in ten_km for a in it["assets"].values()})
    assert len(picked) == len(inside) - 2 == 25


def test_the_work_units_cover_the_area_with_no_gaps() -> None:
    units = select_work_units(ITEMS, AREA)
    step = 100.0
    gx = np.arange(AREA[0] + step / 2, AREA[2], step)
    gy = np.arange(AREA[1] + step / 2, AREA[3], step)
    px, py = (v.ravel() for v in np.meshgrid(gx, gy))
    n = np.zeros(px.shape, int)
    for u in units:
        n += (px >= u.bbox[0]) & (px < u.bbox[2]) & (py >= u.bbox[1]) & (py < u.bbox[3])
    assert n.min() == 1 and n.max() == 1  # no gap, and no cell read twice
    assert len(units) * TILE_M**2 == (AREA[2] - AREA[0]) * (AREA[3] - AREA[1])


def test_nominal_tiles_tile_the_extent() -> None:
    tiles = nominal_tiles(AREA)
    assert len(tiles) == 25
    assert tiles[0][:2] == AREA[:2]
    assert tiles[-1][2:] == AREA[2:]


def test_a_tile_without_a_file_has_no_unit() -> None:
    """Sea or the border."""
    without = [it for it in ITEMS if it["id"] != "20e018-69175_4900_25"]
    assert [u.tile_id for u in select_work_units(without, AREA)] == EXPECTED[1:]


def test_an_extent_is_whole_tiles_of_the_grid() -> None:
    assert check_extent(AREA) == AREA
    for bad in [
        (490000.0, 6917500.0, 501000.0, 6930000.0),
        (502500.0, 6917500.0, 490000.0, 6930000.0),
    ]:
        with pytest.raises(ValueError, match="whole 2500 m tiles"):
            check_extent(bad)
    assert len(select_work_units(ITEMS, (490000.0, 6917500.0, 495000.0, 6920000.0))) == 2


def test_two_files_on_one_tile_are_an_error() -> None:
    twice = copy.deepcopy(ITEMS)
    rescan = copy.deepcopy(twice[0])
    rescan["id"] = "20e099-69175_4900_25"
    rescan["properties"]["proj:bbox"] = [490000.0, 6917500.0, 492500.0, 6920000.0]
    rescan["assets"]["data"]["href"] = (
        "https://dl1.lantmateriet.se/hojd/data/pointcloud/sls/20e099/y20e099-69175_4900_25.copc.laz"
    )
    twice.append(rescan)
    with pytest.raises(ValueError, match="y20e099-69175_4900_25"):
        select_work_units(twice, AREA)


def test_the_fetch_plan_is_every_work_unit_file(tmp_path: Path) -> None:
    plan = files_to_fetch(ITEMS, tmp_path, AREA)
    assert [dest.name for _, dest in plan] == [f"{t}.copc.laz" for t in EXPECTED]
    assert {dest.parent for _, dest in plan} == {tmp_path}
    urls = [u.href for u, _ in plan]
    assert len(set(urls)) == 25
    assert all(urllib.parse.urlsplit(h).scheme == "https" for h in urls)
    assert urls[0] == (
        "https://dl1.lantmateriet.se/hojd/data/pointcloud/sls/20e018/y20e018-69175_4900_25.copc.laz"
    )


def test_files_already_in_the_raw_directory_are_skipped(tmp_path: Path) -> None:
    for tile in EXPECTED[:3]:
        (tmp_path / f"{tile}.copc.laz").write_bytes(b"")
    (tmp_path / f"{EXPECTED[3]}.copc.laz.part").write_bytes(b"")  # interrupted: fetch it again
    plan = files_to_fetch(ITEMS, tmp_path, AREA)
    assert [dest.name for _, dest in plan] == [f"{t}.copc.laz" for t in EXPECTED[3:]]


def test_the_batch_plan_reads_the_fetched_files(tmp_path: Path) -> None:
    for _, dest in files_to_fetch(ITEMS, tmp_path, AREA):
        dest.write_bytes(b"")
    (tmp_path / f"{EXPECTED[0]}.copc.laz.part").write_bytes(b"")  # not a second file
    plan = units_on_disk(tmp_path, AREA)
    # The same tiles and files as the rule over the STAC search, read from the names alone.
    assert [(p.name, bbox) for p, bbox, _ in plan] == [
        (f"{u.tile_id}.copc.laz", u.bbox) for u in select_work_units(ITEMS, AREA)
    ]
    # Neighbours share an edge or a corner: 3 at the extent's corners, 5 along its sides, 8 inside.
    counts = [len(n) for _, _, n in plan]
    assert sorted(counts) == [3] * 4 + [5] * 12 + [8] * 9
    centre = plan[12]  # the middle tile of 5 x 5
    assert {p.name for p, _ in centre[2]} == {
        f"{EXPECTED[i]}.copc.laz" for i in (6, 7, 8, 11, 13, 16, 17, 18)
    }
    own = {p: bbox for p, bbox, _ in plan}
    assert all(own[p] == bbox for _, _, ns in plan for p, bbox in ns)


def test_the_batch_plan_leaves_out_a_tile_with_no_file(tmp_path: Path) -> None:
    for _, dest in files_to_fetch(ITEMS, tmp_path, AREA)[1:]:
        dest.write_bytes(b"")
    plan = units_on_disk(tmp_path, AREA)
    assert [p.name for p, _, _ in plan] == [f"{t}.copc.laz" for t in EXPECTED[1:]]
    assert len(plan[0][2]) == 4  # the south-west corner's east tile lost its west neighbour


def test_the_ground_model_squares_a_box_overlaps() -> None:
    dtm = [Path(f"m{n}_{e}.tif") for n in (691, 692, 693) for e in (48, 49, 50)]
    box = (489999.0, 6917499.0, 502501.0, 6920001.0)  # the v1 area's south row, read 1 m wider
    assert [p.stem for p in overlapping(dtm, box)] == [
        "m691_48", "m691_49", "m691_50", "m692_48", "m692_49", "m692_50"
    ]  # fmt: skip


def test_polygon_tiles_are_the_union_of_the_polygons_less_their_holes() -> None:
    to_wgs84 = Transformer.from_crs(3006, 4326, always_xy=True)

    def ring(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
        """A rectangle in EPSG:3006 tile units from E 490000, N 6917500, as WGS84 positions."""
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return [
            list(to_wgs84.transform(490000 + x * TILE_M, 6917500 + y * TILE_M)) for x, y in corners
        ]

    def corners(geometry: dict[str, Any]) -> list[tuple[int, int]]:
        tiles = polygon_tiles({"type": "Feature", "properties": {}, "geometry": geometry})
        return [(round((t[0] - 490000) / TILE_M), round((t[1] - 6917500) / TILE_M)) for t in tiles]

    # Two 2 x 2 squares overlapping on one tile: the overlap counts once, not as outside.
    two = {"type": "MultiPolygon", "coordinates": [[ring(0, 0, 2, 2)], [ring(1, 1, 3, 3)]]}
    assert corners(two) == [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]
    # A hole takes its tile out of its own polygon.
    holed = {"type": "Polygon", "coordinates": [ring(0, 0, 3, 3), ring(1, 1, 2, 2)]}
    assert (1, 1) not in corners(holed) and len(corners(holed)) == 8
    with pytest.raises(ValueError, match="no Polygon"):
        polygon_tiles({"type": "Point", "coordinates": [15.0, 62.0]})
    with pytest.raises(ValueError, match="no 2.5 km tile"):
        corners({"type": "Polygon", "coordinates": [ring(0.1, 0.1, 0.3, 0.3)]})
