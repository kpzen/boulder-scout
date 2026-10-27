from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from pipeline.extractors.geometry import FILTERS, dropped, geometry, hanging, object_points
from pipeline.ingest.laz_reader import POINT_DTYPE
from pipeline.run_unit import main
from tests.real_fixtures import BOULDERS, FIXTURES


def _rows(tile: Path, out: Path) -> list[dict[str, Any]]:
    main([str(tile), "--out", str(out), "--patches", str(out.parent / "patches")])
    rows: list[dict[str, Any]] = pq.read_table(out).to_pylist()
    return rows


def _geometry(points: np.ndarray, z_norm: np.ndarray, inside: np.ndarray) -> dict[str, float]:
    return geometry(points, z_norm, inside, object_points(points, z_norm, inside))


def _block(stray: int) -> np.ndarray:
    # A 3 m flat-topped block on a 0.5 m grid, all single returns, plus `stray` crown hits
    # at 10-14 m that got through the last-return filter (single-return canopy hits).
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(0.0, 10.0, 0.5), np.arange(0.0, 10.0, 0.5))
    x = np.concatenate([gx.ravel(), rng.uniform(0.0, 10.0, stray)])
    y = np.concatenate([gy.ravel(), rng.uniform(0.0, 10.0, stray)])
    z = np.concatenate([3.0 + rng.normal(0.0, 0.05, gx.size), rng.uniform(10.0, 14.0, stray)])
    points = np.zeros(len(x), dtype=POINT_DTYPE)
    points["x"], points["y"], points["z"] = x, y, z
    points["classification"], points["return_number"], points["num_returns"] = 1, 1, 1
    return points


def test_stray_crown_hits_do_not_raise_height() -> None:
    clean = _block(0)
    clean_feats = _geometry(clean, clean["z"], np.ones(len(clean), dtype=bool))
    # 40 of 440 points (9 %): enough to move H95 if they were kept, as on b04.
    points = _block(40)
    inside = np.ones(len(points), dtype=bool)
    assert points["z"][object_points(points, points["z"], inside)].max() < 3.5
    assert abs(_geometry(points, points["z"], inside)["max_h"] - clean_feats["max_h"]) < 0.05


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_overhanging_crown_kept_out_of_object(seed: int) -> None:
    # A crown 4-8 m up over the 3 m block's edge and the ground beside it, its
    # last returns inside the outline (the proposer's outline spreads under a crown). At the
    # area's 4 pulses/m², most crown hits have a rock or ground hit close beside and far below.
    block = np.concatenate([_shape("block", seed), _shape("block", seed + 10)])
    rng = np.random.default_rng(seed)
    n = 150
    r, t = rng.uniform(3.0, 6.0, n), rng.uniform(0.0, np.pi, n)
    crown = np.zeros(n, dtype=POINT_DTYPE)
    crown["x"], crown["y"], crown["z"] = r * np.cos(t), r * np.sin(t), rng.uniform(4.0, 8.0, n)
    crown["classification"], crown["return_number"], crown["num_returns"] = 1, 1, 1
    points = np.concatenate([block, crown])
    inside = np.hypot(points["x"], points["y"]) < 6.0
    obj = object_points(points, points["z"], inside)
    feats = geometry(points, points["z"], inside, obj)
    assert abs(feats["max_h"] - 3.0) < 0.2
    assert feats["max_dropoff"] < 8.0  # rock's range
    assert points["z"][obj].max() < 3.5  # the crown remnants over the block go


def _shape(kind: str, seed: int) -> np.ndarray:
    # A 3 m shape at 2 pulses/m² (like the fixture crops): a block with vertical walls
    # (radius 4 m) or a rounded paraboloid knoll (radius 6 m). Ground is class 2.
    rng = np.random.default_rng(seed)
    n = 800
    x, y = rng.uniform(-10.0, 10.0, (2, n))
    r = np.hypot(x, y)
    h = (
        np.where(r < 4.0, 3.0, 0.0)
        if kind == "block"
        else np.clip(3.0 * (1 - (r / 6.0) ** 2), 0, None)
    )
    points = np.zeros(n, dtype=POINT_DTYPE)
    points["x"], points["y"], points["z"] = x, y, h + rng.normal(0.0, 0.05, n)
    points["classification"] = np.where(h > 0.0, 1, 2)
    points["return_number"], points["num_returns"] = 1, 1
    return points


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_walled_block_drops_off_more_than_knoll(seed: int) -> None:
    # `inside` is the proposer's 2 m outline.
    block, knoll = _shape("block", seed), _shape("knoll", seed)
    block_drop = _geometry(block, block["z"], block["z"] >= 2.0)["max_dropoff"]
    knoll_drop = _geometry(knoll, knoll["z"], knoll["z"] >= 2.0)["max_dropoff"]
    assert block_drop > 3.0 * knoll_drop


def test_no_base_points_gives_nan_dropoff() -> None:
    points = _block(0)  # all 3 m high: nothing below BASE_MAX
    feats = _geometry(points, points["z"], np.ones(len(points), dtype=bool))
    assert np.isnan(feats["max_dropoff"])
    assert not np.isnan(feats["max_h"])


def test_no_object_points_gives_nan() -> None:
    points = _block(0)
    z_norm = np.full(len(points), 0.5)  # all below OBJ_MIN
    feats = _geometry(points, z_norm, np.ones(len(points), dtype=bool))
    assert all(np.isnan(v) for v in feats.values())


@pytest.mark.parametrize("boulder", BOULDERS, ids=[b["id"] for b in BOULDERS])
def test_fixture_heights_near_field_estimates(tmp_path: Path, boulder: dict[str, Any]) -> None:
    # Author's field estimates are rough; H95 sits within 2 m of them.
    rows = _rows(FIXTURES / boulder["crops"][0]["file"], tmp_path / "f.parquet")
    row = min(rows, key=lambda r: np.hypot(r["x"] - boulder["e"], r["y"] - boulder["n"]))
    assert abs(row["max_h"] - boulder["height_m"]) < 2.0


def _footprint(length: float, width: float, angle: float, seed: int) -> np.ndarray:
    # A 3 m flat-topped block with a length x width footprint turned by `angle` (rad), at
    # 10 pulses/m² over a ground margin of 3 m. Ground is class 2. At the fixtures' 2 pulses/m²
    # sampling noise alone puts a 6 m square anywhere in 0.5-1.0 (median 0.79, 50 seeds).
    rng = np.random.default_rng(seed)
    half = np.array([length, width]) / 2 + 3.0
    n = int(10.0 * 4 * half.prod())
    u, v = rng.uniform(-half, half, (n, 2)).T
    h = np.where((abs(u) < length / 2) & (abs(v) < width / 2), 3.0, 0.0)
    points = np.zeros(n, dtype=POINT_DTYPE)
    points["x"] = u * np.cos(angle) - v * np.sin(angle)
    points["y"] = u * np.sin(angle) + v * np.cos(angle)
    points["z"] = h + rng.normal(0.0, 0.05, n)
    points["classification"] = np.where(h > 0.0, 1, 2)
    points["return_number"], points["num_returns"] = 1, 1
    return points


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_square_block_compact_wall_elongated(seed: int) -> None:
    # An 8 x 8 m block vs a 20 x 2 m wall, both turned off the axes.
    block, wall = _footprint(8.0, 8.0, 0.5, seed), _footprint(20.0, 2.0, 0.5, seed)
    block_feats = _geometry(block, block["z"], block["z"] >= 2.0)
    wall_feats = _geometry(wall, wall["z"], wall["z"] >= 2.0)
    assert block_feats["elongation_2d"] > 0.8
    assert wall_feats["elongation_2d"] < 0.2


def test_points_on_a_line_give_nan_shape() -> None:
    points = _block(0)[:20]  # one row of the grid: all y equal
    feats = _geometry(points, points["z"], np.ones(len(points), dtype=bool))
    assert np.isnan(feats["elongation_2d"]) and np.isnan(feats["tallness"])
    assert not np.isnan(feats["max_h"])


def _crown(seed: int) -> np.ndarray:
    # A dense crown 8 m wide from 3 to 7 m, at 4 pulses/m², like the fixtures' canopy
    # candidates: every pulse ends in a single return, on average 1 m into the foliage.
    rng = np.random.default_rng(seed)
    n = 1600
    x, y = rng.uniform(-10.0, 10.0, (2, n))
    r = np.hypot(x, y)
    top = 3.0 + 4.0 * (1 - r / 4.0)
    h = np.where(r < 4.0, np.maximum(top - rng.exponential(1.0, n), 3.0), 0.0)
    points = np.zeros(n, dtype=POINT_DTYPE)
    points["x"], points["y"], points["z"] = x, y, h + rng.normal(0.0, 0.05, n)
    points["classification"] = np.where(h > 0.0, 1, 2)
    points["return_number"], points["num_returns"] = 1, 1
    return points


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flat_block_smooth_crown_rough(seed: int) -> None:
    # Synthetic part.
    block, crown = _shape("block", seed), _crown(seed)
    assert _geometry(block, block["z"], block["z"] >= 2.0)["plane_rmse"] < 0.12
    assert _geometry(crown, crown["z"], crown["z"] >= 2.0)["plane_rmse"] > 0.2


def test_fixture_rock_smooth(tmp_path: Path) -> None:
    # Real part: the known boulders' candidates are smooth. b09's candidate
    # shows no rock; NaN is no evidence.
    for b in BOULDERS:
        if b["id"] == "b09_dense_slope":
            continue
        rows = _rows(FIXTURES / b["crops"][0]["file"], tmp_path / "f.parquet")
        row = min(rows, key=lambda r: np.hypot(r["x"] - b["e"], r["y"] - b["n"]))
        assert row["plane_rmse"] < 0.12, b["id"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flat_block_level_crown_bumpy(seed: int) -> None:
    # 'Jagged top' at 1-2 m: a flat block barely departs from its neighbours' median.
    block, crown = _shape("block", seed), _crown(seed)
    assert _geometry(block, block["z"], block["z"] >= 2.0)["bump_mad"] < 0.06  # 0.03 at 3 seeds
    assert _geometry(crown, crown["z"], crown["z"] >= 2.0)["bump_mad"] > 0.12  # 0.16-0.28


def _plan(shape: str, seed: int) -> np.ndarray:
    # A 3 m flat-topped block at 10 pulses/m² over a 20 x 20 m square of ground (class 2):
    # an 8 x 8 m square, the square with a 3 m-wide round gap of ground in its middle, or a
    # plus of two 12 x 2 m bars.
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-10.0, 10.0, (2, 4000))
    square = (abs(x) < 4.0) & (abs(y) < 4.0)
    rock = {
        "square": square,
        "holed": square & (np.hypot(x, y) > 1.5),
        "plus": ((abs(x) < 6.0) & (abs(y) < 1.0)) | ((abs(x) < 1.0) & (abs(y) < 6.0)),
    }[shape]
    points = np.zeros(len(x), dtype=POINT_DTYPE)
    points["x"], points["y"] = x, y
    points["z"] = np.where(rock, 3.0, 0.0) + rng.normal(0.0, 0.05, len(x))
    points["classification"] = np.where(rock, 1, 2)
    points["return_number"], points["num_returns"] = 1, 1
    return points


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_outline_solidity_holes_and_tallness(seed: int) -> None:
    # A plus has deep notches, a gap in the middle shows ground, and an 8 m square 3 m
    # high is 3/8 as tall as it is wide: the rescue restores the wall tops
    # the floating filter drops.
    feats = {
        shape: _geometry(p, p["z"], p["z"] >= 2.0)
        for shape in ("square", "holed", "plus")
        for p in [_plan(shape, seed)]
    }
    assert feats["square"]["solidity"] > 0.9
    assert feats["plus"]["solidity"] < 0.6
    assert feats["square"]["ground_inside"] < 0.01  # a ground hit at a wall foot may touch the edge
    assert feats["holed"]["ground_inside"] > 0.1
    assert feats["holed"]["solidity"] > 0.9  # the gap is filled: a hole is not a notch
    assert abs(feats["square"]["tallness"] - 3.0 / 8.0) < 0.03
    assert feats["square"]["max_dropoff"] > 5.5  # the wall tops count


def _beside(block: np.ndarray, low: float, high: float, seed: int) -> np.ndarray:
    # The block plus first returns of two-return pulses at `low`-`high` m within 7 m of its
    # centre, their last returns on the ground (class 2) below.
    rng = np.random.default_rng(seed)
    n = 600
    r, t = rng.uniform(0.0, 7.0, n), rng.uniform(0.0, 2 * np.pi, n)
    first = np.zeros(n, dtype=POINT_DTYPE)
    first["x"], first["y"], first["z"] = r * np.cos(t), r * np.sin(t), rng.uniform(low, high, n)
    first["classification"], first["return_number"], first["num_returns"] = 1, 1, 2
    last = first.copy()
    last["z"], last["classification"], last["return_number"] = 0.0, 2, 2
    hits = np.flatnonzero(r > 4.0)  # beside the block; over it the pulse ends on the rock
    return np.concatenate([block, first, last[hits]])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_halo_counts_vegetation_beside_at_own_heights(seed: int) -> None:
    # A block standing alone has nothing beside it; ringed by vegetation at its own
    # heights it has; canopy above its top (H95) is not beside it.
    alone = _shape("block", seed)
    ringed = _beside(alone, 1.0, 3.0, seed)
    canopy = _beside(alone, 5.0, 9.0, seed)
    halo = {
        name: _geometry(p, p["z"], p["z"] >= 2.0)["halo"]
        for name, p in (("alone", alone), ("ringed", ringed), ("canopy", canopy))
    }
    assert halo["alone"] < 0.05  # 0.01-0.02: wall points the float test keeps out of P_obj
    assert halo["ringed"] > 1.0  # 3.2-3.7
    assert halo["canopy"] < 0.2  # 0.04-0.09: its ground hits float a few more wall points off


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_area_is_the_rock_seen_from_above(seed: int) -> None:
    # An 8 m square covers 64 m² (57.6-60.4: points sample its edges), a plus of two
    # 12 x 2 m bars 44 m², not its 12 x 12 m reach, and a 3 m gap of ground is left out.
    area = {
        shape: _geometry(p, p["z"], p["z"] >= 2.0)["area_m2"]
        for shape in ("square", "holed", "plus")
        for p in [_plan(shape, seed)]
    }
    assert 54.0 < area["square"] < 64.0
    assert 34.0 < area["plus"] < 44.0
    assert area["holed"] < area["square"] - 4.0


def _surface(kind: str, seed: int) -> np.ndarray:
    # 6 pulses/m² over 24 x 24 m, height noise 0.02 m: a 10 x 8 m house, walls 3 m and a
    # ridge at 5 m along its length (two 10 x 4 m roof planes, 80 m² in plan), or a domed
    # block 3 m high and 12 m across. Ground is class 2.
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-12.0, 12.0, (2, 3456))
    if kind == "house":
        h = np.where((abs(x) < 5.0) & (abs(y) < 4.0), 5.0 - 0.5 * abs(y), 0.0)
    else:
        h = np.clip(3.0 * (1 - (np.hypot(x, y) / 6.0) ** 2), 0, None)
    points = np.zeros(len(x), dtype=POINT_DTYPE)
    points["x"], points["y"], points["z"] = x, y, h + rng.normal(0.0, 0.02, len(x))
    points["classification"] = np.where(h > 0.0, 1, 2)
    points["return_number"], points["num_returns"] = 1, 1
    return points


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roof_is_planar_dome_is_not(seed: int) -> None:
    # A 2 m disc on a roof fits one plane, except across the ridge and the eaves; on a
    # dome it sags.
    house, dome = (_surface(kind, seed) for kind in ("house", "dome"))
    planar = {
        name: _geometry(p, p["z"], p["z"] >= 2.0)["planar_area"]
        for name, p in (("house", house), ("dome", dome))
    }
    assert planar["house"] > 30.0
    assert planar["dome"] < 1.0


def _shore(crown: bool, seed: int) -> np.ndarray:
    # 4 pulses/m² of ground (class 2) and rock, nothing east of x = 3.5 m (a lake: water
    # returns nothing): a 3 m rock with 45° sides that reaches the water at its foot, or, on
    # flat ground, a crown 4-6 m up leaning 2.5 m out over the water, 3 m wide (the size of
    # the area run's water-edge candidates: 5-10 m²).
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-10.0, 10.0, (2, 1600))
    h = 0.0 * x if crown else np.clip(4.0 - np.maximum(abs(x), abs(y)), 0, 3.0)
    points = np.zeros(len(x), dtype=POINT_DTYPE)
    points["x"], points["y"], points["z"] = x, y, h + rng.normal(0.0, 0.05, len(x))
    points["classification"] = np.where(h > 0.0, 1, 2)
    points["return_number"], points["num_returns"] = 1, 1
    if crown:
        n = 60
        top = np.zeros(n, dtype=POINT_DTYPE)
        top["x"], top["y"] = rng.uniform(3.5, 6.0, n), rng.uniform(-1.5, 1.5, n)
        top["z"] = rng.uniform(4.0, 6.0, n)
        top["classification"], top["return_number"], top["num_returns"] = 1, 1, 1
        points = np.concatenate([points, top])
    return points[(points["x"] < 3.5) | (points["z"] > 3.5)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_crown_over_water_hangs_rock_at_the_shore_does_not(seed: int) -> None:
    box = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
    share = {}
    for crown in (False, True):
        p = _shore(crown, seed)
        share[crown] = hanging(p, p["z"], object_points(p, p["z"], p["z"] >= 2.0), *box)
    assert share[False] < 0.05  # 0-0.01
    assert share[True] > 0.3  # 0.34-0.52: its edge facing the open water


@pytest.mark.parametrize(("name", "side", "limit"), FILTERS)
def test_each_quick_filter_drops_beyond_its_limit_only(name: str, side: str, limit: float) -> None:
    # Past the limit drops, at it does not; NaN (missing evidence) never drops.
    beyond = limit * 1.01 if side == ">" else limit * 0.99
    assert dropped({name: beyond}) == name
    assert dropped({name: limit}) == ""
    assert dropped({name: float("nan")}) == ""


def test_first_quick_filter_failed_names_the_drop() -> None:
    assert dropped({"halo": 5.0, "hanging": 1.0}) == "halo"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quick_geometry_stops_at_the_first_filter(seed: int) -> None:
    # A block ringed by vegetation at its own heights (halo 3.2-3.7) fails the halo
    # filter, so the outline and local planes are never measured; a block alone passes and
    # gets every feature, as without the filters.
    alone = _shape("block", seed)
    ringed = _beside(alone, 1.0, 3.0, seed)
    obj = object_points(ringed, ringed["z"], ringed["z"] >= 2.0)
    quick = geometry(ringed, ringed["z"], ringed["z"] >= 2.0, obj, quick=True)
    assert dropped(quick) == "halo"
    assert np.isnan(quick["solidity"]) and np.isnan(quick["plane_rmse"])
    full = geometry(ringed, ringed["z"], ringed["z"] >= 2.0, obj)
    assert not np.isnan(full["solidity"]) and not np.isnan(full["plane_rmse"])
    obj = object_points(alone, alone["z"], alone["z"] >= 2.0)
    assert geometry(alone, alone["z"], alone["z"] >= 2.0, obj, quick=True) == geometry(
        alone, alone["z"], alone["z"] >= 2.0, obj
    )


def test_fixture_rocks_pass_the_quick_filters(tmp_path: Path) -> None:
    # Real part: the 20 author-confirmed rocks of the fixture crops (b09 left out)
    # are measured in full and keep their patches.
    for b in BOULDERS:
        if b["id"] == "b09_dense_slope":
            continue
        crop = _rows(FIXTURES / b["crops"][0]["file"], tmp_path / "f.parquet")
        for e, n in [(b["e"], b["n"])] + [(r["e"], r["n"]) for r in b.get("nearby_rocks", [])]:
            row = min(crop, key=lambda r: np.hypot(r["x"] - e, r["y"] - n))
            assert np.hypot(row["x"] - e, row["y"] - n) < 3.0, b["id"]
            assert row["dropped"] == "", (b["id"], row["dropped"])
            assert (tmp_path / "patches" / f"{row['candidate_id']}.bin").exists()
