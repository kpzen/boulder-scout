import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.ingest.dtm_reader import Ground
from pipeline.ingest.laz_reader import POINT_DTYPE, read_tile
from pipeline.proposer import raster_relief
from pipeline.proposer.raster_relief import LowDensityWarning, propose
from tests.real_fixtures import BOULDERS, FIXTURES, NEGATIVES, REFERENCE_BOULDER
from tests.real_fixtures import REFERENCE_CROP as FIXTURE
from tests.synthetic import PULSE_DENSITY, make_synthetic_tile


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_boulders_found_trees_and_stone_not(tmp_path: Path, seed: int) -> None:
    path = tmp_path / "tile.laz"
    truth = make_synthetic_tile(path, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowDensityWarning)
        proposal = propose(read_tile(path))

    assert abs(proposal.pulse_density - PULSE_DENSITY) < 0.05  # measured, not just thresholded
    tall = [b for b in truth if b.height >= 2.0]
    assert {b.kind for b in tall} == {"block", "cone"}
    for b in tall:
        near = [c for c in proposal.candidates if np.hypot(c.x - b.x, c.y - b.y) < 2.0]
        assert len(near) == 1, (b.kind, near)
    # Nothing else: no candidate for the 1.5 m stone or any of the 60 trees.
    assert len(proposal.candidates) == len(tall)


def test_reference_boulder_found_in_real_fixture() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowDensityWarning)
        proposal = propose(read_tile(FIXTURE))

    dist = [
        np.hypot(c.x - REFERENCE_BOULDER[0], c.y - REFERENCE_BOULDER[1])
        for c in proposal.candidates
    ]
    assert min(dist) < 5.0


@pytest.mark.parametrize("boulder", BOULDERS, ids=[b["id"] for b in BOULDERS])
def test_fixture_boulder_found_without_tree(boulder: dict[str, Any]) -> None:
    # Every fixture boulder is a candidate
    # whose relief is not inflated by a tree crown (the tallest boulder is ~8 m). b09's matched
    # candidate is unconfirmed and the canopy gate drops it.
    if boulder["id"] == "b09_dense_slope":
        pytest.skip("the canopy gate drops b09")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowDensityWarning)
        proposal = propose(read_tile(FIXTURES / boulder["crops"][0]["file"]))

    rows, cols = np.indices(proposal.labels.shape)
    dist = np.hypot(
        proposal.x0 + (cols + 0.5) * 0.5 - boulder["e"],
        proposal.y0 + (rows + 0.5) * 0.5 - boulder["n"],
    )
    radius = 5.0 if boulder["location_uncertain"] else 2.0  # b09: phone GPS under canopy
    labels = np.unique(proposal.labels[(dist <= radius) & (proposal.labels > 0)])
    assert len(labels) > 0
    assert all(proposal.candidates[k - 1].max_h <= 8.5 for k in labels)


def test_tall_and_sub_crown_boulders_found_lone_tree_not() -> None:
    # 60 x 60 m of flat ground at 2 pulses/m²: an 11 m block (top above HEIGHT_CAP) in the
    # open (5 m radius: narrower reads too tall for the canopy gate), a 5 m block under a
    # dense 18 m crown and a lone tree, both 4 m in radius.
    rng = np.random.default_rng(0)
    n = 60 * 60 * 2
    x, y = rng.uniform(0.0, 60.0, (2, n))
    open_block = np.hypot(x - 15.0, y - 15.0) < 5.0
    crown_block = np.hypot(x - 45.0, y - 15.0) < 4.0
    tree = np.hypot(x - 30.0, y - 45.0) < 4.0
    crowned = crown_block | tree
    num_returns = np.where(crowned, 3, 1)
    # Lone tree: half the pulses end inside the crown, scattered over 10-18 m.
    stuck = tree & (rng.random(n) < 0.5)
    last_z = np.select([open_block, crown_block, stuck], [11.0, 5.0, rng.uniform(10.0, 18.0, n)])
    ret_z = [18.0 - rng.uniform(0.0, 4.0, n), 18.0 - rng.uniform(4.0, 8.0, n)]
    on_rock = open_block | crown_block

    c = int(crowned.sum())
    m = n + 2 * c
    points = np.zeros(m, dtype=POINT_DTYPE)
    points["x"] = np.concatenate([x, x[crowned], x[crowned]]) + 500000.0
    points["y"] = np.concatenate([y, y[crowned], y[crowned]]) + 6900000.0
    points["z"] = np.concatenate([last_z, ret_z[0][crowned], ret_z[1][crowned]])
    points["z"] += rng.normal(0.0, 0.05, m)
    points["classification"] = np.concatenate([np.where(on_rock | stuck, 1, 2), np.ones(m - n)])
    points["num_returns"] = np.concatenate([num_returns, np.full(m - n, 3)])
    points["return_number"] = np.concatenate([num_returns, np.ones(c), np.full(c, 2)])
    proposal = propose(points)

    open_c, crown_c = sorted(proposal.candidates, key=lambda c: c.x)
    assert np.hypot(open_c.x - 500015.0, open_c.y - 6900015.0) < 1.0
    assert np.hypot(crown_c.x - 500045.0, crown_c.y - 6900015.0) < 1.0
    assert open_c.max_h > 9.5 and abs(crown_c.max_h - 5.0) < 0.5


def test_canopy_gate_drops_single_return_canopy_keeps_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 60 x 60 m of flat ground at 2 pulses/m²: a 3 m block and an 8 x 8 m canopy layer at 6 m
    # that half the pulses pass through to the ground, every return single.
    rng = np.random.default_rng(0)
    n = 60 * 60 * 2
    x, y = rng.uniform(0.0, 60.0, (2, n))
    block = np.hypot(x - 15.0, y - 30.0) < 4.0
    canopy = (np.abs(x - 45.0) < 4.0) & (np.abs(y - 30.0) < 4.0) & (rng.random(n) < 0.5)
    points = np.zeros(n, dtype=POINT_DTYPE)
    points["x"], points["y"] = x + 500000.0, y + 6900000.0
    points["z"] = np.select([block, canopy], [3.0, 6.0]) + rng.normal(0.0, 0.05, n)
    points["classification"] = np.where(block | canopy, 1, 2)
    points["num_returns"] = points["return_number"] = 1

    def found() -> list[tuple[int, int]]:
        return sorted(
            (round(c.x - 500000.0), round(c.y - 6900000.0)) for c in propose(points).candidates
        )

    assert found() == [(15, 30)]
    monkeypatch.setattr(raster_relief, "MAX_GROUND_SHARE", 1.0)  # gate off: the canopy is proposed
    assert found() == [(15, 30), (45, 30)]


def test_canopy_gate_on_real_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    # All 20 author-confirmed rocks (the known
    # boulders but b09, plus their nearby_rocks) stay candidates with every gate on; ground share
    # and shape drop at least half the negative squares' candidates, column spread more.
    def candidates(site: dict[str, Any]) -> list[raster_relief.Candidate]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowDensityWarning)
            return propose(read_tile(FIXTURES / site["crops"][0]["file"])).candidates

    rocks = 0
    for b in BOULDERS:
        if b["id"] == "b09_dense_slope":
            continue
        cands = candidates(b)
        for e, n in [(b["e"], b["n"])] + [(r["e"], r["n"]) for r in b.get("nearby_rocks", [])]:
            assert min(np.hypot(c.x - e, c.y - n) for c in cands) < 3.0, (b["id"], e, n)
            rocks += 1
    gated = sum(len(candidates(s)) for s in NEGATIVES)
    monkeypatch.setattr(raster_relief, "MAX_COLUMN_SPREAD", np.inf)
    without_spread = sum(len(candidates(s)) for s in NEGATIVES)
    monkeypatch.setattr(raster_relief, "MAX_GROUND_SHARE", np.inf)
    monkeypatch.setattr(raster_relief, "MAX_SHAPE_RATIO", np.inf)
    ungated = sum(len(candidates(s)) for s in NEGATIVES)
    print(f"negatives: {ungated} ungated, {without_spread} after shape, {gated} after spread")

    assert rocks == 20
    assert without_spread <= 0.5 * ungated
    # The column-spread gate drops further canopy, rocks all stay.
    assert gated < without_spread


def test_sparse_tile_warns(tmp_path: Path) -> None:
    path = tmp_path / "tile.laz"
    make_synthetic_tile(path)
    points = read_tile(path)
    # Seed 1: the generator draws x from seed 0, so seed 0 here would cut the tile in half.
    pulse = np.random.default_rng(1).random(len(points)) < 0.5  # thin to ~1 pulse/m²

    with pytest.warns(LowDensityWarning, match="pulse density"):
        proposal = propose(points[pulse])
    assert proposal.pulse_density < 1.5


def test_dtm_gap_fill_is_linear_across_blocks_and_nearest_beyond() -> None:
    # 800 px spans two DTM blocks each way; a 100 px void straddles the block corner at 500 px.
    # Linear interpolation reproduces a plane exactly, so blocking must not change any cell.
    rng = np.random.default_rng(0)
    size = 800
    row, col = np.mgrid[:size, :size]
    known = rng.random((size, size)) < 0.3
    known[[0, -1], :] = known[:, [0, -1]] = True
    known[450:550, 450:550] = False

    def dtm_of(known: np.ndarray) -> np.ndarray:
        r, c = np.nonzero(known)
        points = np.zeros(len(r), dtype=POINT_DTYPE)
        points["x"] = 500000.0 + (c + 0.5) * raster_relief.CELL
        points["y"] = 6900000.0 + (r + 0.5) * raster_relief.CELL
        points["z"] = 300.0 + 0.02 * c - 0.01 * r
        points["classification"] = 2
        points["return_number"] = points["num_returns"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowDensityWarning)
            return propose(points).dtm

    np.testing.assert_allclose(dtm_of(known), 300.0 + 0.02 * col - 0.01 * row, atol=1e-6)
    # East of column 600 only the southmost row has ground, which keeps the grid 800 px wide; the
    # northern blocks' hulls stop at column 599, so the nearest cell fills the rest.
    dtm = dtm_of(known & ((col < 600) | (row == 0)))
    assert np.isfinite(dtm).all()
    # Columns 600-649 are nearer column 599 (< 51 px) than the southern blocks' fill (> 100 px).
    assert np.abs(dtm[600:, 600:650] - dtm[600:, 599:600]).max() < 0.05  # a few 0.02 m steps


def test_ground_model_fills_only_empty_cells() -> None:
    # Cells with ground keep their mean; empty ones take the model's bilinear height.
    rng = np.random.default_rng(0)
    size = 400  # cells of 0.5 m
    row, col = np.mgrid[:size, :size]
    known = rng.random((size, size)) < 0.3
    known[[0, -1], :] = known[:, [0, -1]] = True
    known[150:250, 150:250] = False

    def plane(x: np.ndarray, y: np.ndarray) -> np.ndarray:  # 300 + 0.02 col - 0.01 row
        cell = raster_relief.CELL
        return 300.0 + 0.02 * ((x - 500000.0) / cell - 0.5) - 0.01 * ((y - 6900000.0) / cell - 0.5)

    r, c = np.nonzero(known)
    points = np.zeros(len(r), dtype=POINT_DTYPE)
    points["x"] = 500000.0 + (c + 0.5) * raster_relief.CELL
    points["y"] = 6900000.0 + (r + 0.5) * raster_relief.CELL
    points["z"] = plane(points["x"], points["y"])
    points["classification"] = 2
    points["return_number"] = points["num_returns"] = 1
    # The model: 0.5 m above the plane, 1 m cells from (499999, 6900201) at the north-west.
    my, mx = np.mgrid[:203, :203]
    values = plane(499999.5 + mx, 6900200.5 - my) + 0.5
    model = Ground(values.astype(np.float32), 499999.0, 6900201.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowDensityWarning)
        dtm = propose(points, model).dtm
    expected = 300.0 + 0.02 * col - 0.01 * row + np.where(known, 0.0, 0.5)
    np.testing.assert_allclose(dtm, expected, atol=1e-4)  # float32 model heights
