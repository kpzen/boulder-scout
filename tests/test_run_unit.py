import math
from pathlib import Path

import laspy
import numpy as np
import pyarrow.parquet as pq
import pytest

from pipeline.area import Bbox
from pipeline.run_unit import main, run
from tests.real_fixtures import REFERENCE_BOULDER
from tests.real_fixtures import REFERENCE_CROP as FIXTURE
from tests.synthetic import ORIGIN, SIZE, make_synthetic_tile
from tests.test_topo_masker import POLYGON, write_gpkg

COLUMNS = [
    *("candidate_id", "tile_id", "x", "y"),
    *("max_h", "max_dropoff", "elongation_2d"),
    "plane_rmse",
    *("bump_mad", "solidity", "ground_inside", "tallness", "halo", "area_m2"),
    *("planar_area", "hanging"),
    *("single_return_ratio", "mean_intensity_norm", "ground_share"),
    "column_spread",
    *("dropped", "masked", "dist_building", "dist_water"),
]


def test_synthetic_tile_rows_and_stable_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tile = tmp_path / "synth.copc.laz"
    truth = make_synthetic_tile(tile)
    out1, out2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    main([str(tile), "--out", str(out1), "--patches", str(tmp_path / "p")])
    main([str(tile), "--out", str(out2), "--patches", str(tmp_path / "p")])

    captured = capsys.readouterr()
    assert "synth: pulse density 2.0" in captured.out
    assert "WARNING" not in captured.err
    rows = pq.read_table(out1).to_pylist()
    assert pq.read_table(out1).column_names == COLUMNS
    # One row per candidate: the block and the cone (the proposer finds nothing else).
    assert len(rows) == len([b for b in truth if b.height >= 2.0])
    assert {r["tile_id"] for r in rows} == {"synth"}
    for b in truth:
        if b.height >= 2.0:
            (row,) = [r for r in rows if np.hypot(r["x"] - b.x, r["y"] - b.y) < 2.0]
            assert b.height - 1.2 < row["max_h"] < b.height + 0.2  # H95: below a cone's tip
            assert row["dropped"] == ""  # rock passes the quick filters
    ids = [r["candidate_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert ids == pq.read_table(out2).column("candidate_id").to_pylist()


def test_real_fixture_includes_reference_boulder(tmp_path: Path) -> None:
    out = tmp_path / "c.parquet"
    main([str(FIXTURE), "--out", str(out), "--patches", str(tmp_path / "p")])

    rows = pq.read_table(out).to_pylist()
    dist = [np.hypot(r["x"] - REFERENCE_BOULDER[0], r["y"] - REFERENCE_BOULDER[1]) for r in rows]
    assert min(dist) < 5.0
    assert rows[0]["tile_id"] == FIXTURE.name.split(".")[0]


def test_sparse_tile_prints_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tile = tmp_path / "sparse.laz"
    make_synthetic_tile(tile)
    las = laspy.read(tile)
    # Seed 1: the generator draws x from seed 0 (see test_raster_relief).
    las.points = las.points[np.random.default_rng(1).random(len(las.points)) < 0.5]
    las.write(tile)

    main([str(tile), "--out", str(tmp_path / "d.parquet"), "--patches", str(tmp_path / "p")])
    captured = capsys.readouterr()
    assert "sparse: pulse density 1.0" in captured.out
    assert "WARNING sparse: pulse density" in captured.err


def test_boulder_straddling_two_tiles_gives_one_candidate(tmp_path: Path) -> None:
    whole = tmp_path / "whole.laz"
    block = make_synthetic_tile(whole)[0]  # 8 m across
    seam = block.x + 1.0  # the block straddles the edge between tiles a and b
    x0, y0 = ORIGIN
    a_box, b_box = (x0, y0, seam, y0 + SIZE), (seam, y0, x0 + SIZE, y0 + SIZE)
    # Like the real files, each spills 20 m past its nominal tile, with the same pulses in both.
    for name, side in [("a", -1.0), ("b", 1.0)]:
        las = laspy.read(whole)
        las.points = las.points[side * (np.asarray(las.x) - seam) >= -20.0]
        las.write(tmp_path / f"{name}.laz")
    a, b = tmp_path / "a.laz", tmp_path / "b.laz"

    def rows(
        tile: Path, bbox: Bbox | None = None, neighbours: list[tuple[Path, Bbox]] | None = None
    ) -> list[dict[str, float]]:
        out = tmp_path / f"{tile.stem}_{bbox is None}.parquet"
        run(tile, out, tmp_path / "p", bbox, neighbours or [])
        result: list[dict[str, float]] = pq.read_table(out).to_pylist()
        return result

    def at_block(rs: list[dict[str, float]]) -> int:
        return sum(np.hypot(r["x"] - block.x, r["y"] - block.y) < 2.0 for r in rs)

    # Each file read whole sees the block: two candidates for one boulder.
    assert at_block(rows(a)) == at_block(rows(b)) == 1
    units = rows(a, a_box, [(b, b_box)]) + rows(b, b_box, [(a, a_box)])
    assert at_block(units) == 1
    # Nothing else lost or duplicated: the units give the unsplit cloud's candidates.
    reference = rows(whole)
    assert len(units) == len(reference)
    for r in reference:
        assert min(np.hypot(u["x"] - r["x"], u["y"] - r["y"]) for u in units) < 0.5


def test_edge_rings_give_the_same_unit(tmp_path: Path) -> None:
    # A neighbour's edge strip comes from its cached ring, read once from its whole file.
    whole = tmp_path / "whole.laz"
    make_synthetic_tile(whole)
    x0, y0 = ORIGIN
    seam = x0 + SIZE / 2
    a_box, b_box = (x0, y0, seam, y0 + SIZE), (seam, y0, x0 + SIZE, y0 + SIZE)
    for name, side in [("a", -1.0), ("b", 1.0)]:
        las = laspy.read(whole)
        las.points = las.points[side * (np.asarray(las.x) - seam) >= -20.0]
        las.write(tmp_path / f"{name}.laz")
    a, b = tmp_path / "a.laz", tmp_path / "b.laz"
    rings = tmp_path / "rings"

    run(b, tmp_path / "plain.parquet", tmp_path / "p", b_box, [(a, a_box)])
    run(a, tmp_path / "a.parquet", tmp_path / "p", a_box, [(b, b_box)], rings=rings)
    assert sorted(p.name for p in rings.iterdir()) == ["a.npy", "b.npy"]
    a.rename(tmp_path / "gone.laz")  # unit b must now take a's strip from the ring
    run(b, tmp_path / "ringed.parquet", tmp_path / "p", b_box, [(a, a_box)], rings=rings)
    plain = pq.read_table(tmp_path / "plain.parquet")
    assert plain.num_rows and pq.read_table(tmp_path / "ringed.parquet").equals(plain)


def test_masked_candidates_are_not_measured(tmp_path: Path) -> None:
    # A lake over the block masks it; the cone, 100 m off, is measured as before.
    tile = tmp_path / "synth.copc.laz"
    block, cone, _ = make_synthetic_tile(tile)
    topo = tmp_path / "topo"
    topo.mkdir()
    write_gpkg(topo / "byggnadsverk_ln22.gpkg", {"byggnad": [], "byggnadspunkt": []})
    write_gpkg(topo / "kommunikation_ln22.gpkg", {"vaglinje": [], "ralstrafik": []})
    x0, y0, x1, y1 = block.x - 10, block.y - 10, block.x + 10, block.y + 10
    lake = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    write_gpkg(topo / "mark_ln22.gpkg", {"mark": [(POLYGON, lake, "Sjö")]})
    patches = tmp_path / "p"
    main([str(tile), "--out", str(tmp_path / "plain.parquet"), "--patches", str(tmp_path / "q")])
    main(
        [
            str(tile),
            "--out",
            str(tmp_path / "m.parquet"),
            "--patches",
            str(patches),
            "--topo",
            str(topo),
        ]
    )

    rows = pq.read_table(tmp_path / "m.parquet").to_pylist()
    plain = pq.read_table(tmp_path / "plain.parquet").to_pylist()
    (b,) = [r for r in rows if np.hypot(r["x"] - block.x, r["y"] - block.y) < 2.0]
    (c,) = [r for r in rows if np.hypot(r["x"] - cone.x, r["y"] - cone.y) < 2.0]
    assert b["masked"] and b["dist_water"] == 0.0 and b["dropped"] == ""
    assert all(math.isnan(b[k]) for k in COLUMNS[4:-4])
    assert not c["masked"] and c["dist_water"] == math.inf and c["dist_building"] == math.inf
    (c_plain,) = [r for r in plain if r["candidate_id"] == c["candidate_id"]]
    assert c == pytest.approx(c_plain, nan_ok=True)
    assert sorted(p.stem for p in patches.iterdir()) == [c["candidate_id"]]


def test_a_unit_without_candidates_writes_a_typed_table(tmp_path: Path) -> None:
    tile = tmp_path / "synth.copc.laz"
    make_synthetic_tile(tile)
    x0, y0 = ORIGIN
    run(tile, tmp_path / "e.parquet", tmp_path / "p", (x0, y0, x0 + 30, y0 + 30))
    run(tile, tmp_path / "f.parquet", tmp_path / "p")
    empty = pq.read_table(tmp_path / "e.parquet")
    assert empty.num_rows == 0
    assert empty.schema == pq.read_table(tmp_path / "f.parquet").schema
