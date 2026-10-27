from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from pipeline.extractors import geometry
from pipeline.extractors.geometry import object_points
from pipeline.extractors.radiometry import radiometry
from pipeline.ingest.laz_reader import POINT_DTYPE
from pipeline.proposer import raster_relief
from pipeline.run_unit import main
from tests.real_fixtures import BOULDERS, FIXTURES, NEGATIVES
from tests.synthetic import make_synthetic_tile


def _radiometry(points: np.ndarray, inside: np.ndarray) -> dict[str, float]:
    return radiometry(points, points["z"], inside, object_points(points, points["z"], inside))


def _pulses(n: int, rng: np.random.Generator) -> np.ndarray:
    # n pulses at 4 pulses/m² over a 20 x 20 m square around the object, flat ground at z = 0.
    points = np.zeros(n, dtype=POINT_DTYPE)
    points["x"], points["y"] = rng.uniform(-10.0, 10.0, (2, n))
    points["classification"], points["return_number"], points["num_returns"] = 2, 1, 1
    return points


def _first_returns(points: np.ndarray, hit: np.ndarray, z: np.ndarray) -> np.ndarray:
    # Earlier returns at height z for the pulses `hit`, which become 2-return pulses.
    points["num_returns"][hit] = 2
    points["return_number"][hit] = 2
    first = points[hit].copy()
    first["z"], first["classification"], first["return_number"] = z, 1, 1
    return np.concatenate([points, first])


def _tree(seed: int) -> np.ndarray:
    # A crown 3 m in radius from 5 to 12 m. A quarter of its pulses end in a single return in
    # the foliage; the rest split, and 60 % of those end in the lower crown, the rest on ground.
    rng = np.random.default_rng(seed)
    points = _pulses(1600, rng)
    crown = np.hypot(points["x"], points["y"]) < 3.0
    single = crown & (rng.random(len(points)) < 0.25)
    split = crown & ~single
    in_foliage = split & (rng.random(len(points)) < 0.6)
    points["z"][single] = rng.uniform(8.0, 12.0, single.sum())
    points["z"][in_foliage] = rng.uniform(5.0, 8.0, in_foliage.sum())
    points["classification"][single | in_foliage] = 1
    return _first_returns(points, split, rng.uniform(8.0, 12.0, len(points))[split])


def _block(seed: int) -> np.ndarray:
    # A 3 m block 4 m in radius.
    rng = np.random.default_rng(seed)
    points = _pulses(1600, rng)
    rock = np.hypot(points["x"], points["y"]) < 4.0
    points["z"][rock] = 3.0 + rng.normal(0.0, 0.05, rock.sum())
    points["classification"][rock] = 1
    return points


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rock_single_returns_tree_split(seed: int) -> None:
    # single_return_ratio. `inside` is the proposer's 2 m outline.
    block, tree = _block(seed), _tree(seed)
    last = tree["return_number"] == tree["num_returns"]
    assert _radiometry(block, block["z"] >= 2.0)["single_return_ratio"] > 0.8
    assert _radiometry(tree, last & (tree["z"] >= 2.0))["single_return_ratio"] < 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ground_share_block_and_canopy(seed: int) -> None:
    # Synthetic part: a canopy layer at 8-10 m, 4 m in radius, where 70 % of
    # pulses end in a single return in the foliage and the rest on open ground.
    block = _block(seed)
    assert _radiometry(block, block["z"] >= 2.0)["ground_share"] < 0.05
    rng = np.random.default_rng(seed)
    layer = _pulses(1600, rng)
    disc = np.hypot(layer["x"], layer["y"]) < 4.0
    hit = disc & (rng.random(len(layer)) < 0.7)
    layer["z"][hit] = rng.uniform(8.0, 10.0, hit.sum())
    layer["classification"][hit] = 1
    assert _radiometry(layer, disc)["ground_share"] > 0.2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_spread_block_and_canopy(seed: int) -> None:
    # Synthetic part: one surface per spot on a flat-topped block; a
    # single-return canopy layer from 4 to 12 m over open ground has hits at many heights.
    block = _block(seed)
    assert _radiometry(block, block["z"] >= 2.0)["column_spread"] < 0.15
    rng = np.random.default_rng(seed)
    layer = _pulses(1600, rng)
    disc = np.hypot(layer["x"], layer["y"]) < 4.0
    hit = disc & (rng.random(len(layer)) < 0.7)
    layer["z"][hit] = rng.uniform(4.0, 12.0, hit.sum())
    layer["classification"][hit] = 1
    assert _radiometry(layer, disc)["column_spread"] > 0.5


def test_fixture_column_spread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Real part: the 20 author-confirmed rocks (b09 left out) against
    # the negative-square candidates the proposer keeps. The quick filters are off: they drop
    # most canopy candidates before radiometry.
    monkeypatch.setattr(geometry, "FILTERS", ())

    def rows(file: str) -> list[dict[str, float]]:
        main([str(FIXTURES / file), "--out", str(tmp_path / "f.parquet"),
              "--patches", str(tmp_path / "p")])  # fmt: skip
        table: list[dict[str, float]] = pq.read_table(tmp_path / "f.parquet").to_pylist()
        return table

    rocks = []
    for b in BOULDERS:
        if b["id"] == "b09_dense_slope":
            continue
        crop = rows(b["crops"][0]["file"])
        for e, n in [(b["e"], b["n"])] + [(r["e"], r["n"]) for r in b.get("nearby_rocks", [])]:
            row = min(crop, key=lambda r: np.hypot(r["x"] - e, r["y"] - n))
            assert np.hypot(row["x"] - e, row["y"] - n) < 3.0, b["id"]
            rocks.append(row["column_spread"])
    neg = [r["column_spread"] for b in NEGATIVES for r in rows(b["crops"][0]["file"])]
    print(f"column_spread: rocks max {max(rocks):.2f}, negatives median {np.nanmedian(neg):.2f}")
    assert len(rocks) == 20 and max(rocks) < 0.15
    # column_spread counts up to the cleaned P_obj's H95 + 1 m, so less of the crown.
    assert np.nanmedian(neg) > 0.3


def test_mean_intensity_norm_ignores_nan() -> None:
    block = _block(0)
    top = block["z"] >= 2.0
    block["intensity_norm"] = np.where(top, 1.5, -1.0)
    block["intensity_norm"][np.flatnonzero(top)[:10]] = np.nan  # a swath without reference
    assert _radiometry(block, top)["mean_intensity_norm"] == pytest.approx(1.5)
    block["intensity_norm"] = np.nan
    assert np.isnan(_radiometry(block, top)["mean_intensity_norm"])


def test_no_object_points_gives_nan_but_ground_share() -> None:
    # Dropping floating points can leave a canopy candidate without P_obj; ground_share needs none.
    block = _block(0)
    feats = radiometry(
        block, np.full(len(block), 0.5), block["z"] >= 2.0, np.zeros(len(block), bool)
    )
    assert feats.pop("ground_share") == 0.0
    assert all(np.isnan(v) for v in feats.values())


def test_synthetic_tile_rocks(tmp_path: Path) -> None:
    # End to end: the synthetic tile's boulders are open single-return rock.
    truth = make_synthetic_tile(tmp_path / "t.laz", seed=0)
    main(
        [
            str(tmp_path / "t.laz"),
            "--out",
            str(tmp_path / "t.parquet"),
            "--patches",
            str(tmp_path / "p"),
        ]
    )
    rows = pq.read_table(tmp_path / "t.parquet").to_pylist()
    for b in truth:
        if b.height < 2.0:
            continue
        (row,) = [r for r in rows if np.hypot(r["x"] - b.x, r["y"] - b.y) < 2.0]
        assert row["single_return_ratio"] > 0.8
        assert row["ground_share"] < 0.05
        if b.kind == "block":  # the cone's steep sides span ~0.5 m of height per 0.5 m cell
            assert row["column_spread"] < 0.15
        assert not np.isnan(row["mean_intensity_norm"])


def test_fixture_ground_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Real part. b09 stays out until its candidate is confirmed. The
    # canopy gate is off: it drops candidates on this same measure, and so are the quick
    # filters, which drop most canopy candidates before radiometry.
    monkeypatch.setattr(raster_relief, "MAX_GROUND_SHARE", np.inf)
    monkeypatch.setattr(geometry, "FILTERS", ())

    def rows(file: str) -> list[dict[str, float]]:
        main(
            [
                str(FIXTURES / file),
                "--out",
                str(tmp_path / "f.parquet"),
                "--patches",
                str(tmp_path / "p"),
            ]
        )
        table: list[dict[str, float]] = pq.read_table(tmp_path / "f.parquet").to_pylist()
        return table

    for b in BOULDERS:
        if b["id"] == "b09_dense_slope":
            continue
        row = min(
            rows(b["crops"][0]["file"]), key=lambda r: np.hypot(r["x"] - b["e"], r["y"] - b["n"])
        )
        assert np.hypot(row["x"] - b["e"], row["y"] - b["n"]) < 3.0, b["id"]
        assert row["ground_share"] < 0.25, b["id"]
    neg = [r["ground_share"] for b in NEGATIVES for r in rows(b["crops"][0]["file"])]
    assert np.median(neg) > 0.2
