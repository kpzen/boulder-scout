from pathlib import Path

import laspy
import numpy as np
import pytest

from pipeline.ingest.laz_reader import POINT_DTYPE, read_tile
from tests.real_fixtures import REFERENCE_CROP as FIXTURE
from tests.synthetic import make_synthetic_tile


@pytest.fixture(params=["synthetic", "fixture"])
def tile(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    if request.param == "fixture":
        return FIXTURE
    path = tmp_path / "tile.laz"
    make_synthetic_tile(path)
    las = laspy.read(path)  # the generator only makes class 7 noise; add the other dropped classes
    cls = np.asarray(las.classification).copy()
    cls[:30] = np.repeat([9, 17, 18], 10)
    las.classification = cls
    las.write(path)
    return path


def test_read_tile_keeps_classes_1_and_2(tile: Path) -> None:
    las = laspy.read(tile)
    cls = np.asarray(las.classification)
    assert (cls == 7).any()  # something to drop
    if tile != FIXTURE:
        assert {9, 17, 18} <= set(np.unique(cls))
    keep = (cls == 1) | (cls == 2)

    points = read_tile(tile)

    assert points.dtype == POINT_DTYPE
    assert len(points) == keep.sum()  # every class 1/2 point kept, nothing else
    for name, dim in [
        ("x", "x"),
        ("y", "y"),
        ("z", "z"),
        ("intensity", "intensity"),
        ("classification", "classification"),
        ("return_number", "return_number"),
        ("num_returns", "number_of_returns"),
        ("point_source_id", "point_source_id"),
    ]:
        np.testing.assert_array_equal(points[name], np.asarray(las[dim])[keep])


def test_read_tile_normalises_intensity_per_swath(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 4000
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 100, n)
    las.y = rng.uniform(0, 100, n)
    las.z = rng.uniform(0, 1, n)
    swath = rng.integers(1, 3, n)
    rock = rng.random(n) < 0.2  # single-return class 1 rock, the rest single-return ground
    las.classification = np.where(rock, 1, 2)
    las.return_number = np.ones(n, dtype=int)
    las.number_of_returns = np.ones(n, dtype=int)
    raw = np.where(rock, 900.0, 600.0) + rng.normal(0, 60, n)
    las.intensity = np.round(np.where(swath == 2, 1.4 * raw, raw))  # swath 2 is 40 % brighter
    las.point_source_id = swath
    path = tmp_path / "two_swaths.laz"
    las.write(path)

    points = read_tile(path)

    ref = points["classification"] == 2
    one, two = points["point_source_id"] == 1, points["point_source_id"] == 2
    raw_one, raw_two = (np.median(points["intensity"][ref & s]) for s in (one, two))
    assert raw_two / raw_one == pytest.approx(1.4, abs=0.02)
    med_one, med_two = (np.median(points["intensity_norm"][ref & s]) for s in (one, two))
    assert abs(med_one - med_two) < 0.05
    # Rock was not in the reference: its 40 % offset is gone too, up to sampling noise in the IQR.
    rock_one, rock_two = (np.median(points["intensity_norm"][~ref & s]) for s in (one, two))
    assert rock_two == pytest.approx(rock_one, rel=0.05)
    assert not np.isnan(points["intensity_norm"]).any()
