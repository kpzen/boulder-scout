from pathlib import Path

import laspy
import numpy as np

from tests.synthetic import SIZE, make_synthetic_tile


def test_synthetic_tile_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tile.laz"
    truth = make_synthetic_tile(path)
    las = laspy.read(path)

    assert las.header.parse_crs().to_epsg() == 3006
    assert 1.0 <= len(las.points) / SIZE**2 <= 2.5
    cls = np.asarray(las.classification)
    assert set(np.unique(cls)) == {1, 2, 7}
    assert (np.asarray(las.number_of_returns) > 1).any()  # trees

    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    ground = cls == 2
    rock = (cls == 1) & (np.asarray(las.number_of_returns) == 1)
    assert {b.kind for b in truth} == {"block", "cone", "stone"}
    for b in truth:
        r = np.hypot(x - b.x, y - b.y)
        on = rock & (r < 6.0)
        base = np.median(z[ground & (r > 6.0) & (r < 10.0)])
        assert np.hypot(x[on].mean() - b.x, y[on].mean() - b.y) < 0.5
        assert abs(z[on].max() - base - b.height) < 0.5  # cone apex is rarely hit
