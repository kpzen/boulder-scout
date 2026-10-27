"""Synthetic LiDAR tile with known boulders, for tests.

The tile is 200 x 200 m in EPSG:3006, LAS 1.4 point format 6, like Lantmäteriet's files.
Boulder points are class 1 single returns; tree crowns are multi-return class 1
with a class 2 last return underneath; a few class 7 noise points sit far above or below ground.
"""

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS

SIZE = 200.0
ORIGIN = (497000.0, 6925000.0)  # SW corner, near the reference boulder
PULSE_DENSITY = 2.0  # pulses per m²; margin above raster_relief.MIN_PULSE_DENSITY


@dataclass(frozen=True)
class Boulder:
    kind: str  # "block", "cone" or "stone"
    x: float  # EPSG:3006 centroid
    y: float
    height: float  # top above the surrounding ground, m


# kind, offset from origin (m), height (m), base radius (m)
_BOULDERS = [
    ("block", (50.0, 50.0), 2.5, 4.0),
    ("cone", (150.0, 60.0), 4.0, 4.0),
    ("stone", (100.0, 150.0), 1.5, 1.5),
]
_BLOCK_FACE_WIDTH = 0.4  # horizontal run of the block's faces: ~80° steep


def _ground(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = x - ORIGIN[0], y - ORIGIN[1]
    return 300.0 + 0.05 * dx + 0.02 * dy + 0.3 * np.sin(dx / 25.0)


def make_synthetic_tile(path: Path, seed: int = 0) -> list[Boulder]:
    """Write a synthetic LAZ tile to `path` and return its ground-truth boulders."""
    rng = np.random.default_rng(seed)
    n = int(SIZE * SIZE * PULSE_DENSITY)
    x = ORIGIN[0] + rng.uniform(0.0, SIZE, n)
    y = ORIGIN[1] + rng.uniform(0.0, SIZE, n)
    ground = _ground(x, y)
    above = np.zeros(n)
    for kind, (ox, oy), height, radius in _BOULDERS:
        r = np.hypot(x - ORIGIN[0] - ox, y - ORIGIN[1] - oy)
        if kind == "cone":
            h = np.clip(height * (1.0 - r / radius), 0.0, None)
        else:  # flat top, steep faces over the outer _BLOCK_FACE_WIDTH; the stone is a small block
            h = np.clip(height * (radius - r) / _BLOCK_FACE_WIDTH, 0.0, height)
        above = np.maximum(above, h)
    on_rock = above > 0.0

    # Trees: kept 10 m clear of boulders so each crown is plainly vegetation.
    trees: list[tuple[float, float, float, float]] = []
    while len(trees) < 60:
        tx, ty = rng.uniform(5.0, SIZE - 5.0, 2)
        if all(np.hypot(tx - ox, ty - oy) > 10.0 for _, (ox, oy), _, _ in _BOULDERS):
            trees.append((ORIGIN[0] + tx, ORIGIN[1] + ty, rng.uniform(10, 20), rng.uniform(2, 3)))
    crown_top = np.full(n, np.nan)
    for tx, ty, th, tr in trees:
        inside = np.hypot(x - tx, y - ty) < tr
        crown_top[inside] = _ground(np.array([tx]), np.array([ty]))[0] + th
    in_crown = ~np.isnan(crown_top)

    # Last returns: ground or rock. Crown pulses get 1-2 earlier returns in the canopy.
    num_returns = np.where(in_crown, rng.integers(2, 4, n), 1)
    last_z = ground + above
    cls = np.where(on_rock, 1, 2)
    xs, ys, zs, classes = [x], [y], [last_z], [cls]
    rets, nrets = [num_returns], [num_returns]
    for k in (1, 2):
        hit = in_crown & (num_returns > k)
        depth = rng.uniform(0.0, 4.0, hit.sum()) * k
        xs.append(x[hit])
        ys.append(y[hit])
        zs.append(crown_top[hit] - depth)
        classes.append(np.ones(hit.sum(), dtype=int))
        rets.append(np.full(hit.sum(), k))
        nrets.append(num_returns[hit])

    noise = 20
    xs.append(ORIGIN[0] + rng.uniform(0.0, SIZE, noise))
    ys.append(ORIGIN[1] + rng.uniform(0.0, SIZE, noise))
    zs.append(_ground(xs[-1], ys[-1]) + rng.choice([-30.0, 80.0], noise))
    classes.append(np.full(noise, 7))
    rets.append(np.ones(noise, dtype=int))
    nrets.append(np.ones(noise, dtype=int))

    all_cls = np.concatenate(classes)
    m = len(all_cls)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([ORIGIN[0], ORIGIN[1], 0.0])
    header.scales = np.array([0.01, 0.01, 0.01])
    header.add_crs(CRS.from_epsg(3006))
    las = laspy.LasData(header)
    las.x = np.concatenate(xs)
    las.y = np.concatenate(ys)
    las.z = np.concatenate(zs) + rng.normal(0.0, 0.05, m)  # σz
    las.classification = all_cls
    las.return_number = np.concatenate(rets)
    las.number_of_returns = np.concatenate(nrets)
    las.intensity = np.where(all_cls == 1, 900, 600) + rng.integers(0, 200, m)
    las.point_source_id = rng.integers(1, 3, m)  # two overlapping swaths
    las.write(path)

    return [
        Boulder(kind, ORIGIN[0] + ox, ORIGIN[1] + oy, height)
        for kind, (ox, oy), height, _ in _BOULDERS
    ]
