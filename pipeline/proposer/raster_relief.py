"""Propose boulder candidates from a local relief raster."""

import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from pipeline.ingest.dtm_reader import Ground

CELL = 0.5  # m
# DTM gap fill in blocks; each block triangulates its ground cells plus a DTM_MARGIN overlap,
# so a cell by a block edge still has ground on every side. A 100 m fixture crop is one block.
DTM_BLOCK = 500  # px (250 m)
DTM_MARGIN = 100  # px (50 m)
# Blocks triangulate in worker processes: Qhull holds the GIL. Each worker adds ~0.3 GB.
DTM_WORKERS = 6
SMOOTH_SIGMA = 10  # px (5.0 m)
MIN_RELIEF, MAX_RELIEF = 2.0, 15.0  # m
MIN_AREA_PX = 16  # 4.0 m²
# Below this boulders under canopy start to break into pieces under MIN_AREA_PX and vanish.
# Fixture crops thinned at random: all 9 boulders found at 1.4, the dense-canopy one
# lost from 1.2.
MIN_PULSE_DENSITY = 1.4  # first returns per m²
# Solid_DSM: last returns only, as rock is opaque. Points above HEIGHT_CAP are crowns
# unless a dense, flat patch of them supports them (a tall boulder top).
HEIGHT_CAP = 8.0  # m above the DTM
ELEVATED = 1.2  # m above the DTM; elevated cells take a low percentile, not the max
ELEVATED_PERCENTILE = 25
SUPPORT_PX = 5  # 2.5 m window for counting above-cap neighbours
SUPPORT_MIN = 6  # points
SUPPORT_MAX_STD = 0.5  # m
DENSITY_BLOCK = 10  # px (5 m): density is measured over blocks holding any point, not the bbox
# Canopy gate: rock is opaque and seldom much taller than wide; canopy lets pulses reach
# the ground and forms small, tall blobs. Fixture rocks reach 0.21 and 1.42.
GROUND_Z = 0.5  # m: last returns below this z_norm reached the ground
MAX_GROUND_SHARE = 0.35
MAX_SHAPE_RATIO = 1.8  # median relief / sqrt(area)
# Column spread: per cell, the z_norm std of non-ground returns from GROUND_Z up to the
# component's top + SPREAD_GAP; rock is one surface per spot, canopy has hits at many heights.
SPREAD_GAP = 1.0  # m
MAX_COLUMN_SPREAD = 0.25  # m, median over cells with >= 2 points (fixture rocks <= 0.14)
# The median needs this many such cells; on fewer it is no evidence (NaN), not a verdict.
MIN_SPREAD_CELLS = 3


class LowDensityWarning(UserWarning):
    pass


@dataclass(frozen=True)
class Candidate:
    x: float  # EPSG:3006 centroid
    y: float
    max_h: float  # highest relief in the component, m; the unit table's max_h is H95


@dataclass(frozen=True)
class Proposal:
    labels: np.ndarray  # int raster, row 0 at y0 (south), column 0 at x0; 0 = background
    dtm: np.ndarray  # gap-filled ground height per cell, same grid as labels
    x0: float  # grid SW corner, EPSG:3006
    y0: float
    candidates: list[Candidate]  # candidates[i] is label i + 1
    pulse_density: float  # first returns per m² over occupied 5 m blocks


def _fill_block(window: np.ndarray, offset: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a DTM block's empty cells from the ground cells in its window.

    The block starts at `offset` in the window. Returns the empty cells' block indices and
    their values; none filled when the window's ground cells span no triangle.
    """
    block = window[offset[0] : offset[0] + DTM_BLOCK, offset[1] : offset[1] + DTM_BLOCK]
    empty = np.argwhere(np.isnan(block))
    known = np.argwhere(~np.isnan(window))
    if not len(empty) or len(known) < 3:
        return empty[:0], np.empty(0)
    try:
        interp = LinearNDInterpolator(known, window[~np.isnan(window)])
    except QhullError:  # collinear ground cells at a void's edge: left to nearest
        return empty[:0], np.empty(0)
    return empty, interp(empty + offset)


def propose(points: np.ndarray, model: Ground | None = None) -> Proposal:
    """Label connected relief components in a tile of read_tile points.

    With `model` (Lantmäteriet's ground model over the points' extent), the DTM's empty
    cells take its heights instead of a triangulation of the tile's own ground cells.

    Warns with LowDensityWarning when the tile is too sparse for reliable proposals.
    """
    x0 = np.floor(points["x"].min() / CELL) * CELL
    y0 = np.floor(points["y"].min() / CELL) * CELL
    # float32 heights, int32 cell indices and early `del`s below keep a dense unit's memory
    # down.
    col = ((points["x"] - x0) / CELL).astype(np.int32)
    row = ((points["y"] - y0) / CELL).astype(np.int32)
    shape = (int(row.max()) + 1, int(col.max()) + 1)
    size = shape[0] * shape[1]
    flat = row * np.int32(shape[1]) + col
    ground = points["classification"] == 2

    # Whole blocks only: blocks cut by the grid edge would read as sparse.
    nb = (shape[0] // DENSITY_BLOCK, shape[1] // DENSITY_BLOCK)
    first = (points["return_number"] == 1) & (row < nb[0] * DENSITY_BLOCK)
    first &= col < nb[1] * DENSITY_BLOCK
    per_block = np.bincount(
        (row[first] // DENSITY_BLOCK) * nb[1] + col[first] // DENSITY_BLOCK,
        minlength=nb[0] * nb[1],
    )
    del row, col
    occupied = per_block[per_block > 0]
    pulse_density = occupied.sum() / (max(len(occupied), 1) * (DENSITY_BLOCK * CELL) ** 2)
    if pulse_density < MIN_PULSE_DENSITY:
        warnings.warn(
            f"pulse density {pulse_density:.2f}/m² is below {MIN_PULSE_DENSITY}/m²; "
            "boulders may be missed",
            LowDensityWarning,
            stacklevel=2,
        )

    # 1. DTM: mean ground height per cell; empty cells from the ground model, bilinear at
    # the cell centres, or else linearly interpolated from the ground cells block by block (one
    # triangulation of a real tile's ~8 M ground cells needs ~16 GB); then the nearest ground
    # cell where neither reaches.
    n = np.bincount(flat[ground], minlength=size)
    s = np.bincount(flat[ground], points["z"][ground], minlength=size)
    cell_mean = np.full(size, np.nan, np.float32)
    cell_mean[n > 0] = s[n > 0] / n[n > 0]
    cell_mean = cell_mean.reshape(shape)
    del n, s
    dtm = cell_mean.copy()
    if model is not None:
        # Separable bilinear: model rows (north first), then columns, NaN beside unknown cells.
        v = model.values.astype(np.float64)
        r = (model.y1 - y0 - (np.arange(shape[0]) + 0.5) * CELL) / model.cell - 0.5
        c = (x0 - model.x0 + (np.arange(shape[1]) + 0.5) * CELL) / model.cell - 0.5
        r = np.clip(r, 0, v.shape[0] - 1)
        c = np.clip(c, 0, v.shape[1] - 1)
        i = np.minimum(r.astype(np.intp), v.shape[0] - 2)
        j = np.minimum(c.astype(np.intp), v.shape[1] - 2)
        u, w = (r - i)[:, None], c - j
        by_row = v[i] * (1 - u) + v[i + 1] * u
        empty = np.isnan(dtm)
        dtm[empty] = (by_row[:, j] * (1 - w) + by_row[:, j + 1] * w)[empty]
    else:
        blocks = [
            (r, c, max(r - DTM_MARGIN, 0), max(c - DTM_MARGIN, 0))
            for r in range(0, shape[0], DTM_BLOCK)
            for c in range(0, shape[1], DTM_BLOCK)
        ]
        windows = [
            cell_mean[r0 : r + DTM_BLOCK + DTM_MARGIN, c0 : c + DTM_BLOCK + DTM_MARGIN]
            for r, c, r0, c0 in blocks
        ]
        offsets = [(r - r0, c - c0) for r, c, r0, c0 in blocks]
        if len(blocks) > 1:
            with ProcessPoolExecutor(min(DTM_WORKERS, len(blocks))) as pool:
                filled = list(pool.map(_fill_block, windows, offsets))
        else:  # a fixture crop: not worth starting processes
            filled = [_fill_block(windows[0], offsets[0])]
        for (r, c, _, _), (empty, values) in zip(blocks, filled, strict=True):
            dtm[r : r + DTM_BLOCK, c : c + DTM_BLOCK][tuple(empty.T)] = values
    if np.isnan(dtm).any():
        nearest = ndimage.distance_transform_edt(
            np.isnan(dtm), return_distances=False, return_indices=True
        )
        dtm = dtm[tuple(nearest)]
        del nearest
    del cell_mean

    # 2. Smooth_DTM
    smooth_dtm = ndimage.gaussian_filter(dtm, SMOOTH_SIGMA, mode="nearest")

    # 3. Solid_DSM from last returns: drop unsupported points above HEIGHT_CAP, then per cell
    # the ELEVATED_PERCENTILE of elevated points, else the max of the rest; empty cells -inf.
    z = points["z"]
    z_norm = (z - dtm.ravel()[flat]).astype(np.float32)
    last = ground | (points["return_number"] == points["num_returns"])
    high = last & (z_norm > HEIGHT_CAP)
    window = np.ones((SUPPORT_PX, SUPPORT_PX))
    count, s1, s2 = (
        ndimage.correlate(
            np.bincount(flat[high], w, minlength=size).reshape(shape).astype(np.float32),
            window,
            mode="constant",
        )
        for w in (None, z_norm[high], z_norm[high] ** 2)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        var = s2 / count - (s1 / count) ** 2
    supported = (count >= SUPPORT_MIN - 0.5) & (var <= SUPPORT_MAX_STD**2)
    del count, s1, s2, var, high
    solid = last & ((z_norm <= HEIGHT_CAP) | supported.ravel()[flat])
    elevated = solid & (z_norm >= ELEVATED)
    dsm = np.full(size, -np.inf, np.float32)
    np.maximum.at(dsm, flat[solid & ~elevated], z[solid & ~elevated])
    order = np.lexsort((z[elevated], flat[elevated]))
    cells, start, per_cell = np.unique(flat[elevated][order], return_index=True, return_counts=True)
    dsm[cells] = z[elevated][order][start + (per_cell - 1) * ELEVATED_PERCENTILE // 100]
    dsm = dsm.reshape(shape)
    del supported, solid, elevated, order, cells, start, per_cell

    # 4–5. Closing, relief mask, opening.
    relief = ndimage.grey_closing(dsm, size=(3, 3)) - smooth_dtm
    del dsm, smooth_dtm
    mask = (relief >= MIN_RELIEF) & (relief <= MAX_RELIEF)
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), bool))

    # 6. Components of at least MIN_AREA_PX.
    labels, count = ndimage.label(mask)
    area = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero(area >= MIN_AREA_PX)
    keep = keep[keep > 0]

    # 7. Canopy gate: ground share of the last returns inside the outline, and median
    # relief / sqrt(area).
    point_label = labels.ravel()[flat]
    in_outline = last & (point_label > 0)
    total = np.bincount(point_label[in_outline], minlength=count + 1)
    low = np.bincount(point_label[in_outline & (z_norm < GROUND_Z)], minlength=count + 1)
    ground_share = low[keep] / np.maximum(total[keep], 1)
    # Per-component statistics over the labelled cells only, so ndimage never sorts the
    # whole raster's background.
    labelled = labels > 0
    top = np.zeros(count + 1)
    if len(keep):  # ndimage's statistics raise on no cells
        top[keep] = ndimage.median(relief[labelled], labels[labelled], keep)
    shape_ratio = top[keep] / np.sqrt(area[keep] * CELL**2)
    up = (point_label > 0) & ~ground & (z_norm >= GROUND_Z)
    up &= z_norm <= top[point_label] + SPREAD_GAP
    hits, s1, s2 = (
        np.bincount(flat[up], w, minlength=size).astype(np.float32)
        for w in (None, z_norm[up], z_norm[up] ** 2)
    )
    del up, point_label, in_outline, z_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt(np.maximum(s2 / hits - (s1 / hits) ** 2, 0.0))
    spread_labels = np.where(hits >= 2, labels.ravel(), 0)
    del hits, s1, s2
    # A component with fewer than MIN_SPREAD_CELLS cells of >= 2 hits gets NaN: no evidence,
    # kept (ndimage.median also returns an arbitrary value for no cells).
    cells = np.bincount(spread_labels, minlength=count + 1)[keep]
    has = spread_labels > 0
    column_spread = np.full(len(keep), np.nan)
    if has.any():
        enough = cells >= MIN_SPREAD_CELLS
        column_spread[enough] = np.asarray(
            ndimage.median(spread[has], spread_labels[has], keep[enough])
        )
    keep = keep[
        (ground_share <= MAX_GROUND_SHARE)
        & (shape_ratio <= MAX_SHAPE_RATIO)
        & ~(np.asarray(column_spread) > MAX_COLUMN_SPREAD)
    ]
    relabel = np.zeros(count + 1, dtype=labels.dtype)
    relabel[keep] = np.arange(1, len(keep) + 1)
    labels = relabel[labels]
    index = range(1, len(keep) + 1)
    centres = ndimage.center_of_mass(mask, labels, index)
    labelled = labels > 0
    max_h = ndimage.maximum(relief[labelled], labels[labelled], index) if len(keep) else []
    candidates = [
        Candidate(x0 + (c + 0.5) * CELL, y0 + (r + 0.5) * CELL, float(h))
        for (r, c), h in zip(centres, max_h, strict=True)
    ]
    return Proposal(labels, dtm, float(x0), float(y0), candidates, float(pulse_density))
