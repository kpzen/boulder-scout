"""Geometry features of one candidate."""

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

OBJ_MIN = 1.0  # m: object points have z_norm >= OBJ_MIN
# P_obj grows outward from the outline, since the 2 m relief outline cuts off the base.
GROW_RADIUS = 1.0  # m: a point joins if it is this close to a member...
GROW_MAX_RISE = 0.3  # m: ...and at most this much higher than it (the object slopes down)
# Stray crown hits that survive the last-return filter sit far above their neighbours.
SPIKE_RADIUS = 1.5  # m
SPIKE_MAX = 2.0  # m above the neighbours' median
# Rock is opaque, so a last return more than FLOAT_GAP above another within FLOAT_RADIUS
# beside it floats: a branch over open ground or over the rock.
FLOAT_RADIUS = 0.5  # m
FLOAT_GAP = 1.0  # m
# A floating point is restored when a non-floating one within RESCUE_RADIUS sits within
# RESCUE_DZ of its height (one pass): the top edge of a wall, beside the ground at its foot.
RESCUE_RADIUS = 0.5  # m
RESCUE_DZ = 0.3  # m
# P_obj splits into pieces of points linked within PIECE_LINK in 3D; a piece of less than
# PIECE_SMALL of the points whose points sit a median of more than PIECE_GAP above the largest
# piece's points within PIECE_NEAR is a crown remnant and goes.
PIECE_LINK = 1.5  # m
PIECE_SMALL = 0.15
PIECE_GAP = 1.0  # m
PIECE_NEAR = 1.5  # m
BASE_MAX = 0.3  # m: base points have z_norm < BASE_MAX, outside the outline
HIGH_FRAC = 0.7  # max_dropoff looks at object points above HIGH_FRAC * H95
MIN_RUN = 0.25  # m: floor on the horizontal distance to the base
# plane_rmse fits a plane to each object point's neighbours within PLANE_RADIUS, if it
# has at least PLANE_MIN (itself included).
PLANE_RADIUS = 1.0  # m
PLANE_MIN = 5
# bump_mad compares each object point with the median of its neighbours within BUMP_RADIUS.
BUMP_RADIUS = 1.5  # m
# The outline is P_obj's alpha shape (Delaunay triangles with circumradius <= ALPHA_RADIUS),
# drawn on an OUTLINE_PIX grid: its largest piece, holes filled.
ALPHA_RADIUS = 1.0  # m
OUTLINE_PIX = 0.2  # m
# area_m2 sums P_obj's Delaunay triangles with no side longer than AREA_EDGE (the 3D
# view's MAX_EDGE, web/src/view3d.ts).
AREA_EDGE = 2.0  # m
# halo counts the non-ground returns outside P_obj, between HALO_IN and HALO_OUT across
# from it and at its own heights (OBJ_MIN to H95), per P_obj point.
HALO_IN = 0.3  # m
HALO_OUT = 2.0  # m
# planar_area is the share of P_obj whose neighbours within FACET_RADIUS (at least
# FACET_MIN) fit one plane within FACET_RMSE, times area_m2. Roofs are planar over metres; rock
# curves (exported boulders <= 4.2 m², houses mostly 6-300 m²).
FACET_RADIUS = 2.0  # m
FACET_MIN = 8
FACET_RMSE = 0.04  # m
# hanging looks for empty ground beside P_obj: a HANG_CELL cell HANG_IN to HANG_OUT out
# from it with no return of any class within HANG_EMPTY (water returns nothing; class 9 is
# dropped). A P_obj point within HANG_NEAR of such a cell, more than HANG_MIN_H above the
# ground and with no return HANG_DROP lower within HANG_BELOW, hangs over it: a crown leaning
# out over a lake. A rock face has its own lower points below its top edge.
HANG_CELL = 0.5  # m
HANG_IN = 0.5  # m
HANG_OUT = 3.0  # m
HANG_EMPTY = 1.0  # m
HANG_NEAR = 1.75  # m
HANG_MIN_H = 1.5  # m
HANG_BELOW = 0.75  # m
HANG_DROP = 1.0  # m
NAMES = (
    "max_h",
    "max_dropoff",
    "elongation_2d",
    "plane_rmse",
    "bump_mad",
    "solidity",
    "ground_inside",
    "tallness",
    "halo",
    "area_m2",
    "planar_area",
)
# Quick filters: hard drops in the order their columns are known, so a candidate
# that fails one skips the deeper measurements. Each limit lies 50 % beyond the most extreme
# labelled boulder or small rock (tuning labels, hold-outs excluded).
FILTERS = (
    ("halo", ">", 1.70),
    ("tallness", ">", 2.81),
    ("solidity", "<", 0.152),
    ("ground_inside", ">", 0.136),
    ("plane_rmse", ">", 0.233),
    ("bump_mad", ">", 0.537),
    ("planar_area", ">", 11.7),
    ("column_spread", ">", 1.223),
    ("hanging", ">", 0.236),
)


def dropped(feats: dict[str, float]) -> str:
    """The first quick filter (FILTERS) the features fail, or "". A missing or NaN column fails
    none."""
    for name, side, limit in FILTERS:
        v = feats.get(name, np.nan)
        if v > limit if side == ">" else v < limit:
            return name
    return ""


def _pairs(a: cKDTree, b: cKDTree, r: float) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) for every a point i and b point j at most r apart, sorted by i: the sets
    query_ball_point returns, as two flat arrays."""
    d = a.sparse_distance_matrix(b, r, output_type="ndarray")
    order = np.argsort(d["i"], kind="stable")
    return d["i"][order].astype(np.intp), d["j"][order].astype(np.intp)


def _median_by(group: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """np.median of `values` in each group 0..n-1 (NaN when empty), and the group sizes."""
    v = values[np.lexsort((values, group))]
    count = np.bincount(group, minlength=n)
    start = np.cumsum(count) - count
    lo, hi = start + (count - 1) // 2, start + count // 2
    median = np.full(n, np.nan)
    odd, even = count % 2 == 1, (count > 0) & (count % 2 == 0)
    median[odd] = v[lo[odd]]
    median[even] = (v[lo[even]] + v[hi[even]]) / 2
    return median, count


def _local_rmse(tree: cKDTree, xyz: np.ndarray, radius: float, least: int) -> np.ndarray:
    """Per point with >= `least` neighbours within `radius` in plan (itself included), the root
    of the smallest eigenvalue of their 3D covariance (np.cov's, ddof 1)."""
    i, j = _pairs(tree, tree, radius)
    fit = np.bincount(i, minlength=len(xyz))[i] >= least
    centre, group = np.unique(i[fit], return_inverse=True)
    n = np.bincount(group).astype(np.float64)
    near = xyz[j[fit]]
    d = (
        near
        - np.stack([np.bincount(group, near[:, k]) for k in range(3)], axis=1)[group]
        / n[group, None]
    )
    cov = np.empty((len(centre), 3, 3))
    for p in range(3):
        for q in range(p, 3):
            cov[:, p, q] = cov[:, q, p] = np.bincount(group, d[:, p] * d[:, q]) / (n - 1)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[:, 0], 0.0))


def object_points(points: np.ndarray, z_norm: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Mask of P_obj in a candidate's cut-out of read_tile points.

    `inside` marks the points in the candidate's outline. Last returns only: rock is
    opaque, so earlier returns over a boulder are canopy, and so are last returns floating
    above another one close beside them, unless level with solid rock beside them.
    """
    last = (points["classification"] == 2) | (points["return_number"] == points["num_returns"])
    pool = np.flatnonzero(last & (z_norm >= OBJ_MIN))
    xy = np.c_[points["x"][pool], points["y"][pool]]
    # Drop floating points before growing, so growth cannot climb through a crown.
    i, j = _pairs(cKDTree(xy), cKDTree(np.c_[points["x"], points["y"]][last]), FLOAT_RADIUS)
    floor = np.full(len(pool), np.inf)  # every pool point is itself a last return beside it
    np.minimum.at(floor, i, z_norm[last][j])
    z = z_norm[pool]
    keep = z <= floor + FLOAT_GAP
    # but not the top edge of a wall, level with solid rock beside it.
    solid, floating = np.flatnonzero(keep), np.flatnonzero(~keep)
    if len(solid) and len(floating):
        i, j = _pairs(cKDTree(xy[floating]), cKDTree(xy[solid]), RESCUE_RADIUS)
        level = np.abs(z[solid[j]] - z[floating[i]]) <= RESCUE_DZ
        keep[floating] = np.bincount(i[level], minlength=len(floating)) > 0
    pool, xy, z = pool[keep], xy[keep], z[keep]
    # Grow: a point joins from a member within GROW_RADIUS that is at most GROW_MAX_RISE lower,
    # repeatedly, i.e. P_obj takes everything reachable from the outline's points that way.
    member = inside[pool].copy()
    if member.any():
        tree = cKDTree(xy)
        i, j = _pairs(tree, tree, GROW_RADIUS)
        step = z[j] <= z[i] + GROW_MAX_RISE
        start = np.flatnonzero(member)
        n = len(z)  # node n is a source linked to every point in the outline
        graph = coo_matrix(
            (
                np.ones(step.sum() + len(start)),
                (np.r_[i[step], np.full(len(start), n)], np.r_[j[step], start]),
            ),
            shape=(n + 1, n + 1),
        ).tocsr()
        reached = breadth_first_order(graph, n, directed=True, return_predecessors=False)
        member[reached[reached < n]] = True

    pool, xy, z = pool[member], xy[member], z[member]
    spike = np.zeros(len(z), dtype=bool)
    if len(z):
        tree = cKDTree(xy)
        i, j = _pairs(tree, tree, SPIKE_RADIUS)
        other = i != j
        median, count = _median_by(i[other], z[j[other]], len(z))
        spike = (count > 0) & (z > median + SPIKE_MAX)
    pool, xy, z = pool[~spike], xy[~spike], z[~spike]
    # Crown remnants that stay together, which the spike test cannot see, come off whole.
    if len(z):
        pairs = cKDTree(np.c_[xy, z]).query_pairs(PIECE_LINK, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(z), len(z)))
        _, piece = connected_components(graph, directed=False)
        sizes = np.bincount(piece)
        main = piece == np.argmax(sizes)
        main_tree, main_z = cKDTree(xy[main]), z[main]
        for p in np.flatnonzero(sizes < PIECE_SMALL * len(z)):
            idx = np.flatnonzero(piece == p)
            rise = [
                np.median(z[i] - main_z[n])
                for i, n in zip(idx, main_tree.query_ball_point(xy[idx], PIECE_NEAR), strict=True)
                if n
            ]
            if rise and np.median(rise) > PIECE_GAP:
                piece[idx] = -1
        pool = pool[piece >= 0]
    obj = np.zeros(len(points), dtype=bool)
    obj[pool] = True
    return obj


def geometry(
    points: np.ndarray, z_norm: np.ndarray, inside: np.ndarray, obj: np.ndarray, quick: bool = False
) -> dict[str, float]:
    """Height and shape features of a candidate from its cut-out of read_tile points.

    `inside` marks the points in the candidate's outline, `obj` its P_obj (object_points).
    NaN for every feature when P_obj is empty, max_dropoff NaN without base points, the shape
    features (elongation_2d, solidity, ground_inside, tallness, area_m2) NaN when P_obj has
    too few points to span an area, solidity, ground_inside and area_m2 also when its points
    are too sparse for an outline or surface, plane_rmse NaN with fewer than 3 local fits, and
    planar_area NaN with area_m2 or without a point dense enough to fit.
    With `quick`, it stops at the first quick filter (FILTERS) the features known so far
    fail; the features not yet measured are NaN.
    """
    last = (points["classification"] == 2) | (points["return_number"] == points["num_returns"])
    base_xy = np.c_[points["x"], points["y"]][last & (z_norm < BASE_MAX) & ~inside]
    in_obj = obj
    obj_xy, obj = np.c_[points["x"], points["y"]][obj], z_norm[obj]

    if len(obj) == 0:
        return dict.fromkeys(NAMES, float("nan"))
    h95 = np.percentile(obj, 95)
    max_dropoff = elongation_2d = solidity = ground_inside = tallness = area_m2 = np.nan
    plane_rmse = bump_mad = planar_area = np.nan

    def features() -> dict[str, float]:
        return {
            "max_h": float(h95),
            "max_dropoff": float(max_dropoff),
            "elongation_2d": float(elongation_2d),
            "plane_rmse": float(plane_rmse),
            "bump_mad": float(bump_mad),
            "solidity": float(solidity),
            "ground_inside": float(ground_inside),
            "tallness": float(tallness),
            "halo": float(halo),
            "area_m2": float(area_m2),
            "planar_area": float(planar_area),
        }

    # A crown is part of a larger mass of vegetation at its own heights, mostly returns
    # the float test kept out of P_obj; rock stands clear of what is beside it.
    across, _ = cKDTree(obj_xy).query(
        np.c_[points["x"], points["y"]], distance_upper_bound=HALO_OUT
    )
    beside = ~in_obj & (points["classification"] != 2) & (across > HALO_IN) & (across < HALO_OUT)
    halo = np.sum(beside & (z_norm >= OBJ_MIN) & (z_norm <= h95)) / len(obj)
    high = obj > HIGH_FRAC * h95
    if len(base_xy):
        run, _ = cKDTree(base_xy).query(obj_xy[high])
        max_dropoff = np.percentile(obj[high] / np.maximum(run, MIN_RUN), 90)
    try:
        hull = ConvexHull(obj_xy)
    except QhullError:  # fewer than 3 points, or all on a line: no area to measure
        hull = None
    if hull is not None:
        lam = np.linalg.eigvalsh(np.cov(obj_xy.T))  # ascending
        elongation_2d = np.sqrt(lam[0] / lam[1])  # width / length
        tallness = h95 / np.sqrt(hull.volume)  # 'too tall for its base'; 2D volume is area
    if quick and dropped(features()):
        return features()
    if hull is not None:
        # The alpha shape follows notches and gaps that the convex hull spans.
        tri = Delaunay(obj_xy)
        a, b, c = (obj_xy[tri.simplices[:, k]] for k in range(3))
        twice_area = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        sides = np.hypot(*(b - c).T) * np.hypot(*(a - c).T) * np.hypot(*(a - b).T)
        keep = sides <= 2 * ALPHA_RADIUS * twice_area  # circumradius = abc / (2 * twice_area)
        # The map's size is the 3D view's rock surface seen from above.
        longest = np.max([np.hypot(*(b - c).T), np.hypot(*(a - c).T), np.hypot(*(a - b).T)], axis=0)
        if (longest <= AREA_EDGE).any():  # else NaN: no evidence of size, not size 0
            area_m2 = twice_area[longest <= AREA_EDGE].sum() / 2
        lo = obj_xy.min(axis=0) - OUTLINE_PIX
        nx, ny = np.ceil((obj_xy.max(axis=0) + OUTLINE_PIX - lo) / OUTLINE_PIX).astype(int)
        gx, gy = np.meshgrid(
            *(lo[k] + OUTLINE_PIX * (np.arange(n) + 0.5) for k, n in enumerate((nx, ny)))
        )
        simplex = tri.find_simplex(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        pieces, _ = ndimage.label((simplex >= 0) & keep[simplex])
        # No outline when no two neighbouring object points lie within ~2 m.
        if pieces.any():
            largest = np.argmax(np.bincount(pieces.ravel())[1:]) + 1
            outline = ndimage.binary_fill_holes(pieces == largest)
            solidity = outline.sum() / (simplex >= 0).sum()  # 'jagged perimeter'
            # 'holes': ground hits within the outline per object point in it (rock is opaque).
            counts = []
            for xy in (np.c_[points["x"], points["y"]][points["classification"] == 2], obj_xy):
                col, row = np.floor((xy - lo) / OUTLINE_PIX).astype(int).T
                ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
                counts.append(outline[row[ok], col[ok]].sum())
            ground_inside = counts[0] / counts[1] if counts[1] else np.nan
    if quick and dropped(features()):
        return features()
    # One plane misses domed or tilted rock tops by ~0.5 m; local planes follow them.
    obj_tree = cKDTree(obj_xy)
    rmse = _local_rmse(obj_tree, np.c_[obj_xy, obj], PLANE_RADIUS, PLANE_MIN)
    plane_rmse = np.median(rmse) if len(rmse) >= 3 else np.nan
    # A roof stays planar over metres. Heights above sea, not the ground model: the roof
    # is a plane in space.
    facet = _local_rmse(obj_tree, np.c_[obj_xy, points["z"][in_obj]], FACET_RADIUS, FACET_MIN)
    planar_area = np.mean(facet < FACET_RMSE) * area_m2 if len(facet) else np.nan
    # 'Jagged top': rock tops are smooth at 1-2 m, crowns are bumpy.
    i, j = _pairs(obj_tree, obj_tree, BUMP_RADIUS)
    bump_mad = np.median(np.abs(obj - _median_by(i, obj[j], len(obj))[0]))
    return features()


def hanging(
    points: np.ndarray, z_norm: np.ndarray, obj: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """Share of P_obj (`obj`) hanging over empty ground beside it, from every point in
    the box `lo`-`hi` (other candidates' points included, unlike the cut-out: they are not
    empty); only cells inside the box count. NaN when P_obj is empty."""
    if not obj.any():
        return float("nan")
    xy = np.c_[points["x"], points["y"]]
    obj_xy, obj_z = xy[obj], z_norm[obj]
    a, b = (
        np.maximum(obj_xy.min(axis=0) - HANG_OUT, lo),
        np.minimum(obj_xy.max(axis=0) + HANG_OUT, hi),
    )
    gx, gy = np.meshgrid(*(np.arange(a[k], b[k], HANG_CELL) + HANG_CELL / 2 for k in range(2)))
    cells = np.c_[gx.ravel(), gy.ravel()]
    across, _ = cKDTree(obj_xy).query(cells, distance_upper_bound=HANG_OUT)
    cells = cells[(across >= HANG_IN) & (across <= HANG_OUT)]
    tree = cKDTree(xy)
    nearest, _ = tree.query(cells, distance_upper_bound=HANG_EMPTY)
    empty = cells[np.isinf(nearest)]
    if not len(empty):
        return 0.0
    beside, _ = cKDTree(empty).query(obj_xy, distance_upper_bound=HANG_NEAR)
    high = np.flatnonzero(np.isfinite(beside) & (obj_z > HANG_MIN_H))
    if not len(high):
        return 0.0
    i, j = _pairs(cKDTree(obj_xy[high]), tree, HANG_BELOW)
    under = np.bincount(i[z_norm[j] < obj_z[high][i] - HANG_DROP], minlength=len(high))
    return float(np.sum(under == 0) / len(obj_z))
