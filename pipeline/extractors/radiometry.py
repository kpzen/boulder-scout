"""Radiometry features of one candidate."""

import numpy as np

GROUND_MAX = 0.5  # m: last returns below this z_norm reached the ground
SPREAD_CELL = 0.5  # m: column_spread groups returns into cells this wide
SPREAD_GAP = 1.0  # m: column_spread counts returns up to H95 + SPREAD_GAP, as patches do
NAMES = (
    "single_return_ratio",
    "mean_intensity_norm",
    "ground_share",
    "column_spread",
)


def radiometry(
    points: np.ndarray, z_norm: np.ndarray, inside: np.ndarray, obj: np.ndarray
) -> dict[str, float]:
    """Return and intensity features of a candidate from its cut-out of read_tile points.

    `inside` marks the points in the candidate's outline, `obj` its P_obj (object_points).
    NaN for every feature but ground_share when P_obj is empty, and mean_intensity_norm
    NaN when no P_obj point has a normalised intensity.
    """
    # Ground share: rock is opaque, so no pulse over it ends on the ground. It needs no
    # P_obj, so it still counts when dropping floating points leaves a canopy candidate without one.
    last = (points["classification"] == 2) | (points["return_number"] == points["num_returns"])
    ground = z_norm[inside & last] < GROUND_MAX
    ground_share = float(ground.mean()) if len(ground) else float("nan")
    if not obj.any():
        return dict.fromkeys(NAMES, float("nan")) | {"ground_share": ground_share}
    single = (points["num_returns"][obj] == 1) & (points["return_number"][obj] == 1)
    intensity = points["intensity_norm"][obj]
    intensity = intensity[~np.isnan(intensity)]
    h95 = np.percentile(z_norm[obj], 95)
    # Column spread: rock is one surface per spot, canopy has hits at many heights. All
    # non-ground returns in the outline up to just above the object, by 0.5 m cell.
    up = inside & (points["classification"] != 2) & (z_norm >= GROUND_MAX)
    up &= z_norm <= h95 + SPREAD_GAP
    cell = np.c_[np.floor(points["x"][up] / SPREAD_CELL), np.floor(points["y"][up] / SPREAD_CELL)]
    _, idx, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    idx = idx.ravel()
    z = z_norm[up]
    n, s1, s2 = (np.bincount(idx, w) for w in (None, z, z**2))
    var = np.maximum(s2[counts >= 2] / n[counts >= 2] - (s1[counts >= 2] / n[counts >= 2]) ** 2, 0)
    return {
        "single_return_ratio": float(single.mean()),
        "mean_intensity_norm": float(intensity.mean()) if len(intensity) else float("nan"),
        "ground_share": ground_share,
        "column_spread": float(np.median(np.sqrt(var))) if len(var) else float("nan"),
    }
