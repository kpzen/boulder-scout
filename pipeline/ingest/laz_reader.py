"""Read a LAZ tile, keeping LAS classes 1 and 2, and normalise
intensity per swath."""

from pathlib import Path

import laspy
import numpy as np

KEPT_CLASSES = (1, 2)  # drops 7 noise, 9 water, 17 bridge, 18 high noise
POINT_DTYPE = np.dtype(
    [
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        ("intensity", "u2"),
        ("classification", "u1"),
        ("return_number", "u1"),
        ("num_returns", "u1"),
        ("point_source_id", "u2"),
        ("intensity_norm", "f4"),  # robust z-score per swath; NaN when a swath has no reference
    ]
)


def read_tile(path: Path) -> np.ndarray:
    """Return the tile's class 1 and 2 points as a structured array of POINT_DTYPE."""
    las = laspy.read(path)
    keep = np.isin(np.asarray(las.classification), KEPT_CLASSES)
    points = np.empty(int(keep.sum()), dtype=POINT_DTYPE)
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
        points[name] = np.asarray(las[dim])[keep]

    # Reference: single-return ground; I_norm = (I − median) / IQR.
    is_ref = (points["classification"] == 2) & (points["num_returns"] == 1)
    for swath in np.unique(points["point_source_id"]):
        in_swath = points["point_source_id"] == swath
        ref = points["intensity"][in_swath & is_ref].astype(np.float64)
        q25, q50, q75 = np.percentile(ref, [25, 50, 75]) if len(ref) else (0.0, np.nan, 0.0)
        iqr = q75 - q25
        points["intensity_norm"][in_swath] = (
            (points["intensity"][in_swath] - q50) / iqr if iqr > 0 else np.nan
        )
    return points
