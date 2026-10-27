"""Patch serialisation for the 3D view."""

import numpy as np

PATCH_DTYPE = np.dtype(
    [("dx", "<i2"), ("dy", "<i2"), ("dz", "<i2"), ("intensity", "u1"), ("point_type", "u1")]
)
MARGIN = 4.0  # m: patch radius R = max(radius + MARGIN, MIN_RADIUS)
MIN_RADIUS = 8.0  # m
CANOPY_GAP = 1.0  # m: points with z_norm > H95 + CANOPY_GAP are canopy and left out
INTENSITY_STEP = 64  # raw intensity per uint8 step; fixtures' 99th percentile ~12300 -> 192


def patch(
    points: np.ndarray, z_norm: np.ndarray, obj: np.ndarray, origin: np.ndarray, max_h: float
) -> bytes:
    """Encode a candidate's read_tile points within R as patch bytes.

    `obj` marks the candidate's P_obj, the rock points. `origin` is (X_c, Y_c, Z_base).
    With max_h NaN (empty P_obj) no point counts as canopy.
    """
    keep = obj | ~(z_norm > max_h + CANOPY_GAP)
    p = points[keep]
    out = np.empty(len(p), dtype=PATCH_DTYPE)
    for name, coord, o in zip(("dx", "dy", "dz"), ("x", "y", "z"), origin, strict=True):
        out[name] = np.round((p[coord] - o) * 100.0)
    out["intensity"] = np.minimum(p["intensity"] // INTENSITY_STEP, 255)
    # Rock is the object as the features see it (last returns, grown, despiked), so rock
    # under canopy (last of several returns) still counts; other non-ground points are vegetation.
    out["point_type"] = np.where(obj[keep], 1, np.where(p["classification"] == 2, 0, 2))
    return out.tobytes()
