"""Heuristic score: a sum of per-feature piecewise-linear LLRs."""

import json
import math
from pathlib import Path

import numpy as np

# Breakpoints (feature value, LLR) per measure; np.interp holds the end values outside them.
# The fitted terms are in score_table.json, written by `python -m scoring.fit_score`:
# an additive logistic model of the author's boulder labels against vegetation and
# terrain_other, on their candidates' unit-table rows (held-out labels left
# out), knots at the 5/25/50/75/95th percentiles, then the photo terms on top of
# those, which stay as they are, with manmade as a negative too and ndvi_p25's values shifted
# by -2.0. A candidate without photos has NaN for both and scores as without them.
# Manmade is not trained against by the LiDAR terms: the building layer removes it.
LLR_TABLE: dict[str, list[list[float]]] = json.loads(
    Path(__file__).with_name("score_table.json").read_text(encoding="utf-8")
)
# Hand-set, not fitted: a nudge down near buildings Topografi 50 knows, from the
# 40 m mask edge to 0 at 100 m; pipeline/run_unit.py writes inf beyond 100 m.
NEAR_BUILDING = [(40.0, -2.0), (100.0, 0.0)]
# Hand-set penalties with a gate, which never raise a score. Roofs: planar_area (m²)
# from 0 to the full penalty, with the lower ramp within BUILDING_NEAR of a building Topografi
# 50 knows (exported boulders reach 4.2 m², 2.3 near a building; houses mostly 6-300 m²).
# Crowns over water: hanging, only within WATER_NEAR of open water (a shore boulder 0.18).
BUILDING_NEAR = 100.0  # m
PLANAR_RAMP = [(5.0, 0.0), (8.0, -8.5)]
PLANAR_RAMP_NEAR = [(2.5, 0.0), (4.0, -8.5)]
WATER_NEAR = 5.0  # m
HANGING_RAMP = [(0.22, 0.0), (0.35, -6.0)]


def _term(value: float, points: list[tuple[float, float]] | list[list[float]]) -> float:
    if math.isnan(value):
        return 0.0
    xs, llrs = zip(*points, strict=True)
    return float(np.interp(value, xs, llrs))


def score(row: dict[str, float], table: dict[str, list[list[float]]] = LLR_TABLE) -> float:
    """Sum of the LLR terms for one unit-table row; a NaN feature adds exactly 0, and so
    do dist_building and dist_water where no Topografi 50 was read (fixture export, label
    cut-outs): the roof penalty then takes its far ramp and the water-edge one is off."""
    total = sum(_term(row.get(f, math.nan), points) for f, points in table.items())
    total += _term(row.get("dist_building", math.nan), NEAR_BUILDING)
    near = row.get("dist_building", math.inf) <= BUILDING_NEAR
    total += _term(row.get("planar_area", math.nan), PLANAR_RAMP_NEAR if near else PLANAR_RAMP)
    if row.get("dist_water", math.inf) <= WATER_NEAR:
        total += _term(row.get("hanging", math.nan), HANGING_RAMP)
    return total
