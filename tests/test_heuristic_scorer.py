import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from pipeline.run_unit import main as run_unit
from scoring.heuristic_scorer import LLR_TABLE, score
from tests.real_fixtures import BOULDERS, FIXTURES, NEGATIVES

# The negative squares the rocks are scored against.
HELD_OUT = ("n01_medium_flat", "n05_dense_flat")


def _rows(site: dict[str, Any], tmp_path: Path) -> list[dict[str, Any]]:
    out = tmp_path / f"{site['id']}.parquet"
    run_unit(
        [
            str(FIXTURES / site["crops"][0]["file"]),
            "--out",
            str(out),
            "--patches",
            str(out.parent / "p"),
        ]
    )
    rows: list[dict[str, Any]] = pq.read_table(out).to_pylist()
    return rows


def test_nan_feature_adds_exactly_zero() -> None:
    row = {f: points[-1][0] for f, points in LLR_TABLE.items()}
    full = score(row)
    for feature, points in LLR_TABLE.items():
        assert math.isclose(score({**row, feature: math.nan}), full - points[-1][1])
    assert score(dict.fromkeys(LLR_TABLE, math.nan)) == 0.0


def test_photo_terms_read_rock_up_and_green_down() -> None:
    # Bare rock in the photo scores higher than green; without photos the terms are 0.
    assert score({"ndvi_p25": 0.0, "right_angles": 0.05}) > 0.0
    assert score({"ndvi_p25": 0.5, "right_angles": 0.05}) < -1.0
    assert score({"ndvi_p25": math.nan, "right_angles": math.nan}) == 0.0


def test_confirmed_rocks_outscore_held_out_negatives(tmp_path: Path) -> None:
    # >= 90 % of the author-confirmed rocks (the 8 known boulders,
    # not b09, plus their nearby_rocks) score above >= 90 % of the held-out candidates.
    rocks = []
    for b in BOULDERS:
        if b["id"].startswith("b09"):
            continue  # the author can't place its candidates under the dense canopy
        rows = _rows(b, tmp_path)
        for e, n in [(b["e"], b["n"])] + [(r["e"], r["n"]) for r in b.get("nearby_rocks", [])]:
            row = min(rows, key=lambda r: np.hypot(r["x"] - e, r["y"] - n))
            assert np.hypot(row["x"] - e, row["y"] - n) < 3.0
            rocks.append(score(row))
    negatives = [score(r) for s in NEGATIVES if s["id"] in HELD_OUT for r in _rows(s, tmp_path)]

    assert len(rocks) == 20 and len(negatives) == 16
    threshold = np.percentile(negatives, 90)
    assert np.mean(np.array(rocks) > threshold) >= 0.9


def test_roof_and_water_edge_penalties_are_gated_and_never_raise() -> None:
    # planar_area takes the lower ramp within 100 m of a building; hanging counts only
    # within 5 m of water; neither adds anything below its ramp or with NaN.
    def s(**row: float) -> float:
        return score(row)

    assert s(planar_area=0.0) == s(planar_area=4.0) == 0.0
    assert s(planar_area=4.0, dist_building=100.0) == -8.5  # where the building nudge is 0
    assert s(planar_area=4.0, dist_building=math.inf) == 0.0
    assert s(planar_area=8.0) == -8.5
    assert s(hanging=0.35, dist_water=3.0) == -6.0
    assert s(hanging=0.35, dist_water=6.0) == s(hanging=0.35) == 0.0
    assert s(hanging=0.1, dist_water=0.0) == s(hanging=math.nan, dist_water=0.0) == 0.0
