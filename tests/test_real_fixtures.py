import json
from typing import Any

import laspy
import numpy as np
import pytest
from pyproj import Transformer

from tests.real_fixtures import BOULDERS, FIXTURES, MANIFEST, NEGATIVES

HALF = MANIFEST["crop_half_m"]
SITES = BOULDERS + NEGATIVES
CROPS = [(b, c) for b in SITES for c in b["crops"]]


def test_every_site_has_a_crop() -> None:
    assert len(BOULDERS) == 9 and len(NEGATIVES) == 7
    assert all(b["crops"] for b in SITES)


@pytest.mark.parametrize("boulder", SITES, ids=[b["id"] for b in SITES])
def test_manifest_coordinates_agree(boulder: dict[str, Any]) -> None:
    e, n = Transformer.from_crs(4326, 3006, always_xy=True).transform(
        boulder["lon"], boulder["lat"]
    )
    assert np.hypot(e - boulder["e"], n - boulder["n"]) < 5.0


@pytest.mark.parametrize(("boulder", "crop"), CROPS, ids=[c["file"] for _, c in CROPS])
def test_crop_contains_boulder_and_matches_manifest(
    boulder: dict[str, Any], crop: dict[str, Any]
) -> None:
    las = laspy.read(FIXTURES / crop["file"])
    x, y = np.asarray(las.x), np.asarray(las.y)

    assert len(las.points) == crop["points"]
    # A full 100 x 100 m square (not cut off by a file edge) with the boulder well inside it.
    # b09's crop was cut around an earlier estimate 14 m from its current coordinates.
    assert x.max() - x.min() > 2 * HALF - 2 and y.max() - y.min() > 2 * HALF - 2
    assert x.max() - x.min() < 2 * HALF and y.max() - y.min() < 2 * HALF
    margin = min(boulder["e"] - x.min(), x.max() - boulder["e"])
    margin = min(margin, boulder["n"] - y.min(), y.max() - boulder["n"])
    assert margin > 35.0
    first = np.count_nonzero(np.asarray(las.return_number) == 1)
    assert first / (2 * HALF) ** 2 == pytest.approx(crop["pulses_per_m2"], abs=0.01)


def test_label_cutouts_leave_out_the_held_out_labels() -> None:
    # Every non-held-out, non-unsure label has a cut-out unless the build found no
    # candidate near it; no held-out label has one.
    labels = json.loads((FIXTURES / "area_labels.json").read_text(encoding="utf-8"))
    held = set(json.loads((FIXTURES / "area_labels_holdout.json").read_text(encoding="utf-8")))
    manifest = json.loads((FIXTURES / "label_cutouts" / "manifest.json").read_text())
    have = {p.stem for p in (FIXTURES / "label_cutouts").glob("*.npz")}
    want = {x["candidate_id"] for x in labels if x["class"] != "unsure"} - held - {None}
    assert have == want - set(manifest["lost"])
    assert len(have) == manifest["cutouts"]
