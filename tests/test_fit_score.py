"""The score table fitted on labels and unit tables, labels matched by id or position."""

import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scoring.fit_score import FIT_FEATURES, PHOTO_FEATURES, main, match
from scoring.heuristic_scorer import score


def test_a_moved_candidate_is_matched_by_position() -> None:
    # The label's candidate moved 0.9 m (its id moved with it); another sits 4 m away.
    labels = {0: (100.0, 200.0), 1: (150.0, 200.0)}
    centroids = np.array([[104.0, 200.0], [100.6, 200.7]])
    assert match(labels, centroids) == {0: 1}


def test_fit_learns_from_matched_labels_and_writes_a_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 60 labelled candidates: boulders have a low halo, vegetation a high one; every other
    # measure is noise. Label 0's candidate moved 1 m since labelling (a new id), label 1 is
    # held out, label 2's candidate is masked and label 3 has no candidate. Of two missed
    # boulders (no candidate_id), one has had a candidate since: candidate 4 lost its label.
    rng = np.random.default_rng(0)
    n = 60
    boulder = np.arange(n) % 2 == 0
    xy = [(1000.0 + 20 * i, 2000.0) for i in range(n)]
    ids = [f"t_{x:.1f}_{y:.1f}" for x, y in xy]
    now = [f"t_{x + 1:.1f}_{y:.1f}" if i == 0 else ids[i] for i, (x, y) in enumerate(xy)]
    columns = {f: rng.uniform(0, 1, n) for f in FIT_FEATURES}
    columns["halo"] = np.where(boulder, 0.2, 2.0) + rng.uniform(0, 0.5, n)
    pq.write_table(
        pa.table(
            {
                "candidate_id": now,
                "x": [x + (1.0 if i == 0 else 0.0) for i, (x, _) in enumerate(xy)],
                "y": [y for _, y in xy],
                **columns,
                "dropped": [""] * n,
                "masked": [i == 2 for i in range(n)],
            }
        ),
        tmp_path / "units.parquet",
    )
    pq.write_table(
        pa.table({"candidate_id": now, **{f: rng.uniform(0, 1, n) for f in PHOTO_FEATURES}}),
        tmp_path / "photo_measures.parquet",
    )
    labels: list[dict[str, object]] = [
        {"candidate_id": i, "E": x, "N": y, "class": "boulder" if b else "vegetation"}
        for i, (x, y), b in zip(ids, xy, boulder, strict=True)
    ]
    labels[3]["candidate_id"], labels[3]["E"] = "t_9.0_9.0", 9.0
    labels[4] = {"candidate_id": None, "E": xy[4][0] + 0.5, "N": xy[4][1], "class": "boulder"}
    labels.append({"candidate_id": None, "E": 0.0, "N": 0.0, "class": "boulder"})
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "holdout.json").write_text(json.dumps([ids[1]]))
    out = tmp_path / "table.json"
    main(["--labels", str(tmp_path / "labels.json"), "--candidates",
          str(tmp_path / "units.parquet"), "--holdout", str(tmp_path / "holdout.json"),
          "--out", str(out)])  # fmt: skip

    assert (
        "{'by position': 2, 'held out': 1, 'by id': 56, 'no candidate': 2}; "
        "matched: {'kept': 57, 'masked': 1}; fitting on the 57 kept or dropped"
        in capsys.readouterr().out
    )
    table = json.loads(out.read_text())
    assert set(table) == {*FIT_FEATURES, *PHOTO_FEATURES}
    assert table["halo"][0][1] > 0 > table["halo"][-1][1]  # a low halo counts for a boulder
    assert score({"halo": 0.3}, table) > score({"halo": 2.2}, table)
    assert score({}, table) == 0.0 and score({"halo": math.nan}, table) == 0.0
