"""Re-run P_obj and the extractors on the labelled cut-outs and print each measure's separation.

Usage: uv run python scripts/eval_labels.py
Reads tests/fixtures/label_cutouts (scripts/build_label_cutouts.py; held-out labels are not
there) and prints, per measure and the score, the AUC of boulder vs vegetation and vs all
non-boulders: the probability that a boulder reads higher than a non-boulder (< 0.5: lower
means boulder). Seconds, not the area run's 45 min: for trying an extractor change. The
score table is fitted on the unit tables, by scoring/fit_score.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from pipeline.extractors import geometry, radiometry  # noqa: E402
from scoring.fit_score import auc  # noqa: E402
from scoring.heuristic_scorer import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_rows() -> dict[str, tuple[str, dict[str, float]]]:
    labels = json.loads((FIXTURES / "area_labels.json").read_text(encoding="utf-8"))
    rows = {}
    for x in labels:
        path = FIXTURES / "label_cutouts" / f"{x['candidate_id']}.npz"
        if not path.exists():  # held out, unsure, or no candidate
            continue
        c = np.load(path)
        points, z_norm, inside = c["points"], c["z_norm"], c["inside"]
        obj = geometry.object_points(points, z_norm, inside)
        row = geometry.geometry(points, z_norm, inside, obj) | radiometry.radiometry(
            points, z_norm, inside, obj
        )
        row["hanging"] = float(c["hanging"])  # needs points the cut-out leaves out
        rows[x["candidate_id"]] = (x["class"], row)
    return rows


def main() -> None:
    rows = load_rows()
    print(f"{len(rows)} labelled cut-outs")
    print(f"{'feature':22s}{'boulder/veg':>12s}{'boulder/all':>12s}   n boulder/veg/all others")
    scored = {k: (cls, r | {"score": score(r)}) for k, (cls, r) in rows.items()}
    for name in ["score", *geometry.NAMES, *radiometry.NAMES]:
        boulder = [r[name] for cls, r in scored.values() if cls == "boulder"]
        veg = [r[name] for cls, r in scored.values() if cls == "vegetation"]
        other = [r[name] for cls, r in scored.values() if cls != "boulder"]
        (a1, nb, nv), (a2, _, no) = auc(boulder, veg), auc(boulder, other)
        print(f"{name:22s}{a1:12.2f}{a2:12.2f}   {nb}/{nv}/{no}")


if __name__ == "__main__":
    main()
