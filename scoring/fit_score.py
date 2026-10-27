"""Fit the score table on review-mode labels and the unit tables.

Usage: uv run python -m scoring.fit_score [--labels data/labels/labels.json]
       [--candidates data/units] [--photo-measures data/photo_measures.parquet]
       [--holdout tests/fixtures/area_labels_holdout.json] [--out data/score_table.json] [--cv]
A label's measures are its candidate's unit-table row (and photo-measures row): the candidate
with the label's candidate_id, else the nearest within MATCH of its E/N (ids move with the
centroid when the proposer changes); a missed boulder (class boulder, no candidate_id) is
matched by position only, so it counts once a candidate turns up there. Unsure labels, held-out
labels and masked candidates (no measures) are left out; dropped ones are fitted on, with NaN
after the filter they failed: leaving them out put 16 of 153 labelled boulders below the cutoff
out-of-fold. The export reads the table with --table; the committed one is
scoring/score_table.json, fitted on the author's labels (tests/fixtures/area_labels.json):
refit it with --labels tests/fixtures/area_labels.json --out scoring/score_table.json.
--cv prints how the current table and, out-of-fold (5 x 5 folds), this fit score the kept
labelled candidates, and writes nothing.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from scoring.export_geojson import MIN_SCORE
from scoring.heuristic_scorer import score

MATCH = 1.5  # m from a label's position to its candidate's centroid
PHOTO_FEATURES = ["ndvi_p25", "right_angles"]
# Signed off by the author; area_m2 stays a column for the map,
# not a term.
FIT_FEATURES = [
    "plane_rmse", "bump_mad", "solidity", "column_spread", "max_dropoff", "elongation_2d",
    "mean_intensity_norm", "single_return_ratio", "ground_share", "max_h", "tallness",
    "ground_inside", "halo",
]  # fmt: skip
# Which label classes each fit trains boulder against: the LiDAR terms not manmade, which
# the building mask removes; the photo terms manmade too, the photos see buildings.
VOCABULARY = json.loads((Path(__file__).parent / "label_vocabulary.json").read_text())
NEGATIVES, PHOTO_NEGATIVES = tuple(VOCABULARY["negatives"]), tuple(VOCABULARY["photo_negatives"])
# Fitted, the photo terms mostly lift boulders; shifted down, a candidate with photos needs 2.0
# more, which keeps the lowest labelled boulder where it was without photos (2.84 vs 2.87
# out-of-fold) and puts most filter-passing vegetation below the cutoff.
PHOTO_SHIFT = 2.0
KNOTS = (5, 25, 50, 75, 95)  # percentiles of the training values; ends held beyond them
# Penalty on the step between neighbouring knots' LLRs (smooth curves; set by repeated 5-fold
# CV), plus a small ridge that pins each curve's level, so a typical value's term is near 0.
LAMBDA, RIDGE = 0.1, 0.1

Table = dict[str, list[list[float]]]
Row = tuple[str, dict[str, Any]]  # a label's class and its candidate's measures


def auc(pos: list[float], neg: list[float]) -> tuple[float, int, int]:
    a, b = np.array(pos), np.array(neg)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan"), len(a), len(b)
    ranks = rankdata(np.r_[a, b])[: len(a)]  # ties count 1/2
    return (ranks.sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)), len(a), len(b)


def match(labels: dict[int, tuple[float, float]], centroids: np.ndarray) -> dict[int, int]:
    """For each label (index -> E, N), the index of the nearest centroid within MATCH; labels with
    none are left out."""
    if not labels or not len(centroids):
        return {}
    d, k = cKDTree(centroids).query(list(labels.values()), distance_upper_bound=MATCH)
    return {i: int(j) for i, dj, j in zip(labels, d, k, strict=True) if np.isfinite(dj)}


def fit_table(
    rows: list[Row], features: list[str], negatives: tuple[str, ...], base: Table | None = None
) -> Table:
    """Each feature's term is np.interp over its knots, so the logit is linear in the knots'
    LLR values: column j of a feature's block is the interpolation weight of knot j (all 0
    for NaN, so a NaN term is 0 as in score()). With `base` that table's score is a fixed part
    of the logit and only `features` are fitted on top of it."""
    rows = [(cls, r) for cls, r in rows if cls == "boulder" or cls in negatives]
    y = np.array([cls == "boulder" for cls, _ in rows], float)
    offset = np.array([score(r, base) for _, r in rows]) if base else np.zeros(len(rows))
    knots, blocks = {}, []
    for f in features:
        x = np.array([r[f] for _, r in rows], float)
        xs = np.unique(np.percentile(x[~np.isnan(x)], KNOTS))  # ground_inside: mostly 0
        knots[f] = xs
        eye = np.eye(len(xs))
        blocks.append(
            np.array([np.zeros(len(xs)) if np.isnan(v) else [np.interp(v, xs, e) for e in eye]
                      for v in x])
        )  # fmt: skip
    design = np.hstack(blocks)
    steps = block_diag(*[np.diff(np.eye(b.shape[1]), axis=0) for b in blocks])
    penalty = LAMBDA * steps.T @ steps + RIDGE * np.eye(design.shape[1])

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        t = offset + design @ w[1:] + w[0]
        g = 1 / (1 + np.exp(-t)) - y
        value = (np.logaddexp(0, t) - y * t).sum() + w[1:] @ penalty @ w[1:]
        return value, np.r_[g.sum(), design.T @ g + 2 * penalty @ w[1:]]

    w = minimize(loss, np.zeros(design.shape[1] + 1), jac=True, method="L-BFGS-B").x[1:]
    vals = np.split(w, np.cumsum([b.shape[1] for b in blocks])[:-1])
    return {
        f: [[round(float(x), 3), round(float(v), 2)] for x, v in zip(knots[f], vs, strict=True)]
        for f, vs in zip(features, vals, strict=True)
    }


def fit(rows: list[Row]) -> Table:
    """The LiDAR terms, then the photo terms on top of them, which leave them as they are, so
    a candidate without photos scores as with the LiDAR terms alone; ndvi_p25's values carry
    PHOTO_SHIFT (both photo measures are NaN together, so one term carries the shift)."""
    table = fit_table(rows, FIT_FEATURES, NEGATIVES)
    photo = fit_table(rows, PHOTO_FEATURES, PHOTO_NEGATIVES, base=table)
    photo["ndvi_p25"] = [[x, round(v - PHOTO_SHIFT, 2)] for x, v in photo["ndvi_p25"]]
    return table | photo


def report(title: str, scores: np.ndarray, classes: np.ndarray) -> None:
    b, veg = scores[classes == "boulder"], scores[classes == "vegetation"]
    print(f"{title}: AUC boulder/veg {auc(list(b), list(veg))[0]:.3f}; boulders below "
          f"{MIN_SCORE}: {(b < MIN_SCORE).sum()} of {len(b)} (lowest {b.min():.2f})")  # fmt: skip
    for cls in sorted(set(classes) - {"boulder"}):
        sel = scores[classes == cls]
        print(f"  {cls:14s} below {MIN_SCORE}: {np.mean(sel < MIN_SCORE):4.0%} of {len(sel)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=Path("data/labels/labels.json"))
    parser.add_argument("--candidates", type=Path, default=Path("data/units"))
    parser.add_argument("--photo-measures", type=Path)
    parser.add_argument(
        "--holdout", type=Path, default=Path("tests/fixtures/area_labels_holdout.json")
    )
    parser.add_argument("--out", type=Path, default=Path("data/score_table.json"))
    parser.add_argument("--cv", action="store_true")
    args = parser.parse_args(argv)

    held = set(json.loads(args.holdout.read_text())) if args.holdout.exists() else set()
    labels = [
        x for x in json.loads(args.labels.read_text(encoding="utf-8")) if x["class"] != "unsure"
    ]
    photo_path = args.photo_measures or args.candidates.with_name("photo_measures.parquet")
    photo = (
        {r.pop("candidate_id"): r for r in pq.read_table(photo_path).to_pylist()}
        if photo_path.exists()
        else {}
    )
    candidates = pq.read_table(args.candidates).to_pylist()
    by_id = {r["candidate_id"]: i for i, r in enumerate(candidates)}
    moved = match(
        {n: (x["E"], x["N"]) for n, x in enumerate(labels) if x["candidate_id"] not in by_id},
        np.array([(r["x"], r["y"]) for r in candidates]).reshape(-1, 2),
    )
    found: dict[str, int] = {}
    states: dict[str, int] = {}
    rows: list[Row] = []
    kept = []
    for n, x in enumerate(labels):
        if x["candidate_id"] in held:
            how, state = "held out", ""
        elif x["candidate_id"] in by_id:
            how, r = "by id", candidates[by_id[x["candidate_id"]]]
        elif n in moved:
            how, r = "by position", candidates[moved[n]]
        else:
            how, state = "no candidate", ""
        if how in ("by id", "by position"):
            state = "masked" if r["masked"] else "dropped" if r["dropped"] else "kept"
            states[state] = states.get(state, 0) + 1
        found[how] = found.get(how, 0) + 1
        if state not in ("kept", "dropped"):
            continue
        measures = photo.get(r["candidate_id"], {})
        rows.append((x["class"], r | {f: measures.get(f, math.nan) for f in PHOTO_FEATURES}))
        kept.append(state == "kept")
    print(f"{len(labels)} labels (not unsure): {found}; matched: {states};"
          f" fitting on the {len(rows)} kept or dropped")  # fmt: skip

    if args.cv:
        classes, kept_ = np.array([cls for cls, _ in rows]), np.array(kept)
        oof = np.zeros(len(rows))
        for repeat in range(5):
            folds = np.random.default_rng(repeat).permutation(len(rows)) % 5
            for k in range(5):
                table = fit([r for r, f in zip(rows, folds, strict=True) if f != k])
                for i in np.flatnonzero(folds == k):
                    oof[i] += score(rows[i][1], table) / 5
        current = np.array([score(r) for _, r in rows])
        print(f"the {kept_.sum()} kept candidates:")
        report("current table", current[kept_], classes[kept_])
        report("this fit, out-of-fold", oof[kept_], classes[kept_])
        return
    table = fit(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'  "{f}": {json.dumps(points)}' for f, points in table.items()]
    args.out.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    print(f"{len(table)} terms -> {args.out}")


if __name__ == "__main__":
    main()
