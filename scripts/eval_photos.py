"""Measure the aerial photos of each labelled candidate and print how each measure separates
boulders from the other labels.

Usage: uv run python scripts/eval_photos.py [--photos data/photos] [--rows FILE.json]
Reads the label cut-outs (tests/fixtures/label_cutouts, scripts/build_label_cutouts.py) for
area_m2, and the photo stage's crops, {photos}/{label's candidate_id}_rgb.png and
_ir.png (scripts/measure_photos.py), measured with the stage's own `measure`. --rows writes
every label's measures, for a closer look.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from eval_labels import FIXTURES, auc  # noqa: E402
from measure_photos import RES, decode, measure  # noqa: E402

from pipeline.extractors import geometry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", type=Path, default=ROOT / "data" / "photos")
    parser.add_argument("--rows", type=Path)
    args = parser.parse_args(argv)
    labels = json.loads((FIXTURES / "area_labels.json").read_text(encoding="utf-8"))
    rows = []
    for lab in labels:
        i = lab["candidate_id"]
        path = FIXTURES / "label_cutouts" / f"{i}.npz"
        photos = [args.photos / f"{i}_{layer}.png" for layer in RES]
        if i is None or not path.exists() or not all(p.exists() for p in photos):
            continue
        rgb, ir = (decode(p.read_bytes(), str(p)) for p in photos)
        c = np.load(path)
        points, z_norm, inside = c["points"], c["z_norm"], c["inside"]
        obj = geometry.object_points(points, z_norm, inside)
        area = geometry.geometry(points, z_norm, inside, obj)["area_m2"]
        if np.isnan(area):
            continue
        x, y = map(float, i.rsplit("_", 2)[1:])  # the crops are cut around the centroid
        rows.append({"id": i, "class": lab["class"]} | measure(rgb, ir, x, y, x, y, area))
    if args.rows:
        args.rows.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    classes = sorted({r["class"] for r in rows})
    print(f"{len(rows)} labels with photos and a cut-out: "
          + ", ".join(f"{c} {sum(r['class'] == c for r in rows)}" for c in classes))  # fmt: skip
    names = list(dict.fromkeys(k for r in rows for k in r if k not in ("id", "class")))
    negatives = [
        ("vegetation",),
        ("manmade",),
        ("terrain_other",),
        tuple(set(classes) - {"boulder"}),
    ]
    print("\nAUC: P(boulder reads higher); < 0.5 means boulders read lower")
    print(f"{'measure':24s}{'vs veg':>8s}{'vs manmade':>12s}{'vs terrain':>12s}{'vs all':>8s}")
    for name in names:
        boulder = [r[name] for r in rows if r["class"] == "boulder"]
        cols = [auc(boulder, [r[name] for r in rows if r["class"] in cs])[0] for cs in negatives]
        widths = (8, 12, 12, 8)
        print(f"{name:24s}" + "".join(f"{a:>{w}.2f}" for a, w in zip(cols, widths, strict=True)))


if __name__ == "__main__":
    main()
