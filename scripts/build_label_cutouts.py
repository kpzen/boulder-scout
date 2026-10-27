"""Cut out the labelled candidates' points from the area tiles into tests/fixtures.

Usage: uv run python scripts/build_label_cutouts.py [--raw data/raw] [--dtm data/dtm]
For every label in tests/fixtures/area_labels.json that is neither held out nor
unsure, writes label_cutouts/{label's candidate_id}.npz: the cut-out of the candidate nearest
the label's E/N within MATCH (ids move with the centroid), exactly as run_unit builds it
(read_tile points, z_norm, the outline mask `inside`), plus its `hanging`, which needs the points
of other candidates the cut-out leaves out. P_obj and the other extractors can then be re-run on
all labels in seconds (scripts/eval_labels.py). manifest.json maps each label to the candidate's
current id and lists the labels with none. Re-run it when the proposer, the reader or the
cut-out changes. Proposes only the units holding labels; ~30 min.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from pipeline.area import TILE_M, units_on_disk  # noqa: E402
from pipeline.extractors import geometry  # noqa: E402
from pipeline.run_unit import read_unit  # noqa: E402
from scoring.fit_score import MATCH, match  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = FIXTURES / "label_cutouts"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--dtm", type=Path, default=ROOT / "data" / "dtm")
    args = parser.parse_args(argv)
    dtm = sorted(args.dtm.glob("*.tif"))
    if not dtm:
        sys.exit(f"no ground model in {args.dtm}: run scripts/fetch_area.py")
    held = set(json.loads((FIXTURES / "area_labels_holdout.json").read_text(encoding="utf-8")))
    labels = json.loads((FIXTURES / "area_labels.json").read_text(encoding="utf-8"))
    want = {
        x["candidate_id"]: (x["E"], x["N"])
        for x in labels
        if x["candidate_id"] not in held and x["class"] != "unsure"
    }
    want.pop(None, None)  # missed boulders have no candidate
    for old in OUT.glob("*.npz"):
        old.unlink()
    OUT.mkdir(exist_ok=True)
    found: dict[str, str] = {}
    # The whole tiles around the labels, MATCH wider: a candidate's unit holds its centroid.
    es, ns = zip(*want.values(), strict=True)
    extent = (
        (min(es) - MATCH) // TILE_M * TILE_M,
        (min(ns) - MATCH) // TILE_M * TILE_M,
        -((-(max(es) + MATCH)) // TILE_M) * TILE_M,
        -((-(max(ns) + MATCH)) // TILE_M) * TILE_M,
    )
    for tile, bbox, neighbours in units_on_disk(args.raw, extent):
        tile_id = tile.name.split(".")[0]
        x0, y0, x1, y1 = bbox
        # A candidate belongs to the unit holding its centroid, which may lie MATCH away.
        mine = {
            i: (e, n)
            for i, (e, n) in want.items()
            if x0 - MATCH <= e < x1 + MATCH and y0 - MATCH <= n < y1 + MATCH and i not in found
        }
        if not mine:
            continue
        start = time.perf_counter()
        unit = read_unit(tile, bbox, neighbours, dtm)  # as run_unit.run reads it
        points, proposal = unit.points, unit.proposal
        owned = np.array(
            [x0 <= c.x < x1 and y0 <= c.y < y1 for c in proposal.candidates], dtype=bool
        )
        xy = np.array([(c.x, c.y) for c in proposal.candidates]).reshape(-1, 2)
        index = np.flatnonzero(owned)
        matched = {i: int(index[j]) for i, j in match(mine, xy[owned]).items()}
        for i, j in matched.items():
            k = j + 1
            near, cut, box, _ = unit.cut_out(k)
            inside = unit.label[cut] == k
            obj = geometry.object_points(points[cut], unit.z_norm[cut], inside)
            np.savez_compressed(
                OUT / f"{i}.npz",
                points=points[cut],
                z_norm=unit.z_norm[cut],
                inside=inside,
                hanging=geometry.hanging(
                    points[near], unit.z_norm[near], np.isin(near, cut[obj]), *box
                ),
            )
            c = proposal.candidates[j]
            found[i] = f"{tile_id}_{c.x:.1f}_{c.y:.1f}"
        print(f"{tile_id}: {len(matched)} of {len(mine)} labels, "
              f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", "pipeline/proposer", "pipeline/ingest",
         "pipeline/run_unit.py", "pipeline/extractors/geometry.py"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.strip()  # fmt: skip
    manifest = {
        "proposer_commit": commit,
        "cutouts": len(found),
        "lost": sorted(want.keys() - found.keys()),
        "matched": dict(sorted(found.items())),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{len(found)} cut-outs, {len(want) - len(found)} labels without a candidate -> {OUT}")


if __name__ == "__main__":  # required: propose's process pool re-imports this script
    main()
