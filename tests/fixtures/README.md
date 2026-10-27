Real LiDAR test fixtures. © Lantmäteriet, CC BY 4.0, from *Laserdata
Nedladdning, skog*.

- `boulders.json` — the author's 9 known boulders (coordinates, size, height, canopy,
  description), the other rocks the author confirmed in its crop (`nearby_rocks`) and, per
  boulder, its 100 × 100 m crop: source file, flight date, points, pulse density. Every
  boulder has one scan, all leaf-on June 2020.
- `negatives` in `boulders.json` — 7 squares with no boulders to the author's knowledge (flat,
  dense, sloping, hilly, bedrock ridges, a steep ridge), same crop fields; leaf-on June 2020.
- `{site id}__{item id}.laz` — the crops; 0.1–0.4 MB each.
- `area_labels.json` — the author's 667 review-mode labels on the area runs: class,
  tags, note, E/N and the candidate's exported values at labelling time; no point data. They
  match the unit tables `data/units/*.parquet` by `candidate_id` or by position.
  The first 10 carry a `source` field: the 7 marked `queue` were
  random picks, not the author's; the rest are author-picked.
- `area_labels_holdout.json` — the candidate_ids of the 113 labels held out;
  no fit or feature analysis uses them.
- `label_cutouts/{label's candidate_id}.npz` — the cut-out of the candidate within 1.5 m of each
  label's E/N, built by the unit stage's own code (`pipeline/run_unit.py`'s `read_unit`
  and `Unit.cut_out`: read_tile points, `z_norm`, outline mask `inside`, and `hanging`), for
  every label but the held-out and unsure ones. `manifest.json` names the commit they came
  from, each label's current candidate_id and the labels lost. They are a developer tool, needed
  only for extractor work: `scripts/eval_labels.py` and `scripts/eval_photos.py` re-run P_obj
  and the extractors on them in seconds. Refitting the score does not need them: it reads the
  unit tables (`scoring/fit_score.py`). Rebuild them with `scripts/build_label_cutouts.py`
  (~30 min, needs run 2's tiles in `data/raw`) after a proposer, reader or cut-out change.
- `tests/real_fixtures.py` loads the manifest; `b01_reference` is the reference crop.
