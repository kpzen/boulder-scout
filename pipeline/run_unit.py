"""Run the unit stage on one file and write its unit table, one parquet table.

Usage: uv run python -m pipeline.run_unit TILE.laz [--out data/units/TILE.parquet]
    [--patches data/patches] [--topo data/topografi50]
"""

import argparse
import os
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage
from scipy.spatial import cKDTree

from pipeline import serializer
from pipeline.area import Bbox
from pipeline.extractors import geometry, radiometry
from pipeline.ingest.dtm_reader import read_dtm
from pipeline.ingest.laz_reader import read_tile
from pipeline.ingest.topo_masker import LAYERS, WATER, distance, masked
from pipeline.proposer.raster_relief import CELL, LowDensityWarning, Proposal, propose

CUT_BUFFER = 5.0  # m around a candidate's bounding box
# m read past a work unit's edges from its neighbours: the Gaussian's 20 m reach plus a
# straddling component and its cut-out.
EDGE_BUFFER = 50.0
BUILDING_REACH = 100.0  # m: dist_building is inf beyond it (the score's nudge ends there)
WATER_REACH = 20.0  # m: dist_water is inf beyond it (the score's gate is 5 m)
MEASURES = (*geometry.NAMES, "hanging", *radiometry.NAMES)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tile", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patches", type=Path, default=Path("data/patches"))
    parser.add_argument("--topo", type=Path, help="Topografi 50 folder (default: no mask)")
    args = parser.parse_args(argv)
    out = args.out or Path("data/units") / f"{args.tile.name.split('.')[0]}.parquet"
    run(args.tile, out, args.patches, topo=args.topo)


def run(
    tile: Path,
    out: Path,
    patches: Path,
    bbox: Bbox | None = None,
    neighbours: Sequence[tuple[Path, Bbox]] = (),
    dtm: Sequence[Path] = (),
    rings: Path | None = None,
    topo: Path | None = None,
) -> float:
    """Write the unit table of `tile`, a file, and return its pulse density. With `bbox`, the
    file's nominal 2.5 km tile, it is a unit: every file supplies only its own nominal
    tile, read EDGE_BUFFER past `bbox` from `neighbours` (file, nominal tile), and the unit
    keeps the candidates centred in `bbox`. `dtm` are Markhöjdmodell files for the proposer's
    empty DTM cells. With `rings`, a directory, each file's points within EDGE_BUFFER of
    its nominal tile's edges are kept there, so a neighbour's file is read whole only once
    With `topo`, a Topografi 50 folder, candidates near buildings and roads or in open
    water or built land are masked and not measured: NaN measures, no patch.
    Each kept candidate gets a point patch in `patches`."""
    tile_id = tile.name.split(".")[0]  # m20e018-692_49.copc.laz -> m20e018-692_49
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LowDensityWarning)
        unit = read_unit(tile, bbox, neighbours, dtm, rings)
    points, proposal, label, z_norm, tree = (
        unit.points,
        unit.proposal,
        unit.label,
        unit.z_norm,
        unit.tree,
    )
    print(f"{tile_id}: pulse density {proposal.pulse_density:.2f}/m²")
    for w in caught:
        print(f"WARNING {tile_id}: {w.message}", file=sys.stderr)

    # Ownership: a candidate belongs to the one unit whose tile holds its centroid.
    owned = [
        bbox is None or (bbox[0] <= c.x < bbox[2] and bbox[1] <= c.y < bbox[3])
        for c in proposal.candidates
    ]
    cands = [c for c, o in zip(proposal.candidates, owned, strict=True) if o]
    x, y = np.array([c.x for c in cands]), np.array([c.y for c in cands])
    hit = np.zeros(len(cands), bool)
    near_building = near_water = np.full(len(cands), np.inf)
    if topo is not None:
        hit = masked(x, y, topo)
        # The score's building proximity nudge and its water-edge gate.
        theme, tables, *_ = LAYERS[0]
        near_building = distance(x, y, topo, theme, tables, BUILDING_REACH)
        near_water = distance(x, y, topo, *WATER[:2], WATER_REACH, WATER[2])
    # Tile and location. "_" not ":", which Windows forbids in patch file names.
    ids = [f"{tile_id}_{c.x:.1f}_{c.y:.1f}" for c in cands]
    patches.mkdir(parents=True, exist_ok=True)
    feats: list[dict[str, float]] = []
    drops: list[str] = []
    for k in range(1, len(proposal.candidates) + 1):
        if not owned[k - 1]:
            continue
        if hit[len(feats)]:
            feats.append(dict.fromkeys(MEASURES, float("nan")))
            drops.append("")
            continue  # masked: never exported, so neither measured nor patched
        near, cut, box, radius = unit.cut_out(k)
        cut_points, cut_z, inside = points[cut], z_norm[cut], label[cut] == k
        obj = geometry.object_points(cut_points, cut_z, inside)
        # Quick filters: a candidate that fails one skips the rest; NaN for what it skips.
        f = geometry.geometry(cut_points, cut_z, inside, obj, quick=True)
        f |= dict.fromkeys((*radiometry.NAMES, "hanging"), float("nan"))
        if not geometry.dropped(f):
            f |= radiometry.radiometry(cut_points, cut_z, inside, obj)
        if not geometry.dropped(f):
            on_obj = np.isin(near, cut[obj])
            f["hanging"] = geometry.hanging(points[near], z_norm[near], on_obj, *box)
        feats.append(f)
        drops.append(geometry.dropped(f))
        if drops[-1]:
            continue  # no patch: it is never exported
        # Patch: origin is the centroid on the DTM.
        c = proposal.candidates[k - 1]
        centre = (int((c.y - proposal.y0) / CELL), int((c.x - proposal.x0) / CELL))
        disc = np.array(
            tree.query_ball_point(
                (c.x, c.y), max(radius + serializer.MARGIN, serializer.MIN_RADIUS)
            ),
            dtype=np.intp,
        )
        origin = np.array([c.x, c.y, proposal.dtm[centre]])
        patch = serializer.patch(
            points[disc], z_norm[disc], np.isin(disc, cut[obj]), origin, feats[-1]["max_h"]
        )
        (patches / f"{ids[len(feats) - 1]}.bin").write_bytes(patch)

    table = pa.table(
        {
            # Typed even when empty, so the units read as one table.
            "candidate_id": pa.array(ids, pa.string()),
            "tile_id": pa.array([tile_id] * len(cands), pa.string()),
            "x": pa.array(x, pa.float64()),
            "y": pa.array(y, pa.float64()),
            **{name: pa.array([f[name] for f in feats], pa.float64()) for name in MEASURES},
            "dropped": pa.array(drops, pa.string()),
            "masked": pa.array(hit, pa.bool_()),
            "dist_building": pa.array(near_building, pa.float64()),
            "dist_water": pa.array(near_water, pa.float64()),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Whole or absent: a resumed area run skips a unit whose parquet exists. Hidden
    # (".") so a reader of the folder passes over it.
    part = out.with_name(f".{out.name}.part")
    pq.write_table(table, part)
    part.replace(out)
    print(f"{tile_id}: {len(cands)} candidates, {hit.sum()} masked -> {out}, patches -> {patches}")
    return proposal.pulse_density


@dataclass(frozen=True)
class Unit:
    """A unit's points and their proposal, indexed for cut-outs: each
    point's candidate label (0 for none) and z_norm, its height above the DTM."""

    points: np.ndarray
    proposal: Proposal
    label: np.ndarray
    z_norm: np.ndarray
    tree: cKDTree
    slices: list[tuple[slice, slice]]

    def cut_out(
        self, k: int
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], float]:
        """Candidate label `k`'s cut-out: the points within its bounding box + CUT_BUFFER
        (`near`, indices into points), those not inside another candidate's outline (`cut`),
        the square (`box`, SW and NE corners) and its half side less CUT_BUFFER."""
        rows, cols = self.slices[k - 1]
        x0, y0 = self.proposal.x0, self.proposal.y0
        lo = np.array([x0 + cols.start * CELL, y0 + rows.start * CELL])
        hi = np.array([x0 + cols.stop * CELL, y0 + rows.stop * CELL])
        radius = (hi - lo).max() / 2
        near = np.array(
            self.tree.query_ball_point((lo + hi) / 2, radius + CUT_BUFFER, p=np.inf),
            dtype=np.intp,
        )
        cut = near[np.isin(self.label[near], (0, k))]
        box = ((lo + hi) / 2 - radius - CUT_BUFFER, (lo + hi) / 2 + radius + CUT_BUFFER)
        return near, cut, box, float(radius)


def read_unit(
    tile: Path,
    bbox: Bbox | None = None,
    neighbours: Sequence[tuple[Path, Bbox]] = (),
    dtm: Sequence[Path] = (),
    rings: Path | None = None,
) -> Unit:
    """Read `tile` (with `bbox`, the unit, as `run` says) and propose its candidates."""
    if bbox is None:
        points = read_tile(tile)
    else:
        x0, y0, x1, y1 = bbox
        parts = []
        for path, (a0, b0, a1, b1) in [(tile, bbox), *neighbours]:
            ring = rings / f"{path.name.split('.')[0]}.npy" if rings else None
            if ring and path != tile and ring.exists():
                p = np.load(ring)
            else:
                p = read_tile(path)  # whole file: intensity is normalised per file's swaths
                if ring and not ring.exists():
                    x, y = p["x"], p["y"]
                    edge = (x < a0 + EDGE_BUFFER) | (x >= a1 - EDGE_BUFFER)
                    edge |= (y < b0 + EDGE_BUFFER) | (y >= b1 - EDGE_BUFFER)
                    edge &= (x >= a0) & (x < a1) & (y >= b0) & (y < b1)
                    ring.parent.mkdir(parents=True, exist_ok=True)
                    # Whole or absent, never half written. Parallel units may both write
                    # it; Windows refuses the second replace while the first is read.
                    part = ring.with_suffix(f".{os.getpid()}.npy")
                    np.save(part, p[edge])
                    try:
                        part.replace(ring)
                    except PermissionError:
                        part.unlink()
            x, y = p["x"], p["y"]
            keep = (x >= max(a0, x0 - EDGE_BUFFER)) & (x < min(a1, x1 + EDGE_BUFFER))
            keep &= (y >= max(b0, y0 - EDGE_BUFFER)) & (y < min(b1, y1 + EDGE_BUFFER))
            parts.append(p[keep])
        points = np.concatenate(parts)
    model = None
    if dtm:
        x, y = points["x"], points["y"]
        model = read_dtm(dtm, (x.min() - 1, y.min() - 1, x.max() + 1, y.max() + 1))
    proposal = propose(points, model)
    row = ((points["y"] - proposal.y0) / CELL).astype(np.int32)
    col = ((points["x"] - proposal.x0) / CELL).astype(np.int32)
    label = proposal.labels[row, col]
    z_norm = (points["z"] - proposal.dtm[row, col]).astype(np.float32)
    del row, col  # memory
    tree = cKDTree(np.c_[points["x"], points["y"]])
    return Unit(points, proposal, label, z_norm, tree, ndimage.find_objects(proposal.labels))


if __name__ == "__main__":
    main()
