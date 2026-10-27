"""The photo stage: crop and measure the aerial photos of each kept candidate.

Usage: uv run python scripts/measure_photos.py [--candidates data/units]
       [--source data/orthophotos] [--out data] [--rerun]
Reads only the unit tables (a folder of them, or one table) and the photo files in
--source: Lantmäteriet's 4-band orthophotos (`*.tif`, red, green, blue, near infrared, one per
5 km square, scripts/fetch_photos.py), cut at their own resolution, if there are any; else
blocks `{rgb|ir}_{x0}_{y0}_{x1}_{y1}.png` (EPSG:3006 bbox; colour 0.25 m/px, infrared 0.5 m/px
as near infrared, red, green), the newest year's blocks the author collected from Min karta.
With no
photo files it prints so and writes nothing, and the score counts the photo terms as 0
For every kept candidate it writes its photo crops
{out}/photos/{candidate_id}_rgb.png and _ir.png, CUT m squares whose top-left corner lies on the
res grid (`corner`), so their pixels can be matched to LiDAR points, and a row of its photo
measures, `ndvi_p25` and `right_angles`, to {out}/photo_measures.parquet. A candidate whose
crop the photo files do not wholly cover gets neither. The measures look inside a circle of the
candidate's area_m2 around its centroid, widened by BUFFER, since photos are rectified to the
ground, not to the tops of things on it. They use the photos alone, never
LiDAR, which may be from another year. scripts/run_area.py runs it after the units.
A candidate already in photo_measures.parquet keeps its row and crops unless its unit table is
newer than that file (the unit ran since), or --rerun measures every candidate again (after
the photos or the measures change); a candidate the photos did not cover is tried again, and
the rows of candidates no longer in the tables go.
"""

import argparse
import math
import struct
import sys
import time
import zlib
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from pipeline.ingest.dtm_reader import (  # noqa: E402
    BITS,
    COMPRESSION,
    GEO_KEYS,
    HEIGHT,
    PIXEL_SCALE,
    PREDICTOR,
    SAMPLES,
    TIEPOINT,
    TILE_COUNTS,
    TILE_OFFSETS,
    TILE_W,
    WIDTH,
    _File,
    _tags,
)

ROOT = Path(__file__).resolve().parents[1]
RES = {"rgb": 0.25, "ir": 0.5}  # m/px of the PNG blocks
FOUR_BANDS = {"rgb": [0, 1, 2], "ir": [3, 0, 1]}  # a COG's red, green, blue, NIR as the layers
CUT = 40.0  # m: the crop's side, as the photo panel showed it
BUFFER = 1.5  # m around the candidate's circle
RING = 4.0  # m beyond the footprint
BLOCKS_KEPT = 24  # decoded blocks held at once (colour ~33 MB each)
TILES_KEPT = 64  # inflated COG tiles held at once (512 x 512 x 4 bands, 1 MB each)
Block = tuple[float, float, float, float, Path]


def decode(data: bytes, source: str) -> np.ndarray:
    """A PNG (8-bit RGB or RGBA, rows unfiltered as the collected blocks are, or Sub-filtered as
    png() writes them) as an H x W x 3 array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG from {source}: {data[:300]!r}")
    pos, idat = 8, []
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos : pos + 8])
        chunk = data[pos + 8 : pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        pos += 12 + n
    channels = {2: 3, 6: 4}.get(colour)
    if depth != 8 or channels is None or interlace:
        raise ValueError(f"unexpected PNG (depth {depth}, colour type {colour}) from {source}")
    rows = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + w * channels)
    if (rows[:, 0] > 1).any():
        raise ValueError(f"PNG rows filtered other than None or Sub from {source}")
    pixels = rows[:, 1:].reshape(h, w, channels).copy()
    # Sub adds the same channel of the pixel to the left; uint8 cumulative sums wrap mod 256.
    sub = rows[:, 0] == 1
    pixels[sub] = np.cumsum(pixels[sub], axis=1, dtype=np.uint8)
    return pixels[:, :, :3]


def png(image: np.ndarray) -> bytes:
    """An H x W x 3 uint8 array as PNG, rows Sub-filtered (smaller for photos)."""
    h, w, _ = image.shape
    flat = image.reshape(h, w * 3)
    sub = flat.copy()
    sub[:, 3:] = flat[:, 3:] - flat[:, :-3]  # uint8 wraps mod 256, as the filter wants
    raw = np.hstack([np.ones((h, 1), np.uint8), sub]).tobytes()

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + body))
        return struct.pack(">I", len(body)) + kind + body + crc

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = chunk(b"IDAT", zlib.compress(raw, 9))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + idat + chunk(b"IEND", b"")


def corner(x: float, y: float, res: float) -> tuple[float, float]:
    """A crop's top-left corner (EPSG:3006) on the res grid."""
    return round((x - CUT / 2) / res) * res, round((y + CUT / 2) / res) * res


def blocks(source: Path) -> tuple[dict[str, list[Block]], dict[str, float]]:
    """The photo files in source by layer, each with its bbox, and each layer's res: the COGs,
    placed by their GeoTIFF tags, if there are any, else the PNG blocks, by their names."""
    cogs, res = [], set()
    for path in sorted(source.glob("*.tif")) if source.is_dir() else []:
        x0, y1, cell, w, h, *_ = _cog(path)
        cogs.append((x0, y1 - h * cell, x0 + w * cell, y1, path))
        res.add(cell)
    if cogs:
        if len(res) > 1:
            raise ValueError(f"orthophotos in {source} at mixed resolutions {sorted(res)}")
        (cell,) = res
        return {layer: cogs for layer in RES}, {layer: cell for layer in RES}
    found: dict[str, list[Block]] = {layer: [] for layer in RES}
    for path in source.glob("*.png") if source.is_dir() else []:
        layer, *box = path.stem.split("_")
        if layer in RES and len(box) == 4:
            x0, y0, x1, y1 = map(float, box)
            found[layer].append((x0, y0, x1, y1, path))
    return found, RES


@lru_cache(maxsize=BLOCKS_KEPT)
def _block(path: Path) -> np.ndarray:
    return decode(path.read_bytes(), str(path))


@cache
def _cog(path: Path) -> tuple[float, float, float, int, int, int, tuple[int, ...], tuple[int, ...]]:
    """A COG's west and north edge, res, width, height, tile side, tile offsets and byte counts.
    Read as the DTM is, stdlib and numpy: Lantmäteriet's orthophotos are little-endian TIFFs,
    tiled, 4 bands of 8 bits interleaved, deflate with the horizontal predictor (2),
    pixel-is-area EPSG:3006; any other layout raises ValueError rather than being misread."""
    f = _File(path)
    try:
        t = _tags(f)
    finally:
        f.handle.close()
    layout = (t[BITS], t.get(SAMPLES), t[COMPRESSION], t.get(PREDICTOR), t.get(284, (1,)))
    if layout != ((8, 8, 8, 8), (4,), (8,), (2,), (1,)) or TILE_W not in t:
        raise ValueError(f"{path.name}: not a tiled deflated 4-band 8-bit photo {layout}")
    keys = t.get(GEO_KEYS, ())
    geo = {keys[k]: keys[k + 3] for k in range(4, len(keys), 4)}  # key -> inline value
    if geo.get(1025) != 1 or geo.get(3072) != 3006:  # pixel-is-area, SWEREF 99 TM
        raise ValueError(f"{path.name}: not pixel-is-area EPSG:3006 ({geo})")
    sx, sy = t[PIXEL_SCALE][:2]
    if sx != sy:
        raise ValueError(f"{path.name}: {sx} x {sy} m pixels, expected square ones")
    i, j, _, x, y, _ = t[TIEPOINT]
    w, h, side = t[WIDTH][0], t[HEIGHT][0], t[TILE_W][0]
    return x - i * sx, y + j * sy, float(sx), w, h, side, t[TILE_OFFSETS], t[TILE_COUNTS]


@lru_cache(maxsize=TILES_KEPT)
def _cog_tile(path: Path, k: int) -> np.ndarray:
    *_, side, offsets, counts = _cog(path)
    if counts[k] == 0:  # a sparse tile: no data
        return np.zeros((side, side, 4), np.uint8)
    with open(path, "rb") as f:
        f.seek(offsets[k])
        raw = zlib.decompress(f.read(counts[k]))
    # The predictor stores each sample less the same sample of the pixel to its left.
    return np.cumsum(np.frombuffer(raw, np.uint8).reshape(side, side, 4), axis=1, dtype=np.uint8)


def _cog_window(path: Path, rows: slice, cols: slice) -> np.ndarray:
    """A COG's pixels in rows x cols (inside the photo), from only the tiles they touch."""
    _, _, _, w, _, side, _, _ = _cog(path)
    out = np.empty((rows.stop - rows.start, cols.stop - cols.start, 4), np.uint8)
    across = -(-w // side)
    for tr in range(rows.start // side, (rows.stop - 1) // side + 1):
        for tc in range(cols.start // side, (cols.stop - 1) // side + 1):
            a0, a1 = max(tr * side, rows.start), min((tr + 1) * side, rows.stop)
            b0, b1 = max(tc * side, cols.start), min((tc + 1) * side, cols.stop)
            tile = _cog_tile(path, tr * across + tc)
            out[a0 - rows.start : a1 - rows.start, b0 - cols.start : b1 - cols.start] = tile[
                a0 - tr * side : a1 - tr * side, b0 - tc * side : b1 - tc * side
            ]
    return out


def crop(
    layer_blocks: list[Block], layer: str, left: float, top: float, res: float
) -> np.ndarray | None:
    """The CUT m square of `layer` with top-left corner (left, top) from every block it
    overlaps, or None where the blocks leave part of it uncovered or without data."""
    k = round(CUT / res)
    out = np.zeros((k, k, 3), np.uint8)
    covered = np.zeros((k, k), bool)
    for x0, y0, x1, y1, path in layer_blocks:
        c0, c1 = max(0, round((x0 - left) / res)), min(k, round((x1 - left) / res))
        r0, r1 = max(0, round((top - y1) / res)), min(k, round((top - y0) / res))
        if c0 >= c1 or r0 >= r1:
            continue
        bc, br = round((left - x0) / res), round((y1 - top) / res)  # crop origin in the block
        rows, cols = slice(br + r0, br + r1), slice(bc + c0, bc + c1)
        if path.suffix == ".tif":
            part = _cog_window(path, rows, cols)[..., FOUR_BANDS[layer]]
            # No data is 0 in every band; inside the photos no band is 0 (Ortofoto product
            # description, 3.1).
            out[r0:r1, c0:c1] = part
            covered[r0:r1, c0:c1] |= part.any(axis=2)
        else:
            out[r0:r1, c0:c1] = _block(path)[rows, cols]
            covered[r0:r1, c0:c1] = True
    return out if covered.all() else None


def measure(
    rgb: np.ndarray, ir: np.ndarray, x0: float, y0: float, x: float, y: float, area: float
) -> dict[str, float]:
    """ndvi_p25 and right_angles of the crops cut around (x0, y0) (`corner`), inside a circle of
    `area` m² around (x, y) widened by BUFFER, and for right_angles the RING beyond it too."""
    radius = math.sqrt(area / math.pi) + BUFFER
    masks = {}
    for layer, image in (("rgb", rgb), ("ir", ir)):
        k = image.shape[0]
        res = CUT / k
        left, top = corner(x0, y0, res)
        yy, xx = np.mgrid[0:k, 0:k]
        d2 = (left + (xx + 0.5) * res - x) ** 2 + (top - (yy + 0.5) * res - y) ** 2
        masks[layer] = d2 <= radius**2, d2 <= (radius + RING) ** 2
    f = ir.astype(float)
    ndvi = (f[..., 0] - f[..., 1]) / np.maximum(f[..., 0] + f[..., 1], 1)  # NIR, red, green
    # Buildings: straight edges at right angles. Gradient angles taken four times over fold the
    # four sides of a rectangle onto one direction; the weighted mean's length is near 1 for
    # them and near 0 for rock and foliage, whose edges run every way.
    bright = rgb.astype(float).mean(axis=2)
    gy, gx = np.gradient(ndimage.gaussian_filter(bright, 1.0))
    near = masks["rgb"][1]
    m, angle = np.hypot(gx, gy)[near], np.arctan2(gy, gx)[near]
    return {
        "ndvi_p25": float(np.percentile(ndvi[masks["ir"][0]], 25)),
        "right_angles": float(np.abs((m * np.exp(4j * angle)).sum()) / max(m.sum(), 1e-9)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=Path, default=ROOT / "data" / "units")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "orthophotos")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--rerun", action="store_true", help="measure every candidate again")
    args = parser.parse_args(argv)
    found, res = blocks(args.source)
    if not all(found.values()):
        print(f"no photos in {args.source}, skipped")
        return
    start = time.perf_counter()
    path = args.out / "photo_measures.parquet"
    earlier = (
        {}
        if args.rerun or not path.exists()
        else {r.pop("candidate_id"): r for r in pq.read_table(path).to_pylist()}
    )
    since = path.stat().st_mtime if earlier else 0.0
    tables = (
        sorted(args.candidates.glob("*.parquet")) if args.candidates.is_dir() else [args.candidates]
    )
    ids, measures, rows = [], [], []
    for t in tables:
        ran = t.stat().st_mtime > since
        columns = ["candidate_id", "x", "y", "area_m2", "dropped", "masked"]
        for r in pq.read_table(t, columns=columns).to_pylist():
            if r["dropped"] or r["masked"]:
                continue
            if not ran and r["candidate_id"] in earlier:
                ids.append(r["candidate_id"])
                measures.append(earlier[r["candidate_id"]])
            else:
                rows.append(r)
    kept = len(ids)
    rows.sort(key=lambda r: r["candidate_id"])  # tile by tile, so each block is decoded about once
    photos = args.out / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    for r in rows:
        rgb, ir = (
            crop(found[layer], layer, *corner(r["x"], r["y"], res[layer]), res[layer])
            for layer in RES
        )
        if rgb is None or ir is None:
            continue
        for layer, image in (("rgb", rgb), ("ir", ir)):
            (photos / f"{r['candidate_id']}_{layer}.png").write_bytes(png(image))
        ids.append(r["candidate_id"])
        area = r["area_m2"]
        # No P_obj to size the circle: the photo says nothing.
        if math.isnan(area):
            measures.append({"ndvi_p25": math.nan, "right_angles": math.nan})
        else:
            measures.append(measure(rgb, ir, r["x"], r["y"], r["x"], r["y"], area))
    out = pa.table(
        {
            "candidate_id": ids,
            **{
                n: pa.array([m[n] for m in measures], pa.float64())
                for n in ("ndvi_p25", "right_angles")
            },
        }
    )
    pq.write_table(out, path)
    seen = sum(not math.isnan(m["ndvi_p25"]) for m in measures[kept:])
    print(
        f"{len(ids) - kept} of {len(rows)} kept candidates to measure cropped, {seen} seen by the"
        f" photos; {kept} measured earlier, {time.perf_counter() - start:.0f} s -> {photos}, {path}"
    )


if __name__ == "__main__":
    main()
