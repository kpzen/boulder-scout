"""Read Lantmäteriet's Markhöjdmodell (1 m ground model, GeoTIFF/COG) over a bounding box.

Stdlib and numpy only: the files are little-endian classic TIFFs, tiled 512 x 512, one float32
band, deflate with the floating-point predictor (3), pixel-is-area, EPSG:3006 plan coordinates
(RH 2000 heights), nodata -9999. Only the tiles that touch the box are read and inflated. Any
other layout raises ValueError rather than being misread.
"""

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.area import Bbox

# TIFF tags (baseline, then GeoTIFF and GDAL's nodata)
WIDTH, HEIGHT, BITS, COMPRESSION, SAMPLES, PREDICTOR = 256, 257, 258, 259, 277, 317
TILE_W, TILE_H, TILE_OFFSETS, TILE_COUNTS, SAMPLE_FORMAT = 322, 323, 324, 325, 339
PIXEL_SCALE, TIEPOINT, GEO_KEYS, NODATA = 33550, 33922, 34735, 42113
TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 12: "d", 16: "Q"}  # TIFF type -> struct code


@dataclass(frozen=True)
class Ground:
    """A ground model grid: values[0, 0] is the north-west cell; NaN where unknown."""

    values: np.ndarray  # float32, (rows, cols)
    x0: float  # west edge, EPSG:3006
    y1: float  # north edge
    cell: float  # m


def _tags(f: "_File") -> dict[int, tuple]:  # type: ignore[type-arg]
    head = f.read(0, 8)
    if head[:4] != b"II*\x00":
        raise ValueError(f"{f.path.name}: not a little-endian classic TIFF")
    (ifd,) = struct.unpack_from("<I", head, 4)
    (n,) = struct.unpack("<H", f.read(ifd, 2))
    entries = f.read(ifd + 2, 12 * n)
    tags = {}
    for k in range(n):
        tag, typ, count, value = struct.unpack_from("<HHI4s", entries, 12 * k)
        code = TYPES.get(typ)
        if code is None:
            continue  # a type none of the tags read here uses
        size = struct.calcsize("<" + code) * count
        raw = value[:size] if size <= 4 else f.read(struct.unpack("<I", value)[0], size)
        tags[tag] = (
            (raw.rstrip(b"\0").decode(),) if code == "s" else struct.unpack(f"<{count}{code}", raw)
        )
    return tags


class _File:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = open(path, "rb")  # noqa: SIM115 - closed by read_dtm

    def read(self, offset: int, size: int) -> bytes:
        self.handle.seek(offset)
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError(f"{self.path.name}: truncated at byte {offset + len(data)}")
        return data


def _read_window(path: Path, out: Ground) -> None:
    """Fill `out` (NaN where still unknown) with `path`'s cells that fall in it."""
    f = _File(path)
    try:
        t = _tags(f)
        layout = (t[BITS], t[SAMPLE_FORMAT], t.get(SAMPLES, (1,)), t[COMPRESSION])
        if layout[:3] != ((32,), (3,), (1,)) or layout[3] != (8,):
            raise ValueError(f"{path.name}: not one deflated float32 band {layout}")
        predictor = t.get(PREDICTOR, (1,))[0]
        if predictor != 3 or TILE_W not in t:
            raise ValueError(f"{path.name}: predictor {predictor} or untiled layout unsupported")
        keys = t.get(GEO_KEYS, ())
        geo = {keys[k]: keys[k + 3] for k in range(4, len(keys), 4)}  # key -> inline value
        if geo.get(1025) != 1 or geo.get(3072) != 3006:  # pixel-is-area, SWEREF 99 TM
            raise ValueError(f"{path.name}: not pixel-is-area EPSG:3006 ({geo})")
        sx, sy = t[PIXEL_SCALE][:2]
        if (sx, sy) != (out.cell, out.cell):
            raise ValueError(f"{path.name}: {sx} x {sy} m cells, expected {out.cell} m")
        i, j, _, X0, Y1, _ = t[TIEPOINT]
        X0, Y1 = X0 - i * sx, Y1 + j * sy
        nodata = float(t[NODATA][0]) if NODATA in t else np.nan
        width, height, tw, th = t[WIDTH][0], t[HEIGHT][0], t[TILE_W][0], t[TILE_H][0]
        offsets, counts = t[TILE_OFFSETS], t[TILE_COUNTS]
        # File pixel (r, c) is out's cell (r + dr, c + dc); out's grid must be the file's.
        dr, dc = (out.y1 - Y1) / out.cell, (X0 - out.x0) / out.cell
        if dr != round(dr) or dc != round(dc):
            raise ValueError(f"{path.name}: grid not aligned to whole {out.cell} m cells")
        dr, dc = round(dr), round(dc)
        rows, cols = out.values.shape
        r0, r1 = max(-dr, 0), min(rows - dr, height)
        c0, c1 = max(-dc, 0), min(cols - dc, width)
        across = -(-width // tw)
        for tr in range(r0 // th, -(-r1 // th) if r1 > r0 else 0):
            for tc in range(c0 // tw, -(-c1 // tw) if c1 > c0 else 0):
                k = tr * across + tc
                raw = zlib.decompress(f.read(offsets[k], counts[k]))
                # Adobe TN3: byte planes, most significant first, differenced
                planes = np.cumsum(
                    np.frombuffer(raw, np.uint8).reshape(th, 4 * tw), axis=1, dtype=np.uint8
                )
                tile = planes.reshape(th, 4, tw).transpose(0, 2, 1).copy().view(">f4")[..., 0]
                a0, a1 = max(tr * th, r0), min((tr + 1) * th, r1)
                b0, b1 = max(tc * tw, c0), min((tc + 1) * tw, c1)
                part = tile[a0 - tr * th : a1 - tr * th, b0 - tc * tw : b1 - tc * tw]
                part = np.where(part == nodata, np.nan, part).astype(np.float32)
                dest = out.values[a0 + dr : a1 + dr, b0 + dc : b1 + dc]
                unknown = np.isnan(dest)
                dest[unknown] = part[unknown]
    finally:
        f.handle.close()


def read_dtm(paths: Sequence[Path], bbox: Bbox, cell: float = 1.0) -> Ground:
    """The ground model over bbox (EPSG:3006, snapped outward to whole cells) from whichever of
    `paths` cover it; NaN where none does or the file has no data."""
    x0, y0 = np.floor(bbox[0] / cell) * cell, np.floor(bbox[1] / cell) * cell
    x1, y1 = np.ceil(bbox[2] / cell) * cell, np.ceil(bbox[3] / cell) * cell
    out = Ground(
        np.full((round((y1 - y0) / cell), round((x1 - x0) / cell)), np.nan, np.float32),
        float(x0),
        float(y1),
        cell,
    )
    for path in paths:
        _read_window(path, out)
    return out
