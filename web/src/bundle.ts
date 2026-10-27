// The export is bundled per tile: tiles.json lists the tiles with the [offset,
// length] of each one's head (its boulders' GeoJSON) in tiles/{tile}.boulders.bin, and a
// boulder's `patch` and `photo` are its ranges in that file and in tiles/{tile}.photos.bin.
export type Range = [number, number];

// One range of a file, by HTTP range request; a server that ignores the range sends the whole
// file, which is cut here.
export async function readRange(url: string, [offset, length]: Range): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  return res.status === 206 ? buf : buf.slice(offset, offset + length);
}
