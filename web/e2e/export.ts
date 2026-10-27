import { readFileSync } from "node:fs";
import type { Page } from "@playwright/test";

// The export bundled per tile: tiles.json, and per tile a boulders file of patches and
// then the head, the tile's GeoJSON, whose `patch` and `photo` are [offset, length] ranges.
type Range = [number, number];
export type Feature = {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: { candidate_id: string; score: number; max_h: number | null; area_m2: number | null; tile: string; patch?: Range; photo?: Range };
};

// Every boulder of the export in dir, with the tile it is in, as the map reads them.
export function boulders(dir: string): Feature[] {
  const tiles: { tile: string; head: Range }[] = JSON.parse(readFileSync(`${dir}/tiles.json`, "utf8"));
  return tiles.flatMap(({ tile, head: [offset, length] }) => {
    const head = readFileSync(`${dir}/tiles/${tile}.boulders.bin`).subarray(offset, offset + length);
    return JSON.parse(head.toString()).features.map((f: Feature) => ({ ...f, properties: { ...f.properties, tile } }));
  });
}

// Serve these boulders as the map's whole export, one tile "e2e" whose boulders file holds
// their patches, read from the export in dir, and their head; a range request gets its range.
export async function serve(page: Page, features: { properties: object }[], dir?: string): Promise<void> {
  const patches: Buffer[] = [];
  let at = 0;
  const shown = features.map((f) => {
    const { tile, patch, ...properties } = f.properties as { tile?: string; patch?: Range };
    if (!patch || !dir) return { ...f, properties };
    const bytes = readFileSync(`${dir}/tiles/${tile}.boulders.bin`).subarray(patch[0], patch[0] + patch[1]);
    patches.push(bytes);
    at += bytes.length;
    return { ...f, properties: { ...properties, patch: [at - bytes.length, bytes.length] } };
  });
  const head = Buffer.from(JSON.stringify({ type: "FeatureCollection", features: shown }));
  const file = Buffer.concat([...patches, head]);
  await page.route("**/tiles.json", (r) => r.fulfill({ json: [{ tile: "e2e", head: [at, head.length] }] }));
  await page.route("**/tiles/e2e.boulders.bin", (r) => {
    const range = /^bytes=(\d+)-(\d+)$/.exec(r.request().headers().range ?? "");
    if (!range) return r.fulfill({ body: file });
    return r.fulfill({ status: 206, body: file.subarray(Number(range[1]), Number(range[2]) + 1) });
  });
}
