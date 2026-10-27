import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { toSweref, toWgs84 } from "../src/sweref";

const AREA = process.env.AREA_FILE!;
const UNITS = process.env.UNITS_DIR!;
const root = resolve(import.meta.dirname, "../..");
const python = (code: string, ...args: string[]) =>
  execFileSync("uv", ["run", "python", "-c", code, ...args], { cwd: root, encoding: "utf-8" });

test("the page's SWEREF 99 TM conversion matches pyproj's", () => {
  const points: [number, number][] = [[14.95158, 62.43372], [11.2, 58.3], [23.5, 67.9], [15, 55.4]];
  const expected: [number, number][] = JSON.parse(
    python(
      "import json, sys; from pyproj import Transformer; t = Transformer.from_crs(4326, 3006, always_xy=True);" +
        " print(json.dumps([t.transform(*p) for p in json.loads(sys.argv[1])]))",
      JSON.stringify(points),
    ),
  );
  points.forEach((p, i) => {
    const [e, n] = toSweref(p);
    expect(Math.abs(e - expected[i][0])).toBeLessThan(0.001);
    expect(Math.abs(n - expected[i][1])).toBeLessThan(0.001);
    const [lon, lat] = toWgs84(expected[i]);
    expect(Math.abs(lon - p[0])).toBeLessThan(1e-8);
    expect(Math.abs(lat - p[1])).toBeLessThan(1e-8);
  });
});

test("drawn areas save as run_area.py's polygons, count their tiles and keep run tiles shown", async ({ page }) => {
  rmSync(AREA, { force: true });
  rmSync(UNITS, { recursive: true, force: true });
  mkdirSync(UNITS);
  await page.goto("/area.html");
  const summary = page.locator("#summary");
  const fresh = async () => Number((await summary.textContent())!.split(" ")[0]);
  await expect(summary).toHaveText("0 new tiles · about 0.0 h · 0 run already");
  // Two small triangles side by side in the middle of Sweden's view: some dozens of tiles each.
  const box = (await page.locator("#app").boundingBox())!;
  const [cx, cy] = [box.x + box.width / 2, box.y + box.height / 2];
  for (const [dx, dy] of [[0, 0], [14, 0], [7, 12], [0, 0]]) await page.mouse.click(cx + dx, cy + dy);
  await expect(page.locator("#hint")).toHaveText(/^Click on the map to start an area/); // closed
  const first = await fresh();
  expect(first).toBeGreaterThan(10);
  for (const [dx, dy] of [[40, 0], [54, 0], [47, 12]]) await page.mouse.click(cx + dx, cy + dy);
  await page.keyboard.press("Enter");
  await expect.poll(fresh).toBeGreaterThan(first + 10); // a second area, not the first stretched
  const both = await fresh();
  // Dragging a corner moves it and adds none.
  await page.mouse.move(cx + 47, cy + 12);
  await page.mouse.down();
  await page.mouse.move(cx + 47, cy + 30, { steps: 5 });
  await page.mouse.up();
  await expect.poll(fresh).toBeGreaterThan(both);
  const tiles = await fresh();
  await page.getByRole("button", { name: "Save area" }).click();
  await expect(page.locator("#status")).toHaveText("Saved to data/area.geojson.");

  // The pipeline takes the same tiles from the saved file: two areas of three corners.
  const saved = JSON.parse(readFileSync(AREA, "utf-8")).features[0].geometry;
  expect(saved.type).toBe("MultiPolygon");
  expect(saved.coordinates.map((p: number[][][]) => p[0].length)).toEqual([4, 4]);
  const corners: [number, number][] = JSON.parse(
    python(
      "import json, sys; from pipeline.area import polygon_tiles;" +
        " print(json.dumps([[round(t[0]), round(t[1])] for t in polygon_tiles(json.load(open(sys.argv[1])))]))",
      AREA,
    ),
  );
  expect(corners).toHaveLength(tiles);

  // One tile gets a unit table: the reopened page shows the saved areas with it as run, and
  // Clear removes the areas but not the run tile.
  const [e, n] = corners[0];
  writeFileSync(resolve(UNITS, `y20e018-${n / 100}_${e / 100}_25.parquet`), "");
  await page.reload();
  await expect(summary).toHaveText(new RegExp(`^${tiles - 1} new tiles · .* · 1 run already$`));
  await expect(page.locator("#status")).toHaveText("Opened the saved area, data/area.geojson.");
  await page.getByRole("button", { name: "Clear" }).click();
  await expect(summary).toHaveText("0 new tiles · about 0.0 h · 1 run already");
});
