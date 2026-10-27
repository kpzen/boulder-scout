import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { type PreviewServer, preview } from "vite";
import { boulders } from "./export";

// The demo build (scripts/build_demo.py) of four of the fixture export's tiles, served as
// Pages serves it: static files under the sub-path, range requests answered with 206.
const BASE = "/boulder-scout-demo/";
const BBOX = ["495000", "6922500", "500000", "6927500"];
const exportDir = process.env.FIXTURE_EXPORT!;
const out = resolve(exportDir, "../demo");
let server: PreviewServer;

test.beforeAll(async () => {
  test.setTimeout(120_000);
  const root = resolve(import.meta.dirname, "../..");
  execFileSync("uv", ["run", "python", "scripts/build_demo.py", "--export", exportDir, "--bbox", ...BBOX,
    "--out", out, "--base", BASE], { cwd: root, stdio: "inherit", env: { ...process.env, VITE_ARCGIS_KEY: "e2e" } });
  server = await preview({ configFile: false, base: BASE, build: { outDir: out }, preview: { port: 4174 }, logLevel: "warn" });
});
test.afterAll(() => server?.close());

test("the demo build holds the chosen tiles unchanged and the map reads them under the base path", async ({ page }) => {
  // Only the tiles whose nominal tile lies in the bbox, byte for byte the export's.
  const chosen = ["b01_reference__20e029-69250_4950_25", "b02__20e018-69225_4950_25",
    "b04_valley_pair__20e018-69225_4975_25", "b05__20e029-69250_4975_25"];
  const all: { tile: string }[] = JSON.parse(readFileSync(`${exportDir}/tiles.json`, "utf8"));
  expect(JSON.parse(readFileSync(`${out}/tiles.json`, "utf8"))).toEqual(all.filter((t) => chosen.includes(t.tile)));
  const files = readdirSync(`${out}/tiles`);
  expect(files.sort()).toEqual(chosen.flatMap((t) => [`${t}.boulders.bin`, `${t}.photos.bin`]).sort());
  for (const f of files) expect(readFileSync(`${out}/tiles/${f}`).equals(readFileSync(`${exportDir}/tiles/${f}`)), f).toBe(true);
  expect(readdirSync(out)).toContain(".nojekyll");
  expect(readFileSync(`${out}/README.md`, "utf8")).toEqual(readFileSync(resolve(import.meta.dirname, "../demo/README.md"), "utf8"));

  // One boulder alone on the map (?ids=); ?review does nothing. The fixtures have no photo
  // crops (their photos files are empty), so the photo crops are checked on the v1 build.
  const demo = boulders(out);
  const one = demo.find((f) => f.properties.patch)!;
  const reads: { url: string; status: number }[] = [];
  page.on("response", (r) => {
    if (r.url().includes("/tiles")) reads.push({ url: new URL(r.url()).pathname, status: r.status() });
  });
  const requests: string[] = [];
  page.on("request", (r) => requests.push(r.url()));
  await page.route("https://tile.openstreetmap.org/**", (r) => r.abort()); // no network in tests
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto(`${server.resolvedUrls!.local[0]}?review&ids=${one.properties.candidate_id}`);
  await expect(page.locator("#count")).toHaveText(`1 of ${demo.length} shown`);
  await expect(page.locator("#review")).toHaveCount(0);
  await expect(page.locator("input[value=imagery]")).toBeChecked(); // the key is in the build

  // fitBounds puts a lone boulder at the middle of its padding (main.ts: 260 left, 80 right,
  // 60 top and bottom); the pin's head is about 28 px above it at any size.
  const { width } = page.viewportSize()!;
  const main = await page.locator("#main").boundingBox();
  await page.mouse.click(260 + (width - 340) / 2, main!.y + 60 + (main!.height - 120) / 2 - 28);
  const info = page.locator("#info");
  await expect(info.locator("b")).toHaveText(one.properties.candidate_id);
  await info.getByRole("button", { name: "3D view" }).click();
  await expect(page.locator("#view3d canvas")).toBeVisible();
  await expect(page.locator("#caption3d")).not.toContainText("could not load");
  await page.screenshot({ path: "test-results/demo.png" });

  expect(reads.map((r) => r.url)).toContain(`${BASE}tiles.json`);
  for (const r of reads.filter((r) => r.url.endsWith(".bin"))) {
    expect(r.url.startsWith(`${BASE}tiles/`), r.url).toBe(true);
    expect(r.status, r.url).toBe(206);
  }
  expect(reads.filter((r) => r.url.endsWith(".boulders.bin")).length).toBeGreaterThan(chosen.length); // heads + patch
  expect(requests.filter((u) => u.includes("labels.json"))).toEqual([]);
});
