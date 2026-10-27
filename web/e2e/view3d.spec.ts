import { expect, type Route, test } from "@playwright/test";
import { boulders, serve } from "./export";

test("clicking the synthetic block shows its values and opens its 3D view", async ({ page }) => {
  const dir = process.env.SYNTHETIC_EXPORT!;
  // The block sits at the tile origin + (50, 50) m (tests/synthetic.py); ids end in _x_y.
  const block = boulders(dir).filter((f) => {
    const [x, y] = f.properties.candidate_id.split("_").slice(-2).map(Number);
    return Math.hypot(x - 497050, y - 6925050) < 3;
  });
  expect(block).toHaveLength(1);
  await serve(page, block, dir);
  // First a failed load, below: the block's patch is the served file's first bytes.
  const fail = (r: Route) => (r.request().headers().range?.startsWith("bytes=0-") ? r.abort() : r.fallback());
  await page.route("**/tiles/e2e.boulders.bin", fail);
  await page.route("https://tile.openstreetmap.org/**", (r) => r.abort()); // no network in tests
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");

  // A click on the marker (found by its green: score ≥ 5) opens the info panel with the
  // candidate's exported values, rounded to 2 decimals.
  const marker = await page.evaluate(async (src) => {
    const img = new Image();
    img.src = src;
    await img.decode();
    const ctx = Object.assign(document.createElement("canvas"), {
      width: img.width,
      height: img.height,
    }).getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, img.width, img.height).data;
    let [sx, sy, n] = [0, 0, 0];
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 1] > 120 && d[i] < 90 && d[i + 1] > d[i + 2] + 40) {
        sx += (i / 4) % img.width;
        sy += Math.floor(i / 4 / img.width);
        n++;
      }
    }
    return { x: sx / n, y: sy / n, n };
  }, `data:image/png;base64,${(await page.screenshot()).toString("base64")}`);
  expect(marker.n).toBeGreaterThan(50);
  await page.mouse.click(marker.x, marker.y);
  const info = page.locator("#info");
  await expect(info).toBeVisible();
  const props = block[0].properties;
  await expect(info.locator("b")).toHaveText(props.candidate_id);
  const keys = ["score", "area_m2", "max_h"] as const;
  await expect(info.locator("tr")).toHaveCount(keys.length);
  for (const k of keys) {
    expect(typeof props[k], k).toBe("number");
    await expect(info.locator(`tr[data-prop=${k}] td`), k).toHaveText(props[k]!.toFixed(2));
  }
  await page.screenshot({ path: "test-results/info.png" });

  // A patch that fails to load shows the error in the 3D panel, without a scene.
  await info.getByRole("button", { name: "3D view" }).click();
  await expect(page.locator("#view3d")).toBeVisible();
  await expect(page.locator("#caption3d")).toContainText("could not load patch");
  await expect(page.locator("#view3d canvas")).toBeHidden();
  await page.screenshot({ path: "test-results/view3d-error.png" });
  await page.locator("#close3d").click();

  await page.unroute("**/tiles/e2e.boulders.bin", fail);
  await info.getByRole("button", { name: "3D view" }).click();
  await expect(page.locator("#view3d canvas")).toBeVisible();
  const caption = await page.locator("#caption3d").textContent();
  const counts = /(\d+) ground points · (\d+) rock points · (\d+) rock triangles · (\d+) wall quads · (\d+) loose/;
  const [, ground, rock, triangles, walls, loose] = caption!.match(counts)!.map(Number);
  expect(ground).toBeGreaterThan(100);
  expect(rock).toBeGreaterThan(50);
  expect(triangles).toBeGreaterThan(rock); // a filled top: ~2 triangles per point
  expect(walls).toBeGreaterThan(10); // the outline drops to the ground all round
  expect(loose).toBe(0);
  await page.screenshot({ path: "test-results/view3d.png" });
  const png = await page.locator("#view3d").screenshot();

  // The panel's screenshot shows each layer: count pixels by colour below the caption.
  const pixels = await page.evaluate(async (src) => {
    const img = new Image();
    img.src = src;
    await img.decode();
    const ctx = Object.assign(document.createElement("canvas"), {
      width: img.width,
      height: img.height,
    }).getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const top = 40;
    const d = ctx.getImageData(0, top, img.width, img.height - top).data;
    const n = { rock: 0, ground: 0, all: d.length / 4 };
    for (let i = 0; i < d.length; i += 4) {
      const [r, g, b] = [d[i], d[i + 1], d[i + 2]];
      const grey = Math.abs(r - g) < 15 && Math.abs(g - b) < 15;
      if (grey && r > 60 && r < 200) n.rock++; // rock surface and walls, 0x8e8b86
      else if (g > r + 10 && g > b + 20 && g > 120) n.ground++; // green mesh, 0x7f9f5f
    }
    return n;
  }, `data:image/png;base64,${png.toString("base64")}`);
  expect(pixels.ground / pixels.all).toBeGreaterThan(0.1);
  expect(pixels.rock / pixels.all).toBeGreaterThan(0.005);
});

test("a missing tile file is a 404, not the index page", async ({ request }) => {
  expect((await request.get("/tiles/no_such_tile.boulders.bin")).status()).toBe(404);
});
