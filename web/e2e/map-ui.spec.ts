import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import { boulders, serve } from "./export";

// One high-scoring fixture candidate, at the centre of the padded map (see review.spec.ts).
const all = boulders(process.env.FIXTURE_EXPORT!);
const one = all.find((f) => f.properties.score >= 5)!;
const centre = { x: (260 + 1280 - 80) / 2, y: 48 + (720 - 48 - 8) / 2 };

test.beforeEach(async ({ page }) => {
  await serve(page, [one], process.env.FIXTURE_EXPORT);
  await page.route("https://tile.openstreetmap.org/**", (r) => r.abort()); // no network in tests
});

// The pixels of one colour in a screenshot (green: a score ≥ 5 pin, blue: a saved pin, red: the
// test's imagery tile): their count, centroid and lowest row.
async function pixels(page: Page, colour: "green" | "blue" | "red") {
  return page.evaluate(async ([src, colour]) => {
    const img = new Image();
    img.src = src;
    await img.decode();
    const ctx = Object.assign(document.createElement("canvas"), {
      width: img.width,
      height: img.height,
    }).getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, img.width, img.height).data;
    let [sx, sy, n, bottom] = [0, 0, 0, 0];
    for (let i = 0; i < d.length; i += 4) {
      const [r, g, b] = [d[i], d[i + 1], d[i + 2]];
      const hit = {
        green: g > 120 && r < 90 && g > b + 40,
        blue: b > 200 && r < 80 && g < 160,
        red: r > 200 && g < 60 && b < 60,
      }[colour];
      if (hit) {
        const [x, y] = [(i / 4) % img.width, Math.floor(i / 4 / img.width)];
        [sx, sy, n, bottom] = [sx + x, sy + y, n + 1, Math.max(bottom, y)];
      }
    }
    return { x: sx / n, y: sy / n, n, bottom, share: n / (img.width * img.height) };
  }, [`data:image/png;base64,${(await page.screenshot()).toString("base64")}`, colour] as const);
}

test("a green pin points at its candidate from just above it, and opens it", async ({ page }) => {
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");
  const pin = await pixels(page, "green");
  expect(pin.n).toBeGreaterThan(100); // score ≥ 5 is green
  expect(Math.abs(pin.x - centre.x)).toBeLessThan(2);
  expect(pin.bottom).toBeLessThan(centre.y - 2); // the tip stops short of the candidate
  expect(pin.bottom).toBeGreaterThan(centre.y - 12);
  await page.mouse.click(pin.x, pin.y);
  await expect(page.locator("#info b")).toHaveText(one.properties.candidate_id);
});

test("below zoom 12 a candidate is a dot on it, which opens it; from 12 a pin", async ({ page }) => {
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");
  // fit() frames the one candidate at z17; each click zooms by one once the last has ended.
  const zoom = async (button: "in" | "out", times: number) => {
    for (let i = 0; i < times; i++) {
      await page.locator(`.maplibregl-ctrl-zoom-${button}`).click();
      await page.waitForTimeout(600);
    }
  };
  await zoom("out", 6); // z11
  await expect.poll(async () => (await pixels(page, "green")).n).toBeGreaterThan(10);
  const dot = await pixels(page, "green");
  expect(dot.n).toBeLessThan(80); // a 4 px dot, not a pin
  await page.mouse.click(dot.x, dot.y);
  await expect(page.locator("#info b")).toHaveText(one.properties.candidate_id);

  await zoom("in", 1); // z12
  await expect.poll(async () => (await pixels(page, "green")).n).toBeGreaterThan(100);
});

test("of overlapping pins, a click opens the one drawn where it lands", async ({ page }) => {
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  // Two copies of the fixture 15 px apart at z18, where fit() frames them; the higher score
  // draws on top.
  const [lon, lat] = one.geometry.coordinates;
  const dLon = 15 / ((256 * 2 ** 18) / 360);
  const pin = (id: string, score: number, x: number) => ({
    ...one,
    geometry: { type: "Point", coordinates: [x, lat] },
    properties: { ...one.properties, candidate_id: id, score },
  });
  const features = [pin("top_1_2", 9, lon), pin("under_3_4", 3, lon + dLon)];
  await serve(page, features);
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("2 of 2 shown");
  // fit() centres the pair; the heads are ~29 px above the candidates (icon-size ≈ 1.03).
  // The top pin is the green one (score 9); fit() centres the pair, so the other is its mirror.
  const top = await pixels(page, "green");
  expect(top.n).toBeGreaterThan(100);
  const [topX, underX, headY] = [top.x, 2 * centre.x - top.x, top.y - 2];
  expect(underX - topX).toBeGreaterThan(10);
  expect(underX - topX).toBeLessThan(20);
  await page.mouse.click(topX + 9, headY); // on both heads: the top one
  await expect(page.locator("#info b")).toHaveText("top_1_2");
  await page.mouse.click(underX + 8, headY); // on the lower head only
  await expect(page.locator("#info b")).toHaveText("under_3_4");
  await page.mouse.click(topX - 8, headY); // on the top head only
  await expect(page.locator("#info b")).toHaveText("top_1_2");
});

test("z19 imagery loads where Esri has it; elsewhere z18 is enlarged", async ({ page }) => {
  await page.goto("about:blank");
  const red = await page.evaluate(() => {
    const canvas = Object.assign(document.createElement("canvas"), { width: 256, height: 256 });
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#f00";
    ctx.fillRect(0, 0, 256, 256);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  for (const has19 of [false, true]) {
    const zooms = new Set<number>();
    await page.unroute("https://ibasemaps-api.arcgis.com/**");
    await page.route("https://ibasemaps-api.arcgis.com/**", (r) => {
      const parts = new URL(r.request().url()).pathname.split("/");
      if (parts.includes("tilemap")) return r.fulfill({ json: { data: Array(32 * 32).fill(has19 ? 1 : 0) } });
      zooms.add(Number(parts.at(-3)));
      return r.fulfill({ contentType: "image/png", body: Buffer.from(red, "base64") });
    });
    await page.goto("/");
    await expect(page.locator("#count")).toHaveText("1 of 1 shown");
    for (let i = 0; i < 4; i++) await page.locator(".maplibregl-ctrl-zoom-in").click(); // 17 → 21
    await expect.poll(() => zooms.has(18)).toBe(true);
    await page.waitForTimeout(1500);
    expect(Math.max(...zooms)).toBe(has19 ? 19 : 18);
    // Either way the imagery covers the map: no gap where a z19 tile is missing.
    expect((await pixels(page, "red")).share).toBeGreaterThan(0.7);
  }
});

test("3D view and aerial photos close on a second click; saved boulders export as KML", async ({ page }) => {
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");
  const head = { x: centre.x, y: centre.y - 25 };
  await page.mouse.click(head.x, head.y);
  const info = page.locator("#info");
  for (const [name, panel] of [["3D view", "#view3d"], ["Aerial photos", "#photos"]]) {
    await info.getByRole("button", { name }).click();
    await expect(page.locator(panel)).toBeVisible();
    await info.getByRole("button", { name }).click();
    await expect(page.locator(panel)).toBeHidden();
  }

  const exportButton = page.locator("#export");
  await expect(exportButton).toBeDisabled();
  await expect(exportButton).toHaveText("Export as KML");
  await info.getByRole("button", { name: "Save" }).click();
  await expect(info.locator("#save")).toHaveText("Saved");
  await expect(exportButton).toHaveText("Export as KML (1)");
  // A saved pin is blue, not its score's green.
  await expect.poll(async () => (await pixels(page, "blue")).n).toBeGreaterThan(100);
  expect((await pixels(page, "green")).n).toBe(0);
  await page.screenshot({ path: "test-results/saved.png" });

  // Saved survives a reload (localStorage).
  await page.reload();
  await expect(page.locator("#count")).toHaveText("1 of 1 shown");
  await page.mouse.click(head.x, head.y);
  await expect(info.locator("#save")).toHaveText("Saved");

  const [download] = await Promise.all([page.waitForEvent("download"), exportButton.click()]);
  expect(download.suggestedFilename()).toBe("boulder-scout.kml");
  const kml = readFileSync(await download.path(), "utf8");
  const p = one.properties;
  const [lon, lat] = one.geometry.coordinates;
  expect(kml.match(/<Placemark>/g)).toHaveLength(1);
  expect(kml).toContain(`<name>${p.max_h!.toFixed(1)} m high · ${p.area_m2!.toFixed(0)} m²</name>`);
  expect(kml).toContain(`<coordinates>${lon.toFixed(7)},${lat.toFixed(7)}</coordinates>`);
  expect(kml).toContain(p.candidate_id);
});
