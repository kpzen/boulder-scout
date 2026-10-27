import { expect, test } from "@playwright/test";
import { serve } from "./export";

// Five candidates near the known boulder; e has no height or area (null).
const rows: [string, number | null, number | null, number][] = [
  // id, max_h, area_m2, score
  ["a", 1.5, 5, 2.0],
  ["b", 3, 10, 2.5],
  ["c", 5, 30, 4],
  ["d", 8, 80, 6],
  ["e", null, null, 3],
];
const geojson = {
  type: "FeatureCollection",
  features: rows.map(([candidate_id, max_h, area_m2, score], i) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [14.944 + 0.001 * i, 62.457] },
    properties: { candidate_id, max_h, area_m2, score },
  })),
};

test("each slider hides candidates below its value", async ({ page }) => {
  await serve(page, geojson.features);
  await page.route("https://tile.openstreetmap.org/**", (r) => r.abort()); // no network in tests
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto("/");
  const count = page.locator("#count");
  // Only height, area and score have sliders. Height and area start at 2 m and 8 m², which
  // hides a; score starts at its minimum.
  await expect(page.locator("#filters input")).toHaveCount(3);
  await expect(page.locator("input[name=max_h]")).toHaveValue("2");
  await expect(page.locator("input[name=area_m2]")).toHaveValue("8");
  await expect(page.locator("input[name=score]")).toHaveValue("2");
  await expect(count).toHaveText("4 of 5 shown");

  // A minimum between b's and c's value leaves c, d and e, whose null always passes;
  // score is never null, so there e (score 3) goes too. Each slider then goes back.
  const moves: [string, string, number][] = [
    ["max_h", "4", 3],
    ["area_m2", "20", 3],
    ["score", "3.5", 2],
  ];
  for (const [name, value, shown] of moves) {
    const slider = page.locator(`input[name=${name}]`);
    const start = await slider.inputValue();
    await slider.fill(value);
    await expect(count, name).toHaveText(`${shown} of 5 shown`);
    await slider.fill(start);
    await expect(count, name).toHaveText("4 of 5 shown");
  }

  // Sliders combine: area ≥ 20 leaves c, d, e (a's height is under 2); score ≥ 3.5 then
  // leaves c and d.
  await page.locator("input[name=area_m2]").fill("20");
  await page.locator("input[name=score]").fill("3.5");
  await expect(count).toHaveText("2 of 5 shown");
  await page.screenshot({ path: "test-results/filters.png" });
});
