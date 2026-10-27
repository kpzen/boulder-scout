import { expect, type Page, test } from "@playwright/test";
import { boulders, serve } from "./export";

// The 16 real fixture crops' export (e2e/synthetic-setup.ts). "N of M shown" counts the
// candidates drawn inside the view, so it equals the slider-passing count only when the view
// holds every one of them; the scale bar's text changes with the zoom.
// Height and area start at 2 m and 8 m² (null passes), so some fixture candidates start hidden.
const all = boulders(process.env.FIXTURE_EXPORT!);
const total = all.length;
const passing = (min: number) =>
  all.filter(
    ({ properties: p }) =>
      p.score >= min && (p.max_h ?? 2) >= 2 && (p.area_m2 ?? 8) >= 8,
  ).length;
const start = passing(-Infinity);

async function zoomIn(page: Page, times: number): Promise<void> {
  for (let i = 0; i < times; i++) await page.locator(".maplibregl-ctrl-zoom-in").click();
}

// The scale bar's text once the zoom animation has ended (unchanged for 0.5 s).
async function settledScale(page: Page): Promise<string> {
  const scale = page.locator(".maplibregl-ctrl-scale");
  let last = "";
  for (;;) {
    const text = (await scale.textContent())!;
    if (text === last) return text;
    last = text;
    await page.waitForTimeout(500);
  }
}

test("zoom buttons change the view and fit frames every shown candidate", async ({ page }) => {
  await serve(page, all);
  await page.route("https://tile.openstreetmap.org/**", (r) => r.abort()); // no network in tests
  await page.route("https://ibasemaps-api.arcgis.com/**", (r) => r.abort());
  await page.goto("/");
  const count = page.locator("#count");
  const fit = page.getByRole("button", { name: "Fit to shown candidates" });
  const score = page.locator("input[name=score]");

  // The first view frames every candidate; score starts at its minimum.
  await expect(count).toHaveText(`${start} of ${total} shown`);
  const first = await settledScale(page);

  await zoomIn(page, 3);
  const zoomed = await settledScale(page);
  expect(zoomed).not.toBe(first);
  await expect.poll(async () => parseInt((await count.textContent())!)).toBeLessThan(start);
  await page.locator(".maplibregl-ctrl-zoom-out").click();
  expect(await settledScale(page)).not.toBe(zoomed);

  await fit.click();
  await expect(count).toHaveText(`${start} of ${total} shown`);

  // After the score slider moves, fit frames the new set from a zoomed-in view.
  for (const min of [Number(await score.getAttribute("min")), 6]) {
    await zoomIn(page, 3);
    await score.fill(String(min));
    await expect.poll(async () => parseInt((await count.textContent())!)).toBeLessThan(passing(min));
    await fit.click();
    await expect(count).toHaveText(`${passing(min)} of ${total} shown`);
  }
  expect(passing(6)).toBeLessThan(start);
  await page.screenshot({ path: "test-results/navigation.png" });
});
