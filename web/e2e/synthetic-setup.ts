import { execFileSync } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

// Run the real pipeline on the synthetic tile (tests/synthetic.py) and on the fixture crops into
// temp public exports; tests find them through SYNTHETIC_EXPORT and FIXTURE_EXPORT, which
// Playwright passes on to its workers.
export default function setup(): void {
  const dir = mkdtempSync(resolve(tmpdir(), "boulder-scout-e2e-"));
  const root = resolve(import.meta.dirname, "../..");
  const py = (...args: string[]) =>
    execFileSync("uv", ["run", "python", ...args], { cwd: root, stdio: "inherit" });
  py(
    "-c",
    "import sys; from pathlib import Path; from tests.synthetic import make_synthetic_tile; make_synthetic_tile(Path(sys.argv[1]))",
    `${dir}/t.laz`,
  );
  py("-m", "pipeline.run_unit", `${dir}/t.laz`, "--out", `${dir}/units/t.parquet`,
    "--patches", `${dir}/unit_patches`);
  py("-m", "scoring.export_geojson", "--candidates", `${dir}/units`,
    "--patches", `${dir}/unit_patches`, "--out", dir, "--min-height", "0", "--min-area", "0");
  process.env.SYNTHETIC_EXPORT = dir;
  // The 16 real fixture crops (tests/fixtures), exported as for the author's local viewer.
  py("scripts/export_fixtures.py", "--data", `${dir}/fixtures`);
  process.env.FIXTURE_EXPORT = `${dir}/fixtures/public_export`;
}
