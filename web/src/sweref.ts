// SWEREF 99 TM (EPSG:3006) <-> WGS84 for the area page's tile preview only: which tiles a
// run takes is pipeline/area.py's. Lantmäteriet's Gauss-Krüger formulas on GRS 80,
// central meridian 15° E, scale 0.9996, false easting 500 000 m; good to well under a millimetre.
const A = 6378137;
const F = 1 / 298.257222101;
const LON0 = (15 * Math.PI) / 180;
const K0 = 0.9996;
const FE = 500000;

const E2 = F * (2 - F);
const N = F / (2 - F);
const AR = (A / (1 + N)) * (1 + N ** 2 / 4 + N ** 4 / 64);
const BETA = [
  N / 2 - (2 * N ** 2) / 3 + (5 * N ** 3) / 16 + (41 * N ** 4) / 180,
  (13 * N ** 2) / 48 - (3 * N ** 3) / 5 + (557 * N ** 4) / 1440,
  (61 * N ** 3) / 240 - (103 * N ** 4) / 140,
  (49561 * N ** 4) / 161280,
];
const DELTA = [
  N / 2 - (2 * N ** 2) / 3 + (37 * N ** 3) / 96 - N ** 4 / 360,
  N ** 2 / 48 + N ** 3 / 15 - (437 * N ** 4) / 1440,
  (17 * N ** 3) / 480 - (37 * N ** 4) / 840,
  (4397 * N ** 4) / 161280,
];

/** [lon, lat] in degrees to [easting, northing] in metres. */
export function toSweref([lon, lat]: [number, number]): [number, number] {
  const phi = (lat * Math.PI) / 180;
  const s2 = Math.sin(phi) ** 2;
  const conformal =
    phi -
    Math.sin(phi) * Math.cos(phi) *
      (E2 + ((5 * E2 ** 2 - E2 ** 3) / 6) * s2 + ((104 * E2 ** 3 - 45 * E2 ** 4) / 120) * s2 ** 2 +
        ((1237 * E2 ** 4) / 1260) * s2 ** 3);
  const dl = (lon * Math.PI) / 180 - LON0;
  const xi = Math.atan(Math.tan(conformal) / Math.cos(dl));
  const eta = Math.atanh(Math.cos(conformal) * Math.sin(dl));
  let north = xi;
  let east = eta;
  BETA.forEach((b, i) => {
    const j = 2 * (i + 1);
    north += b * Math.sin(j * xi) * Math.cosh(j * eta);
    east += b * Math.cos(j * xi) * Math.sinh(j * eta);
  });
  return [K0 * AR * east + FE, K0 * AR * north];
}

/** [easting, northing] in metres to [lon, lat] in degrees. */
export function toWgs84([east, north]: [number, number]): [number, number] {
  const xi = north / (K0 * AR);
  const eta = (east - FE) / (K0 * AR);
  let x = xi;
  let e = eta;
  DELTA.forEach((d, i) => {
    const j = 2 * (i + 1);
    x -= d * Math.sin(j * xi) * Math.cosh(j * eta);
    e -= d * Math.cos(j * xi) * Math.sinh(j * eta);
  });
  const conformal = Math.asin(Math.sin(x) / Math.cosh(e));
  const s2 = Math.sin(conformal) ** 2;
  const phi =
    conformal +
    Math.sin(conformal) * Math.cos(conformal) *
      (E2 + E2 ** 2 + E2 ** 3 + E2 ** 4 - ((7 * E2 ** 2 + 17 * E2 ** 3 + 30 * E2 ** 4) / 6) * s2 +
        ((224 * E2 ** 3 + 889 * E2 ** 4) / 120) * s2 ** 2 - ((4279 * E2 ** 4) / 1260) * s2 ** 3);
  return [((LON0 + Math.atan(Math.sinh(e) / Math.cos(x))) * 180) / Math.PI, (phi * 180) / Math.PI];
}
