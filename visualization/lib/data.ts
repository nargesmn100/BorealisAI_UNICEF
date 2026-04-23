/**
 * Synthetic data for the poverty-mapping demo.
 *
 * Conceptual model:
 *  - A stylised country is divided into 5 coarse administrative regions.
 *  - Each region has an official poverty score from a household survey (MICS / DHS).
 *  - A fine grid of cells covers the country. Each cell is assigned to a region
 *    via point-in-polygon and receives a *predicted* vulnerability score.
 *  - The weighted average of predicted cell scores should approximately recover
 *    the official regional poverty score — the core weakly-supervised constraint.
 */

export const SVG_WIDTH  = 640;
export const SVG_HEIGHT = 460;

// ─── Types ────────────────────────────────────────────────────────────────────

export type Resolution    = 'coarse' | 'medium' | 'fine';
export type ViewMode      = 'coarse' | 'fine' | 'both';
export type SettlementType = 'Urban' | 'Peri-urban' | 'Rural';

export interface Region {
  id: string;
  name: string;
  /** Official poverty headcount ratio from survey, 0–1 */
  officialPoverty: number;
  totalPopulation: number;
  polygon: [number, number][];
  /** Approximate centroid label position (auto-computed if omitted) */
  labelPos?: [number, number];
}

export interface GridCell {
  id: string;
  col: number;
  row: number;
  /** Top-left corner in SVG coordinates */
  x: number;
  y: number;
  width: number;
  height: number;
  regionId: string;
  predictedPoverty: number;
  population: number;
  childPopulation: number;
  /** VIIRS night-time light radiance (0–100) */
  nightLights: number;
  /** Fraction of cell covered by built-up area (0–1) */
  buildingDensity: number;
  /** Accessibility score: 1 = high, near city; 0 = remote (0–1) */
  accessibility: number;
  settlementType: SettlementType;
}

export interface RegionStats {
  count: number;
  /** Population-weighted mean of predicted cell scores */
  aggregatedPoverty: number;
  totalPopulation: number;
  totalChildPopulation: number;
  minPoverty: number;
  maxPoverty: number;
  /** 10-bin histogram of predicted poverty scores */
  histogram: { binStart: number; count: number }[];
}

// ─── Region definitions ───────────────────────────────────────────────────────
//
// Five synthetic regions loosely evoking a sub-Saharan country layout:
//  - North:  large Sahelian belt, highest poverty
//  - East:   upland plateau, medium-high poverty
//  - West:   coastal urban corridor, lowest poverty
//  - Center: river basin transition zone
//  - South:  delta / southern farming belt
//
// Polygons tile SVG_WIDTH × SVG_HEIGHT with deliberate 1–2 px overlap at
// shared edges — white strokes render the boundaries cleanly.

export const REGIONS: Region[] = [
  {
    id: 'north',
    name: 'Northern Sahel',
    officialPoverty: 0.64,
    totalPopulation: 2_840_000,
    polygon: [
      [0, 0], [640, 0], [640, 158],
      [502, 178], [342, 185], [200, 175], [80, 181], [0, 156],
    ],
    labelPos: [320, 80],
  },
  {
    id: 'east',
    name: 'Eastern Plateau',
    officialPoverty: 0.51,
    totalPopulation: 3_120_000,
    polygon: [
      [342, 185], [502, 178], [640, 158],
      [640, 460], [442, 460], [382, 322], [402, 232],
    ],
    labelPos: [530, 330],
  },
  {
    id: 'west',
    name: 'Western Coast',
    officialPoverty: 0.22,
    totalPopulation: 4_580_000,
    polygon: [
      [0, 156], [80, 181], [200, 175],
      [238, 252], [198, 362], [120, 402], [0, 460],
    ],
    labelPos: [68, 340],
  },
  {
    id: 'center',
    name: 'River Basin',
    officialPoverty: 0.38,
    totalPopulation: 3_740_000,
    polygon: [
      [80, 181], [200, 175], [342, 185],
      [402, 232], [382, 322], [260, 342], [198, 362], [238, 252],
    ],
    labelPos: [265, 272],
  },
  {
    id: 'south',
    name: 'Southern Delta',
    officialPoverty: 0.33,
    totalPopulation: 2_960_000,
    polygon: [
      [120, 402], [198, 362], [260, 342],
      [382, 322], [442, 460], [0, 460],
    ],
    labelPos: [240, 420],
  },
];

// ─── Grid resolution configs ──────────────────────────────────────────────────

const RESOLUTION_GRID: Record<Resolution, { cols: number; rows: number }> = {
  coarse: { cols: 14, rows: 10 },
  medium: { cols: 21, rows: 15 },
  fine:   { cols: 28, rows: 20 },
};

// ─── Seeded RNG (LCG) ─────────────────────────────────────────────────────────
// Ensures reproducible synthetic data across renders.

function makeRng(seed: number) {
  let s = seed >>> 0;
  return (): number => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ─── Point-in-polygon (ray casting) ──────────────────────────────────────────

function pointInPolygon(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (((yi > py) !== (yj > py)) &&
        (px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

// ─── Cell generation ─────────────────────────────────────────────────────────

export function generateCells(resolution: Resolution): GridCell[] {
  const { cols, rows } = RESOLUTION_GRID[resolution];
  const cellW = SVG_WIDTH  / cols;
  const cellH = SVG_HEIGHT / rows;
  const rng   = makeRng(42);

  const cells: GridCell[] = [];

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      // Cell centre for polygon assignment
      const cx = col * cellW + cellW * 0.5;
      const cy = row * cellH + cellH * 0.5;

      let region: Region | undefined;
      for (const r of REGIONS) {
        if (pointInPolygon(cx, cy, r.polygon)) { region = r; break; }
      }
      if (!region) { rng(); rng(); rng(); rng(); rng(); continue; } // advance rng to keep sequence

      // Predicted poverty: region mean ± spatial noise
      // Noise amplitude is intentionally wide to show intra-region heterogeneity
      const noise = (rng() - 0.5) * 0.30;
      const pred  = clamp(region.officialPoverty + noise, 0.04, 0.94);

      // Settlement type driven by poverty (heuristic for demo realism)
      const settlement: SettlementType =
        pred < 0.28 ? 'Urban' : pred < 0.50 ? 'Peri-urban' : 'Rural';

      // Proxy features negatively correlated with poverty
      const nightLights    = round2(clamp((1 - pred) * 82 + rng() * 18,  2, 97));
      const buildingDensity = round2(clamp((1 - pred) * 0.72 + rng() * 0.20, 0.01, 0.97));
      const accessibility   = round2(clamp((1 - pred) * 0.78 + rng() * 0.18, 0.04, 0.96));

      const basePopPerCell = settlement === 'Urban' ? 6800 : settlement === 'Peri-urban' ? 3400 : 1100;
      const population     = Math.round(basePopPerCell * (0.4 + rng() * 1.2));
      const childFrac      = clamp(0.16 + pred * 0.12 + (rng() - 0.5) * 0.04, 0.10, 0.32);
      const childPopulation = Math.round(population * childFrac);

      cells.push({
        id: `${region.id}-${col}-${row}`,
        col,
        row,
        x: col * cellW,
        y: row * cellH,
        width:  cellW,
        height: cellH,
        regionId: region.id,
        predictedPoverty: round3(pred),
        population,
        childPopulation,
        nightLights:     Math.round(nightLights),
        buildingDensity: round2(buildingDensity),
        accessibility:   round2(accessibility),
        settlementType:  settlement,
      });
    }
  }

  return cells;
}

// ─── Aggregation helper ───────────────────────────────────────────────────────

export function getRegionStats(cells: GridCell[], regionId: string): RegionStats {
  const rc = cells.filter(c => c.regionId === regionId);
  if (!rc.length) {
    return { count: 0, aggregatedPoverty: 0, totalPopulation: 0, totalChildPopulation: 0,
             minPoverty: 0, maxPoverty: 0, histogram: [] };
  }

  const totalPop       = rc.reduce((s, c) => s + c.population, 0);
  const totalChildPop  = rc.reduce((s, c) => s + c.childPopulation, 0);
  const weightedSum    = rc.reduce((s, c) => s + c.predictedPoverty * c.population, 0);
  const aggPoverty     = totalPop > 0 ? weightedSum / totalPop : 0;

  // 10 equal-width bins over [0, 1]
  const bins = Array.from({ length: 10 }, (_, i) => ({ binStart: i / 10, count: 0 }));
  rc.forEach(c => {
    const idx = Math.min(9, Math.floor(c.predictedPoverty * 10));
    bins[idx].count++;
  });

  return {
    count: rc.length,
    aggregatedPoverty:   round3(aggPoverty),
    totalPopulation:     totalPop,
    totalChildPopulation: totalChildPop,
    minPoverty: round3(Math.min(...rc.map(c => c.predictedPoverty))),
    maxPoverty: round3(Math.max(...rc.map(c => c.predictedPoverty))),
    histogram:  bins,
  };
}

// ─── Color scale ─────────────────────────────────────────────────────────────
// Sequential scale: pale amber → amber → orange → red → dark red

const COLOR_STOPS: [number, [number, number, number]][] = [
  [0.00, [254, 251, 240]],
  [0.20, [253, 230, 138]],
  [0.40, [251, 191,  36]],
  [0.55, [249, 115,  22]],
  [0.70, [220,  38,  38]],
  [0.85, [153,  27,  27]],
  [1.00, [120,  15,  15]],
];

export function povertyColor(score: number, alpha = 1): string {
  const t = clamp(score, 0, 1);
  let lo = COLOR_STOPS[0], hi = COLOR_STOPS[COLOR_STOPS.length - 1];
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (t >= COLOR_STOPS[i][0] && t <= COLOR_STOPS[i + 1][0]) {
      lo = COLOR_STOPS[i]; hi = COLOR_STOPS[i + 1]; break;
    }
  }
  const range = hi[0] - lo[0];
  const frac  = range === 0 ? 0 : (t - lo[0]) / range;
  const [r, g, b] = lo[1].map((c, i) => Math.round(c + (hi[1][i] - c) * frac));
  return alpha < 1 ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
}

/** Text color (dark/light) for readability on a poverty-colored background */
export function povertyTextColor(score: number): string {
  return score > 0.45 ? '#ffffff' : '#1e293b';
}

// ─── Formatters ───────────────────────────────────────────────────────────────

export function fmtPct(v: number, decimals = 1): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtNum(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000)     return `${(v / 1_000).toFixed(0)}K`;
  return `${v}`;
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
function round2(v: number): number { return Math.round(v * 100) / 100; }
function round3(v: number): number { return Math.round(v * 1000) / 1000; }
