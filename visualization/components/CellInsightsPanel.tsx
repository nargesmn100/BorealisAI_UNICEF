'use client';

import React, { useMemo } from 'react';
import { GridCell, REGIONS, Region, povertyColor, fmtPct, fmtNum } from '@/lib/data';

interface Props {
  cells: GridCell[];
  selectedCell: GridCell | null;
  selectedRegion: Region | null;
}

// ─── Scatter Plot: Night Lights vs Poverty ────────────────────────────────────

function ScatterPlot({
  cells,
  selectedCellId,
}: {
  cells: GridCell[];
  selectedCellId: string | null;
}) {
  const W = 260, H = 180;
  const PAD = { t: 12, r: 12, b: 32, l: 38 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;

  const scaleX = (v: number) => PAD.l + (v / 100) * innerW;
  const scaleY = (v: number) => PAD.t + (1 - v) * innerH;

  // Grid lines
  const xTicks = [0, 25, 50, 75, 100];
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div>
      <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-2">
        Night lights vs. poverty
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="overflow-visible">
        {/* Grid lines */}
        {xTicks.map(v => (
          <line
            key={`xg-${v}`}
            x1={scaleX(v)} y1={PAD.t}
            x2={scaleX(v)} y2={PAD.t + innerH}
            stroke="#e2e8f0" strokeWidth={0.5}
          />
        ))}
        {yTicks.map(v => (
          <line
            key={`yg-${v}`}
            x1={PAD.l} y1={scaleY(v)}
            x2={PAD.l + innerW} y2={scaleY(v)}
            stroke="#e2e8f0" strokeWidth={0.5}
          />
        ))}

        {/* Axis lines */}
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={PAD.t + innerH}
              stroke="#cbd5e1" strokeWidth={1} />
        <line x1={PAD.l} y1={PAD.t + innerH} x2={PAD.l + innerW} y2={PAD.t + innerH}
              stroke="#cbd5e1" strokeWidth={1} />

        {/* X-axis labels */}
        {xTicks.map(v => (
          <text key={`xl-${v}`} x={scaleX(v)} y={H - 6}
                textAnchor="middle" fontSize={7.5} fill="#94a3b8"
                fontFamily="Inter, system-ui, sans-serif">
            {v}
          </text>
        ))}

        {/* Y-axis labels */}
        {yTicks.map(v => (
          <text key={`yl-${v}`} x={PAD.l - 4} y={scaleY(v) + 2.5}
                textAnchor="end" fontSize={7.5} fill="#94a3b8"
                fontFamily="Inter, system-ui, sans-serif">
            {(v * 100).toFixed(0)}
          </text>
        ))}

        {/* Axis titles */}
        <text x={PAD.l + innerW / 2} y={H - 0.5}
              textAnchor="middle" fontSize={7} fill="#94a3b8"
              fontFamily="Inter, system-ui, sans-serif">
          Night lights (0–100)
        </text>
        <text
          x={8} y={PAD.t + innerH / 2}
          textAnchor="middle" fontSize={7} fill="#94a3b8"
          fontFamily="Inter, system-ui, sans-serif"
          transform={`rotate(-90, 8, ${PAD.t + innerH / 2})`}
        >
          Poverty %
        </text>

        {/* Data points — non-selected first, selected on top */}
        {cells
          .filter(c => c.id !== selectedCellId)
          .map(c => (
            <circle
              key={c.id}
              cx={scaleX(c.nightLights)}
              cy={scaleY(c.predictedPoverty)}
              r={2.5}
              fill={povertyColor(c.predictedPoverty)}
              fillOpacity={0.65}
            />
          ))}

        {/* Selected cell on top */}
        {cells
          .filter(c => c.id === selectedCellId)
          .map(c => (
            <g key={c.id}>
              <circle
                cx={scaleX(c.nightLights)}
                cy={scaleY(c.predictedPoverty)}
                r={6}
                fill="none"
                stroke="#2563eb"
                strokeWidth={1.5}
              />
              <circle
                cx={scaleX(c.nightLights)}
                cy={scaleY(c.predictedPoverty)}
                r={3.5}
                fill={povertyColor(c.predictedPoverty)}
              />
            </g>
          ))}
      </svg>
      <p className="text-[9px] text-slate-400 italic mt-1">
        Each dot = one grid cell · blue ring = selected cell
      </p>
    </div>
  );
}

// ─── Feature Profile: selected cell vs. region average ───────────────────────

interface Feature {
  label: string;
  cellValue: number;
  regionAvg: number;
  maxValue: number;
  unit: string;
  invert?: boolean; // true = lower is better
}

function FeatureProfileBar({ feature }: { feature: Feature }) {
  const W = 160;
  const cellFrac   = Math.min(1, feature.cellValue  / feature.maxValue);
  const regionFrac = Math.min(1, feature.regionAvg  / feature.maxValue);

  const cellColor   = feature.invert
    ? (cellFrac > regionFrac ? '#ef4444' : '#22c55e')
    : (cellFrac > regionFrac ? '#22c55e' : '#ef4444');

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-slate-500 w-24 flex-shrink-0 truncate">
        {feature.label}
      </span>
      <div className="flex-1 relative h-3">
        {/* Region avg bar (background) */}
        <div
          className="absolute top-0 left-0 h-full rounded-sm bg-slate-200"
          style={{ width: `${regionFrac * 100}%` }}
        />
        {/* Cell value bar */}
        <div
          className="absolute top-0 left-0 h-full rounded-sm opacity-80"
          style={{ width: `${cellFrac * 100}%`, backgroundColor: cellColor }}
        />
        {/* Region avg marker line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-slate-500 opacity-60"
          style={{ left: `${regionFrac * 100}%` }}
        />
      </div>
      <span className="text-[10px] font-mono text-slate-600 w-12 text-right flex-shrink-0">
        {feature.unit === '%'
          ? `${(feature.cellValue * (feature.maxValue === 1 ? 100 : 1)).toFixed(0)}%`
          : feature.cellValue.toFixed(0)}
      </span>
    </div>
  );
}

function FeatureProfile({ cell, regionCells }: { cell: GridCell; regionCells: GridCell[] }) {
  const avg = (key: keyof GridCell) =>
    regionCells.reduce((s, c) => s + (c[key] as number), 0) / regionCells.length;

  const features: Feature[] = [
    {
      label: 'Poverty score',
      cellValue: cell.predictedPoverty,
      regionAvg: avg('predictedPoverty'),
      maxValue: 1,
      unit: '%',
      invert: true,
    },
    {
      label: 'Night lights',
      cellValue: cell.nightLights,
      regionAvg: avg('nightLights'),
      maxValue: 100,
      unit: '',
    },
    {
      label: 'Building density',
      cellValue: cell.buildingDensity,
      regionAvg: avg('buildingDensity'),
      maxValue: 1,
      unit: '%',
    },
    {
      label: 'Accessibility',
      cellValue: cell.accessibility,
      regionAvg: avg('accessibility'),
      maxValue: 1,
      unit: '%',
    },
    {
      label: 'Population',
      cellValue: cell.population,
      regionAvg: avg('population'),
      maxValue: Math.max(...regionCells.map(c => c.population)),
      unit: '',
    },
  ];

  return (
    <div>
      <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-2">
        Cell vs. region avg
        <span className="ml-1.5 text-slate-300 font-normal normal-case">
          (bar = cell · line = avg)
        </span>
      </p>
      <div className="space-y-2">
        {features.map(f => <FeatureProfileBar key={f.label} feature={f} />)}
      </div>
      <p className="text-[9px] text-slate-400 italic mt-2">
        Green = better than avg · Red = worse than avg
      </p>
    </div>
  );
}

// ─── Top Cells Ranking ────────────────────────────────────────────────────────

function TopCellsRanking({
  cells,
  selectedCellId,
  onSelect,
}: {
  cells: GridCell[];
  selectedCellId: string | null;
  onSelect: (cell: GridCell) => void;
}) {
  const sorted = useMemo(
    () => [...cells].sort((a, b) => b.predictedPoverty - a.predictedPoverty).slice(0, 8),
    [cells],
  );
  const maxPov = sorted[0]?.predictedPoverty ?? 1;

  return (
    <div>
      <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-2">
        Top 8 most deprived cells
      </p>
      <div className="space-y-1">
        {sorted.map((cell, i) => {
          const isSelected = cell.id === selectedCellId;
          const barW = (cell.predictedPoverty / maxPov) * 100;
          return (
            <div
              key={cell.id}
              onClick={() => onSelect(cell)}
              className={`flex items-center gap-2 cursor-pointer rounded px-1.5 py-1 transition-colors ${
                isSelected ? 'bg-blue-50' : 'hover:bg-slate-50'
              }`}
            >
              {/* Rank number */}
              <span className="text-[9px] text-slate-400 w-4 text-right flex-shrink-0">
                {i + 1}
              </span>
              {/* Bar */}
              <div className="flex-1 h-3 bg-slate-100 rounded-sm overflow-hidden relative">
                <div
                  className="absolute top-0 left-0 h-full rounded-sm"
                  style={{
                    width: `${barW}%`,
                    backgroundColor: povertyColor(cell.predictedPoverty),
                  }}
                />
              </div>
              {/* Score */}
              <span className={`text-[10px] font-mono w-10 text-right flex-shrink-0 ${
                isSelected ? 'text-blue-600 font-semibold' : 'text-slate-600'
              }`}>
                {fmtPct(cell.predictedPoverty)}
              </span>
              {/* Cell ID */}
              <span className="text-[9px] text-slate-400 font-mono w-24 truncate">
                {cell.id}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Empty state ─────────────────────────────────────────────────────────────

function EmptyState({ hasRegion }: { hasRegion: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center h-40 text-center px-6">
      <p className="text-[11px] text-slate-400 leading-relaxed">
        {hasRegion
          ? 'Click a cell in the table or on the map to see its feature profile and how it compares to the regional average.'
          : 'Select a region first, then click a cell to see detailed insights.'}
      </p>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function CellInsightsPanel({ cells, selectedCell, selectedRegion }: Props) {
  // Cells for the active region (or all if none selected)
  const regionCells = useMemo(
    () => selectedRegion ? cells.filter(c => c.regionId === selectedRegion.id) : cells,
    [cells, selectedRegion],
  );

  if (!selectedRegion) {
    return (
      <div className="border border-slate-200 rounded-xl p-5 bg-white">
        <p className="text-[10px] uppercase tracking-widest text-slate-400 mb-3">Cell insights</p>
        <EmptyState hasRegion={false} />
      </div>
    );
  }

  return (
    <div className="border border-slate-200 rounded-xl p-5 bg-white">
      <div className="flex items-baseline justify-between mb-4">
        <p className="text-[10px] uppercase tracking-widest text-slate-400">
          Cell insights · {selectedRegion.name}
        </p>
        <span className="text-[10px] text-slate-400">
          {regionCells.length} cells
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 divide-x divide-slate-100">
        {/* Panel 1: Scatter */}
        <div className="lg:pr-6">
          <ScatterPlot cells={regionCells} selectedCellId={selectedCell?.id ?? null} />
        </div>

        {/* Panel 2: Feature profile (only when cell selected) */}
        <div className="lg:px-6">
          {selectedCell && selectedCell.regionId === selectedRegion.id ? (
            <>
              {/* Selected cell header */}
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: povertyColor(selectedCell.predictedPoverty) }}
                />
                <span className="text-[10px] font-mono text-slate-600">{selectedCell.id}</span>
                <span className={`ml-auto text-[10px] px-1.5 py-0.5 rounded font-medium ${
                  selectedCell.settlementType === 'Urban'
                    ? 'bg-blue-50 text-blue-600'
                    : selectedCell.settlementType === 'Peri-urban'
                    ? 'bg-amber-50 text-amber-600'
                    : 'bg-green-50 text-green-700'
                }`}>
                  {selectedCell.settlementType}
                </span>
              </div>
              <FeatureProfile cell={selectedCell} regionCells={regionCells} />
            </>
          ) : (
            <EmptyState hasRegion />
          )}
        </div>

        {/* Panel 3: Top cells ranking */}
        <div className="lg:pl-6">
          <TopCellsRanking
            cells={regionCells}
            selectedCellId={selectedCell?.id ?? null}
            onSelect={() => {}} // wired in page.tsx via prop
          />
        </div>
      </div>
    </div>
  );
}
