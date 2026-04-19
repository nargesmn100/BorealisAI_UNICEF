'use client';

import React from 'react';
import { GridCell, Region, povertyColor, fmtPct } from '@/lib/data';

interface Props {
  cell: GridCell;
  region: Region;
  screenX: number;
  screenY: number;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-slate-400 text-[10px]">{label}</span>
      <span className="text-slate-100 text-[10px] font-medium font-mono">{value}</span>
    </div>
  );
}

export default function CellTooltip({ cell, region, screenX, screenY }: Props) {
  // Offset so tooltip doesn't cover cursor; flip if near right/bottom edge
  const offsetX = 14;
  const offsetY = -14;

  const dotColor = povertyColor(cell.predictedPoverty);

  return (
    <div
      className="tooltip-enter pointer-events-none fixed z-50"
      style={{
        left: screenX + offsetX,
        top:  screenY + offsetY,
        transform: 'translateY(-100%)',
      }}
    >
      <div
        className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 min-w-[190px]"
        style={{ backdropFilter: 'blur(6px)' }}
      >
        {/* Cell ID + region */}
        <div className="flex items-center gap-1.5 mb-2 pb-1.5 border-b border-slate-700">
          <span
            className="w-2 h-2 rounded-sm flex-shrink-0"
            style={{ backgroundColor: dotColor }}
          />
          <span className="text-white text-[11px] font-semibold truncate">{region.name}</span>
        </div>

        {/* Poverty score — prominent */}
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-slate-400 text-[10px]">Predicted poverty</span>
          <span
            className="text-[14px] font-bold font-mono"
            style={{ color: dotColor === 'rgb(254, 251, 240)' ? '#fbbf24' : dotColor }}
          >
            {fmtPct(cell.predictedPoverty)}
          </span>
        </div>

        {/* Feature summary */}
        <div className="space-y-0.5">
          <Row label="Population"      value={cell.population.toLocaleString()} />
          <Row label="Children"        value={cell.childPopulation.toLocaleString()} />
          <Row label="Night lights"    value={`${cell.nightLights}/100`} />
          <Row label="Building dens."  value={`${(cell.buildingDensity * 100).toFixed(0)}%`} />
          <Row label="Accessibility"   value={`${(cell.accessibility * 100).toFixed(0)}%`} />
          <Row label="Settlement"      value={cell.settlementType} />
        </div>

        {/* Footer caption */}
        <p className="mt-2 pt-1.5 border-t border-slate-700 text-[9px] text-slate-500 italic">
          Inferred score — not a direct survey label
        </p>
      </div>
    </div>
  );
}
