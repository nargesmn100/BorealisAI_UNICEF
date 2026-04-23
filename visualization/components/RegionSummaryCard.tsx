'use client';

import React from 'react';
import { Region, RegionStats, povertyColor, fmtPct, fmtNum } from '@/lib/data';

interface Props {
  region: Region;
  stats: RegionStats;
}

// Mini histogram: SVG bar chart of predicted poverty distribution
function ScoreHistogram({ histogram }: { histogram: RegionStats['histogram'] }) {
  const W = 200, H = 54;
  const maxCount = Math.max(1, ...histogram.map(b => b.count));
  const binW = W / histogram.length;

  return (
    <div>
      <p className="text-[9px] font-medium uppercase tracking-widest text-slate-400 mb-1.5">
        Cell score distribution
      </p>
      <svg viewBox={`0 0 ${W} ${H + 12}`} width="100%" style={{ display: 'block' }}>
        {histogram.map((bar, i) => {
          const barH = Math.max(1, (bar.count / maxCount) * H);
          const x = i * binW;
          const y = H - barH;
          return (
            <g key={i}>
              <rect
                x={x + 0.5}
                y={y}
                width={binW - 1}
                height={barH}
                fill={povertyColor(bar.binStart + 0.05)}
                rx={1}
              />
            </g>
          );
        })}
        {/* X-axis ticks at 0%, 50%, 100% */}
        {[0, 0.5, 1.0].map(v => (
          <text
            key={v}
            x={v * W}
            y={H + 10}
            textAnchor="middle"
            fontSize={7.5}
            fill="#94a3b8"
            fontFamily="Inter, system-ui, sans-serif"
          >
            {(v * 100).toFixed(0)}%
          </text>
        ))}
        <line x1={0} y1={H + 0.5} x2={W} y2={H + 0.5} stroke="#e2e8f0" strokeWidth={0.5} />
      </svg>
    </div>
  );
}

// Small key-value row
function Stat({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between items-baseline gap-2">
      <span className="text-[11px] text-slate-500">{label}</span>
      <span className={`text-[12px] font-semibold text-slate-800 ${mono ? 'font-mono' : ''}`}>
        {value}
      </span>
    </div>
  );
}

export default function RegionSummaryCard({ region, stats }: Props) {
  const dot = (
    <span
      className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5 flex-shrink-0"
      style={{ backgroundColor: povertyColor(region.officialPoverty) }}
    />
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div>
        <div className="flex items-center gap-1 mb-0.5">
          {dot}
          <h2 className="text-sm font-semibold text-slate-900">{region.name}</h2>
        </div>
        <p className="text-[10px] text-slate-400 ml-4">
          {stats.count} grid cells · {fmtNum(stats.totalPopulation)} total population
        </p>
      </div>

      {/* Poverty scores */}
      <div className="border border-slate-100 rounded-lg p-3 bg-slate-50 space-y-1.5">
        <p className="text-[9px] font-medium uppercase tracking-widest text-slate-400 mb-2">
          Poverty scores
        </p>
        <Stat label="Official survey score" value={fmtPct(region.officialPoverty)} mono />
        <Stat label="Avg. predicted score"  value={fmtPct(stats.aggregatedPoverty)} mono />
        <div className="h-px bg-slate-200 my-1" />
        <Stat label="Min cell score" value={fmtPct(stats.minPoverty)} mono />
        <Stat label="Max cell score" value={fmtPct(stats.maxPoverty)} mono />
      </div>

      {/* Population */}
      <div className="space-y-1.5">
        <p className="text-[9px] font-medium uppercase tracking-widest text-slate-400">
          Population
        </p>
        <Stat label="Total"    value={fmtNum(stats.totalPopulation)} />
        <Stat label="Children" value={fmtNum(stats.totalChildPopulation)} />
      </div>

      {/* Histogram */}
      <ScoreHistogram histogram={stats.histogram} />

      {/* Aggregation accuracy indicator */}
      {(() => {
        const diff = Math.abs(region.officialPoverty - stats.aggregatedPoverty);
        const diffPp = (diff * 100).toFixed(1);
        const good = diff < 0.04;
        return (
          <div
            className={`rounded-lg px-3 py-2 text-[10px] leading-snug ${
              good
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                : 'bg-amber-50 text-amber-700 border border-amber-100'
            }`}
          >
            <span className="font-semibold">
              {good ? 'Good aggregation match' : 'Aggregation offset'}
            </span>
            {' '}—{' '}
            predicted mean is {diffPp} pp {stats.aggregatedPoverty > region.officialPoverty ? 'above' : 'below'} survey.
          </div>
        );
      })()}
    </div>
  );
}
