'use client';

import React from 'react';
import { Region, RegionStats, fmtPct } from '@/lib/data';

interface Props {
  region: Region;
  stats: RegionStats;
}

export default function AggregationPanel({ region, stats }: Props) {
  const diff   = Math.abs(region.officialPoverty - stats.aggregatedPoverty);
  const diffPp = (diff * 100).toFixed(1);

  return (
    <div className="border border-slate-200 rounded-xl p-5 bg-white">
      {/* Section title */}
      <p className="text-[10px] font-medium uppercase tracking-widest text-slate-400 mb-4">
        Aggregation check · {region.name}
      </p>

      {/* Score comparison */}
      <div className="flex items-center gap-4 mb-5">
        {/* Official */}
        <div className="flex-1 text-center border border-slate-200 rounded-lg py-3 bg-slate-50">
          <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-1">Official survey</p>
          <p className="text-2xl font-semibold font-mono text-slate-900">
            {fmtPct(region.officialPoverty, 1)}
          </p>
          <p className="text-[9px] text-slate-400 mt-0.5">from MICS / DHS</p>
        </div>

        {/* Arrow */}
        <div className="flex flex-col items-center gap-0.5 flex-shrink-0">
          <span className="text-slate-300 text-lg">≈</span>
          <span className="text-[9px] text-slate-400">{diffPp} pp</span>
        </div>

        {/* Predicted */}
        <div className="flex-1 text-center border border-slate-200 rounded-lg py-3 bg-slate-50">
          <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-1">Aggregated cells</p>
          <p className="text-2xl font-semibold font-mono text-slate-900">
            {fmtPct(stats.aggregatedPoverty, 1)}
          </p>
          <p className="text-[9px] text-slate-400 mt-0.5">pop.-weighted mean</p>
        </div>
      </div>

      {/* Formula */}
      <div className="bg-slate-900 rounded-lg px-4 py-3 mb-4 overflow-x-auto">
        <pre className="text-[11px] font-mono text-slate-200 leading-relaxed whitespace-pre">
{`Ŷ_region  =  Σ ( ŷᵢ × popᵢ )  /  Σ ( popᵢ )

           =  ${fmtPct(stats.aggregatedPoverty, 2)}   (${stats.count} cells, ${formatPopM(stats.totalPopulation)} people)

Official   =  ${fmtPct(region.officialPoverty, 2)}   Δ = ${diffPp} pp`}
        </pre>
      </div>

      {/* Concept note */}
      <div className="bg-blue-50 border border-blue-100 rounded-lg px-4 py-3 text-[11px] text-blue-800 leading-relaxed">
        <span className="font-semibold">Weak supervision:</span>{' '}
        The model is not trained on true poverty for each cell — those labels don&rsquo;t exist.
        Instead, it learns cell-level vulnerability scores such that the population-weighted
        aggregate of predictions within each region recovers the trusted coarse survey total.
        Fine-scale predictions are inferred structure, not observed ground truth.
      </div>
    </div>
  );
}

function formatPopM(n: number): string {
  return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : `${(n / 1_000).toFixed(0)}K`;
}
