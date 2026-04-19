'use client';

import React from 'react';

const STEPS = [
  {
    number: '01',
    title: 'Align datasets to a shared grid',
    body: 'Population rasters, satellite-derived proxies (night lights, building density, vegetation), travel-time maps, and conflict data are resampled to a common spatial resolution — typically 1 km² grid cells.',
  },
  {
    number: '02',
    title: 'Predict vulnerability per grid cell',
    body: 'A supervised model (Ridge, GBM, GAM, or neural net) learns a mapping from cell-level proxy features to a vulnerability score. Crucially, the supervision signal comes from coarse regional poverty totals, not from cell-level labels.',
  },
  {
    number: '03',
    title: 'Aggregate to match official totals',
    body: 'Population-weighted averages of predicted cell scores are constrained to match each region\'s official poverty headcount. This reconciliation step ensures policy-coherent estimates at every administrative level.',
  },
];

export default function StepsExplainer() {
  return (
    <div>
      <p className="text-[10px] font-medium uppercase tracking-widest text-slate-400 mb-4">
        How it works
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {STEPS.map((step, i) => (
          <div key={i} className="flex gap-3">
            {/* Step number */}
            <div className="flex-shrink-0">
              <span className="block text-[22px] font-light text-slate-200 leading-none font-mono">
                {step.number}
              </span>
            </div>
            {/* Content */}
            <div>
              <h3 className="text-[12px] font-semibold text-slate-800 mb-1 leading-snug">
                {step.title}
              </h3>
              <p className="text-[11px] text-slate-500 leading-relaxed">{step.body}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Fine-scale disclaimer */}
      <div className="mt-5 border-t border-slate-100 pt-4">
        <p className="text-[10px] text-slate-400 italic leading-relaxed">
          <span className="font-medium not-italic text-slate-500">Note: </span>
          Fine-scale predictions are not directly observed labels; they are inferred under a
          regional aggregation constraint. Accuracy at the cell level depends on how well proxy
          features correlate with true deprivation — and can only be validated against independent
          point-level survey data (e.g. DHS GPS clusters or LSMS household geo-coordinates).
        </p>
      </div>
    </div>
  );
}
