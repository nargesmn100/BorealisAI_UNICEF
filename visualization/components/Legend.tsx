'use client';

import React from 'react';
import { povertyColor } from '@/lib/data';

const TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
const GRADIENT_STOPS = Array.from({ length: 21 }, (_, i) => i / 20);

export default function Legend() {
  const W = 220;
  const H = 14;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[10px] font-medium uppercase tracking-widest text-slate-400">
        Poverty score
      </span>
      <svg viewBox={`0 0 ${W} ${H + 18}`} width={W} height={H + 18} style={{ overflow: 'visible' }}>
        <defs>
          <linearGradient id="legend-grad" x1="0" y1="0" x2="1" y2="0">
            {GRADIENT_STOPS.map(v => (
              <stop
                key={v}
                offset={`${v * 100}%`}
                stopColor={povertyColor(v)}
              />
            ))}
          </linearGradient>
        </defs>

        {/* Gradient bar */}
        <rect x={0} y={0} width={W} height={H} rx={3} fill="url(#legend-grad)" />

        {/* Tick marks and labels */}
        {TICKS.map(v => {
          const x = v * W;
          return (
            <g key={v}>
              <line x1={x} y1={H} x2={x} y2={H + 4} stroke="#94a3b8" strokeWidth={1} />
              <text
                x={x}
                y={H + 13}
                textAnchor="middle"
                fontSize={8.5}
                fill="#64748b"
                fontFamily="Inter, system-ui, sans-serif"
              >
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
      </svg>
      <span className="text-[9px] text-slate-400 italic">
        Low ← poverty headcount ratio → High
      </span>
    </div>
  );
}
