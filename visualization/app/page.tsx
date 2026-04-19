'use client';

import React, { useState, useMemo, useCallback } from 'react';
import {
  REGIONS,
  generateCells,
  getRegionStats,
  Resolution,
  ViewMode,
  GridCell,
} from '@/lib/data';
import PovertyMapSVG from '@/components/PovertyMapSVG';
import Legend from '@/components/Legend';
import RegionSummaryCard from '@/components/RegionSummaryCard';
import AggregationPanel from '@/components/AggregationPanel';
import CellTooltip from '@/components/CellTooltip';
import StepsExplainer from '@/components/StepsExplainer';

// ─── View / Resolution toggles ────────────────────────────────────────────────

const VIEW_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: 'coarse', label: 'Coarse regions' },
  { value: 'fine',   label: 'Fine grid' },
  { value: 'both',   label: 'Overlay both' },
];

const RES_OPTIONS: { value: Resolution; label: string; sub: string }[] = [
  { value: 'coarse', label: 'Coarse',  sub: '140 cells'  },
  { value: 'medium', label: 'Medium',  sub: '315 cells'  },
  { value: 'fine',   label: 'Fine',    sub: '560 cells'  },
];

// ─── Tooltip state ────────────────────────────────────────────────────────────

interface TooltipInfo {
  cell: GridCell;
  screenX: number;
  screenY: number;
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Page() {
  const [viewMode,         setViewMode]         = useState<ViewMode>('coarse');
  const [resolution,       setResolution]       = useState<Resolution>('medium');
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [hoveredRegionId,  setHoveredRegionId]  = useState<string | null>(null);
  const [tooltip,          setTooltip]          = useState<TooltipInfo | null>(null);

  // Regenerate cells when resolution changes
  const cells = useMemo(() => generateCells(resolution), [resolution]);

  // Derived: region + stats for selected region
  const selectedRegion = useMemo(
    () => REGIONS.find(r => r.id === selectedRegionId) ?? null,
    [selectedRegionId],
  );
  const selectedStats = useMemo(
    () => selectedRegion ? getRegionStats(cells, selectedRegion.id) : null,
    [cells, selectedRegion],
  );

  // Tooltip region
  const tooltipRegion = useMemo(
    () => tooltip ? REGIONS.find(r => r.id === tooltip.cell.regionId) ?? null : null,
    [tooltip],
  );

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handleRegionClick = useCallback((id: string) => {
    setSelectedRegionId(prev => (prev === id ? null : id));
    setTooltip(null);
    // Automatically switch to coarse view when selecting a region
    if (viewMode === 'fine') setViewMode('coarse');
  }, [viewMode]);

  const handleRegionEnter = useCallback((id: string) => {
    setHoveredRegionId(id);
  }, []);

  const handleRegionLeave = useCallback(() => {
    setHoveredRegionId(null);
  }, []);

  const handleCellEnter = useCallback((cell: GridCell, screenX: number, screenY: number) => {
    setTooltip({ cell, screenX, screenY });
    setHoveredRegionId(cell.regionId);
  }, []);

  const handleCellLeave = useCallback(() => {
    setTooltip(null);
    setHoveredRegionId(null);
  }, []);

  const handleViewChange = useCallback((v: ViewMode) => {
    setViewMode(v);
    setTooltip(null);
    // Clear region selection when switching to fine-only (no region borders visible)
    if (v === 'fine') setSelectedRegionId(null);
  }, []);

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-[1280px] mx-auto px-6 py-8">

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div className="mb-8 border-b border-slate-100 pb-6">
          <div className="flex items-start justify-between gap-6 flex-wrap">
            <div>
              <h1 className="text-xl font-semibold text-slate-900 tracking-tight">
                Fine-Scale Poverty Estimation from Coarse Regional Labels
              </h1>
              <p className="mt-1 text-sm text-slate-500 max-w-2xl">
                Interactive demonstration of weakly supervised spatial prediction and aggregation.
                Click a region to explore its grid-cell decomposition.
              </p>
            </div>
            {/* Legend */}
            <div className="flex-shrink-0 pt-1">
              <Legend />
            </div>
          </div>
        </div>

        {/* ── Controls ───────────────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center gap-4 mb-5">
          {/* View toggle */}
          <div>
            <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-1.5">View</p>
            <div className="inline-flex border border-slate-200 rounded-lg overflow-hidden">
              {VIEW_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => handleViewChange(opt.value)}
                  className={`toggle-btn border-r last:border-r-0 border-slate-200 ${
                    viewMode === opt.value ? 'active' : 'inactive'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Resolution toggle (only relevant in fine/both modes) */}
          <div>
            <p className="text-[9px] uppercase tracking-widest text-slate-400 mb-1.5">Resolution</p>
            <div className="inline-flex border border-slate-200 rounded-lg overflow-hidden">
              {RES_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setResolution(opt.value)}
                  className={`toggle-btn border-r last:border-r-0 border-slate-200 flex flex-col items-center leading-none gap-0.5 py-1.5 ${
                    resolution === opt.value ? 'active' : 'inactive'
                  } ${viewMode === 'coarse' ? 'opacity-40 cursor-not-allowed' : ''}`}
                  disabled={viewMode === 'coarse'}
                >
                  <span>{opt.label}</span>
                  <span className={`text-[8px] ${resolution === opt.value ? 'text-slate-300' : 'text-slate-400'}`}>
                    {opt.sub}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Hint text */}
          {viewMode === 'coarse' && (
            <p className="text-[11px] text-slate-400 italic ml-2">
              Click a region to select it → then switch to &quot;Fine grid&quot; or &quot;Overlay both&quot;
            </p>
          )}
          {viewMode !== 'coarse' && !selectedRegionId && (
            <p className="text-[11px] text-slate-400 italic ml-2">
              Hover over cells to see predicted features
            </p>
          )}
        </div>

        {/* ── Main content: map + side panel ─────────────────────────────── */}
        <div className="flex gap-5 items-start">

          {/* Map container */}
          <div
            className="flex-1 min-w-0 border border-slate-200 rounded-xl overflow-hidden bg-slate-50"
            style={{ aspectRatio: '640 / 460' }}
          >
            <PovertyMapSVG
              cells={cells}
              viewMode={viewMode}
              selectedRegionId={selectedRegionId}
              hoveredRegionId={hoveredRegionId}
              onRegionClick={handleRegionClick}
              onRegionEnter={handleRegionEnter}
              onRegionLeave={handleRegionLeave}
              onCellEnter={handleCellEnter}
              onCellLeave={handleCellLeave}
            />
          </div>

          {/* Right panel */}
          <div className="w-72 flex-shrink-0 flex flex-col gap-4">
            {selectedRegion && selectedStats ? (
              <div className="border border-slate-200 rounded-xl p-4 bg-white">
                <RegionSummaryCard region={selectedRegion} stats={selectedStats} />
              </div>
            ) : (
              /* Empty state */
              <div className="border border-dashed border-slate-200 rounded-xl p-5 bg-slate-50 text-center">
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Select a region on the map to see its cell-level breakdown, predicted
                  vulnerability distribution, and aggregation accuracy.
                </p>
                <div className="mt-4 space-y-2">
                  {REGIONS.map(r => (
                    <button
                      key={r.id}
                      onClick={() => handleRegionClick(r.id)}
                      className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:border-slate-300 transition-colors text-left"
                    >
                      <span className="text-[11px] text-slate-700 font-medium">{r.name}</span>
                      <span className="text-[11px] font-mono text-slate-500">
                        {(r.officialPoverty * 100).toFixed(0)}%
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Aggregation panel (shows when region selected) ─────────────── */}
        {selectedRegion && selectedStats && (
          <div className="mt-5">
            <AggregationPanel region={selectedRegion} stats={selectedStats} />
          </div>
        )}

        {/* ── Steps explainer ────────────────────────────────────────────── */}
        <div className="mt-8 border-t border-slate-100 pt-6">
          <StepsExplainer />
        </div>

        {/* ── Footer ─────────────────────────────────────────────────────── */}
        <div className="mt-8 border-t border-slate-100 pt-4 flex justify-between items-center flex-wrap gap-2">
          <p className="text-[10px] text-slate-400">
            All data is synthetic and generated for demonstration purposes only.
          </p>
          <p className="text-[10px] text-slate-400">
            BorealisAI × UNICEF · Poverty Mapping Research Demo
          </p>
        </div>
      </div>

      {/* ── Floating cell tooltip ───────────────────────────────────────── */}
      {tooltip && tooltipRegion && (
        <CellTooltip
          cell={tooltip.cell}
          region={tooltipRegion}
          screenX={tooltip.screenX}
          screenY={tooltip.screenY}
        />
      )}
    </div>
  );
}
