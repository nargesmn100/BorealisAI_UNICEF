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
import CellTable from '@/components/CellTable';
import CellInsightsPanel from '@/components/CellInsightsPanel';

// ─── View / Resolution toggles ────────────────────────────────────────────────

const VIEW_OPTIONS: { value: ViewMode; label: string }[] = [
  { value: 'coarse', label: 'Coarse regions' },
  { value: 'fine',   label: 'Fine grid'      },
  { value: 'both',   label: 'Overlay both'   },
];

const RES_OPTIONS: { value: Resolution; label: string; sub: string }[] = [
  { value: 'coarse', label: 'Coarse', sub: '~140 cells' },
  { value: 'medium', label: 'Medium', sub: '~315 cells' },
  { value: 'fine',   label: 'Fine',   sub: '~560 cells' },
];

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
  const [selectedCellId,   setSelectedCellId]   = useState<string | null>(null);
  const [tooltip,          setTooltip]          = useState<TooltipInfo | null>(null);

  const cells = useMemo(() => generateCells(resolution), [resolution]);

  const selectedRegion = useMemo(
    () => REGIONS.find(r => r.id === selectedRegionId) ?? null,
    [selectedRegionId],
  );
  const selectedStats = useMemo(
    () => selectedRegion ? getRegionStats(cells, selectedRegion.id) : null,
    [cells, selectedRegion],
  );

  const selectedCell = useMemo(
    () => selectedCellId ? cells.find(c => c.id === selectedCellId) ?? null : null,
    [cells, selectedCellId],
  );

  const tooltipRegion = useMemo(
    () => tooltip ? REGIONS.find(r => r.id === tooltip.cell.regionId) ?? null : null,
    [tooltip],
  );

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handleRegionClick = useCallback((id: string) => {
    setSelectedRegionId(prev => (prev === id ? null : id));
    setSelectedCellId(null);
    setTooltip(null);
    if (viewMode === 'fine') setViewMode('coarse');
  }, [viewMode]);

  const handleRegionEnter  = useCallback((id: string) => setHoveredRegionId(id), []);
  const handleRegionLeave  = useCallback(() => setHoveredRegionId(null), []);

  const handleCellEnter = useCallback((cell: GridCell, screenX: number, screenY: number) => {
    setTooltip({ cell, screenX, screenY });
    setHoveredRegionId(cell.regionId);
  }, []);

  const handleCellLeave = useCallback(() => {
    setTooltip(null);
    setHoveredRegionId(null);
  }, []);

  /** Click on map cell — select cell + its region, switch to fine/both view */
  const handleMapCellClick = useCallback((cell: GridCell) => {
    setSelectedCellId(prev => (prev === cell.id ? null : cell.id));
    setSelectedRegionId(cell.regionId);
    if (viewMode === 'coarse') setViewMode('fine');
  }, [viewMode]);

  /** Click on table row — select cell + its region, ensure grid is visible */
  const handleTableCellSelect = useCallback((cell: GridCell) => {
    setSelectedCellId(prev => (prev === cell.id ? null : cell.id));
    setSelectedRegionId(cell.regionId);
    if (viewMode === 'coarse') setViewMode('fine');
  }, [viewMode]);

  const handleViewChange = useCallback((v: ViewMode) => {
    setViewMode(v);
    setTooltip(null);
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

          {/* Resolution toggle */}
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

          {/* Context hint */}
          {viewMode === 'coarse' && !selectedRegionId && (
            <p className="text-[11px] text-slate-400 italic ml-2">
              Click a region to select it
            </p>
          )}
          {viewMode !== 'coarse' && !selectedCellId && (
            <p className="text-[11px] text-slate-400 italic ml-2">
              Click a cell to pin it · hover to preview
            </p>
          )}
          {selectedCellId && (
            <button
              onClick={() => setSelectedCellId(null)}
              className="ml-auto text-[10px] text-slate-400 hover:text-slate-600 border border-slate-200 rounded px-2 py-1"
            >
              Clear cell selection
            </button>
          )}
        </div>

        {/* ── Main content: map + side panel ─────────────────────────────── */}
        <div className="flex gap-5 items-start">
          {/* Map */}
          <div
            className="flex-1 min-w-0 border border-slate-200 rounded-xl overflow-hidden bg-slate-50"
            style={{ aspectRatio: '640 / 460' }}
          >
            <PovertyMapSVG
              cells={cells}
              viewMode={viewMode}
              selectedRegionId={selectedRegionId}
              hoveredRegionId={hoveredRegionId}
              selectedCellId={selectedCellId}
              onRegionClick={handleRegionClick}
              onRegionEnter={handleRegionEnter}
              onRegionLeave={handleRegionLeave}
              onCellEnter={handleCellEnter}
              onCellLeave={handleCellLeave}
              onCellClick={handleMapCellClick}
            />
          </div>

          {/* Right panel */}
          <div className="w-72 flex-shrink-0 flex flex-col gap-4">
            {selectedRegion && selectedStats ? (
              <div className="border border-slate-200 rounded-xl p-4 bg-white">
                <RegionSummaryCard region={selectedRegion} stats={selectedStats} />
              </div>
            ) : (
              <div className="border border-dashed border-slate-200 rounded-xl p-5 bg-slate-50 text-center">
                <p className="text-[11px] text-slate-400 leading-relaxed mb-4">
                  Select a region to see cell breakdown, score distribution, and aggregation accuracy.
                </p>
                <div className="space-y-2">
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

            {/* Selected cell mini-card */}
            {selectedCell && (
              <div className="border border-blue-200 rounded-xl p-4 bg-blue-50">
                <p className="text-[9px] uppercase tracking-widest text-blue-400 mb-2">
                  Selected cell
                </p>
                <p className="text-[11px] font-mono text-blue-700 font-semibold mb-2 truncate">
                  {selectedCell.id}
                </p>
                <div className="space-y-1">
                  {[
                    ['Predicted poverty', `${(selectedCell.predictedPoverty * 100).toFixed(1)}%`],
                    ['Settlement',        selectedCell.settlementType],
                    ['Population',        selectedCell.population.toLocaleString()],
                    ['Night lights',      `${selectedCell.nightLights}/100`],
                    ['Bldg density',      `${(selectedCell.buildingDensity * 100).toFixed(0)}%`],
                    ['Accessibility',     `${(selectedCell.accessibility  * 100).toFixed(0)}%`],
                  ].map(([label, value]) => (
                    <div key={label} className="flex justify-between">
                      <span className="text-[10px] text-blue-500">{label}</span>
                      <span className="text-[10px] font-mono text-blue-800 font-medium">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Aggregation panel ──────────────────────────────────────────── */}
        {selectedRegion && selectedStats && (
          <div className="mt-5">
            <AggregationPanel region={selectedRegion} stats={selectedStats} />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            CELL EXPLORER — table + insights
        ══════════════════════════════════════════════════════════════════ */}
        <div className="mt-8 border-t border-slate-100 pt-6">
          <div className="flex items-baseline gap-3 mb-5">
            <h2 className="text-sm font-semibold text-slate-800">Cell Explorer</h2>
            <span className="text-[11px] text-slate-400">
              {cells.length.toLocaleString()} cells at {resolution} resolution ·
              click any row to select and highlight on the map
            </span>
          </div>

          {/* Table + insights side by side on large screens */}
          <div className="flex flex-col xl:flex-row gap-6 items-start">
            {/* Table — takes ~60% */}
            <div className="flex-1 min-w-0">
              <CellTable
                cells={cells}
                selectedCellId={selectedCellId}
                onCellSelect={handleTableCellSelect}
              />
            </div>

            {/* Insights panel — 40% */}
            <div className="xl:w-[400px] flex-shrink-0 w-full">
              <CellInsightsPanel
                cells={cells}
                selectedCell={selectedCell}
                selectedRegion={selectedRegion}
              />
            </div>
          </div>
        </div>

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

      {/* Floating cell tooltip */}
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
