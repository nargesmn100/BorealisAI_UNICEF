'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { GridCell, REGIONS, povertyColor, fmtPct } from '@/lib/data';

const PAGE_SIZE = 15;

type SortKey = 'predictedPoverty' | 'population' | 'nightLights' | 'buildingDensity' | 'accessibility';
type SortDir  = 'asc' | 'desc';

interface Props {
  cells: GridCell[];
  selectedCellId: string | null;
  onCellSelect: (cell: GridCell) => void;
}

// ─── Column config ────────────────────────────────────────────────────────────

const COLUMNS: { key: SortKey | 'id' | 'regionId' | 'settlementType'; label: string; sortable: boolean; align: 'left' | 'right' }[] = [
  { key: 'id',              label: 'Cell ID',      sortable: false, align: 'left'  },
  { key: 'regionId',        label: 'Region',       sortable: false, align: 'left'  },
  { key: 'settlementType',  label: 'Settlement',   sortable: false, align: 'left'  },
  { key: 'predictedPoverty',label: 'Poverty',      sortable: true,  align: 'right' },
  { key: 'population',      label: 'Population',   sortable: true,  align: 'right' },
  { key: 'nightLights',     label: 'Night Lights', sortable: true,  align: 'right' },
  { key: 'buildingDensity', label: 'Bldg Density', sortable: true,  align: 'right' },
  { key: 'accessibility',   label: 'Access',       sortable: true,  align: 'right' },
];

// ─── Small poverty dot ────────────────────────────────────────────────────────

function PovertyDot({ score }: { score: number }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block w-2 h-2 rounded-sm flex-shrink-0"
        style={{ backgroundColor: povertyColor(score) }}
      />
      <span className="font-mono">{fmtPct(score)}</span>
    </span>
  );
}

// ─── Sort indicator ───────────────────────────────────────────────────────────

function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
  return (
    <span className={`ml-1 text-[9px] ${active ? 'text-slate-700' : 'text-slate-300'}`}>
      {active ? (dir === 'asc' ? '↑' : '↓') : '↕'}
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function CellTable({ cells, selectedCellId, onCellSelect }: Props) {
  const [filterRegion, setFilterRegion] = useState<string>('all');
  const [sortKey,      setSortKey]      = useState<SortKey>('predictedPoverty');
  const [sortDir,      setSortDir]      = useState<SortDir>('desc');
  const [page,         setPage]         = useState(0);
  const [search,       setSearch]       = useState('');

  const selectedRowRef = useRef<HTMLTableRowElement>(null);
  const tbodyRef       = useRef<HTMLTableSectionElement>(null);

  // ── Filtered + sorted data ──────────────────────────────────────────────────
  const filtered = useMemo(() => {
    let data = cells;
    if (filterRegion !== 'all') data = data.filter(c => c.regionId === filterRegion);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      data = data.filter(c =>
        c.id.toLowerCase().includes(q) ||
        c.regionId.toLowerCase().includes(q) ||
        c.settlementType.toLowerCase().includes(q),
      );
    }
    return [...data].sort((a, b) => {
      const factor = sortDir === 'asc' ? 1 : -1;
      return factor * (a[sortKey] - b[sortKey]);
    });
  }, [cells, filterRegion, sortKey, sortDir, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageItems  = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  // ── When selectedCellId changes externally (e.g. click on map) → jump page ──
  useEffect(() => {
    if (!selectedCellId) return;
    const idx = filtered.findIndex(c => c.id === selectedCellId);
    if (idx === -1) return;
    const targetPage = Math.floor(idx / PAGE_SIZE);
    setPage(targetPage);
  }, [selectedCellId, filtered]);

  // ── Scroll selected row into view after page change ──────────────────────────
  useEffect(() => {
    if (selectedRowRef.current) {
      selectedRowRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [selectedCellId, page]);

  // ── Sort handler ─────────────────────────────────────────────────────────────
  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
    setPage(0);
  }, [sortKey]);

  // ── Region filter options ─────────────────────────────────────────────────────
  const regionOptions = useMemo(() => REGIONS.map(r => ({ id: r.id, name: r.name })), []);

  return (
    <div className="flex flex-col gap-3">
      {/* ── Controls bar ──────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Region filter */}
        <div className="flex items-center gap-1.5">
          <label className="text-[10px] uppercase tracking-widest text-slate-400">Region</label>
          <select
            value={filterRegion}
            onChange={e => { setFilterRegion(e.target.value); setPage(0); }}
            className="text-xs border border-slate-200 rounded-md px-2 py-1 bg-white text-slate-700 focus:outline-none focus:ring-1 focus:ring-slate-300"
          >
            <option value="all">All regions</option>
            {regionOptions.map(r => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </select>
        </div>

        {/* Search */}
        <div className="flex items-center gap-1.5">
          <label className="text-[10px] uppercase tracking-widest text-slate-400">Search</label>
          <input
            type="text"
            placeholder="Cell ID, region, settlement…"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            className="text-xs border border-slate-200 rounded-md px-2.5 py-1 w-48 bg-white text-slate-700 placeholder-slate-300 focus:outline-none focus:ring-1 focus:ring-slate-300"
          />
        </div>

        {/* Count */}
        <span className="text-[10px] text-slate-400 ml-auto">
          {filtered.length.toLocaleString()} cells
          {selectedCellId && ' · 1 selected'}
        </span>
      </div>

      {/* ── Table ─────────────────────────────────────────────────────────── */}
      <div className="border border-slate-200 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                {COLUMNS.map(col => (
                  <th
                    key={col.key}
                    className={`px-3 py-2 font-medium text-slate-500 whitespace-nowrap select-none ${
                      col.align === 'right' ? 'text-right' : 'text-left'
                    } ${col.sortable ? 'cursor-pointer hover:text-slate-700' : ''}`}
                    onClick={col.sortable ? () => handleSort(col.key as SortKey) : undefined}
                  >
                    {col.label}
                    {col.sortable && (
                      <SortIcon
                        active={sortKey === col.key}
                        dir={sortDir}
                      />
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody ref={tbodyRef} className="divide-y divide-slate-100">
              {pageItems.length === 0 ? (
                <tr>
                  <td colSpan={COLUMNS.length} className="px-3 py-8 text-center text-slate-400 text-xs">
                    No cells match the current filter.
                  </td>
                </tr>
              ) : (
                pageItems.map(cell => {
                  const isSelected = cell.id === selectedCellId;
                  const regionName = REGIONS.find(r => r.id === cell.regionId)?.name ?? cell.regionId;
                  return (
                    <tr
                      key={cell.id}
                      ref={isSelected ? selectedRowRef : undefined}
                      onClick={() => onCellSelect(cell)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-blue-50 hover:bg-blue-50'
                          : 'hover:bg-slate-50'
                      }`}
                    >
                      {/* Cell ID */}
                      <td className="px-3 py-1.5 font-mono text-slate-500">
                        <span className={`${isSelected ? 'text-blue-600 font-semibold' : ''}`}>
                          {cell.id}
                        </span>
                        {isSelected && (
                          <span className="ml-1.5 inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
                        )}
                      </td>
                      {/* Region */}
                      <td className="px-3 py-1.5 text-slate-700">{regionName}</td>
                      {/* Settlement */}
                      <td className="px-3 py-1.5">
                        <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-medium ${
                          cell.settlementType === 'Urban'
                            ? 'bg-blue-50 text-blue-600'
                            : cell.settlementType === 'Peri-urban'
                            ? 'bg-amber-50 text-amber-600'
                            : 'bg-green-50 text-green-700'
                        }`}>
                          {cell.settlementType}
                        </span>
                      </td>
                      {/* Poverty */}
                      <td className="px-3 py-1.5 text-right">
                        <PovertyDot score={cell.predictedPoverty} />
                      </td>
                      {/* Population */}
                      <td className="px-3 py-1.5 text-right font-mono text-slate-600">
                        {cell.population.toLocaleString()}
                      </td>
                      {/* Night lights */}
                      <td className="px-3 py-1.5 text-right">
                        <NightLightsBar value={cell.nightLights} />
                      </td>
                      {/* Building density */}
                      <td className="px-3 py-1.5 text-right font-mono text-slate-600">
                        {(cell.buildingDensity * 100).toFixed(0)}%
                      </td>
                      {/* Accessibility */}
                      <td className="px-3 py-1.5 text-right font-mono text-slate-600">
                        {(cell.accessibility * 100).toFixed(0)}%
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Pagination ────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-400">
          Page {page + 1} of {totalPages} · showing {pageItems.length} of {filtered.length}
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setPage(0)}
            disabled={page === 0}
            className="px-2 py-1 text-[10px] border border-slate-200 rounded disabled:opacity-30 hover:bg-slate-50 transition-colors"
          >
            «
          </button>
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className="px-2 py-1 text-[10px] border border-slate-200 rounded disabled:opacity-30 hover:bg-slate-50 transition-colors"
          >
            ‹
          </button>
          {/* Page numbers — show up to 5 around current */}
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            const start = Math.max(0, Math.min(page - 2, totalPages - 5));
            const p = start + i;
            return (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={`px-2 py-1 text-[10px] border rounded transition-colors ${
                  p === page
                    ? 'bg-slate-900 text-white border-slate-900'
                    : 'border-slate-200 hover:bg-slate-50'
                }`}
              >
                {p + 1}
              </button>
            );
          })}
          <button
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className="px-2 py-1 text-[10px] border border-slate-200 rounded disabled:opacity-30 hover:bg-slate-50 transition-colors"
          >
            ›
          </button>
          <button
            onClick={() => setPage(totalPages - 1)}
            disabled={page >= totalPages - 1}
            className="px-2 py-1 text-[10px] border border-slate-200 rounded disabled:opacity-30 hover:bg-slate-50 transition-colors"
          >
            »
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Night lights mini-bar ────────────────────────────────────────────────────

function NightLightsBar({ value }: { value: number }) {
  const W = 48, H = 8;
  const fill = Math.round((value / 100) * W);
  const color = value > 60 ? '#fbbf24' : value > 30 ? '#f97316' : '#cbd5e1';
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width={W} height={H} className="flex-shrink-0">
        <rect x={0} y={0} width={W} height={H} rx={2} fill="#f1f5f9" />
        <rect x={0} y={0} width={fill} height={H} rx={2} fill={color} />
      </svg>
      <span className="font-mono text-slate-500 text-[10px]">{value}</span>
    </span>
  );
}
