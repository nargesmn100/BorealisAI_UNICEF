'use client';

import React, { useRef, useCallback } from 'react';
import {
  REGIONS,
  SVG_WIDTH,
  SVG_HEIGHT,
  ViewMode,
  GridCell,
  Region,
  povertyColor,
  povertyTextColor,
  fmtPct,
} from '@/lib/data';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface TooltipState {
  visible: boolean;
  screenX: number;
  screenY: number;
  cell: GridCell | null;
  region: Region | null;
}

interface Props {
  cells: GridCell[];
  viewMode: ViewMode;
  selectedRegionId: string | null;
  hoveredRegionId: string | null;
  onRegionClick: (id: string) => void;
  onRegionEnter: (id: string) => void;
  onRegionLeave: () => void;
  onCellEnter: (cell: GridCell, screenX: number, screenY: number) => void;
  onCellLeave: () => void;
}

// ─── Helper: polygon points string ───────────────────────────────────────────

function pts(poly: [number, number][]): string {
  return poly.map(([x, y]) => `${x},${y}`).join(' ');
}

// ─── Region label positions (label centroid fallback) ─────────────────────────

function regionCentroid(poly: [number, number][]): [number, number] {
  const n = poly.length;
  return [
    poly.reduce((s, [x]) => s + x, 0) / n,
    poly.reduce((s, [, y]) => s + y, 0) / n,
  ];
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function PovertyMapSVG({
  cells,
  viewMode,
  selectedRegionId,
  hoveredRegionId,
  onRegionClick,
  onRegionEnter,
  onRegionLeave,
  onCellEnter,
  onCellLeave,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  // Convert screen coords → SVG coords for cell hover hit-testing
  const toSvgCoords = useCallback((clientX: number, clientY: number): [number, number] => {
    const el = svgRef.current;
    if (!el) return [0, 0];
    const rect = el.getBoundingClientRect();
    return [
      ((clientX - rect.left) / rect.width)  * SVG_WIDTH,
      ((clientY - rect.top)  / rect.height) * SVG_HEIGHT,
    ];
  }, []);

  const handleSvgMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (viewMode === 'coarse') return;
    const [svgX, svgY] = toSvgCoords(e.clientX, e.clientY);
    // Find the cell under the cursor
    const hit = cells.find(
      c => svgX >= c.x && svgX < c.x + c.width && svgY >= c.y && svgY < c.y + c.height,
    );
    if (hit) onCellEnter(hit, e.clientX, e.clientY);
    else onCellLeave();
  }, [cells, viewMode, toSvgCoords, onCellEnter, onCellLeave]);

  const showRegions = viewMode === 'coarse' || viewMode === 'both';
  const showCells   = viewMode === 'fine'   || viewMode === 'both';

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
      className="w-full h-full"
      style={{ display: 'block' }}
      onMouseMove={handleSvgMouseMove}
      onMouseLeave={onCellLeave}
    >
      {/* ── Clip-path defs (one per region) ─────────────────────────────── */}
      <defs>
        {REGIONS.map(r => (
          <clipPath key={`clip-${r.id}`} id={`clip-${r.id}`}>
            <polygon points={pts(r.polygon)} />
          </clipPath>
        ))}
        {/* Subtle inset shadow for selected region */}
        <filter id="selected-glow" x="-5%" y="-5%" width="110%" height="110%">
          <feFlood floodColor="#2563eb" floodOpacity="0.25" result="flood" />
          <feComposite in="flood" in2="SourceGraphic" operator="in" result="masked" />
          <feMorphology in="masked" operator="dilate" radius="3" result="dilated" />
          <feMerge>
            <feMergeNode in="dilated" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* ── Background ───────────────────────────────────────────────────── */}
      <rect x={0} y={0} width={SVG_WIDTH} height={SVG_HEIGHT} fill="#f1f5f9" rx={4} />

      {/* ── Region fills (coarse layer) ──────────────────────────────────── */}
      {showRegions && REGIONS.map(region => {
        const isSelected = selectedRegionId === region.id;
        const isHovered  = hoveredRegionId  === region.id;
        const otherSelected = selectedRegionId && selectedRegionId !== region.id;

        let fillOpacity = 1;
        if (viewMode === 'both') fillOpacity = 0.30; // semi-transparent when cells overlay
        if (otherSelected)       fillOpacity = viewMode === 'both' ? 0.18 : 0.40;
        if (isHovered && viewMode === 'coarse') fillOpacity = 0.85;

        return (
          <polygon
            key={`region-${region.id}`}
            className="region-polygon"
            points={pts(region.polygon)}
            fill={povertyColor(region.officialPoverty)}
            fillOpacity={fillOpacity}
            stroke="#ffffff"
            strokeWidth={viewMode === 'both' ? 1 : 2}
            strokeLinejoin="round"
            style={{
              outline: isSelected ? '2px solid #2563eb' : 'none',
              cursor: 'pointer',
            }}
            onClick={() => onRegionClick(region.id)}
            onMouseEnter={() => onRegionEnter(region.id)}
            onMouseLeave={onRegionLeave}
          />
        );
      })}

      {/* ── Selected region highlight ring ───────────────────────────────── */}
      {selectedRegionId && viewMode === 'coarse' && (() => {
        const r = REGIONS.find(x => x.id === selectedRegionId);
        if (!r) return null;
        return (
          <polygon
            points={pts(r.polygon)}
            fill="none"
            stroke="#2563eb"
            strokeWidth={3}
            strokeLinejoin="round"
            pointerEvents="none"
          />
        );
      })()}

      {/* ── Grid cells (fine layer) ──────────────────────────────────────── */}
      {showCells && REGIONS.map(region => {
        const regionCells = cells.filter(c => c.regionId === region.id);
        const isRegionSelected = selectedRegionId === region.id;
        const anySelected      = selectedRegionId !== null;

        return (
          <g key={`cells-${region.id}`} clipPath={`url(#clip-${region.id})`}>
            {regionCells.map(cell => {
              const dimmed = anySelected && !isRegionSelected;
              return (
                <rect
                  key={cell.id}
                  className="cell-rect"
                  x={cell.x}
                  y={cell.y}
                  width={cell.width - 0.4}
                  height={cell.height - 0.4}
                  fill={povertyColor(cell.predictedPoverty)}
                  fillOpacity={dimmed ? 0.30 : 0.92}
                  stroke={viewMode === 'both' ? '#fff' : '#e2e8f0'}
                  strokeWidth={0.3}
                />
              );
            })}
          </g>
        );
      })}

      {/* ── Region outlines on fine/both views (for spatial context) ─────── */}
      {(viewMode === 'fine' || viewMode === 'both') && REGIONS.map(region => (
        <polygon
          key={`outline-${region.id}`}
          points={pts(region.polygon)}
          fill="none"
          stroke={selectedRegionId === region.id ? '#2563eb' : '#94a3b8'}
          strokeWidth={selectedRegionId === region.id ? 2.5 : 1}
          strokeLinejoin="round"
          pointerEvents="none"
        />
      ))}

      {/* ── Region labels (coarse view only) ────────────────────────────── */}
      {viewMode === 'coarse' && REGIONS.map(region => {
        const [lx, ly] = region.labelPos ?? regionCentroid(region.polygon);
        const otherSelected = selectedRegionId && selectedRegionId !== region.id;
        const opacity = otherSelected ? 0.45 : 1;

        return (
          <g
            key={`label-${region.id}`}
            style={{ opacity, cursor: 'pointer', pointerEvents: 'none' }}
          >
            {/* Name */}
            <text
              x={lx}
              y={ly - 7}
              textAnchor="middle"
              fontSize={11}
              fontWeight={600}
              fontFamily="Inter, system-ui, sans-serif"
              fill={povertyTextColor(region.officialPoverty)}
              style={{ userSelect: 'none' }}
            >
              {region.name}
            </text>
            {/* Official poverty score badge */}
            <text
              x={lx}
              y={ly + 9}
              textAnchor="middle"
              fontSize={13}
              fontWeight={700}
              fontFamily="'JetBrains Mono', monospace"
              fill={povertyTextColor(region.officialPoverty)}
              style={{ userSelect: 'none' }}
            >
              {fmtPct(region.officialPoverty, 0)}
            </text>
          </g>
        );
      })}

      {/* ── Scale bar (bottom-left) ──────────────────────────────────────── */}
      <text
        x={8}
        y={SVG_HEIGHT - 6}
        fontSize={8}
        fill="#94a3b8"
        fontFamily="Inter, system-ui, sans-serif"
        style={{ userSelect: 'none' }}
      >
        Stylised synthetic territory · not geographic data
      </text>
    </svg>
  );
}
