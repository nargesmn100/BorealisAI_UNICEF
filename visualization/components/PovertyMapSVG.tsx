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
  /** Cell that was clicked (persistent selection) */
  selectedCellId: string | null;
  onRegionClick: (id: string) => void;
  onRegionEnter: (id: string) => void;
  onRegionLeave: () => void;
  onCellEnter: (cell: GridCell, screenX: number, screenY: number) => void;
  onCellLeave: () => void;
  /** Fired when user clicks a cell in fine/both view */
  onCellClick: (cell: GridCell) => void;
}

function pts(poly: [number, number][]): string {
  return poly.map(([x, y]) => `${x},${y}`).join(' ');
}

function regionCentroid(poly: [number, number][]): [number, number] {
  const n = poly.length;
  return [
    poly.reduce((s, [x]) => s + x, 0) / n,
    poly.reduce((s, [, y]) => s + y, 0) / n,
  ];
}

export default function PovertyMapSVG({
  cells,
  viewMode,
  selectedRegionId,
  hoveredRegionId,
  selectedCellId,
  onRegionClick,
  onRegionEnter,
  onRegionLeave,
  onCellEnter,
  onCellLeave,
  onCellClick,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  const toSvgCoords = useCallback((clientX: number, clientY: number): [number, number] => {
    const el = svgRef.current;
    if (!el) return [0, 0];
    const rect = el.getBoundingClientRect();
    return [
      ((clientX - rect.left) / rect.width)  * SVG_WIDTH,
      ((clientY - rect.top)  / rect.height) * SVG_HEIGHT,
    ];
  }, []);

  /** Find the cell under a given SVG coordinate */
  const cellAtCoords = useCallback((svgX: number, svgY: number): GridCell | undefined => {
    return cells.find(
      c => svgX >= c.x && svgX < c.x + c.width && svgY >= c.y && svgY < c.y + c.height,
    );
  }, [cells]);

  const handleSvgMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (viewMode === 'coarse') return;
    const [svgX, svgY] = toSvgCoords(e.clientX, e.clientY);
    const hit = cellAtCoords(svgX, svgY);
    if (hit) onCellEnter(hit, e.clientX, e.clientY);
    else onCellLeave();
  }, [viewMode, toSvgCoords, cellAtCoords, onCellEnter, onCellLeave]);

  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (viewMode === 'coarse') return;
    const [svgX, svgY] = toSvgCoords(e.clientX, e.clientY);
    const hit = cellAtCoords(svgX, svgY);
    if (hit) onCellClick(hit);
  }, [viewMode, toSvgCoords, cellAtCoords, onCellClick]);

  const showRegions = viewMode === 'coarse' || viewMode === 'both';
  const showCells   = viewMode === 'fine'   || viewMode === 'both';

  // Find selected cell object for rendering highlight
  const selectedCell = selectedCellId ? cells.find(c => c.id === selectedCellId) : null;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
      className="w-full h-full"
      style={{ display: 'block' }}
      onMouseMove={handleSvgMouseMove}
      onMouseLeave={onCellLeave}
      onClick={handleSvgClick}
    >
      <defs>
        {REGIONS.map(r => (
          <clipPath key={`clip-${r.id}`} id={`clip-${r.id}`}>
            <polygon points={pts(r.polygon)} />
          </clipPath>
        ))}
      </defs>

      {/* Background */}
      <rect x={0} y={0} width={SVG_WIDTH} height={SVG_HEIGHT} fill="#f1f5f9" rx={4} />

      {/* ── Region fills (coarse layer) ──────────────────────────────────── */}
      {showRegions && REGIONS.map(region => {
        const isSelected    = selectedRegionId === region.id;
        const otherSelected = selectedRegionId !== null && !isSelected;

        let fillOpacity = 1;
        if (viewMode === 'both')  fillOpacity = 0.30;
        if (otherSelected)        fillOpacity = viewMode === 'both' ? 0.18 : 0.40;

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
            style={{ cursor: 'pointer' }}
            onClick={() => onRegionClick(region.id)}
            onMouseEnter={() => onRegionEnter(region.id)}
            onMouseLeave={onRegionLeave}
          />
        );
      })}

      {/* Selected region ring */}
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
        const regionCells      = cells.filter(c => c.regionId === region.id);
        const isRegionSelected = selectedRegionId === region.id;
        const anySelected      = selectedRegionId !== null;

        return (
          <g key={`cells-${region.id}`} clipPath={`url(#clip-${region.id})`}>
            {regionCells.map(cell => {
              const dimmed      = anySelected && !isRegionSelected;
              const isCellSel   = cell.id === selectedCellId;
              return (
                <rect
                  key={cell.id}
                  className="cell-rect"
                  x={cell.x}
                  y={cell.y}
                  width={cell.width  - 0.4}
                  height={cell.height - 0.4}
                  fill={povertyColor(cell.predictedPoverty)}
                  fillOpacity={dimmed ? 0.28 : isCellSel ? 1 : 0.92}
                  stroke={viewMode === 'both' ? '#fff' : '#e2e8f0'}
                  strokeWidth={0.3}
                />
              );
            })}
          </g>
        );
      })}

      {/* ── Selected cell highlight ring ─────────────────────────────────── */}
      {selectedCell && showCells && (
        <rect
          x={selectedCell.x + 0.5}
          y={selectedCell.y + 0.5}
          width={selectedCell.width  - 1}
          height={selectedCell.height - 1}
          fill="none"
          stroke="#2563eb"
          strokeWidth={1.8}
          rx={1}
          pointerEvents="none"
        />
      )}

      {/* ── Region outlines on fine/both views ──────────────────────────── */}
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
        const [lx, ly]     = region.labelPos ?? regionCentroid(region.polygon);
        const otherSelected = selectedRegionId && selectedRegionId !== region.id;
        const opacity       = otherSelected ? 0.45 : 1;

        return (
          <g key={`label-${region.id}`} style={{ opacity, pointerEvents: 'none' }}>
            <text
              x={lx} y={ly - 7}
              textAnchor="middle" fontSize={11} fontWeight={600}
              fontFamily="Inter, system-ui, sans-serif"
              fill={povertyTextColor(region.officialPoverty)}
              style={{ userSelect: 'none' }}
            >
              {region.name}
            </text>
            <text
              x={lx} y={ly + 9}
              textAnchor="middle" fontSize={13} fontWeight={700}
              fontFamily="'JetBrains Mono', monospace"
              fill={povertyTextColor(region.officialPoverty)}
              style={{ userSelect: 'none' }}
            >
              {fmtPct(region.officialPoverty, 0)}
            </text>
          </g>
        );
      })}

      <text
        x={8} y={SVG_HEIGHT - 6}
        fontSize={8} fill="#94a3b8"
        fontFamily="Inter, system-ui, sans-serif"
        style={{ userSelect: 'none' }}
      >
        Stylised synthetic territory · not geographic data
      </text>
    </svg>
  );
}
