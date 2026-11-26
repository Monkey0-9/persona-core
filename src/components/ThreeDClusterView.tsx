import { useMemo, useRef, useState, type PointerEvent } from 'react';
import { RotateCcw } from 'lucide-react';
import {
  SEGMENT_PALETTE,
  type AlgorithmPoint,
  type CustomerRecord,
  type PersonaProfile,
  type ScoreResult,
} from '../types';
import type { AlgorithmView } from './Charts';

interface ProjectedPoint {
  customer: CustomerRecord;
  sx: number;
  sy: number;
  depth: number;
  color: string;
  clusterId: number;
  isNoise: boolean;
}

const VIEW_PRESETS = [
  { label: '3D Isometric', yaw: -0.68, pitch: 0.44 },
  { label: 'Income × Spending', yaw: 0.0, pitch: 0.0 },
  { label: 'Age × Spending', yaw: Math.PI / 2, pitch: 0.0 },
  { label: 'Age × Income', yaw: 0.0, pitch: Math.PI / 2 },
] as const;

export function ThreeDClusterView({
  customers,
  personas,
  colorMap,
  algorithmView = 'kmeans',
  algorithmPoints = [],
  highlightedClusterId = null,
  livePin = null,
  onHover,
}: {
  customers: CustomerRecord[];
  personas: PersonaProfile[];
  colorMap: Map<string, string>;
  algorithmView?: AlgorithmView;
  algorithmPoints?: AlgorithmPoint[];
  highlightedClusterId?: number | null;
  livePin?: ScoreResult | null;
  onHover?: (c: CustomerRecord | null) => void;
}) {
  const [yaw, setYaw] = useState<number>(-0.68);
  const [pitch, setPitch] = useState<number>(0.44);
  const [hovered, setHovered] = useState<CustomerRecord | null>(null);
  const dragRef = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);

  const width = 620;
  const height = 400;
  const cx = width / 2;
  const cy = height / 2 + 4;
  const scale = 104;

  const bounds = useMemo(() => {
    const ages = customers.map((c) => c.age);
    const incs = customers.map((c) => c.annual_income_k);
    return {
      ageMin: Math.min(18, ...(ages.length ? ages : [18])),
      ageMax: Math.max(70, ...(ages.length ? ages : [70])),
      incMin: Math.min(15, ...(incs.length ? incs : [15])),
      incMax: Math.max(135, ...(incs.length ? incs : [135])),
      spMin: 0,
      spMax: 100,
    };
  }, [customers]);

  const project = useMemo(() => {
    const cosY = Math.cos(yaw);
    const sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch);
    const sinP = Math.sin(pitch);

    return (age: number, inc: number, sp: number) => {
      // Normalize coordinates to [-1, 1] cube: X=Income, Y=Spending, Z=Age
      const nx = ((inc - bounds.incMin) / Math.max(bounds.incMax - bounds.incMin, 1)) * 2 - 1;
      const ny = ((sp - bounds.spMin) / Math.max(bounds.spMax - bounds.spMin, 1)) * 2 - 1;
      const nz = ((age - bounds.ageMin) / Math.max(bounds.ageMax - bounds.ageMin, 1)) * 2 - 1;

      const x1 = nx * cosY - nz * sinY;
      const z1 = nx * sinY + nz * cosY;
      const y2 = ny * cosP - z1 * sinP;
      const z2 = ny * sinP + z1 * cosP;

      const persp = 3.2 / (3.2 - z2 * 0.55);
      return {
        sx: cx + x1 * scale * persp,
        sy: cy - y2 * scale * persp,
        depth: z2,
      };
    };
  }, [yaw, pitch, bounds, cx, cy]);

  const algoLookup = useMemo(() => {
    const map = new Map<string, AlgorithmPoint>();
    for (const ap of algorithmPoints) {
      if (ap.customer_ref) map.set(ap.customer_ref, ap);
    }
    return map;
  }, [algorithmPoints]);

  const projectedPoints = useMemo<ProjectedPoint[]>(() => {
    const items = customers.map((pt) => {
      const p = project(pt.age, pt.annual_income_k, pt.spending_score);
      let color =
        colorMap.get(pt.persona_name) ??
        SEGMENT_PALETTE[Math.abs(pt.cluster_id) % SEGMENT_PALETTE.length];
      let clusterId = pt.cluster_id;
      let isNoise = false;

      if (algorithmView !== 'kmeans') {
        const match = pt.customer_ref ? algoLookup.get(pt.customer_ref) : undefined;
        if (match) {
          if (algorithmView === 'ward') {
            clusterId = match.ward_cluster_id;
            color = SEGMENT_PALETTE[Math.abs(clusterId) % SEGMENT_PALETTE.length];
          } else if (algorithmView === 'gmm') {
            clusterId = match.gmm_cluster_id;
            color = SEGMENT_PALETTE[Math.abs(clusterId) % SEGMENT_PALETTE.length];
          } else if (algorithmView === 'dbscan') {
            clusterId = match.dbscan_cluster_id;
            isNoise = clusterId === -1;
            color = isNoise ? '#BE123C' : SEGMENT_PALETTE[Math.abs(clusterId) % SEGMENT_PALETTE.length];
          } else if (algorithmView === 'hdbscan') {
            clusterId = match.hdbscan_cluster_id ?? -1;
            isNoise = clusterId === -1;
            color = isNoise ? '#BE123C' : SEGMENT_PALETTE[Math.abs(clusterId) % SEGMENT_PALETTE.length];
          }
        }
      }
      return {
        customer: pt,
        sx: p.sx,
        sy: p.sy,
        depth: p.depth,
        color,
        clusterId,
        isNoise,
      };
    });
    return items.sort((a, b) => a.depth - b.depth);
  }, [customers, project, colorMap, algorithmView, algoLookup]);

  const projectedCentroids = useMemo(() => {
    if (algorithmView !== 'kmeans') return [];
    return personas.map((p, idx) => {
      const coords = project(p.mean_age, p.mean_income, p.mean_spending);
      return {
        persona: p,
        color: colorMap.get(p.persona_name) ?? SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length],
        ...coords,
      };
    });
  }, [algorithmView, personas, project, colorMap]);

  // 3D wireframe box edges and axis labels
  const boxEdges = useMemo(() => {
    const corners = [
      project(bounds.ageMin, bounds.incMin, bounds.spMin), // 0: min all
      project(bounds.ageMin, bounds.incMax, bounds.spMin), // 1: +Income
      project(bounds.ageMin, bounds.incMin, bounds.spMax), // 2: +Spend
      project(bounds.ageMin, bounds.incMax, bounds.spMax), // 3
      project(bounds.ageMax, bounds.incMin, bounds.spMin), // 4: +Age
      project(bounds.ageMax, bounds.incMax, bounds.spMin), // 5
      project(bounds.ageMax, bounds.incMin, bounds.spMax), // 6
      project(bounds.ageMax, bounds.incMax, bounds.spMax), // 7
    ];
    const pairs = [
      [0, 1],
      [0, 2],
      [0, 4],
      [1, 3],
      [1, 5],
      [2, 3],
      [2, 6],
      [3, 7],
      [4, 5],
      [4, 6],
      [5, 7],
      [6, 7],
    ];
    return { corners, pairs };
  }, [project, bounds]);

  const handlePointerDown = (e: PointerEvent<SVGSVGElement>) => {
    dragRef.current = { x: e.clientX, y: e.clientY, yaw, pitch };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    setYaw(dragRef.current.yaw + dx * 0.008);
    setPitch(Math.max(-1.45, Math.min(1.45, dragRef.current.pitch + dy * 0.008)));
  };

  const handlePointerUp = () => {
    dragRef.current = null;
  };

  const liveProjected = livePin
    ? project(livePin.age, livePin.annual_income_k, livePin.spending_score)
    : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {VIEW_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => {
                setYaw(preset.yaw);
                setPitch(preset.pitch);
              }}
              className="px-2.5 py-1 text-[11px] font-medium bg-[#FAF9F6] hover:bg-[#EFECE5] border border-[#E4E2DC] rounded-md text-[#141413] cursor-pointer transition-colors"
            >
              {preset.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => {
            setYaw(-0.68);
            setPitch(0.44);
          }}
          className="inline-flex items-center gap-1 px-2 py-1 text-[11px] text-[#64635E] hover:text-[#141413] cursor-pointer"
        >
          <RotateCcw className="w-3 h-3" />
          <span>Reset Orbit</span>
        </button>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        className="w-full h-auto bg-[#FAF9F6] rounded-lg border border-[#E4E2DC] cursor-grab active:cursor-grabbing select-none touch-none"
      >
        {boxEdges.pairs.map(([a, b], idx) => {
          const isPrimaryAxis = idx < 3;
          return (
            <line
              key={`${a}-${b}`}
              x1={boxEdges.corners[a].sx}
              y1={boxEdges.corners[a].sy}
              x2={boxEdges.corners[b].sx}
              y2={boxEdges.corners[b].sy}
              stroke={isPrimaryAxis ? '#9E9B93' : '#DFDDD6'}
              strokeWidth={isPrimaryAxis ? '1.3' : '0.9'}
              strokeDasharray={isPrimaryAxis ? undefined : '3 3'}
            />
          );
        })}

        {/* Axis labels placed outside the bounding box */}
        {[
          { pt: boxEdges.corners[1], label: `Income ($${bounds.incMax}k)`, dxOff: 44, dyOff: 18 },
          { pt: boxEdges.corners[2], label: 'Spending (100)', dxOff: -8, dyOff: -14 },
          { pt: boxEdges.corners[4], label: `Age (${bounds.ageMax}y)`, dxOff: -44, dyOff: 18 },
        ].map(({ pt, label, dxOff, dyOff }) => (
          <text
            key={label}
            x={pt.sx + dxOff}
            y={pt.sy + dyOff}
            textAnchor="middle"
            fill="#141413"
            fontSize="10"
            fontWeight="600"
          >
            {label}
          </text>
        ))}

        {/* Customers sorted back-to-front */}
        {projectedPoints.map((item) => {
          const { customer, sx, sy, depth, color, clusterId, isNoise } = item;
          const dimmed = highlightedClusterId !== null && highlightedClusterId !== clusterId;
          const opacity = dimmed ? 0.14 : 0.72 + ((depth + 1) / 2) * 0.24;
          const r = 3.4 + ((depth + 1) / 2) * 1.4;
          const isMale = customer.gender === 'Male';

          return (
            <g
              key={customer.id}
              onMouseEnter={() => {
                setHovered(customer);
                onHover?.(customer);
              }}
              onMouseLeave={() => {
                setHovered(null);
                onHover?.(null);
              }}
              className="cursor-pointer"
            >
              {isNoise ? (
                <polygon
                  points={`${sx},${sy - 5} ${sx + 5},${sy} ${sx},${sy + 5} ${sx - 5},${sy}`}
                  fill="#FFFFFF"
                  stroke="#BE123C"
                  strokeWidth="1.5"
                  opacity={opacity}
                />
              ) : isMale ? (
                <polygon
                  points={`${sx},${sy - r - 0.8} ${sx + r + 0.6},${sy + r} ${sx - r - 0.6},${sy + r}`}
                  fill={color}
                  fillOpacity={opacity}
                  stroke="#FFFFFF"
                  strokeWidth="0.7"
                />
              ) : (
                <circle
                  cx={sx}
                  cy={sy}
                  r={r}
                  fill={color}
                  fillOpacity={opacity}
                  stroke="#FFFFFF"
                  strokeWidth="0.7"
                />
              )}
              {customer.is_boundary_customer && !dimmed && algorithmView !== 'dbscan' && algorithmView !== 'hdbscan' && (
                <circle
                  cx={sx}
                  cy={sy}
                  r={r + 2.6}
                  fill="none"
                  stroke="#141413"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />
              )}
            </g>
          );
        })}

        {/* Cluster centroids */}
        {projectedCentroids.map((c) => {
          const dimmed =
            highlightedClusterId !== null && highlightedClusterId !== c.persona.cluster_id;
          return (
            <g key={c.persona.cluster_id} opacity={dimmed ? 0.2 : 1} pointerEvents="none">
              <circle
                cx={c.sx}
                cy={c.sy}
                r={7.5}
                fill={c.color}
                stroke="#FFFFFF"
                strokeWidth="2"
              />
              <text
                x={c.sx}
                y={c.sy - 11}
                textAnchor="middle"
                fill="#141413"
                fontSize="9.5"
                fontWeight="700"
              >
                {c.persona.persona_name}
              </text>
            </g>
          );
        })}

        {liveProjected && (
          <g pointerEvents="none">
            <circle
              cx={liveProjected.sx}
              cy={liveProjected.sy}
              r={9}
              fill="none"
              stroke="#141413"
              strokeWidth="2"
            />
            <circle cx={liveProjected.sx} cy={liveProjected.sy} r={4.2} fill="#141413" />
          </g>
        )}

        {hovered && (() => {
          const hp = project(hovered.age, hovered.annual_income_k, hovered.spending_score);
          const boxW = 205;
          const boxH = 44;
          const tx = hp.sx + boxW + 12 > width ? hp.sx - boxW - 10 : hp.sx + 10;
          const ty = hp.sy - boxH - 8 < 10 ? hp.sy + 10 : hp.sy - boxH - 6;
          return (
            <g pointerEvents="none">
              <rect
                x={tx}
                y={ty}
                width={boxW}
                height={boxH}
                rx={6}
                fill="#141413"
                fillOpacity={0.94}
              />
              <text x={tx + 10} y={ty + 17} fill="#FFFFFF" fontSize="10" fontWeight="600">
                {hovered.customer_ref || `ID-${hovered.id}`} · {hovered.persona_name}
              </text>
              <text
                x={tx + 10}
                y={ty + 33}
                fill="#D6D3CB"
                fontSize="9"
                className="font-mono-tabular"
              >
                {hovered.gender} · {hovered.age}y · ${hovered.annual_income_k}k · Sp {hovered.spending_score}
              </text>
            </g>
          );
        })()}
      </svg>

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#64635E]">
        <span>Drag anywhere on the 3D manifold to rotate Age × Income × Spending space</span>
        <span className="font-mono-tabular">
          ● Female · ▲ Male · Dashed ring = Boundary Holdout
        </span>
      </div>
    </div>
  );
}
