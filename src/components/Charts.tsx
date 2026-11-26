import { useState } from 'react';
import {
  SEGMENT_PALETTE,
  type AlgorithmPoint,
  type CustomerRecord,
  type GmmEllipse,
  type PersonaProfile,
  type ScoreResult,
  type SweepPoint,
  type WardDendrogram,
} from '../types';

export type AlgorithmView = 'kmeans' | 'ward' | 'gmm' | 'dbscan';

function resolvePointColorAndNoise(
  pt: CustomerRecord,
  algorithmView: AlgorithmView,
  algoLookup: Map<string, AlgorithmPoint>,
  colorMap: Map<string, string>
): { color: string; clusterId: number; isNoise: boolean } {
  if (algorithmView === 'kmeans') {
    return {
      color: colorMap.get(pt.persona_name) ?? SEGMENT_PALETTE[pt.cluster_id % SEGMENT_PALETTE.length],
      clusterId: pt.cluster_id,
      isNoise: false,
    };
  }
  const match = pt.customer_ref ? algoLookup.get(pt.customer_ref) : undefined;
  if (!match) {
    return {
      color: colorMap.get(pt.persona_name) ?? SEGMENT_PALETTE[pt.cluster_id % SEGMENT_PALETTE.length],
      clusterId: pt.cluster_id,
      isNoise: false,
    };
  }
  if (algorithmView === 'ward') {
    const cid = match.ward_cluster_id;
    return { color: SEGMENT_PALETTE[Math.abs(cid) % SEGMENT_PALETTE.length], clusterId: cid, isNoise: false };
  }
  if (algorithmView === 'gmm') {
    const cid = match.gmm_cluster_id;
    return { color: SEGMENT_PALETTE[Math.abs(cid) % SEGMENT_PALETTE.length], clusterId: cid, isNoise: false };
  }
  const dbId = match.dbscan_cluster_id;
  if (dbId === -1) {
    return { color: '#BE123C', clusterId: -1, isNoise: true };
  }
  return {
    color: SEGMENT_PALETTE[Math.abs(dbId) % SEGMENT_PALETTE.length],
    clusterId: dbId,
    isNoise: false,
  };
}

export function IncomeSpendingQuadrantChart({
  customers,
  ellipses,
  colorMap,
  livePin,
  algorithmView = 'kmeans',
  algorithmPoints = [],
  highlightedClusterId = null,
  onHover,
}: {
  customers: CustomerRecord[];
  ellipses: GmmEllipse[];
  colorMap: Map<string, string>;
  livePin?: ScoreResult | null;
  algorithmView?: AlgorithmView;
  algorithmPoints?: AlgorithmPoint[];
  highlightedClusterId?: number | null;
  onHover?: (c: CustomerRecord | null) => void;
}) {
  const [hovered, setHovered] = useState<CustomerRecord | null>(null);

  const width = 560;
  const height = 370;
  const padL = 48;
  const padR = 20;
  const padT = 24;
  const padB = 42;

  const xValues = [
    ...customers.map((c) => c.annual_income_k),
    ...ellipses.flatMap((e) => [e.cx - e.rx, e.cx + e.rx]),
    ...(livePin ? [livePin.annual_income_k] : []),
  ];
  const rawMinX = xValues.length ? Math.min(...xValues) : 15;
  const rawMaxX = xValues.length ? Math.max(...xValues) : 140;
  const minX = Math.max(0, Math.floor((rawMinX - 8) / 10) * 10);
  const maxX = Math.max(minX + 40, Math.ceil((rawMaxX + 8) / 10) * 10);
  const minY = 0;
  const maxY = 102;

  const mapX = (x: number) => padL + ((x - minX) / Math.max(maxX - minX, 1)) * (width - padL - padR);
  const mapY = (y: number) => height - padB - ((y - minY) / (maxY - minY)) * (height - padT - padB);
  const scaleRx = (rx: number) => (rx / Math.max(maxX - minX, 1)) * (width - padL - padR);
  const scaleRy = (ry: number) => (ry / (maxY - minY)) * (height - padT - padB);

  const tickStep = maxX - minX > 220 ? 50 : maxX - minX > 120 ? 25 : 20;
  const firstTick = Math.ceil(minX / tickStep) * tickStep;
  const xTicks: number[] = [];
  for (let t = firstTick; t <= maxX; t += tickStep) {
    xTicks.push(t);
  }
  const yTicks = [20, 40, 60, 80, 100];

  const algoLookup = new Map<string, AlgorithmPoint>();
  for (const ap of algorithmPoints) {
    if (ap.customer_ref) algoLookup.set(ap.customer_ref, ap);
  }

  const showEllipses = algorithmView === 'kmeans' || algorithmView === 'gmm';

  const handleEnter = (pt: CustomerRecord) => {
    setHovered(pt);
    onHover?.(pt);
  };
  const handleLeave = () => {
    setHovered(null);
    onHover?.(null);
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto bg-[#FAF9F6] rounded-lg border border-[#E4E2DC]"
    >
      {xTicks.map((t) => (
        <g key={`x-${t}`}>
          <line
            x1={mapX(t)}
            y1={padT}
            x2={mapX(t)}
            y2={height - padB}
            stroke="#EAE8E1"
            strokeWidth="1"
          />
          <text
            x={mapX(t)}
            y={height - padB + 16}
            textAnchor="middle"
            fill="#64635E"
            fontSize="10"
            className="font-mono-tabular"
          >
            ${t}k
          </text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`y-${t}`}>
          <line
            x1={padL}
            y1={mapY(t)}
            x2={width - padR}
            y2={mapY(t)}
            stroke="#EAE8E1"
            strokeWidth="1"
          />
          <text
            x={padL - 8}
            y={mapY(t) + 3}
            textAnchor="end"
            fill="#64635E"
            fontSize="10"
            className="font-mono-tabular"
          >
            {t}
          </text>
        </g>
      ))}

      <line
        x1={padL}
        y1={height - padB}
        x2={width - padR}
        y2={height - padB}
        stroke="#C9C6BE"
        strokeWidth="1"
      />
      <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="#C9C6BE" strokeWidth="1" />

      {showEllipses &&
        ellipses.map((el) => {
          const cx = mapX(el.cx);
          const cy = mapY(el.cy);
          const rx = scaleRx(el.rx);
          const ry = scaleRy(el.ry);
          const color = colorMap.get(el.persona_name) ?? SEGMENT_PALETTE[el.cluster_id % SEGMENT_PALETTE.length];
          const dimmed = highlightedClusterId !== null && highlightedClusterId !== el.cluster_id;
          return (
            <ellipse
              key={el.cluster_id}
              cx={cx}
              cy={cy}
              rx={rx}
              ry={ry}
              fill={color}
              fillOpacity={dimmed ? 0.015 : algorithmView === 'gmm' ? 0.11 : 0.06}
              stroke={color}
              strokeOpacity={dimmed ? 0.12 : algorithmView === 'gmm' ? 0.55 : 0.35}
              strokeWidth={algorithmView === 'gmm' ? '1.5' : '1.1'}
              strokeDasharray="4 3"
              transform={`rotate(${-el.angle_deg} ${cx} ${cy})`}
            />
          );
        })}

      {customers.map((pt) => {
        const cx = mapX(pt.annual_income_k);
        const cy = mapY(pt.spending_score);
        const { color, clusterId, isNoise } = resolvePointColorAndNoise(
          pt,
          algorithmView,
          algoLookup,
          colorMap
        );
        const dimmed = highlightedClusterId !== null && highlightedClusterId !== clusterId;
        const opacity = dimmed ? 0.16 : 0.88;

        return (
          <g
            key={pt.id}
            onMouseEnter={() => handleEnter(pt)}
            onMouseLeave={handleLeave}
            className="cursor-pointer"
          >
            {isNoise ? (
              <polygon
                points={`${cx},${cy - 5} ${cx + 5},${cy} ${cx},${cy + 5} ${cx - 5},${cy}`}
                fill="#FFFFFF"
                stroke="#BE123C"
                strokeWidth="1.6"
                opacity={opacity}
              />
            ) : (
              <circle
                cx={cx}
                cy={cy}
                r={pt.source === 'batch' ? 4.6 : 3.9}
                fill={color}
                fillOpacity={opacity}
                stroke="#FFFFFF"
                strokeWidth="0.8"
              />
            )}
            {pt.is_boundary_customer && !dimmed && algorithmView !== 'dbscan' && (
              <circle
                cx={cx}
                cy={cy}
                r={7}
                fill="none"
                stroke="#141413"
                strokeWidth="1.2"
                strokeDasharray="2 2"
              />
            )}
          </g>
        );
      })}

      {livePin && (
        <g>
          <line
            x1={mapX(livePin.annual_income_k)}
            y1={padT}
            x2={mapX(livePin.annual_income_k)}
            y2={height - padB}
            stroke="#141413"
            strokeWidth="1"
            strokeDasharray="3 3"
            strokeOpacity={0.45}
          />
          <line
            x1={padL}
            y1={mapY(livePin.spending_score)}
            x2={width - padR}
            y2={mapY(livePin.spending_score)}
            stroke="#141413"
            strokeWidth="1"
            strokeDasharray="3 3"
            strokeOpacity={0.45}
          />
          <circle
            cx={mapX(livePin.annual_income_k)}
            cy={mapY(livePin.spending_score)}
            r={8.5}
            fill="none"
            stroke="#141413"
            strokeWidth="1.8"
          />
          <circle
            cx={mapX(livePin.annual_income_k)}
            cy={mapY(livePin.spending_score)}
            r={4.5}
            fill="#141413"
          />
        </g>
      )}

      {/* Floating SVG Hover Tooltip */}
      {hovered && (() => {
        const px = mapX(hovered.annual_income_k);
        const py = mapY(hovered.spending_score);
        const boxW = 190;
        const boxH = 54;
        const tx = px + boxW + 14 > width - padR ? px - boxW - 10 : px + 10;
        const ty = py - boxH - 8 < padT ? py + 10 : py - boxH - 6;
        const { clusterId, isNoise } = resolvePointColorAndNoise(
          hovered,
          algorithmView,
          algoLookup,
          colorMap
        );
        const label =
          algorithmView === 'kmeans'
            ? hovered.persona_name
            : isNoise
              ? 'DBSCAN Noise (-1)'
              : `${algorithmView.toUpperCase()} Cluster ${clusterId}`;

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
            <text x={tx + 10} y={ty + 17} fill="#FFFFFF" fontSize="10.5" fontWeight="600">
              {hovered.customer_ref || `ID-${hovered.id}`} · {label}
            </text>
            <text
              x={tx + 10}
              y={ty + 33}
              fill="#D6D3CB"
              fontSize="9.5"
              className="font-mono-tabular"
            >
              {hovered.gender} · Age {hovered.age} · ${hovered.annual_income_k}k · Sp {hovered.spending_score}
            </text>
            <text
              x={tx + 10}
              y={ty + 46}
              fill="#93C5FD"
              fontSize="9"
              className="font-mono-tabular"
            >
              Confidence {(hovered.gmm_confidence * 100).toFixed(0)}% · Entropy {hovered.normalized_entropy.toFixed(3)}
            </text>
          </g>
        );
      })()}

      <text
        x={(padL + width - padR) / 2}
        y={height - 10}
        textAnchor="middle"
        fill="#5C5B56"
        fontSize="11"
        fontWeight="500"
      >
        Annual Income (k$)
      </text>
      <text
        x={15}
        y={(padT + height - padB) / 2}
        textAnchor="middle"
        fill="#5C5B56"
        fontSize="11"
        fontWeight="500"
        transform={`rotate(-90 15 ${(padT + height - padB) / 2})`}
      >
        Spending Score (1–100)
      </text>
    </svg>
  );
}

export function PcaManifoldChart({
  customers,
  colorMap,
  explainedVariance,
  livePin,
  algorithmView = 'kmeans',
  algorithmPoints = [],
  highlightedClusterId = null,
  onHover,
}: {
  customers: CustomerRecord[];
  colorMap: Map<string, string>;
  explainedVariance: number[];
  livePin?: ScoreResult | null;
  algorithmView?: AlgorithmView;
  algorithmPoints?: AlgorithmPoint[];
  highlightedClusterId?: number | null;
  onHover?: (c: CustomerRecord | null) => void;
}) {
  const [hovered, setHovered] = useState<CustomerRecord | null>(null);

  const width = 560;
  const height = 370;
  const padL = 46;
  const padR = 20;
  const padT = 24;
  const padB = 42;

  const xs = [...customers.map((p) => p.pca_1), ...(livePin ? [livePin.pca_1] : [])];
  const ys = [...customers.map((p) => p.pca_2), ...(livePin ? [livePin.pca_2] : [])];
  const minX = (xs.length ? Math.min(...xs) : -3) - 0.35;
  const maxX = (xs.length ? Math.max(...xs) : 3) + 0.35;
  const minY = (ys.length ? Math.min(...ys) : -3) - 0.35;
  const maxY = (ys.length ? Math.max(...ys) : 3) + 0.35;

  const mapX = (x: number) => padL + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (width - padL - padR);
  const mapY = (y: number) =>
    height - padB - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (height - padT - padB);

  const algoLookup = new Map<string, AlgorithmPoint>();
  for (const ap of algorithmPoints) {
    if (ap.customer_ref) algoLookup.set(ap.customer_ref, ap);
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto bg-[#FAF9F6] rounded-lg border border-[#E4E2DC]"
    >
      {minX < 0 && maxX > 0 && (
        <line
          x1={mapX(0)}
          y1={padT}
          x2={mapX(0)}
          y2={height - padB}
          stroke="#E4E2DC"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
      )}
      {minY < 0 && maxY > 0 && (
        <line
          x1={padL}
          y1={mapY(0)}
          x2={width - padR}
          y2={mapY(0)}
          stroke="#E4E2DC"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
      )}

      <line
        x1={padL}
        y1={height - padB}
        x2={width - padR}
        y2={height - padB}
        stroke="#C9C6BE"
        strokeWidth="1"
      />
      <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="#C9C6BE" strokeWidth="1" />

      {customers.map((pt) => {
        const cx = mapX(pt.pca_1);
        const cy = mapY(pt.pca_2);
        const radius = 3.2 + pt.normalized_entropy * 6.0;
        const { color, clusterId, isNoise } = resolvePointColorAndNoise(
          pt,
          algorithmView,
          algoLookup,
          colorMap
        );
        const dimmed = highlightedClusterId !== null && highlightedClusterId !== clusterId;
        const opacity = dimmed ? 0.16 : 0.82;

        return (
          <g
            key={pt.id}
            onMouseEnter={() => {
              setHovered(pt);
              onHover?.(pt);
            }}
            onMouseLeave={() => {
              setHovered(null);
              onHover?.(null);
            }}
            className="cursor-pointer"
          >
            {isNoise ? (
              <polygon
                points={`${cx},${cy - 5} ${cx + 5},${cy} ${cx},${cy + 5} ${cx - 5},${cy}`}
                fill="#FFFFFF"
                stroke="#BE123C"
                strokeWidth="1.6"
                opacity={opacity}
              />
            ) : (
              <circle
                cx={cx}
                cy={cy}
                r={radius}
                fill={color}
                fillOpacity={opacity}
                stroke="#FFFFFF"
                strokeWidth="0.7"
              />
            )}
            {pt.is_boundary_customer && !dimmed && algorithmView !== 'dbscan' && (
              <circle
                cx={cx}
                cy={cy}
                r={radius + 2.8}
                fill="none"
                stroke="#141413"
                strokeWidth="1.2"
                strokeDasharray="2 2"
              />
            )}
          </g>
        );
      })}

      {livePin && (
        <g>
          <circle
            cx={mapX(livePin.pca_1)}
            cy={mapY(livePin.pca_2)}
            r={8.5}
            fill="none"
            stroke="#141413"
            strokeWidth="1.8"
          />
          <circle cx={mapX(livePin.pca_1)} cy={mapY(livePin.pca_2)} r={4.2} fill="#141413" />
        </g>
      )}

      {hovered && (() => {
        const px = mapX(hovered.pca_1);
        const py = mapY(hovered.pca_2);
        const boxW = 185;
        const boxH = 44;
        const tx = px + boxW + 14 > width - padR ? px - boxW - 10 : px + 10;
        const ty = py - boxH - 8 < padT ? py + 10 : py - boxH - 6;
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
              PC1 {hovered.pca_1.toFixed(2)} · PC2 {hovered.pca_2.toFixed(2)} · H {hovered.normalized_entropy.toFixed(3)}
            </text>
          </g>
        );
      })()}

      <text
        x={(padL + width - padR) / 2}
        y={height - 10}
        textAnchor="middle"
        fill="#5C5B56"
        fontSize="11"
        fontWeight="500"
      >
        PC1 ({((explainedVariance[0] ?? 0) * 100).toFixed(1)}% variance)
      </text>
      <text
        x={15}
        y={(padT + height - padB) / 2}
        textAnchor="middle"
        fill="#5C5B56"
        fontSize="11"
        fontWeight="500"
        transform={`rotate(-90 15 ${(padT + height - padB) / 2})`}
      >
        PC2 ({((explainedVariance[1] ?? 0) * 100).toFixed(1)}% variance)
      </text>
    </svg>
  );
}

export function FeatureHistogramChart({
  title,
  values,
  mean,
  median,
  accent,
  unitPrefix = '',
  unitSuffix = '',
}: {
  title: string;
  values: number[];
  mean: number;
  median: number;
  accent: string;
  unitPrefix?: string;
  unitSuffix?: string;
}) {
  const [hoveredBin, setHoveredBin] = useState<number | null>(null);
  const width = 260;
  const height = 148;
  const padL = 26;
  const padR = 12;
  const padT = 16;
  const padB = 26;

  const binCount = 10;
  const rawMin = values.length ? Math.min(...values) : 0;
  const rawMax = values.length ? Math.max(...values) : 100;
  const span = Math.max(rawMax - rawMin, 1);
  const binWidth = span / binCount;

  const bins = Array.from({ length: binCount }, (_, idx) => {
    const start = rawMin + idx * binWidth;
    const end = start + binWidth;
    const count = values.filter((v) =>
      idx === binCount - 1 ? v >= start && v <= end + 1e-9 : v >= start && v < end
    ).length;
    return { idx, start, end, count };
  });
  const maxCount = Math.max(1, ...bins.map((b) => b.count));

  const mapX = (v: number) => padL + ((v - rawMin) / span) * (width - padL - padR);
  const plotW = width - padL - padR;
  const barW = plotW / binCount;
  const plotH = height - padT - padB;

  return (
    <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-[#141413]">{title}</span>
        <span className="font-mono-tabular text-[#64635E]">
          {hoveredBin !== null
            ? `${unitPrefix}${bins[hoveredBin].start.toFixed(0)}–${bins[hoveredBin].end.toFixed(0)}${unitSuffix}: ${bins[hoveredBin].count}`
            : `μ ${unitPrefix}${mean.toFixed(1)}${unitSuffix}`}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        {bins.map((b) => {
          const barH = (b.count / maxCount) * plotH;
          const x = padL + b.idx * barW + 1.5;
          const y = height - padB - barH;
          return (
            <rect
              key={b.idx}
              x={x}
              y={y}
              width={Math.max(2, barW - 3)}
              height={barH}
              rx={2}
              fill={accent}
              fillOpacity={hoveredBin === b.idx ? 0.95 : 0.72}
              onMouseEnter={() => setHoveredBin(b.idx)}
              onMouseLeave={() => setHoveredBin(null)}
              className="cursor-pointer"
            />
          );
        })}
        <line
          x1={mapX(mean)}
          y1={padT}
          x2={mapX(mean)}
          y2={height - padB}
          stroke="#141413"
          strokeWidth="1.3"
        />
        <line
          x1={mapX(median)}
          y1={padT}
          x2={mapX(median)}
          y2={height - padB}
          stroke="#1E40AF"
          strokeWidth="1.2"
          strokeDasharray="3 2"
        />
        <line
          x1={padL}
          y1={height - padB}
          x2={width - padR}
          y2={height - padB}
          stroke="#C9C6BE"
          strokeWidth="1"
        />
        <text
          x={padL}
          y={height - 8}
          textAnchor="start"
          fill="#64635E"
          fontSize="9"
          className="font-mono-tabular"
        >
          {unitPrefix}
          {rawMin.toFixed(0)}
          {unitSuffix}
        </text>
        <text
          x={width - padR}
          y={height - 8}
          textAnchor="end"
          fill="#64635E"
          fontSize="9"
          className="font-mono-tabular"
        >
          {unitPrefix}
          {rawMax.toFixed(0)}
          {unitSuffix}
        </text>
        <text
          x={padL - 4}
          y={padT + 6}
          textAnchor="end"
          fill="#8A8880"
          fontSize="8.5"
          className="font-mono-tabular"
        >
          {maxCount}
        </text>
      </svg>
    </div>
  );
}

export function GenderDemographicChart({ personas }: { personas: PersonaProfile[] }) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between text-xs text-[#64635E]">
        <span>Post-Hoc Gender Mix by Segment</span>
        <span className="font-mono-tabular flex items-center gap-3">
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-xs bg-[#1D4ED8] inline-block" /> Female
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2 h-2 rounded-xs bg-[#C9C6BE] inline-block" /> Male
          </span>
        </span>
      </div>
      <div className="space-y-2">
        {personas.map((p) => (
          <div key={p.cluster_id} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-[#141413] font-medium truncate pr-2">{p.persona_name}</span>
              <span className="font-mono-tabular text-[#5C5B56] shrink-0">
                {(p.female_share * 100).toFixed(0)}% F · {(p.male_share * 100).toFixed(0)}% M
              </span>
            </div>
            <div className="h-2.5 w-full rounded-sm overflow-hidden flex bg-[#E4E2DC]">
              <div
                className="h-full bg-[#1D4ED8]"
                style={{ width: `${p.female_share * 100}%` }}
              />
              <div
                className="h-full bg-[#9CA3AF]"
                style={{ width: `${p.male_share * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function CorrelationMatrixChart({
  matrix,
  mode,
}: {
  matrix: Record<string, Record<string, number>>;
  mode: 'pearson' | 'spearman';
}) {
  const keys = Object.keys(matrix);
  const shortLabel = (name: string) =>
    name.includes('Income') ? 'Income (k$)' : name.includes('Spending') ? 'Spending' : 'Age';

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-1.5 text-xs">
        <div className="p-2 text-[#8A8880] font-mono-tabular uppercase text-[10px] flex items-center">
          {mode}
        </div>
        {keys.map((col) => (
          <div
            key={`head-${col}`}
            className="p-2 text-center font-medium text-[#5C5B56] bg-[#FAF9F6] rounded border border-[#E4E2DC]"
          >
            {shortLabel(col)}
          </div>
        ))}
        {keys.map((row) => (
          <div key={`row-${row}`} className="contents">
            <div className="p-2 font-medium text-[#5C5B56] bg-[#FAF9F6] rounded border border-[#E4E2DC] flex items-center">
              {shortLabel(row)}
            </div>
            {keys.map((col) => {
              const val = matrix[row]?.[col] ?? 0;
              const absVal = Math.abs(val);
              const bg =
                val >= 0
                  ? `rgba(29, 78, 216, ${Math.min(0.85, 0.08 + absVal * 0.72)})`
                  : `rgba(190, 18, 60, ${Math.min(0.85, 0.08 + absVal * 0.72)})`;
              const textColor = absVal > 0.45 ? '#FFFFFF' : '#141413';
              return (
                <div
                  key={`${row}-${col}`}
                  style={{ backgroundColor: bg, color: textColor }}
                  className="p-2.5 rounded text-center font-mono-tabular font-semibold border border-[#E4E2DC]/60"
                >
                  {val.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export function WardDendrogramChart({
  dendro,
  targetK = 5,
}: {
  dendro: WardDendrogram;
  targetK?: number;
}) {
  const width = 620;
  const height = 270;
  const padL = 48;
  const padR = 20;
  const padT = 20;
  const padB = 34;

  const allX = dendro.icoord.flat();
  const allY = dendro.dcoord.flat();
  const minX = Math.min(...allX) - 5;
  const maxX = Math.max(...allX) + 5;
  const maxY = Math.max(...allY, dendro.cut_height) * 1.08;

  const mapX = (x: number) => padL + ((x - minX) / Math.max(maxX - minX, 1)) * (width - padL - padR);
  const mapY = (y: number) => height - padB - (y / Math.max(maxY, 1e-6)) * (height - padT - padB);

  const cutY = mapY(dendro.cut_height);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto bg-[#FAF9F6] rounded-lg border border-[#E4E2DC]"
    >
      <line
        x1={padL}
        y1={height - padB}
        x2={width - padR}
        y2={height - padB}
        stroke="#C9C6BE"
        strokeWidth="1"
      />
      <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="#C9C6BE" strokeWidth="1" />

      {dendro.icoord.map((xs, idx) => {
        const ys = dendro.dcoord[idx];
        const path = `M ${mapX(xs[0]).toFixed(1)} ${mapY(ys[0]).toFixed(1)} L ${mapX(xs[1]).toFixed(1)} ${mapY(ys[1]).toFixed(1)} L ${mapX(xs[2]).toFixed(1)} ${mapY(ys[2]).toFixed(1)} L ${mapX(xs[3]).toFixed(1)} ${mapY(ys[3]).toFixed(1)}`;
        const belowCut = ys[1] < dendro.cut_height;
        return (
          <path
            key={idx}
            d={path}
            fill="none"
            stroke={belowCut ? '#1D4ED8' : '#8A8880'}
            strokeWidth={belowCut ? '1.6' : '1.2'}
          />
        );
      })}

      <line
        x1={padL}
        y1={cutY}
        x2={width - padR}
        y2={cutY}
        stroke="#BE123C"
        strokeWidth="1.4"
        strokeDasharray="5 4"
      />
      <text
        x={width - padR - 4}
        y={cutY - 6}
        textAnchor="end"
        fill="#BE123C"
        fontSize="10"
        fontWeight="600"
        className="font-mono-tabular"
      >
        k={targetK} cut ({dendro.cut_height.toFixed(2)})
      </text>

      <text
        x={(padL + width - padR) / 2}
        y={height - 10}
        textAnchor="middle"
        fill="#5C5B56"
        fontSize="10.5"
      >
        Truncated Ward Linkage Leaves (Top 24 Merges)
      </text>
      <text
        x={14}
        y={(padT + height - padB) / 2}
        textAnchor="middle"
        fill="#5C5B56"
        fontSize="10.5"
        transform={`rotate(-90 14 ${(padT + height - padB) / 2})`}
      >
        Ward Distance
      </text>
    </svg>
  );
}

export function MetricSweepChart({
  title,
  points,
  targetK,
  stroke,
  accessor,
}: {
  title: string;
  points: SweepPoint[];
  targetK: number;
  stroke: string;
  accessor: (pt: SweepPoint) => number;
}) {
  const width = 360;
  const height = 195;
  const padL = 44;
  const padR = 16;
  const padT = 18;
  const padB = 30;

  const xs = points.map((p) => p.k);
  const ys = points.map(accessor);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const mapX = (x: number) => padL + ((x - minX) / Math.max(maxX - minX, 1)) * (width - padL - padR);
  const mapY = (y: number) =>
    height - padB - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (height - padT - padB);

  const d = points
    .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${mapX(pt.k).toFixed(1)} ${mapY(accessor(pt)).toFixed(1)}`)
    .join(' ');

  const targetX = mapX(targetK);

  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-[#5C5B56]">{title}</p>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto bg-[#FAF9F6] rounded-lg border border-[#E4E2DC]"
      >
        <line
          x1={padL}
          y1={height - padB}
          x2={width - padR}
          y2={height - padB}
          stroke="#C9C6BE"
          strokeWidth="1"
        />
        <line
          x1={targetX}
          y1={padT}
          x2={targetX}
          y2={height - padB}
          stroke="#BE123C"
          strokeWidth="1.2"
          strokeDasharray="4 3"
        />
        <path d={d} fill="none" stroke={stroke} strokeWidth="2" />
        {points.map((pt) => (
          <g key={pt.k}>
            <circle
              cx={mapX(pt.k)}
              cy={mapY(accessor(pt))}
              r={pt.k === targetK ? 4.5 : 3}
              fill={pt.k === targetK ? '#BE123C' : stroke}
            />
            <text
              x={mapX(pt.k)}
              y={height - 10}
              textAnchor="middle"
              fill="#64635E"
              fontSize="9.5"
              className="font-mono-tabular"
            >
              {pt.k}
            </text>
          </g>
        ))}
        <text
          x={padL - 4}
          y={padT + 4}
          textAnchor="end"
          fill="#8A8880"
          fontSize="9"
          className="font-mono-tabular"
        >
          {maxY >= 100 ? maxY.toFixed(0) : maxY.toFixed(2)}
        </text>
        <text
          x={padL - 4}
          y={height - padB}
          textAnchor="end"
          fill="#8A8880"
          fontSize="9"
          className="font-mono-tabular"
        >
          {minY >= 100 ? minY.toFixed(0) : minY.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}
