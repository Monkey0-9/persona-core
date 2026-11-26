import { BarChart3, Box, Compass, Layers } from 'lucide-react';
import {
  FeaturePairScatterChart,
  IncomeSpendingQuadrantChart,
  MetricSweepChart,
  PcaManifoldChart,
  WardDendrogramChart,
  type AlgorithmView,
} from './Charts';
import { ThreeDClusterView } from './ThreeDClusterView';
import {
  SEGMENT_PALETTE,
  type CustomerRecord,
  type ScoreResult,
  type SegmentsPayload,
} from '../types';

export function GeometrySection({
  segments,
  customers,
  colorMap,
  liveScore,
  algorithmView,
  highlightedClusterId,
  hoveredCustomer,
  onChangeAlgorithm,
  onHighlightCluster,
  onHoverCustomer,
}: {
  segments: SegmentsPayload;
  customers: CustomerRecord[];
  colorMap: Map<string, string>;
  liveScore: ScoreResult | null;
  algorithmView: AlgorithmView;
  highlightedClusterId: number | null;
  hoveredCustomer: CustomerRecord | null;
  onChangeAlgorithm: (alg: AlgorithmView) => void;
  onHighlightCluster: (cid: number | null) => void;
  onHoverCustomer: (c: CustomerRecord | null) => void;
}) {
  return (
    <div className="space-y-6">
      {/* Multi-Algorithm Visual Switcher */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="space-y-0.5">
          <h2 className="text-sm font-semibold text-[#141413]">Algorithm Comparison Overlay</h2>
          <p className="text-xs text-[#64635E]">
            Compare K-Means++, Ward Hierarchical, GMM contours, DBSCAN (k-NN knee), and HDBSCAN density hierarchy
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {(
            [
              { id: 'kmeans', label: 'K-Means++' },
              { id: 'ward', label: 'Ward Hierarchical' },
              { id: 'gmm', label: 'GMM Full Cov' },
              {
                id: 'dbscan',
                label: `DBSCAN (eps ${segments.dbscan_eps?.toFixed(2) ?? 'knee'})`,
              },
              { id: 'hdbscan', label: 'HDBSCAN' },
            ] as const
          ).map((alg) => (
            <button
              key={alg.id}
              type="button"
              onClick={() => onChangeAlgorithm(alg.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border cursor-pointer transition-colors whitespace-nowrap ${
                algorithmView === alg.id
                  ? 'bg-[#141413] text-white border-[#141413]'
                  : 'bg-[#FAF9F6] text-[#5C5B56] border-[#E4E2DC] hover:text-[#141413]'
              }`}
            >
              {alg.label}
            </button>
          ))}
        </div>
      </section>

      {/* Interactive 3D Manifold + Pairwise Projections */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-7 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-[#141413]">
                Interactive 3D Behavioral Manifold (Age × Income × Spending)
              </h2>
              <p className="text-xs text-[#64635E]">
                Drag to orbit in 3D space or click orthographic projections
              </p>
            </div>
            <Box className="w-4 h-4 text-[#8A8880]" />
          </div>
          <ThreeDClusterView
            customers={customers}
            personas={segments.personas}
            colorMap={colorMap}
            algorithmView={algorithmView}
            algorithmPoints={segments.algorithm_points ?? []}
            highlightedClusterId={highlightedClusterId}
            livePin={liveScore}
            onHover={onHoverCustomer}
          />
        </section>

        <section className="lg:col-span-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-[#141413]">
                Age-Driven Subspace Projections
              </h2>
              <p className="text-xs text-[#64635E]">
                ● Female · ▲ Male across Age, Income, and Spending Score
              </p>
            </div>
          </div>
          <div className="space-y-3">
            <FeaturePairScatterChart
              customers={customers}
              colorMap={colorMap}
              xAxis="age"
              yAxis="spending_score"
              algorithmView={algorithmView}
              algorithmPoints={segments.algorithm_points ?? []}
              highlightedClusterId={highlightedClusterId}
              onHover={onHoverCustomer}
            />
            <FeaturePairScatterChart
              customers={customers}
              colorMap={colorMap}
              xAxis="age"
              yAxis="annual_income_k"
              algorithmView={algorithmView}
              algorithmPoints={segments.algorithm_points ?? []}
              highlightedClusterId={highlightedClusterId}
              onHover={onHoverCustomer}
            />
          </div>
        </section>
      </div>

      {/* 2D Quadrant + 2D PCA */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-[#141413]">
                Income vs. Spending ({algorithmView.toUpperCase()} Overlay)
              </h2>
              <p className="text-xs text-[#64635E]">
                {algorithmView === 'dbscan' || algorithmView === 'hdbscan'
                  ? 'Crimson hollow diamonds mark density noise (-1) outliers'
                  : `Dashed rings mark boundary customers exceeding entropy ${segments.entropy_threshold}`}
              </p>
            </div>
            <Compass className="w-4 h-4 text-[#8A8880]" />
          </div>

          <IncomeSpendingQuadrantChart
            customers={customers}
            ellipses={segments.gmm_ellipses ?? []}
            colorMap={colorMap}
            livePin={liveScore}
            algorithmView={algorithmView}
            algorithmPoints={segments.algorithm_points ?? []}
            highlightedClusterId={highlightedClusterId}
            onHover={onHoverCustomer}
          />

          <div className="min-h-[20px] text-xs font-mono-tabular text-[#5C5B56]">
            {hoveredCustomer ? (
              <span>
                {hoveredCustomer.customer_ref || `ID ${hoveredCustomer.id}`} ·{' '}
                {hoveredCustomer.persona_name} · Age {hoveredCustomer.age} · $
                {hoveredCustomer.annual_income_k}k · Spend {hoveredCustomer.spending_score} · Entropy{' '}
                {hoveredCustomer.normalized_entropy.toFixed(3)}
              </span>
            ) : (
              <span className="text-[#8A8880]">
                Hover any point to inspect cluster assignment, coordinates, and entropy
              </span>
            )}
          </div>
        </section>

        <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-[#141413]">
                2D PCA Manifold (Point Radius Scaled by GMM Entropy)
              </h2>
              <p className="text-xs text-[#64635E]">
                Cumulative explained variance:{' '}
                {(segments.features.cumulative_variance_ratio * 100).toFixed(1)}%
              </p>
            </div>
            <Layers className="w-4 h-4 text-[#8A8880]" />
          </div>

          <PcaManifoldChart
            customers={customers}
            colorMap={colorMap}
            explainedVariance={segments.features.explained_variance_ratio}
            livePin={liveScore}
            algorithmView={algorithmView}
            algorithmPoints={segments.algorithm_points ?? []}
            highlightedClusterId={highlightedClusterId}
            onHover={onHoverCustomer}
          />

          <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-[#5C5B56]">
            {segments.personas.map((p, idx) => {
              const active = highlightedClusterId === p.cluster_id;
              return (
                <button
                  key={p.cluster_id}
                  type="button"
                  onClick={() => onHighlightCluster(active ? null : p.cluster_id)}
                  className={`inline-flex items-center gap-1.5 cursor-pointer ${
                    active ? 'font-semibold text-[#141413] underline' : 'hover:text-[#141413]'
                  }`}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-xs inline-block"
                    style={{ backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length] }}
                  />
                  <span>{p.persona_name}</span>
                </button>
              );
            })}
          </div>
        </section>
      </div>

      {/* Ward Dendrogram + 4-panel Model Selection Sweep */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#141413]">
              Ward Hierarchical Clustering Dendrogram
            </h2>
            <span className="text-xs font-mono-tabular text-[#BE123C]">
              k={segments.target_k} cut {segments.ward_dendrogram?.cut_height.toFixed(2)}
            </span>
          </div>
          {segments.ward_dendrogram && (
            <WardDendrogramChart dendro={segments.ward_dendrogram} targetK={segments.target_k} />
          )}
        </section>

        <section className="lg:col-span-7 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#141413]">
              Multi-Criteria Model Selection Sweeps (k = 2..10)
            </h2>
            <BarChart3 className="w-4 h-4 text-[#8A8880]" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <MetricSweepChart
              title="K-Means Inertia (Elbow)"
              points={segments.kmeans_sweep}
              targetK={segments.target_k}
              stroke="#1D4ED8"
              accessor={(pt) => pt.inertia ?? 0}
            />
            <MetricSweepChart
              title="K-Means Silhouette (Higher Better)"
              points={segments.kmeans_sweep}
              targetK={segments.target_k}
              stroke="#B45309"
              accessor={(pt) => pt.silhouette ?? 0}
            />
            <MetricSweepChart
              title="Calinski-Harabasz Index (Higher Better)"
              points={segments.kmeans_sweep}
              targetK={segments.target_k}
              stroke="#6D28D9"
              accessor={(pt) => pt.calinski_harabasz ?? 0}
            />
            <MetricSweepChart
              title="GMM BIC Curve (Lower Better)"
              points={segments.gmm_sweep}
              targetK={segments.target_k}
              stroke="#047857"
              accessor={(pt) => pt.bic ?? 0}
            />
          </div>
        </section>
      </div>
    </div>
  );
}
