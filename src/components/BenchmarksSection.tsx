import { ShieldCheck } from 'lucide-react';
import { ConsensusMatrixChart, CorrelationMatrixChart } from './Charts';
import type { SegmentsPayload } from '../types';

export function BenchmarksSection({
  segments,
  corrMode,
  onChangeCorrMode,
}: {
  segments: SegmentsPayload;
  corrMode: 'pearson' | 'spearman';
  onChangeCorrMode: (m: 'pearson' | 'spearman') => void;
}) {
  return (
    <div className="space-y-6">
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#E4E2DC] flex items-center justify-between">
          <h2 className="text-sm font-semibold text-[#141413]">
            Multi-Algorithm Clustering Benchmark (5 Algorithms)
          </h2>
          <span className="text-xs font-mono-tabular text-[#64635E]">
            Scaled space: {segments.features.feature_names.join(', ')}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                <th className="py-3 px-5 font-medium">Algorithm</th>
                <th className="py-3 px-4 text-right font-medium">Clusters</th>
                <th className="py-3 px-4 text-right font-medium">Silhouette</th>
                <th className="py-3 px-4 text-right font-medium">Calinski-Harabasz</th>
                <th className="py-3 px-4 text-right font-medium">Davies-Bouldin</th>
                <th className="py-3 px-5 text-right font-medium">Noise Points</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E4E2DC]">
              {segments.benchmarks.map((b) => (
                <tr key={b.algorithm} className="hover:bg-[#FAF9F6]">
                  <td className="py-3 px-5 font-medium text-[#141413]">{b.algorithm}</td>
                  <td className="py-3 px-4 text-right font-mono-tabular">{b.n_clusters}</td>
                  <td className="py-3 px-4 text-right font-mono-tabular font-semibold text-[#1E40AF]">
                    {b.silhouette !== null ? b.silhouette.toFixed(4) : 'N/A'}
                  </td>
                  <td className="py-3 px-4 text-right font-mono-tabular">
                    {b.calinski_harabasz !== null ? b.calinski_harabasz.toFixed(2) : 'N/A'}
                  </td>
                  <td className="py-3 px-4 text-right font-mono-tabular">
                    {b.davies_bouldin !== null ? b.davies_bouldin.toFixed(4) : 'N/A'}
                  </td>
                  <td className="py-3 px-5 text-right font-mono-tabular text-[#64635E]">
                    {b.noise_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Cross-Algorithm Consensus ARI + Gender Sensitivity Audit */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {segments.algorithm_consensus && (
          <section className="lg:col-span-7 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[#141413]">
                Cross-Algorithm Consensus (Adjusted Rand Index)
              </h2>
              <span className="text-xs font-mono-tabular text-[#64635E]">
                1.00 = identical partition
              </span>
            </div>
            <ConsensusMatrixChart consensus={segments.algorithm_consensus} />
          </section>
        )}

        {segments.gender_sensitivity && (
          <section className="lg:col-span-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-[#047857]" />
                <h2 className="text-sm font-semibold text-[#141413]">
                  Gender Privacy & Fairness Audit
                </h2>
              </div>
              <span className="text-xs font-mono-tabular text-[#047857]">
                {segments.gender_sensitivity.active_mode === 'profiling_only'
                  ? 'Profiling Only'
                  : 'Included in Geometry'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg">
                <span className="text-[#64635E] block">Silhouette (No Gender)</span>
                <span className="text-lg font-semibold font-mono-tabular text-[#141413]">
                  {segments.gender_sensitivity.silhouette_without_gender.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg">
                <span className="text-[#64635E] block">Silhouette (With Gender)</span>
                <span className="text-lg font-semibold font-mono-tabular text-[#141413]">
                  {segments.gender_sensitivity.silhouette_with_gender.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg">
                <span className="text-[#64635E] block">ARI Partition Agreement</span>
                <span className="text-lg font-semibold font-mono-tabular text-[#1E40AF]">
                  {segments.gender_sensitivity.ari_agreement.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg">
                <span className="text-[#64635E] block">Reassigned Customers</span>
                <span className="text-lg font-semibold font-mono-tabular text-[#B45309]">
                  {segments.gender_sensitivity.reassigned_customer_pct.toFixed(1)}%
                </span>
              </div>
            </div>

            <p className="text-xs text-[#5C5B56] leading-relaxed">
              Chi-square independence test across active segments yields{' '}
              <span className="font-mono-tabular font-semibold text-[#141413]">
                p = {segments.gender_sensitivity.gender_cluster_chi2_p_value.toFixed(4)}
              </span>{' '}
              (
              {segments.gender_sensitivity.gender_independent_at_05
                ? 'clusters are statistically balanced across genders'
                : 'mild gender skew detected across segments'}
              ).
            </p>
          </section>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#141413]">
              Bootstrap Stability per Cluster (B={segments.stability.n_bootstrap})
            </h2>
            <span className="text-xs font-mono-tabular text-[#047857]">
              Mean Jaccard {segments.stability.mean_jaccard.toFixed(4)}
            </span>
          </div>
          <div className="space-y-3">
            {Object.entries(segments.stability.per_cluster_jaccard).map(([cid, jaccard]) => {
              const persona = segments.personas.find((p) => String(p.cluster_id) === cid);
              return (
                <div key={cid} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-[#141413] font-medium">
                      Cluster {cid} · {persona?.persona_name}
                    </span>
                    <span className="font-mono-tabular text-[#5C5B56]">{jaccard.toFixed(4)}</span>
                  </div>
                  <div className="h-2 bg-[#F1EFE9] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#1E40AF] rounded-full"
                      style={{ width: `${Math.min(100, jaccard * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-sm font-semibold text-[#141413]">
              EDA Summary & Correlation Matrix
            </h2>
            <div className="flex items-center gap-1">
              {(['pearson', 'spearman'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => onChangeCorrMode(m)}
                  className={`px-2 py-1 text-[11px] font-medium rounded border cursor-pointer ${
                    corrMode === m
                      ? 'bg-[#141413] text-white border-[#141413]'
                      : 'bg-[#FAF9F6] text-[#5C5B56] border-[#E4E2DC]'
                  }`}
                >
                  {m === 'pearson' ? 'Pearson r' : 'Spearman ρ'}
                </button>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#E4E2DC] text-[#64635E]">
                  <th className="py-2 pr-3 font-medium">Feature</th>
                  <th className="py-2 px-2 text-right font-medium">Mean</th>
                  <th className="py-2 px-2 text-right font-medium">Std</th>
                  <th className="py-2 px-2 text-right font-medium">Median</th>
                  <th className="py-2 pl-2 text-right font-medium">IQR Outliers</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E4E2DC]">
                {segments.eda.numeric_summaries.map((s) => (
                  <tr key={s.feature}>
                    <td className="py-2.5 pr-3 font-medium text-[#141413]">{s.feature}</td>
                    <td className="py-2.5 px-2 text-right font-mono-tabular">
                      {s.mean.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-2 text-right font-mono-tabular">{s.std.toFixed(2)}</td>
                    <td className="py-2.5 px-2 text-right font-mono-tabular">
                      {s.median.toFixed(1)}
                    </td>
                    <td className="py-2.5 pl-2 text-right font-mono-tabular">
                      {s.iqr_outlier_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pt-2 border-t border-[#E4E2DC]">
            <CorrelationMatrixChart
              matrix={
                corrMode === 'pearson' ? segments.eda.pearson_matrix : segments.eda.spearman_matrix
              }
              mode={corrMode}
            />
          </div>
        </section>
      </div>

      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-[#141413]">
            Surrogate Decision Tree IF-THEN Rules
          </h2>
          <span className="text-xs font-mono-tabular text-[#64635E]">
            Fidelity {(segments.explainability.fidelity_score * 100).toFixed(1)}% · Max Depth{' '}
            {segments.explainability.max_depth}
          </span>
        </div>
        <div className="space-y-2">
          {segments.explainability.rules.map((rule) => (
            <div
              key={rule.rule_id}
              className="flex flex-col md:flex-row md:items-center justify-between gap-2 py-2.5 px-3.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-xs"
            >
              <span className="font-mono-tabular text-[#141413]">{rule.condition_text}</span>
              <span className="font-mono-tabular text-[#5C5B56] whitespace-nowrap shrink-0">
                Support {rule.support_count} · Purity {(rule.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
