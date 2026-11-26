import { ArrowUpRight } from 'lucide-react';
import {
  AgeCohortChart,
  FeatureHistogramChart,
  GenderDemographicChart,
  IncomeSpendingQuadrantChart,
} from './Charts';
import {
  SEGMENT_PALETTE,
  type CustomerRecord,
  type ScoreResult,
  type SegmentsPayload,
} from '../types';

export function BriefSection({
  segments,
  customers,
  colorMap,
  liveScore,
  highlightedClusterId,
  onHighlightCluster,
  onHoverCustomer,
  onSelectCohort,
}: {
  segments: SegmentsPayload;
  customers: CustomerRecord[];
  colorMap: Map<string, string>;
  liveScore: ScoreResult | null;
  highlightedClusterId: number | null;
  onHighlightCluster: (cid: number | null) => void;
  onHoverCustomer: (c: CustomerRecord | null) => void;
  onSelectCohort: (personaName: string) => void;
}) {
  const analyticsByCluster = new Map(
    (segments.segment_analytics ?? []).map((item) => [item.cluster_id, item])
  );

  return (
    <div className="space-y-6">
      {/* Interactive Behavioral Map + EDA Characteristics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-7 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-[#141413]">
                Behavioral Segment Map (Income vs. Spending Score)
              </h2>
              <p className="text-xs text-[#64635E]">
                Click a persona below to isolate cluster geometry & 1.65σ GMM contours
              </p>
            </div>
            {highlightedClusterId !== null && (
              <button
                type="button"
                onClick={() => onHighlightCluster(null)}
                className="px-2.5 py-1 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded text-[#141413] cursor-pointer"
              >
                Show All Clusters
              </button>
            )}
          </div>

          <IncomeSpendingQuadrantChart
            customers={customers}
            ellipses={segments.gmm_ellipses ?? []}
            colorMap={colorMap}
            livePin={liveScore}
            highlightedClusterId={highlightedClusterId}
            onHover={onHoverCustomer}
          />

          <div className="flex flex-wrap gap-1.5 pt-1">
            {segments.personas.map((p, idx) => {
              const active = highlightedClusterId === p.cluster_id;
              const color = SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length];
              return (
                <button
                  key={p.cluster_id}
                  type="button"
                  onClick={() => onHighlightCluster(active ? null : p.cluster_id)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md border cursor-pointer transition-colors ${
                    active
                      ? 'bg-[#141413] text-white border-[#141413]'
                      : 'bg-[#FAF9F6] text-[#141413] border-[#E4E2DC] hover:bg-[#EFECE5]'
                  }`}
                >
                  <span
                    className="w-2 h-2 rounded-xs inline-block"
                    style={{ backgroundColor: color }}
                  />
                  <span>{p.persona_name}</span>
                  <span className="font-mono-tabular opacity-75">
                    {(p.share * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="lg:col-span-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#141413]">
              Exploratory Characteristics & Gender Split
            </h2>
            <span className="text-xs font-mono-tabular text-[#64635E]">
              n={segments.row_count}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3 gap-2.5">
            {segments.eda.numeric_summaries.map((sum) => {
              const isIncome = sum.feature.includes('Income');
              const isSpend = sum.feature.includes('Spending');
              const vals = customers.map((c) =>
                isIncome ? c.annual_income_k : isSpend ? c.spending_score : c.age
              );
              return (
                <FeatureHistogramChart
                  key={sum.feature}
                  title={isIncome ? 'Income' : isSpend ? 'Spending' : 'Age'}
                  values={vals}
                  mean={sum.mean}
                  median={sum.median}
                  accent={isIncome ? '#047857' : isSpend ? '#B45309' : '#1D4ED8'}
                  unitPrefix={isIncome ? '$' : ''}
                  unitSuffix={isIncome ? 'k' : ''}
                />
              );
            })}
          </div>

          <div className="pt-2 border-t border-[#E4E2DC]">
            <GenderDemographicChart personas={segments.personas} />
          </div>

          {segments.eda.age_cohorts && segments.eda.age_cohorts.length > 0 && (
            <div className="pt-2 border-t border-[#E4E2DC]">
              <AgeCohortChart cohorts={segments.eda.age_cohorts} />
            </div>
          )}
        </section>
      </div>

      {/* Gender Statistical Hypothesis Testing */}
      {segments.eda.gender_hypothesis_tests && segments.eda.gender_hypothesis_tests.length > 0 && (
        <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
          <div className="px-5 py-3.5 border-b border-[#E4E2DC] flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-[#141413]">
                Gender Behavioral Hypothesis Audit (Female vs. Male)
              </h2>
              <p className="text-xs text-[#64635E]">
                Welch two-sample t-test, Mann-Whitney U, and Cohen&apos;s d effect size across core attributes
              </p>
            </div>
            <span className="text-xs font-mono-tabular text-[#64635E]">
              {Object.entries(segments.eda.gender_breakdown.counts)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' · ')}
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                  <th className="py-2.5 px-5 font-medium">Feature</th>
                  <th className="py-2.5 px-4 text-right font-medium">Female Mean</th>
                  <th className="py-2.5 px-4 text-right font-medium">Male Mean</th>
                  <th className="py-2.5 px-4 text-right font-medium">Welch t</th>
                  <th className="py-2.5 px-4 text-right font-medium">Welch p</th>
                  <th className="py-2.5 px-4 text-right font-medium">Mann-Whitney p</th>
                  <th className="py-2.5 px-4 text-right font-medium">Cohen&apos;s d</th>
                  <th className="py-2.5 px-5 font-medium">Significance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E4E2DC]">
                {segments.eda.gender_hypothesis_tests.map((test) => (
                  <tr key={test.feature} className="hover:bg-[#FAF9F6]">
                    <td className="py-2.5 px-5 font-medium text-[#141413]">{test.feature}</td>
                    <td className="py-2.5 px-4 text-right font-mono-tabular">
                      {test.female_mean.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono-tabular">
                      {test.male_mean.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono-tabular">
                      {test.welch_t_stat.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono-tabular">
                      {test.welch_p_value.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono-tabular">
                      {test.mannwhitney_p_value.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono-tabular">
                      {test.cohens_d.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-5 font-mono-tabular">
                      {test.significant_at_05 ? (
                        <span className="text-[#B45309] font-medium">p &lt; 0.05</span>
                      ) : (
                        <span className="text-[#047857]">Parity (NS)</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Proportional Allocation Bar */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-[#141413]">Cohort Share Allocation</h2>
          <span className="text-xs font-mono-tabular text-[#64635E]">
            100% deterministic centroid alignment
          </span>
        </div>
        <div className="h-3.5 w-full rounded-md overflow-hidden flex bg-[#F1EFE9]">
          {segments.personas.map((p, idx) => (
            <div
              key={p.cluster_id}
              style={{
                width: `${p.share * 100}%`,
                backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length],
              }}
              title={`${p.persona_name}: ${(p.share * 100).toFixed(1)}%`}
            />
          ))}
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-2 pt-1 text-xs text-[#5C5B56]">
          {segments.personas.map((p, idx) => (
            <span key={p.cluster_id} className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-xs inline-block"
                style={{ backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length] }}
              />
              <span className="font-medium text-[#141413]">{p.persona_name}</span>
              <span className="font-mono-tabular text-[#64635E]">
                {(p.share * 100).toFixed(1)}%
              </span>
            </span>
          ))}
        </div>
      </section>

      {/* Playbook Table */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#E4E2DC] flex items-center justify-between">
          <h2 className="text-sm font-semibold text-[#141413]">
            Actionable Retail Personas, Quartile Profiles & Campaign Playbooks
          </h2>
          <span className="text-xs font-mono-tabular text-[#64635E]">
            {segments.personas.length} behavioral segments
          </span>
        </div>
        <div className="divide-y divide-[#E4E2DC]">
          {segments.personas.map((p, idx) => {
            const extra = analyticsByCluster.get(p.cluster_id);
            return (
              <div
                key={p.cluster_id}
                className="p-5 grid grid-cols-1 lg:grid-cols-12 gap-4 items-start hover:bg-[#FAF9F6] transition-colors"
              >
                <div className="lg:col-span-4 space-y-1">
                  <div className="flex items-center gap-2 text-xs text-[#64635E] font-mono-tabular">
                    <span
                      className="w-2.5 h-2.5 rounded-xs inline-block"
                      style={{ backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length] }}
                    />
                    <span>Cluster {p.cluster_id}</span>
                    <span aria-hidden="true">·</span>
                    <span>{p.customer_count} customers</span>
                    {extra && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="text-[#047857] font-medium">
                          {extra.revenue_share_pct.toFixed(1)}% Rev Share
                        </span>
                      </>
                    )}
                  </div>
                  <h3 className="text-base font-semibold text-[#141413]">{p.persona_name}</h3>
                  <p className="text-xs text-[#5C5B56]">{p.tagline}</p>
                </div>

                <div className="lg:col-span-4 grid grid-cols-4 gap-3 text-xs">
                  <div>
                    <span className="text-[#8A8880] block">Mean Age</span>
                    <span className="font-mono-tabular font-medium text-[#141413]">
                      {p.mean_age.toFixed(1)}
                    </span>
                    {extra && (
                      <span className="block text-[10px] font-mono-tabular text-[#8A8880]">
                        IQR {extra.age_q25.toFixed(0)}–{extra.age_q75.toFixed(0)}
                      </span>
                    )}
                  </div>
                  <div>
                    <span className="text-[#8A8880] block">Income</span>
                    <span className="font-mono-tabular font-medium text-[#141413]">
                      ${p.mean_income.toFixed(1)}k
                    </span>
                    {extra && (
                      <span className="block text-[10px] font-mono-tabular text-[#8A8880]">
                        IQR ${extra.income_q25.toFixed(0)}–{extra.income_q75.toFixed(0)}k
                      </span>
                    )}
                  </div>
                  <div>
                    <span className="text-[#8A8880] block">Spend</span>
                    <span className="font-mono-tabular font-medium text-[#141413]">
                      {p.mean_spending.toFixed(1)}
                    </span>
                    {extra && (
                      <span className="block text-[10px] font-mono-tabular text-[#8A8880]">
                        IQR {extra.spending_q25.toFixed(0)}–{extra.spending_q75.toFixed(0)}
                      </span>
                    )}
                  </div>
                  <div>
                    <span className="text-[#8A8880] block">Female</span>
                    <span className="font-mono-tabular font-medium text-[#141413]">
                      {(p.female_share * 100).toFixed(0)}%
                    </span>
                    {extra && (
                      <span className="block text-[10px] font-mono-tabular text-[#8A8880]">
                        Holdout {extra.boundary_rate_pct.toFixed(0)}%
                      </span>
                    )}
                  </div>
                </div>

                <div className="lg:col-span-4 space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium text-[#1E40AF]">{p.target_channel}</span>
                    <button
                      type="button"
                      onClick={() => onSelectCohort(p.persona_name)}
                      className="text-xs text-[#5C5B56] hover:text-[#141413] inline-flex items-center gap-0.5 cursor-pointer"
                    >
                      <span>View Cohort</span>
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <p className="text-xs text-[#5C5B56] leading-relaxed">{p.recommended_action}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
