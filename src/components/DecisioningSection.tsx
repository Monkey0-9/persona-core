import { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Compass,
  SlidersHorizontal,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { IncomeSpendingQuadrantChart, PcaManifoldChart } from './Charts';
import type {
  CounterfactualResult,
  CustomerRecord,
  ScoreResult,
  ScoringEventRecord,
  SegmentsPayload,
} from '../types';

const SCENARIO_PRESETS = [
  { label: 'Luxury VIP', gender: 'Female' as const, age: 32, income: 92, spending: 86 },
  { label: 'Cautious Affluent', gender: 'Male' as const, age: 44, income: 96, spending: 18 },
  { label: 'Boundary Corridor', gender: 'Female' as const, age: 29, income: 44, spending: 68 },
  { label: 'Young Trendsetter', gender: 'Female' as const, age: 24, income: 26, spending: 81 },
  { label: 'Budget Pragmatist', gender: 'Male' as const, age: 48, income: 25, spending: 19 },
];

export function DecisioningSection({
  segments,
  customers,
  colorMap,
  customerRefIn,
  genderIn,
  ageIn,
  incomeIn,
  spendingIn,
  maxIncomeSlider,
  liveScore,
  scoringBusy,
  scoreError,
  scoringEvents,
  onChangeRef,
  onChangeGender,
  onChangeAge,
  onChangeIncome,
  onChangeSpending,
  onSubmitScore,
}: {
  segments: SegmentsPayload;
  customers: CustomerRecord[];
  colorMap: Map<string, string>;
  customerRefIn: string;
  genderIn: 'Female' | 'Male';
  ageIn: number;
  incomeIn: number;
  spendingIn: number;
  maxIncomeSlider: number;
  liveScore: ScoreResult | null;
  scoringBusy: boolean;
  scoreError: string | null;
  scoringEvents: ScoringEventRecord[];
  onChangeRef: (v: string) => void;
  onChangeGender: (v: 'Female' | 'Male') => void;
  onChangeAge: (v: number) => void;
  onChangeIncome: (v: number) => void;
  onChangeSpending: (v: number) => void;
  onSubmitScore: (
    g?: 'Female' | 'Male',
    a?: number,
    inc?: number,
    sp?: number,
    ref?: string
  ) => Promise<void>;
}) {
  const [targetPersonaKey, setTargetPersonaKey] = useState<string>('affluent_vip');
  const [cfResult, setCfResult] = useState<CounterfactualResult | null>(null);
  const [cfBusy, setCfBusy] = useState<boolean>(false);
  const [cfError, setCfError] = useState<string | null>(null);

  const runCounterfactual = async () => {
    setCfBusy(true);
    setCfError(null);
    try {
      const res = await apiClient.getCounterfactual({
        customer_ref: customerRefIn.trim() || undefined,
        gender: genderIn,
        age: ageIn,
        annual_income_k: incomeIn,
        spending_score: spendingIn,
        target_persona_key: targetPersonaKey,
      });
      setCfResult(res);
    } catch (err) {
      setCfError(err instanceof Error ? err.message : 'Counterfactual simulation failed.');
    } finally {
      setCfBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-[#1E40AF]" />
              <h2 className="text-sm font-semibold text-[#141413]">Customer Profile Parameters</h2>
            </div>
            <span className="text-xs font-mono-tabular text-[#8A8880]">POST /api/v1/score</span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {SCENARIO_PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => {
                  onChangeGender(preset.gender);
                  onChangeAge(preset.age);
                  onChangeIncome(preset.income);
                  onChangeSpending(preset.spending);
                  void onSubmitScore(preset.gender, preset.age, preset.income, preset.spending);
                }}
                className="px-2.5 py-1 text-xs bg-[#F7F6F2] hover:bg-[#EAE8E1] border border-[#E4E2DC] rounded-md text-[#141413] cursor-pointer transition-colors"
              >
                {preset.label}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-xs text-[#5C5B56] mb-1">Customer Reference</label>
              <input
                type="text"
                maxLength={64}
                value={customerRefIn}
                onChange={(e) => onChangeRef(e.target.value)}
                placeholder="WALKIN-01"
                className="w-full px-3 py-1.5 text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[#141413] focus:outline-none focus:border-[#1E40AF]"
              />
            </div>

            <div>
              <span className="block text-xs text-[#5C5B56] mb-1.5">
                Gender ({segments.include_gender ? 'Included in distance' : 'Profiling only'})
              </span>
              <div className="grid grid-cols-2 gap-2">
                {(['Female', 'Male'] as const).map((g) => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => onChangeGender(g)}
                    className={`py-2 px-3 text-xs font-medium rounded-lg border cursor-pointer transition-colors ${
                      genderIn === g
                        ? 'bg-[#141413] text-white border-[#141413]'
                        : 'bg-[#FAF9F6] text-[#5C5B56] border-[#E4E2DC] hover:text-[#141413]'
                    }`}
                  >
                    {g}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-[#5C5B56]">Age (16–100 yrs)</span>
                <input
                  type="number"
                  min={16}
                  max={100}
                  value={ageIn}
                  onChange={(e) =>
                    onChangeAge(Math.min(100, Math.max(16, Number(e.target.value) || 16)))
                  }
                  className="w-20 px-2 py-0.5 text-right text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded"
                />
              </div>
              <input
                type="range"
                min={16}
                max={85}
                value={Math.min(85, ageIn)}
                onChange={(e) => onChangeAge(Number(e.target.value))}
                className="w-full accent-[#1E40AF]"
              />
            </div>

            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-[#5C5B56]">Annual Income (k$)</span>
                <input
                  type="number"
                  min={5}
                  max={500}
                  value={incomeIn}
                  onChange={(e) =>
                    onChangeIncome(Math.min(500, Math.max(5, Number(e.target.value) || 5)))
                  }
                  className="w-20 px-2 py-0.5 text-right text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded"
                />
              </div>
              <input
                type="range"
                min={10}
                max={maxIncomeSlider}
                value={Math.min(incomeIn, maxIncomeSlider)}
                onChange={(e) => onChangeIncome(Number(e.target.value))}
                className="w-full accent-[#1E40AF]"
              />
            </div>

            <div>
              <div className="flex justify-between items-center text-xs mb-1">
                <span className="text-[#5C5B56]">Spending Score (1–100)</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={spendingIn}
                  onChange={(e) =>
                    onChangeSpending(Math.min(100, Math.max(1, Number(e.target.value) || 1)))
                  }
                  className="w-20 px-2 py-0.5 text-right text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded"
                />
              </div>
              <input
                type="range"
                min={1}
                max={100}
                value={spendingIn}
                onChange={(e) => onChangeSpending(Number(e.target.value))}
                className="w-full accent-[#1E40AF]"
              />
            </div>

            <button
              type="button"
              disabled={scoringBusy}
              onClick={() => void onSubmitScore()}
              className="w-full py-2.5 px-4 text-xs font-semibold bg-[#1E40AF] text-white rounded-lg hover:bg-[#1E3A8A] disabled:opacity-60 transition-colors cursor-pointer"
            >
              {scoringBusy ? 'Scoring via FastAPI…' : 'Score & Persist Event'}
            </button>

            {scoreError && <p className="text-xs text-[#BE123C]">{scoreError}</p>}
          </div>
        </section>

        <section className="lg:col-span-7 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-6">
          {liveScore ? (
            <>
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <p className="text-xs font-mono-tabular text-[#64635E]">
                    Cluster {liveScore.cluster_id} · PCA ({liveScore.pca_1.toFixed(2)},{' '}
                    {liveScore.pca_2.toFixed(2)})
                  </p>
                  <h2 className="text-2xl font-bold text-[#141413]">{liveScore.persona_name}</h2>
                </div>
                <div className="text-xs font-mono-tabular">
                  {liveScore.is_boundary_customer ? (
                    <span className="inline-flex items-center gap-1.5 text-[#B45309] font-semibold">
                      <AlertTriangle className="w-4 h-4" />
                      <span>Boundary Holdout</span>
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 text-[#047857] font-semibold">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Clear Assignment</span>
                    </span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 border-y border-[#E4E2DC] py-4">
                <div>
                  <span className="text-xs text-[#64635E] block">GMM Confidence</span>
                  <span className="text-xl font-semibold font-mono-tabular text-[#141413]">
                    {(liveScore.gmm_confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-xs text-[#64635E] block">Normalized Entropy</span>
                  <span className="text-xl font-semibold font-mono-tabular text-[#1E40AF]">
                    {liveScore.normalized_entropy.toFixed(4)}
                  </span>
                </div>
                <div>
                  <span className="text-xs text-[#64635E] block">Centroid Distance</span>
                  <span className="text-xl font-semibold font-mono-tabular text-[#141413]">
                    {liveScore.distance_to_centroid.toFixed(3)}σ
                  </span>
                </div>
                <div>
                  <span className="text-xs text-[#64635E] block">Baseline Drift</span>
                  <span
                    className={`text-sm font-semibold font-mono-tabular ${
                      liveScore.drift_flag ? 'text-[#BE123C]' : 'text-[#047857]'
                    }`}
                  >
                    {liveScore.drift_flag ? 'Out-of-Range' : 'In Baseline'}
                  </span>
                </div>
              </div>

              {/* Aligned GMM Posterior Probability Distribution */}
              <div className="space-y-2.5">
                <h3 className="text-xs font-semibold text-[#141413]">
                  GMM Posterior Probability Distribution
                </h3>
                <div className="space-y-2">
                  {Object.entries(liveScore.posterior_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([personaName, prob]) => (
                      <div key={personaName} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-[#5C5B56]">{personaName}</span>
                          <span className="font-mono-tabular text-[#141413]">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-[#F1EFE9] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, prob * 100)}%`,
                              backgroundColor: colorMap.get(personaName) ?? '#1D4ED8',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              <div className="pt-2 border-t border-[#E4E2DC] space-y-2">
                <div>
                  <p className="text-xs font-medium text-[#64635E]">Recommended Playbook</p>
                  <p className="text-xs text-[#141413] leading-relaxed">
                    {liveScore.recommended_action}
                  </p>
                </div>
                {(() => {
                  const matchingRules = segments.explainability.rules.filter(
                    (r) => r.cluster_id === liveScore.cluster_id
                  );
                  if (!matchingRules.length) return null;
                  return (
                    <div className="p-2.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[11px] font-mono-tabular text-[#5C5B56]">
                      <span className="text-[#141413] font-semibold">Surrogate Rule: </span>
                      {matchingRules[0].condition_text}
                    </div>
                  );
                })()}
              </div>
            </>
          ) : (
            <p className="text-xs text-[#64635E]">Submit parameters to score customer.</p>
          )}
        </section>
      </div>

      {/* Counterfactual "What-If" Segment Upgrade Simulator */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Compass className="w-4 h-4 text-[#047857]" />
              <h2 className="text-sm font-semibold text-[#141413]">
                Counterfactual What-If Transition Simulator
              </h2>
            </div>
            <p className="text-xs text-[#64635E]">
              Computes the minimum Age / Income / Spending delta required to migrate this customer into a target persona
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={targetPersonaKey}
              onChange={(e) => setTargetPersonaKey(e.target.value)}
              className="px-3 py-1.5 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[#141413]"
            >
              {segments.personas.map((p) => (
                <option key={p.persona_key} value={p.persona_key}>
                  Target: {p.persona_name}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={cfBusy}
              onClick={() => void runCounterfactual()}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold bg-[#047857] text-white rounded-lg hover:bg-[#065F46] cursor-pointer disabled:opacity-60"
            >
              <span>{cfBusy ? 'Simulating…' : 'Compute Minimum Path'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {cfError && <p className="text-xs text-[#BE123C]">{cfError}</p>}

        {cfResult && (
          <div className="p-4 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg space-y-3">
            <p className="text-xs font-medium text-[#141413]">{cfResult.action_summary}</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <span className="text-[#64635E] block">Delta Age</span>
                <span className="font-mono-tabular font-semibold text-[#141413]">
                  {cfResult.delta_age >= 0 ? `+${cfResult.delta_age}` : cfResult.delta_age} yrs (→{' '}
                  {cfResult.counterfactual_profile.age})
                </span>
              </div>
              <div>
                <span className="text-[#64635E] block">Delta Income</span>
                <span className="font-mono-tabular font-semibold text-[#141413]">
                  {cfResult.delta_income_k >= 0
                    ? `+${cfResult.delta_income_k}`
                    : cfResult.delta_income_k}
                  k$ (→ ${cfResult.counterfactual_profile.annual_income_k}k)
                </span>
              </div>
              <div>
                <span className="text-[#64635E] block">Delta Spending</span>
                <span className="font-mono-tabular font-semibold text-[#1E40AF]">
                  {cfResult.delta_spending_score >= 0
                    ? `+${cfResult.delta_spending_score}`
                    : cfResult.delta_spending_score}{' '}
                  pts (→ {cfResult.counterfactual_profile.spending_score})
                </span>
              </div>
              <div>
                <span className="text-[#64635E] block">Target Posterior</span>
                <span className="font-mono-tabular font-semibold text-[#047857]">
                  {(cfResult.target_posterior_probability * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-6 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#141413]">
              Live Position: Income vs. Spending & PCA Manifold
            </h2>
            <Activity className="w-4 h-4 text-[#8A8880]" />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <IncomeSpendingQuadrantChart
              customers={customers}
              ellipses={segments.gmm_ellipses ?? []}
              colorMap={colorMap}
              livePin={liveScore}
            />
            <PcaManifoldChart
              customers={customers}
              colorMap={colorMap}
              explainedVariance={segments.features.explained_variance_ratio}
              livePin={liveScore}
            />
          </div>
        </section>

        <section className="lg:col-span-6 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-[#E4E2DC] flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#141413]">
              Persisted Scoring Events ({scoringEvents.length})
            </h2>
            <span className="text-xs font-mono-tabular text-[#64635E]">scoring_events table</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                  <th className="py-2.5 px-4 font-medium">Profile</th>
                  <th className="py-2.5 px-3 font-medium">Persona</th>
                  <th className="py-2.5 px-3 text-right font-medium">Entropy</th>
                  <th className="py-2.5 px-4 font-medium">Gate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E4E2DC]">
                {scoringEvents.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-6 text-center text-[#8A8880]">
                      Score a customer above to persist events.
                    </td>
                  </tr>
                ) : (
                  scoringEvents.slice(0, 8).map((ev) => (
                    <tr key={ev.id} className="hover:bg-[#FAF9F6]">
                      <td className="py-2.5 px-4 font-mono-tabular text-[#141413]">
                        {ev.age}y · ${ev.annual_income_k}k · {ev.spending_score}
                      </td>
                      <td className="py-2.5 px-3 text-[#141413]">{ev.persona_name}</td>
                      <td className="py-2.5 px-3 text-right font-mono-tabular">
                        {ev.normalized_entropy.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-4 font-mono-tabular">
                        {ev.is_boundary_customer ? (
                          <span className="text-[#B45309]">Holdout</span>
                        ) : (
                          <span className="text-[#047857]">Clear</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
