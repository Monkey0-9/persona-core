import { useState, type ChangeEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Database,
  Filter,
  GitBranch,
  Layers,
  RotateCcw,
  Search,
  Upload,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type {
  ActiveModelMetadata,
  CustomerRecord,
  DriftReportItem,
  ModelComparisonResult,
  SegmentsPayload,
} from '../types';

export function DriftSection({
  segments,
  customers,
  customerTotal,
  driftReports,
  activeModel,
  segmentFilter,
  boundaryOnly,
  searchQuery,
  batchBusy,
  batchNotice,
  onChangeSegmentFilter,
  onToggleBoundaryOnly,
  onChangeSearchQuery,
  onRunScenarioBatch,
  onCsvUpload,
  onModelChanged,
}: {
  segments: SegmentsPayload;
  customers: CustomerRecord[];
  customerTotal: number;
  driftReports: DriftReportItem[];
  activeModel: ActiveModelMetadata | undefined;
  segmentFilter: string;
  boundaryOnly: boolean;
  searchQuery: string;
  batchBusy: boolean;
  batchNotice: string | null;
  onChangeSegmentFilter: (v: string) => void;
  onToggleBoundaryOnly: () => void;
  onChangeSearchQuery: (v: string) => void;
  onRunScenarioBatch: (s: 'nominal' | 'luxury_drift' | 'youth_wave') => Promise<void>;
  onCsvUpload: (e: ChangeEvent<HTMLInputElement>) => Promise<void>;
  onModelChanged: () => void;
}) {
  const [candVersion, setCandVersion] = useState<string>('v1.1.0');
  const [candK, setCandK] = useState<number>(5);
  const [candGender, setCandGender] = useState<boolean>(false);
  const [registryBusy, setRegistryBusy] = useState<boolean>(false);
  const [registryMsg, setRegistryMsg] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ModelComparisonResult | null>(null);

  const modelsQuery = useQuery({
    queryKey: ['model-versions', activeModel?.version],
    queryFn: () => apiClient.listModelVersions(),
  });

  const obsQuery = useQuery({
    queryKey: ['observability-summary'],
    queryFn: () => apiClient.getObservabilitySummary(),
    refetchInterval: 15_000,
  });

  const filteredCustomers = customers.filter((c) => {
    if (segmentFilter !== 'All' && c.persona_name !== segmentFilter) return false;
    if (boundaryOnly && !c.is_boundary_customer) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      const refMatch = (c.customer_ref || '').toLowerCase().includes(q);
      const segMatch = c.persona_name.toLowerCase().includes(q);
      if (!refMatch && !segMatch) return false;
    }
    return true;
  });

  const handleRegisterCandidate = async () => {
    setRegistryBusy(true);
    setRegistryMsg(null);
    try {
      const res = await apiClient.registerCandidateModel({
        version: candVersion.trim(),
        target_k: candK,
        include_gender: candGender,
        notes: `UI candidate k=${candK} gender=${candGender}`,
      });
      setComparison(res.comparison);
      setRegistryMsg(`Registered candidate ${res.version} (run ${res.run_key}).`);
      void modelsQuery.refetch();
    } catch (err) {
      setRegistryMsg(err instanceof Error ? err.message : 'Candidate registration failed.');
    } finally {
      setRegistryBusy(false);
    }
  };

  const handlePromote = async (version: string) => {
    setRegistryBusy(true);
    setRegistryMsg(null);
    try {
      await apiClient.promoteModelVersion(version);
      setRegistryMsg(`Promoted ${version} to active production model.`);
      void modelsQuery.refetch();
      onModelChanged();
    } catch (err) {
      setRegistryMsg(err instanceof Error ? err.message : 'Promotion failed.');
    } finally {
      setRegistryBusy(false);
    }
  };

  const handleRollback = async () => {
    setRegistryBusy(true);
    setRegistryMsg(null);
    try {
      const mv = await apiClient.rollbackModelVersion();
      setRegistryMsg(`Rolled back active model to ${mv.version}.`);
      void modelsQuery.refetch();
      onModelChanged();
    } catch (err) {
      setRegistryMsg(err instanceof Error ? err.message : 'Rollback failed.');
    } finally {
      setRegistryBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Batch Scoring & PSI Audit */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-[#141413]">
              Batch Scoring & Population Stability Index (PSI) Audit
            </h2>
            <p className="text-xs text-[#64635E]">
              Executes POST /api/v1/customers/batch-score, computes per-feature PSI & KS tests, and persists drift reports
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={batchBusy}
              onClick={() => void onRunScenarioBatch('nominal')}
              className="px-3 py-1.5 text-xs font-medium bg-[#F7F6F2] hover:bg-[#EAE8E1] border border-[#E4E2DC] rounded-lg cursor-pointer"
            >
              Run Nominal Batch
            </button>
            <button
              type="button"
              disabled={batchBusy}
              onClick={() => void onRunScenarioBatch('youth_wave')}
              className="px-3 py-1.5 text-xs font-medium bg-[#F7F6F2] hover:bg-[#EAE8E1] border border-[#E4E2DC] rounded-lg cursor-pointer"
            >
              Test Youth Shift
            </button>
            <button
              type="button"
              disabled={batchBusy}
              onClick={() => void onRunScenarioBatch('luxury_drift')}
              className="px-3 py-1.5 text-xs font-medium bg-[#141413] text-white hover:bg-[#2B2B29] rounded-lg cursor-pointer"
            >
              Test Luxury Drift
            </button>
            <label className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#1E40AF] text-white hover:bg-[#1E3A8A] rounded-lg cursor-pointer">
              <Upload className="w-3.5 h-3.5" />
              <span>Upload CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => void onCsvUpload(e)}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {batchNotice && (
          <div className="px-3.5 py-2.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-xs font-mono-tabular text-[#141413]">
            {batchNotice}
          </div>
        )}

        {activeModel && (
          <div className="p-4 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-[#64635E] block">Active Model Version</span>
              <span className="font-mono-tabular font-semibold text-[#141413]">
                {activeModel.version}
              </span>
            </div>
            <div>
              <span className="text-[#64635E] block">Feature Schema Hash</span>
              <span className="font-mono-tabular font-semibold text-[#141413]">
                {activeModel.feature_schema_hash}
              </span>
            </div>
            <div>
              <span className="text-[#64635E] block">Registered Algorithm</span>
              <span className="font-medium text-[#141413]">{activeModel.algorithm}</span>
            </div>
            <div>
              <span className="text-[#64635E] block">Artifact Path</span>
              <span className="font-mono-tabular text-[#5C5B56] truncate block">
                {activeModel.artifact_path}
              </span>
            </div>
          </div>
        )}

        {driftReports.length > 0 ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {driftReports[0].features.map((feat) => (
                <div
                  key={feat.feature}
                  className="p-4 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg space-y-2"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-[#141413]">{feat.feature}</span>
                    <span
                      className={`font-mono-tabular font-semibold ${
                        feat.severity === 'stable'
                          ? 'text-[#047857]'
                          : feat.severity === 'monitor'
                            ? 'text-[#B45309]'
                            : 'text-[#BE123C]'
                      }`}
                    >
                      {feat.severity.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-baseline justify-between">
                    <span className="text-xl font-semibold font-mono-tabular text-[#141413]">
                      PSI {feat.psi.toFixed(4)}
                    </span>
                    <span className="text-xs font-mono-tabular text-[#64635E]">
                      μ {feat.baseline_mean.toFixed(1)} → {feat.current_mean.toFixed(1)}
                    </span>
                  </div>
                  {typeof feat.ks_statistic === 'number' && (
                    <div className="text-[11px] font-mono-tabular text-[#5C5B56]">
                      KS D={feat.ks_statistic.toFixed(4)} · p={feat.ks_pvalue?.toFixed(4)}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="overflow-x-auto border border-[#E4E2DC] rounded-lg">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                    <th className="py-2.5 px-4 font-medium">Audit ID</th>
                    <th className="py-2.5 px-4 text-right font-medium">Batch Size</th>
                    <th className="py-2.5 px-4 text-right font-medium">Max PSI</th>
                    <th className="py-2.5 px-4 font-medium">Status</th>
                    <th className="py-2.5 px-4 font-medium">Recorded</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E4E2DC]">
                  {driftReports.map((rep) => (
                    <tr key={rep.id}>
                      <td className="py-2.5 px-4 font-mono-tabular">DRIFT-{rep.id}</td>
                      <td className="py-2.5 px-4 text-right font-mono-tabular">{rep.batch_size}</td>
                      <td className="py-2.5 px-4 text-right font-mono-tabular">
                        {rep.max_psi.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-4 font-mono-tabular font-semibold">
                        <span
                          className={
                            rep.overall_status === 'stable'
                              ? 'text-[#047857]'
                              : rep.overall_status === 'monitor'
                                ? 'text-[#B45309]'
                                : 'text-[#BE123C]'
                          }
                        >
                          {rep.overall_status}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 font-mono-tabular text-[#64635E]">
                        {new Date(rep.created_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center text-xs text-[#8A8880] border border-dashed border-[#E4E2DC] rounded-lg">
            <GitBranch className="w-4 h-4 mx-auto mb-1.5 text-[#8A8880]" />
            Run a batch scenario or upload a CSV to compute and persist PSI drift telemetry.
          </div>
        )}
      </section>

      {/* Model Registry Governance + Live Observability Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-8 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#1E40AF]" />
              <h2 className="text-sm font-semibold text-[#141413]">
                Model Registry Governance & Promotion
              </h2>
            </div>
            <button
              type="button"
              disabled={registryBusy}
              onClick={() => void handleRollback()}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium bg-[#FAF9F6] hover:bg-[#EFECE5] border border-[#E4E2DC] rounded-md cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Rollback Previous</span>
            </button>
          </div>

          <div className="flex flex-wrap items-end gap-2.5 p-3.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-xs">
            <div>
              <label className="block text-[11px] text-[#64635E] mb-1">Candidate Tag</label>
              <input
                type="text"
                value={candVersion}
                onChange={(e) => setCandVersion(e.target.value)}
                placeholder="v1.1.0"
                className="w-28 px-2.5 py-1.5 font-mono-tabular bg-[#FFFFFF] border border-[#E4E2DC] rounded"
              />
            </div>
            <div>
              <label className="block text-[11px] text-[#64635E] mb-1">k Clusters</label>
              <input
                type="number"
                min={2}
                max={10}
                value={candK}
                onChange={(e) => setCandK(Number(e.target.value))}
                className="w-20 px-2.5 py-1.5 font-mono-tabular bg-[#FFFFFF] border border-[#E4E2DC] rounded"
              />
            </div>
            <label className="inline-flex items-center gap-1.5 py-1.5 px-2 text-xs text-[#5C5B56] cursor-pointer">
              <input
                type="checkbox"
                checked={candGender}
                onChange={(e) => setCandGender(e.target.checked)}
              />
              <span>Include Gender</span>
            </label>
            <button
              type="button"
              disabled={registryBusy}
              onClick={() => void handleRegisterCandidate()}
              className="px-3 py-1.5 text-xs font-semibold bg-[#141413] text-white rounded hover:bg-[#2B2B29] cursor-pointer disabled:opacity-60"
            >
              {registryBusy ? 'Training…' : 'Register Candidate'}
            </button>
          </div>

          {registryMsg && (
            <p className="text-xs font-mono-tabular text-[#047857]">{registryMsg}</p>
          )}

          {comparison && (
            <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-xs font-mono-tabular flex flex-wrap gap-4">
              <span>Candidate {comparison.candidate.version} vs Baseline:</span>
              <span>
                Δ Silhouette:{' '}
                {comparison.deltas.kmeans_silhouette !== null
                  ? comparison.deltas.kmeans_silhouette.toFixed(4)
                  : 'N/A'}
              </span>
              <span>
                Δ Jaccard:{' '}
                {comparison.deltas.mean_jaccard_stability !== null
                  ? comparison.deltas.mean_jaccard_stability.toFixed(4)
                  : 'N/A'}
              </span>
              <span>
                L2 Centroid Shift:{' '}
                {comparison.centroid_shift_l2 !== null
                  ? comparison.centroid_shift_l2.toFixed(4)
                  : 'N/A'}
              </span>
            </div>
          )}

          <div className="overflow-x-auto border border-[#E4E2DC] rounded-lg">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                  <th className="py-2 px-3 font-medium">Version</th>
                  <th className="py-2 px-3 font-medium">Status</th>
                  <th className="py-2 px-3 font-medium">Schema Hash</th>
                  <th className="py-2 px-3 text-right font-medium">Silhouette</th>
                  <th className="py-2 px-3 text-right font-medium">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E4E2DC]">
                {(modelsQuery.data?.items ?? []).map((mv) => (
                  <tr key={mv.id}>
                    <td className="py-2 px-3 font-mono-tabular font-semibold text-[#141413]">
                      {mv.version}
                    </td>
                    <td className="py-2 px-3 font-mono-tabular">
                      <span
                        className={
                          mv.is_active ? 'text-[#047857] font-semibold' : 'text-[#64635E]'
                        }
                      >
                        {mv.is_active ? 'ACTIVE' : mv.status}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono-tabular text-[#64635E]">
                      {mv.feature_schema_hash}
                    </td>
                    <td className="py-2 px-3 text-right font-mono-tabular">
                      {typeof mv.metrics.kmeans_silhouette === 'number'
                        ? mv.metrics.kmeans_silhouette.toFixed(4)
                        : '—'}
                    </td>
                    <td className="py-2 px-3 text-right">
                      {!mv.is_active && (
                        <button
                          type="button"
                          disabled={registryBusy}
                          onClick={() => void handlePromote(mv.version)}
                          className="px-2 py-1 text-[11px] font-medium bg-[#1E40AF] text-white rounded hover:bg-[#1E3A8A] cursor-pointer"
                        >
                          Promote
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="lg:col-span-4 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#047857]" />
            <h2 className="text-sm font-semibold text-[#141413]">Service Observability</h2>
          </div>

          {obsQuery.data ? (
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg">
                  <span className="text-[#64635E] block">Total Requests</span>
                  <span className="text-lg font-semibold font-mono-tabular text-[#141413]">
                    {obsQuery.data.total_requests}
                  </span>
                </div>
                <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg">
                  <span className="text-[#64635E] block">p95 / p99 Latency</span>
                  <span className="text-lg font-semibold font-mono-tabular text-[#1E40AF]">
                    {obsQuery.data.p95_latency_ms} / {obsQuery.data.p99_latency_ms} ms
                  </span>
                </div>
              </div>
              <div className="p-3 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg space-y-1.5 font-mono-tabular">
                <div className="flex justify-between">
                  <span className="text-[#64635E]">Database</span>
                  <span className={obsQuery.data.db_up ? 'text-[#047857]' : 'text-[#BE123C]'}>
                    {obsQuery.data.db_up ? 'CONNECTED' : 'DOWN'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#64635E]">Redis Limiter</span>
                  <span className="text-[#141413]">
                    {obsQuery.data.redis_up === null
                      ? 'IN-MEMORY'
                      : obsQuery.data.redis_up
                        ? 'CONNECTED'
                        : 'FALLBACK'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#64635E]">Active Bundle</span>
                  <span className="text-[#141413]">
                    {obsQuery.data.active_model_version ?? 'v1.0.0'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-[#64635E]">
                Prometheus exposition active at <span className="font-mono-tabular">/api/metrics</span>
              </p>
            </div>
          ) : (
            <p className="text-xs text-[#64635E]">Loading telemetry…</p>
          )}
        </section>
      </div>

      {/* Scored Customer Ledger */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-[#E4E2DC] flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-[#1E40AF]" />
            <h2 className="text-sm font-semibold text-[#141413]">
              Scored Customer Ledger ({filteredCustomers.length} of {customerTotal})
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#8A8880] absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => onChangeSearchQuery(e.target.value)}
                placeholder="Search ref or persona"
                className="pl-8 pr-3 py-1.5 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[#141413] placeholder:text-[#8A8880] focus:outline-none focus:border-[#1E40AF]"
              />
            </div>

            <div className="flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-[#8A8880]" />
              <select
                value={segmentFilter}
                onChange={(e) => onChangeSegmentFilter(e.target.value)}
                className="px-2.5 py-1.5 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[#141413] focus:outline-none focus:border-[#1E40AF]"
              >
                <option value="All">All Personas</option>
                {segments.personas.map((p) => (
                  <option key={p.cluster_id} value={p.persona_name}>
                    {p.persona_name}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="button"
              onClick={onToggleBoundaryOnly}
              className={`px-3 py-1.5 text-xs rounded-lg border cursor-pointer transition-colors whitespace-nowrap shrink-0 ${
                boundaryOnly
                  ? 'bg-[#B45309]/10 border-[#B45309] text-[#B45309] font-medium'
                  : 'bg-[#FAF9F6] border-[#E4E2DC] text-[#5C5B56]'
              }`}
            >
              Boundary Holdouts Only
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                <th className="py-2.5 px-4 font-medium">Ref</th>
                <th className="py-2.5 px-3 font-medium">Source</th>
                <th className="py-2.5 px-3 font-medium">Gender</th>
                <th className="py-2.5 px-3 text-right font-medium">Age</th>
                <th className="py-2.5 px-3 text-right font-medium">Income</th>
                <th className="py-2.5 px-3 text-right font-medium">Score</th>
                <th className="py-2.5 px-3 font-medium">Persona</th>
                <th className="py-2.5 px-3 text-right font-medium">Confidence</th>
                <th className="py-2.5 px-3 text-right font-medium">Entropy</th>
                <th className="py-2.5 px-4 font-medium">Gate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E4E2DC]">
              {filteredCustomers.slice(0, 50).map((row) => (
                <tr key={row.id} className="hover:bg-[#FAF9F6]">
                  <td className="py-2.5 px-4 font-mono-tabular text-[#141413]">
                    {row.customer_ref || `BATCH-${row.id}`}
                  </td>
                  <td className="py-2.5 px-3 font-mono-tabular text-[#64635E]">{row.source}</td>
                  <td className="py-2.5 px-3 text-[#5C5B56]">{row.gender}</td>
                  <td className="py-2.5 px-3 text-right font-mono-tabular">{row.age}</td>
                  <td className="py-2.5 px-3 text-right font-mono-tabular">
                    ${row.annual_income_k.toFixed(0)}k
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono-tabular">
                    {row.spending_score.toFixed(0)}
                  </td>
                  <td className="py-2.5 px-3 font-medium text-[#141413]">{row.persona_name}</td>
                  <td className="py-2.5 px-3 text-right font-mono-tabular">
                    {(row.gmm_confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono-tabular">
                    {row.normalized_entropy.toFixed(3)}
                  </td>
                  <td className="py-2.5 px-4 font-mono-tabular">
                    {row.is_boundary_customer ? (
                      <span className="text-[#B45309] font-medium">Holdout</span>
                    ) : (
                      <span className="text-[#047857]">Clear</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
