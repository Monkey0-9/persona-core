import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, KeyRound, LogOut, RefreshCw, ShieldCheck } from 'lucide-react';
import { apiClient, getAuthSession, subscribeAuthSession } from './api/client';
import { BenchmarksSection } from './components/BenchmarksSection';
import { BriefSection } from './components/BriefSection';
import { type AlgorithmView } from './components/Charts';
import { DataWorkspace } from './components/DataWorkspace';
import { DecisioningSection } from './components/DecisioningSection';
import { DriftSection } from './components/DriftSection';
import { ErrorBoundary } from './components/ErrorBoundary';
import { GeometrySection } from './components/GeometrySection';
import {
  SEGMENT_PALETTE,
  type AuthSession,
  type CustomerRecord,
  type ScoreResult,
  type SegmentsPayload,
} from './types';

type WorkspaceSection = 'data' | 'brief' | 'geometry' | 'benchmarks' | 'decisioning' | 'drift';

const NAV_ITEMS: { id: WorkspaceSection; label: string; shortLabel: string }[] = [
  { id: 'data', label: 'Data & Pipeline', shortLabel: 'Data' },
  { id: 'brief', label: 'Executive Brief', shortLabel: 'Brief' },
  { id: 'geometry', label: 'Cluster Geometry', shortLabel: 'Geometry' },
  { id: 'benchmarks', label: 'Benchmarks & Rules', shortLabel: 'Benchmarks' },
  { id: 'decisioning', label: 'Live Decisioning', shortLabel: 'Decisioning' },
  { id: 'drift', label: 'Drift & Governance', shortLabel: 'Drift' },
];

export default function App() {
  const queryClient = useQueryClient();
  const [section, setSection] = useState<WorkspaceSection>('brief');

  // Geometry interactive state
  const [hoveredCustomer, setHoveredCustomer] = useState<CustomerRecord | null>(null);
  const [algorithmView, setAlgorithmView] = useState<AlgorithmView>('kmeans');
  const [highlightedClusterId, setHighlightedClusterId] = useState<number | null>(null);
  const [corrMode, setCorrMode] = useState<'pearson' | 'spearman'>('pearson');

  // Live scoring state (real POST /api/v1/customers/score)
  const [customerRefIn, setCustomerRefIn] = useState<string>('WALKIN-01');
  const [genderIn, setGenderIn] = useState<'Female' | 'Male'>('Female');
  const [ageIn, setAgeIn] = useState<number>(32);
  const [incomeIn, setIncomeIn] = useState<number>(88);
  const [spendingIn, setSpendingIn] = useState<number>(82);
  const [manualLiveScore, setManualLiveScore] = useState<ScoreResult | null>(null);
  const [scoringBusy, setScoringBusy] = useState<boolean>(false);
  const [scoreError, setScoreError] = useState<string | null>(null);

  // Customer ledger & batch state
  const [segmentFilter, setSegmentFilter] = useState<string>('All');
  const [boundaryOnly, setBoundaryOnly] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [batchBusy, setBatchBusy] = useState<boolean>(false);
  const [batchNotice, setBatchNotice] = useState<string | null>(null);

  // JWT session state (in-memory Bearer token)
  const [authSession, setAuthState] = useState<AuthSession | null>(() => getAuthSession());
  const [authModalOpen, setAuthModalOpen] = useState<boolean>(false);
  const [authEmail, setAuthEmail] = useState<string>('');
  const [authPassword, setAuthPassword] = useState<string>('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authBusy, setAuthBusy] = useState<boolean>(false);

  useEffect(() => subscribeAuthSession(setAuthState), []);

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
  });

  const segmentsQuery = useQuery({
    queryKey: ['segments'],
    queryFn: () => apiClient.getSegments(),
  });

  const customersQuery = useQuery({
    queryKey: ['customers', segmentsQuery.data?.run_key],
    queryFn: () => apiClient.getCustomers(500, 0),
    enabled: Boolean(segmentsQuery.data),
  });

  const scoringEventsQuery = useQuery({
    queryKey: ['scoring-events', segmentsQuery.data?.run_key],
    queryFn: () => apiClient.getScoringEvents(15, 0),
    enabled: Boolean(segmentsQuery.data),
  });

  const driftReportsQuery = useQuery({
    queryKey: ['drift-reports', segmentsQuery.data?.run_key],
    queryFn: () => apiClient.getDriftReports(10, 0),
    enabled: Boolean(segmentsQuery.data),
  });

  const activeModelQuery = useQuery({
    queryKey: ['active-model', segmentsQuery.data?.run_key],
    queryFn: () => apiClient.getActiveModel(),
    enabled: Boolean(segmentsQuery.data),
  });

  const authStatusQuery = useQuery({
    queryKey: ['auth-status'],
    queryFn: () => apiClient.getAuthStatus(),
  });

  const initialScoreQuery = useQuery({
    queryKey: ['initial-live-score', segmentsQuery.data?.run_key],
    queryFn: () =>
      apiClient.scoreCustomer({
        customer_ref: 'WALKIN-01',
        gender: 'Female',
        age: 32,
        annual_income_k: 88,
        spending_score: 82,
      }),
    enabled: Boolean(segmentsQuery.data) && manualLiveScore === null,
  });

  const liveScore = manualLiveScore ?? initialScoreQuery.data ?? null;
  const segments = segmentsQuery.data ?? null;
  const health = healthQuery.data ?? null;
  const customers = useMemo(() => customersQuery.data?.items ?? [], [customersQuery.data]);
  const customerTotal = customersQuery.data?.total ?? 0;
  const scoringEvents = scoringEventsQuery.data?.items ?? [];
  const driftReports = driftReportsQuery.data?.items ?? [];

  const reloadWorkspace = useCallback(async () => {
    await Promise.all([
      healthQuery.refetch(),
      segmentsQuery.refetch(),
      customersQuery.refetch(),
      scoringEventsQuery.refetch(),
      driftReportsQuery.refetch(),
      activeModelQuery.refetch(),
    ]);
  }, [
    healthQuery,
    segmentsQuery,
    customersQuery,
    scoringEventsQuery,
    driftReportsQuery,
    activeModelQuery,
  ]);

  const submitLiveScore = useCallback(
    async (
      genderVal = genderIn,
      ageVal = ageIn,
      incomeVal = incomeIn,
      spendingVal = spendingIn,
      refVal = customerRefIn
    ) => {
      setScoringBusy(true);
      setScoreError(null);
      try {
        const result = await apiClient.scoreCustomer({
          customer_ref: refVal.trim() || `STUDIO-${Date.now().toString().slice(-4)}`,
          gender: genderVal,
          age: ageVal,
          annual_income_k: incomeVal,
          spending_score: spendingVal,
        });
        setManualLiveScore(result);
        await scoringEventsQuery.refetch();
      } catch (err) {
        setScoreError(err instanceof Error ? err.message : 'Live scoring request failed');
      } finally {
        setScoringBusy(false);
      }
    },
    [genderIn, ageIn, incomeIn, spendingIn, customerRefIn, scoringEventsQuery]
  );

  const handlePipelineUpdated = useCallback(
    async (nextSegments: SegmentsPayload) => {
      queryClient.setQueryData(['segments'], nextSegments);
      setSegmentFilter('All');
      setBoundaryOnly(false);
      setHoveredCustomer(null);
      setManualLiveScore(null);
      await Promise.all([
        healthQuery.refetch(),
        customersQuery.refetch(),
        activeModelQuery.refetch(),
        scoringEventsQuery.refetch(),
      ]);
    },
    [queryClient, healthQuery, customersQuery, activeModelQuery, scoringEventsQuery]
  );

  const runScenarioBatch = async (scenario: 'nominal' | 'luxury_drift' | 'youth_wave') => {
    if (!customers.length) return;
    setBatchBusy(true);
    setBatchNotice(null);
    try {
      const sample = customers.slice(0, 40).map((c) => {
        if (scenario === 'luxury_drift') {
          return {
            gender: c.gender,
            age: Math.min(90, c.age + 14),
            annual_income_k: Math.min(250, c.annual_income_k + 34),
            spending_score: Math.min(100, c.spending_score + 12),
          };
        }
        if (scenario === 'youth_wave') {
          return {
            gender: c.gender,
            age: Math.max(18, c.age - 11),
            annual_income_k: Math.max(15, c.annual_income_k - 18),
            spending_score: Math.min(100, c.spending_score + 18),
          };
        }
        return {
          gender: c.gender,
          age: c.age,
          annual_income_k: c.annual_income_k,
          spending_score: c.spending_score,
        };
      });

      const resp = await apiClient.batchScore(sample);
      setBatchNotice(
        `Persisted ${resp.count} scored customers (${resp.boundary_count} boundary holdouts) · Batch drift ${resp.drift_report.overall_status.toUpperCase()} (max PSI ${resp.drift_report.max_psi.toFixed(4)})`
      );
      await Promise.all([
        customersQuery.refetch(),
        driftReportsQuery.refetch(),
        scoringEventsQuery.refetch(),
      ]);
    } catch (err) {
      setBatchNotice(err instanceof Error ? err.message : 'Batch scoring failed');
    } finally {
      setBatchBusy(false);
    }
  };

  const handleCsvUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBatchBusy(true);
    setBatchNotice(null);
    try {
      const text = await file.text();
      const lines = text
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
      if (lines.length < 2) {
        throw new Error('CSV must include a header row and at least one customer.');
      }
      const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
      const gIdx = headers.findIndex((h) => h === 'gender');
      const aIdx = headers.findIndex((h) => h === 'age');
      const iIdx = headers.findIndex((h) => h.includes('income'));
      const sIdx = headers.findIndex((h) => h.includes('spending'));
      if (gIdx < 0 || aIdx < 0 || iIdx < 0 || sIdx < 0) {
        throw new Error('CSV columns must include gender, age, annual_income_k, and spending_score.');
      }
      const parsed = lines.slice(1, 1001).map((line) => {
        const cols = line.split(',').map((c) => c.trim());
        const gRaw: 'Female' | 'Male' = cols[gIdx] === 'Male' ? 'Male' : 'Female';
        return {
          gender: gRaw,
          age: Number(cols[aIdx]),
          annual_income_k: Number(cols[iIdx]),
          spending_score: Number(cols[sIdx]),
        };
      });
      const resp = await apiClient.batchScore(parsed);
      setBatchNotice(
        `Uploaded & scored ${resp.count} customers · Drift ${resp.drift_report.overall_status.toUpperCase()} (max PSI ${resp.drift_report.max_psi.toFixed(4)})`
      );
      await Promise.all([
        customersQuery.refetch(),
        driftReportsQuery.refetch(),
        scoringEventsQuery.refetch(),
      ]);
    } catch (err) {
      setBatchNotice(err instanceof Error ? err.message : 'CSV upload failed');
    } finally {
      setBatchBusy(false);
      event.target.value = '';
    }
  };

  const handleJwtSubmit = async () => {
    setAuthBusy(true);
    setAuthError(null);
    try {
      if (authStatusQuery.data?.has_users) {
        await apiClient.login(authEmail, authPassword);
      } else {
        await apiClient.bootstrapAdmin(authEmail, authPassword);
        void authStatusQuery.refetch();
      }
      setAuthModalOpen(false);
      setAuthPassword('');
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : 'Authentication failed');
    } finally {
      setAuthBusy(false);
    }
  };

  const colorMap = useMemo(() => {
    const map = new Map<string, string>();
    if (!segments) return map;
    segments.personas.forEach((p, idx) => {
      map.set(p.persona_name, SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length]);
    });
    return map;
  }, [segments]);

  const maxIncomeSlider = useMemo(() => {
    const peak = customers.reduce((acc, c) => Math.max(acc, c.annual_income_k), 140);
    return Math.min(500, Math.max(150, Math.ceil((peak * 1.15) / 10) * 10));
  }, [customers]);

  if (segmentsQuery.isLoading && !segments) {
    return (
      <div className="min-h-screen bg-[#F7F6F2] flex">
        <aside className="hidden lg:block w-60 border-r border-[#E4E2DC] bg-[#FFFFFF] p-6 space-y-4">
          <div className="h-6 bg-[#EFECE5] rounded animate-pulse w-36" />
          <div className="space-y-2 pt-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-9 bg-[#F7F6F2] rounded animate-pulse" />
            ))}
          </div>
        </aside>
        <main className="flex-1 p-8 space-y-6">
          <div className="h-10 bg-[#FFFFFF] border border-[#E4E2DC] rounded-lg animate-pulse" />
          <div className="grid grid-cols-5 gap-4">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-24 bg-[#FFFFFF] border border-[#E4E2DC] rounded-lg animate-pulse" />
            ))}
          </div>
        </main>
      </div>
    );
  }

  if (segmentsQuery.error || !segments) {
    const errMsg =
      segmentsQuery.error instanceof Error
        ? segmentsQuery.error.message
        : 'Unable to load segment metadata from FastAPI.';
    return (
      <div className="min-h-screen bg-[#F7F6F2] flex items-center justify-center p-6">
        <div className="max-w-lg w-full bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 text-[#BE123C] font-semibold">
            <AlertTriangle className="w-5 h-5" />
            <span>Inference Backend Connection Error</span>
          </div>
          <p className="text-sm text-[#5C5B56] leading-relaxed">
            {errMsg} Ensure the training pipeline has run and the FastAPI service is reachable.
          </p>
          <button
            type="button"
            onClick={() => void reloadWorkspace()}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold bg-[#141413] text-white rounded-lg hover:bg-[#2B2B29] cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Retry Connection</span>
          </button>
        </div>
      </div>
    );
  }

  const activeNav = NAV_ITEMS.find((n) => n.id === section) ?? NAV_ITEMS[0];

  return (
    <div className="min-h-screen bg-[#F7F6F2] text-[#141413] flex flex-col lg:flex-row">
      {/* Left Workspace Navigation Rail (240px) */}
      <aside className="hidden lg:flex lg:w-60 lg:flex-col border-r border-[#E4E2DC] bg-[#FFFFFF] shrink-0">
        <div className="h-14 px-5 flex items-center border-b border-[#E4E2DC]">
          <span className="text-sm font-bold tracking-tight text-[#141413] whitespace-nowrap">
            Mall Segmentation
          </span>
        </div>

        <div className="p-4 flex-1 space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = section === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setSection(item.id)}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                  active
                    ? 'bg-[#141413] text-white'
                    : 'text-[#5C5B56] hover:bg-[#F7F6F2] hover:text-[#141413]'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        <div className="p-5 border-t border-[#E4E2DC] space-y-2 text-xs text-[#5C5B56]">
          <div className="flex items-center justify-between">
            <span>Run Key</span>
            <span className="font-mono-tabular text-[#141413]">
              {segments.run_key.replace('mall-seg-', '')}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Gender Metric</span>
            <span className="font-mono-tabular text-[#141413]">
              {segments.include_gender ? 'Included' : 'Profiling Only'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Entropy Gate</span>
            <span className="font-mono-tabular text-[#141413]">
              &gt; {segments.entropy_threshold.toFixed(2)}
            </span>
          </div>
          {activeModelQuery.data && (
            <div className="flex items-center justify-between">
              <span>Model Version</span>
              <span className="font-mono-tabular text-[#141413]">
                {activeModelQuery.data.version}
              </span>
            </div>
          )}
          <div className="pt-2 border-t border-[#E4E2DC]">
            {authSession ? (
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-medium text-[#141413] truncate">{authSession.user.email}</div>
                  <div className="text-[11px] font-mono-tabular text-[#5C5B56]">
                    Role · {authSession.user.role}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void apiClient.logout()}
                  title="Sign out JWT session"
                  className="p-1.5 text-[#5C5B56] hover:text-[#BE123C] cursor-pointer"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setAuthModalOpen(true)}
                className="w-full flex items-center justify-center gap-1.5 py-1.5 px-2.5 bg-[#F7F6F2] hover:bg-[#EFECE5] border border-[#E4E2DC] rounded-lg text-xs font-medium text-[#141413] cursor-pointer transition-colors"
              >
                <KeyRound className="w-3.5 h-3.5" />
                <span>{authStatusQuery.data?.has_users ? 'JWT Sign In' : 'Claim Admin JWT'}</span>
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 bg-[#FFFFFF] border-b border-[#E4E2DC] px-4 sm:px-6 flex items-center justify-between gap-3 sticky top-0 z-20">
          <div className="flex items-center gap-2 text-xs text-[#5C5B56] min-w-0">
            <span className="font-semibold text-[#141413] lg:hidden shrink-0">Mall Segmentation</span>
            <span className="hidden sm:inline lg:hidden" aria-hidden="true">
              /
            </span>
            <span className="hidden lg:inline text-[#8A8880]">Workspace</span>
            <span className="hidden lg:inline text-[#8A8880]" aria-hidden="true">
              /
            </span>
            <span className="hidden sm:inline font-medium text-[#141413] truncate">
              {activeNav.label}
            </span>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <span className="hidden sm:inline text-xs font-mono-tabular text-[#5C5B56] whitespace-nowrap">
              {health?.status === 'ok' ? 'API & DB Ready' : 'Degraded'}
            </span>
            <button
              type="button"
              onClick={() => setSection('decisioning')}
              className="px-4 py-1.5 text-xs font-semibold bg-[#1E40AF] text-white rounded-lg hover:bg-[#1E3A8A] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              Score Customer
            </button>
          </div>
        </header>

        {authModalOpen && (
          <div className="fixed inset-0 z-50 bg-[#141413]/40 flex items-center justify-center p-4">
            <div className="max-w-sm w-full bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-[#141413]">
                  {authStatusQuery.data?.has_users
                    ? 'Sign In with Role JWT'
                    : 'Initialize Admin Account'}
                </h2>
                <button
                  type="button"
                  onClick={() => setAuthModalOpen(false)}
                  className="text-xs text-[#64635E] hover:text-[#141413] cursor-pointer"
                >
                  Close
                </button>
              </div>
              <p className="text-xs text-[#5C5B56]">
                {authStatusQuery.data?.has_users
                  ? 'Authenticate with scrypt-verified email and password to receive a short-lived HS256 Bearer token.'
                  : 'Create the first administrator credential stored in SQLite/PostgreSQL with salted scrypt hashing.'}
              </p>
              <div className="space-y-3">
                <input
                  type="email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  placeholder="analyst@mall-seg.local"
                  className="w-full px-3 py-2 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg"
                />
                <input
                  type="password"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder="Password (min 8 characters)"
                  className="w-full px-3 py-2 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg"
                />
                {authError && <p className="text-xs text-[#BE123C]">{authError}</p>}
                <button
                  type="button"
                  disabled={authBusy}
                  onClick={() => void handleJwtSubmit()}
                  className="w-full py-2 px-4 text-xs font-semibold bg-[#141413] text-white rounded-lg hover:bg-[#2B2B29] disabled:opacity-60 cursor-pointer"
                >
                  {authBusy
                    ? 'Authenticating…'
                    : authStatusQuery.data?.has_users
                      ? 'Sign In'
                      : 'Create Admin & Issue Token'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Mobile Nav Bar */}
        <div className="flex lg:hidden gap-1.5 px-4 py-2.5 bg-[#FFFFFF] border-b border-[#E4E2DC] overflow-x-auto">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setSection(item.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md whitespace-nowrap shrink-0 ${
                section === item.id
                  ? 'bg-[#141413] text-white'
                  : 'bg-[#F7F6F2] text-[#5C5B56]'
              }`}
            >
              {item.shortLabel}
            </button>
          ))}
        </div>

        <main className="p-6 lg:p-8 max-w-[1280px] w-full mx-auto space-y-8">
          {/* Editorial Page Header */}
          <section className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[#E4E2DC] pb-5">
            <div className="space-y-1 max-w-2xl">
              <div className="flex items-center gap-2 text-xs text-[#64635E] font-mono-tabular">
                <span>Source {segments.dataset_source}</span>
                <span aria-hidden="true">·</span>
                <span>{segments.row_count} validated training profiles</span>
                <span aria-hidden="true">·</span>
                <span>{customerTotal} persisted records</span>
              </div>
              <h1 className="text-2xl font-bold text-[#141413] [text-wrap:balance]">
                Customer Segmentation & Uncertainty Decisioning
              </h1>
            </div>

            <div className="flex items-center gap-2 text-xs font-mono-tabular text-[#047857]">
              <ShieldCheck className="w-4 h-4 shrink-0" />
              <span>
                Bootstrap Gate {segments.passed_stability_gate ? 'Passed' : 'Failed'} · Jaccard{' '}
                {segments.mean_jaccard_stability.toFixed(4)}
              </span>
            </div>
          </section>

          {/* Quantitative KPI Ledger */}
          <section className="grid grid-cols-2 lg:grid-cols-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl divide-y lg:divide-y-0 lg:divide-x divide-[#E4E2DC]">
            {[
              {
                label: 'Silhouette Score',
                value: segments.silhouette_score.toFixed(4),
                note: `K-Means k=${segments.target_k}`,
              },
              {
                label: 'Bootstrap Jaccard',
                value: segments.mean_jaccard_stability.toFixed(4),
                note: `B=${segments.stability.n_bootstrap} Hungarian`,
              },
              {
                label: 'Surrogate Fidelity',
                value: `${(segments.surrogate_fidelity * 100).toFixed(1)}%`,
                note: `Depth-${segments.explainability.max_depth} tree`,
              },
              {
                label: 'PCA Variance',
                value: `${(segments.features.cumulative_variance_ratio * 100).toFixed(1)}%`,
                note: '2 components',
              },
              {
                label: 'Boundary Holdouts',
                value: String(segments.boundary_customer_count),
                note: `Entropy > ${segments.entropy_threshold}`,
              },
            ].map((item) => (
              <div key={item.label} className="p-4 space-y-1">
                <p className="text-xs text-[#64635E]">{item.label}</p>
                <p className="text-2xl font-semibold font-mono-tabular text-[#141413]">
                  {item.value}
                </p>
                <p className="text-xs font-mono-tabular text-[#8A8880]">{item.note}</p>
              </div>
            ))}
          </section>

          <ErrorBoundary key={section} title="Workspace View Error">
            {section === 'data' && (
              <DataWorkspace
                segments={segments}
                activeCustomers={customers}
                onPipelineUpdated={handlePipelineUpdated}
                onNavigateBrief={() => setSection('brief')}
                onNavigateGeometry={() => setSection('geometry')}
              />
            )}

            {section === 'brief' && (
              <BriefSection
                segments={segments}
                customers={customers}
                colorMap={colorMap}
                liveScore={liveScore}
                highlightedClusterId={highlightedClusterId}
                onHighlightCluster={setHighlightedClusterId}
                onHoverCustomer={setHoveredCustomer}
                onSelectCohort={(personaName) => {
                  setSegmentFilter(personaName);
                  setSection('drift');
                }}
              />
            )}

            {section === 'geometry' && (
              <GeometrySection
                segments={segments}
                customers={customers}
                colorMap={colorMap}
                liveScore={liveScore}
                algorithmView={algorithmView}
                highlightedClusterId={highlightedClusterId}
                hoveredCustomer={hoveredCustomer}
                onChangeAlgorithm={setAlgorithmView}
                onHighlightCluster={setHighlightedClusterId}
                onHoverCustomer={setHoveredCustomer}
              />
            )}

            {section === 'benchmarks' && (
              <BenchmarksSection
                segments={segments}
                corrMode={corrMode}
                onChangeCorrMode={setCorrMode}
              />
            )}

            {section === 'decisioning' && (
              <DecisioningSection
                segments={segments}
                customers={customers}
                colorMap={colorMap}
                customerRefIn={customerRefIn}
                genderIn={genderIn}
                ageIn={ageIn}
                incomeIn={incomeIn}
                spendingIn={spendingIn}
                maxIncomeSlider={maxIncomeSlider}
                liveScore={liveScore}
                scoringBusy={scoringBusy}
                scoreError={scoreError}
                scoringEvents={scoringEvents}
                onChangeRef={setCustomerRefIn}
                onChangeGender={setGenderIn}
                onChangeAge={setAgeIn}
                onChangeIncome={setIncomeIn}
                onChangeSpending={setSpendingIn}
                onSubmitScore={submitLiveScore}
              />
            )}

            {section === 'drift' && (
              <DriftSection
                segments={segments}
                customers={customers}
                customerTotal={customerTotal}
                driftReports={driftReports}
                activeModel={activeModelQuery.data}
                segmentFilter={segmentFilter}
                boundaryOnly={boundaryOnly}
                searchQuery={searchQuery}
                batchBusy={batchBusy}
                batchNotice={batchNotice}
                onChangeSegmentFilter={setSegmentFilter}
                onToggleBoundaryOnly={() => setBoundaryOnly((prev) => !prev)}
                onChangeSearchQuery={setSearchQuery}
                onRunScenarioBatch={runScenarioBatch}
                onCsvUpload={handleCsvUpload}
                onModelChanged={() => {
                  void reloadWorkspace();
                }}
              />
            )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
