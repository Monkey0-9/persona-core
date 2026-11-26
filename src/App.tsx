import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Compass,
  Database,
  Filter,
  GitBranch,
  KeyRound,
  Layers,
  LogOut,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Upload,
} from 'lucide-react';
import {
  apiClient,
  getAuthSession,
  subscribeAuthSession,
} from './api/client';
import {
  CorrelationMatrixChart,
  FeatureHistogramChart,
  GenderDemographicChart,
  IncomeSpendingQuadrantChart,
  MetricSweepChart,
  PcaManifoldChart,
  WardDendrogramChart,
  type AlgorithmView,
} from './components/Charts';
import { DataWorkspace } from './components/DataWorkspace';
import { ErrorBoundary } from './components/ErrorBoundary';
import {
  SEGMENT_PALETTE,
  type AuthSession,
  type CustomerRecord,
  type DriftReportItem,
  type HealthStatus,
  type ScoreResult,
  type ScoringEventRecord,
  type SegmentsPayload,
} from './types';

type WorkspaceSection = 'data' | 'brief' | 'geometry' | 'benchmarks' | 'decisioning' | 'drift';

const NAV_ITEMS: { id: WorkspaceSection; label: string; shortLabel: string }[] = [
  { id: 'data', label: '00. Data & Pipeline', shortLabel: 'Data' },
  { id: 'brief', label: '01. Executive Brief', shortLabel: 'Brief' },
  { id: 'geometry', label: '02. Cluster Geometry', shortLabel: 'Geometry' },
  { id: 'benchmarks', label: '03. Benchmarks & Rules', shortLabel: 'Benchmarks' },
  { id: 'decisioning', label: '04. Live Decisioning', shortLabel: 'Decisioning' },
  { id: 'drift', label: '05. Drift & Ledger', shortLabel: 'Drift' },
];

const SCENARIO_PRESETS = [
  { label: 'Luxury VIP', gender: 'Female' as const, age: 32, income: 92, spending: 86 },
  { label: 'Cautious Affluent', gender: 'Male' as const, age: 44, income: 96, spending: 18 },
  { label: 'Boundary Corridor', gender: 'Female' as const, age: 29, income: 44, spending: 68 },
  { label: 'Young Trendsetter', gender: 'Female' as const, age: 24, income: 26, spending: 81 },
  { label: 'Budget Pragmatist', gender: 'Male' as const, age: 48, income: 25, spending: 19 },
];

export default function App() {
  const [section, setSection] = useState<WorkspaceSection>('brief');
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [segments, setSegments] = useState<SegmentsPayload | null>(null);
  const [customers, setCustomers] = useState<CustomerRecord[]>([]);
  const [customerTotal, setCustomerTotal] = useState<number>(0);
  const [scoringEvents, setScoringEvents] = useState<ScoringEventRecord[]>([]);
  const [driftReports, setDriftReports] = useState<DriftReportItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);

  // Geometry interactive state
  const [hoveredCustomer, setHoveredCustomer] = useState<CustomerRecord | null>(null);
  const [algorithmView, setAlgorithmView] = useState<AlgorithmView>('kmeans');
  const [highlightedClusterId, setHighlightedClusterId] = useState<number | null>(null);
  const [corrMode, setCorrMode] = useState<'pearson' | 'spearman'>('pearson');

  // Live scoring state (real POST /api/v1/score)
  const [customerRefIn, setCustomerRefIn] = useState<string>('WALKIN-01');
  const [genderIn, setGenderIn] = useState<'Female' | 'Male'>('Female');
  const [ageIn, setAgeIn] = useState<number>(32);
  const [incomeIn, setIncomeIn] = useState<number>(88);
  const [spendingIn, setSpendingIn] = useState<number>(82);
  const [liveScore, setLiveScore] = useState<ScoreResult | null>(null);
  const [scoringBusy, setScoringBusy] = useState<boolean>(false);
  const [scoreError, setScoreError] = useState<string | null>(null);

  // Customer ledger filters
  const [segmentFilter, setSegmentFilter] = useState<string>('All');
  const [boundaryOnly, setBoundaryOnly] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Batch scoring state
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

  const activeModelQuery = useQuery({
    queryKey: ['active-model', segments?.run_key],
    queryFn: () => apiClient.getActiveModel(),
    enabled: Boolean(segments),
  });

  const authStatusQuery = useQuery({
    queryKey: ['auth-status'],
    queryFn: () => apiClient.getAuthStatus(),
  });

  const loadWorkspaceData = useCallback(async () => {
    setLoading(true);
    setApiError(null);
    try {
      const [h, seg, cust, ev, dr] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getSegments(),
        apiClient.getCustomers(500, 0),
        apiClient.getScoringEvents(15, 0),
        apiClient.getDriftReports(10, 0),
      ]);
      setHealth(h);
      setSegments(seg);
      setCustomers(cust.items);
      setCustomerTotal(cust.total);
      setScoringEvents(ev.items);
      setDriftReports(dr.items);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : 'Unable to reach FastAPI backend.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadWorkspaceData();
  }, [loadWorkspaceData]);

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
        setLiveScore(result);
        const ev = await apiClient.getScoringEvents(15, 0);
        setScoringEvents(ev.items);
      } catch (err) {
        setScoreError(err instanceof Error ? err.message : 'Live scoring request failed');
      } finally {
        setScoringBusy(false);
      }
    },
    [genderIn, ageIn, incomeIn, spendingIn, customerRefIn]
  );

  // Trigger initial real API score once segments are loaded
  useEffect(() => {
    if (segments && !liveScore && !scoringBusy) {
      void submitLiveScore(genderIn, ageIn, incomeIn, spendingIn);
    }
  }, [segments, liveScore, scoringBusy, submitLiveScore, genderIn, ageIn, incomeIn, spendingIn]);

  const handlePipelineUpdated = useCallback(
    async (nextSegments: SegmentsPayload) => {
      setSegments(nextSegments);
      setSegmentFilter('All');
      setBoundaryOnly(false);
      setHoveredCustomer(null);
      const [h, cust] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getCustomers(500, 0),
      ]);
      setHealth(h);
      setCustomers(cust.items);
      setCustomerTotal(cust.total);
      setLiveScore(null);
      void activeModelQuery.refetch();
    },
    [activeModelQuery]
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
      const [cust, dr] = await Promise.all([
        apiClient.getCustomers(500, 0),
        apiClient.getDriftReports(10, 0),
      ]);
      setCustomers(cust.items);
      setCustomerTotal(cust.total);
      setDriftReports(dr.items);
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
      if (lines.length < 2) throw new Error('CSV must include a header row and at least one customer.');
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
      const [cust, dr] = await Promise.all([
        apiClient.getCustomers(500, 0),
        apiClient.getDriftReports(10, 0),
      ]);
      setCustomers(cust.items);
      setCustomerTotal(cust.total);
      setDriftReports(dr.items);
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

  const filteredCustomers = useMemo(() => {
    return customers.filter((c) => {
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
  }, [customers, segmentFilter, boundaryOnly, searchQuery]);

  const maxIncomeSlider = useMemo(() => {
    const peak = customers.reduce((acc, c) => Math.max(acc, c.annual_income_k), 140);
    return Math.min(500, Math.max(150, Math.ceil((peak * 1.15) / 10) * 10));
  }, [customers]);

  if (loading && !segments) {
    return (
      <div className="min-h-screen bg-[#F7F6F2] flex">
        <aside className="hidden lg:block w-60 border-r border-[#E4E2DC] bg-[#FFFFFF] p-6 space-y-4">
          <div className="h-6 bg-[#EFECE5] rounded animate-pulse w-36" />
          <div className="space-y-2 pt-6">
            {Array.from({ length: 5 }).map((_, i) => (
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

  if (apiError || !segments) {
    return (
      <div className="min-h-screen bg-[#F7F6F2] flex items-center justify-center p-6">
        <div className="max-w-lg w-full bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 text-[#BE123C] font-semibold">
            <AlertTriangle className="w-5 h-5" />
            <span>Inference Backend Connection Error</span>
          </div>
          <p className="text-sm text-[#5C5B56] leading-relaxed">
            {apiError || 'Unable to load segment metadata from FastAPI.'} Ensure the training
            pipeline has run and the FastAPI service is reachable.
          </p>
          <button
            type="button"
            onClick={() => void loadWorkspaceData()}
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
                <span>
                  {authStatusQuery.data?.has_users ? 'JWT Sign In' : 'Claim Admin JWT'}
                </span>
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Clean Single-Row Top Header */}
        <header className="h-14 bg-[#FFFFFF] border-b border-[#E4E2DC] px-6 flex items-center justify-between gap-4 sticky top-0 z-20">
          <div className="flex items-center gap-2 text-xs text-[#5C5B56] whitespace-nowrap">
            <span className="font-semibold text-[#141413] lg:hidden">Mall Segmentation</span>
            <span className="lg:hidden" aria-hidden="true">
              /
            </span>
            <span className="hidden lg:inline text-[#8A8880]">Workspace</span>
            <span className="hidden lg:inline text-[#8A8880]" aria-hidden="true">
              /
            </span>
            <span className="font-medium text-[#141413]">{activeNav.label}</span>
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
                  {authStatusQuery.data?.has_users ? 'Sign In with Role JWT' : 'Initialize Admin Account'}
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
            {/* 00. DATA & PIPELINE */}
          {section === 'data' && (
            <DataWorkspace
              segments={segments}
              activeCustomers={customers}
              onPipelineUpdated={handlePipelineUpdated}
              onNavigateBrief={() => setSection('brief')}
              onNavigateGeometry={() => setSection('geometry')}
            />
          )}

          {/* 01. EXECUTIVE BRIEF */}
          {section === 'brief' && (
            <div className="space-y-6">
              {/* Interactive Behavioral Map + EDA Distributions */}
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
                        onClick={() => setHighlightedClusterId(null)}
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
                    onHover={setHoveredCustomer}
                  />

                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {segments.personas.map((p, idx) => {
                      const active = highlightedClusterId === p.cluster_id;
                      const color = SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length];
                      return (
                        <button
                          key={p.cluster_id}
                          type="button"
                          onClick={() =>
                            setHighlightedClusterId((prev) =>
                              prev === p.cluster_id ? null : p.cluster_id
                            )
                          }
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
                </section>
              </div>

              {/* Proportional Allocation Bar */}
              <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-[#141413]">
                    Cohort Share Allocation
                  </h2>
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
                        style={{
                          backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length],
                        }}
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
                    Actionable Retail Personas & Campaign Playbooks
                  </h2>
                  <span className="text-xs font-mono-tabular text-[#64635E]">
                    {segments.personas.length} behavioral segments
                  </span>
                </div>
                <div className="divide-y divide-[#E4E2DC]">
                  {segments.personas.map((p, idx) => (
                    <div
                      key={p.cluster_id}
                      className="p-5 grid grid-cols-1 lg:grid-cols-12 gap-4 items-start hover:bg-[#FAF9F6] transition-colors"
                    >
                      <div className="lg:col-span-4 space-y-1">
                        <div className="flex items-center gap-2 text-xs text-[#64635E] font-mono-tabular">
                          <span
                            className="w-2.5 h-2.5 rounded-xs inline-block"
                            style={{
                              backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length],
                            }}
                          />
                          <span>Cluster {p.cluster_id}</span>
                          <span aria-hidden="true">·</span>
                          <span>{p.customer_count} customers</span>
                        </div>
                        <h3 className="text-base font-semibold text-[#141413]">
                          {p.persona_name}
                        </h3>
                        <p className="text-xs text-[#5C5B56]">{p.tagline}</p>
                      </div>

                      <div className="lg:col-span-4 grid grid-cols-4 gap-3 text-xs">
                        <div>
                          <span className="text-[#8A8880] block">Mean Age</span>
                          <span className="font-mono-tabular font-medium text-[#141413]">
                            {p.mean_age.toFixed(1)}
                          </span>
                        </div>
                        <div>
                          <span className="text-[#8A8880] block">Income</span>
                          <span className="font-mono-tabular font-medium text-[#141413]">
                            ${p.mean_income.toFixed(1)}k
                          </span>
                        </div>
                        <div>
                          <span className="text-[#8A8880] block">Spend</span>
                          <span className="font-mono-tabular font-medium text-[#141413]">
                            {p.mean_spending.toFixed(1)}
                          </span>
                        </div>
                        <div>
                          <span className="text-[#8A8880] block">Female</span>
                          <span className="font-mono-tabular font-medium text-[#141413]">
                            {(p.female_share * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>

                      <div className="lg:col-span-4 space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-xs font-medium text-[#1E40AF]">
                            {p.target_channel}
                          </span>
                          <button
                            type="button"
                            onClick={() => {
                              setSegmentFilter(p.persona_name);
                              setSection('drift');
                            }}
                            className="text-xs text-[#5C5B56] hover:text-[#141413] inline-flex items-center gap-0.5 cursor-pointer"
                          >
                            <span>View Cohort</span>
                            <ArrowUpRight className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <p className="text-xs text-[#5C5B56] leading-relaxed">
                          {p.recommended_action}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}

          {/* 02. CLUSTER GEOMETRY */}
          {section === 'geometry' && (
            <div className="space-y-6">
              {/* Multi-Algorithm Visual Switcher */}
              <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <h2 className="text-sm font-semibold text-[#141413]">
                    Algorithm Comparison Overlay
                  </h2>
                  <p className="text-xs text-[#64635E]">
                    Compare K-Means++, Ward Hierarchical, DBSCAN density noise (-1 diamonds), and GMM contours
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  {(
                    [
                      { id: 'kmeans', label: 'K-Means++' },
                      { id: 'ward', label: 'Ward Hierarchical' },
                      { id: 'dbscan', label: `DBSCAN (eps ${segments.dbscan_eps?.toFixed(2) ?? 'knee'})` },
                      { id: 'gmm', label: 'GMM Full Cov' },
                    ] as const
                  ).map((alg) => (
                    <button
                      key={alg.id}
                      type="button"
                      onClick={() => setAlgorithmView(alg.id)}
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

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-sm font-semibold text-[#141413]">
                        Income vs. Spending ({algorithmView.toUpperCase()} Overlay)
                      </h2>
                      <p className="text-xs text-[#64635E]">
                        {algorithmView === 'dbscan'
                          ? 'Crimson hollow diamonds mark DBSCAN density noise (-1) outliers'
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
                    onHover={setHoveredCustomer}
                  />

                  <div className="min-h-[20px] text-xs font-mono-tabular text-[#5C5B56]">
                    {hoveredCustomer ? (
                      <span>
                        {hoveredCustomer.customer_ref || `ID ${hoveredCustomer.id}`} ·{' '}
                        {hoveredCustomer.persona_name} · Age {hoveredCustomer.age} · $
                        {hoveredCustomer.annual_income_k}k · Spend{' '}
                        {hoveredCustomer.spending_score} · Entropy{' '}
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
                    onHover={setHoveredCustomer}
                  />

                  <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-[#5C5B56]">
                    {segments.personas.map((p, idx) => {
                      const active = highlightedClusterId === p.cluster_id;
                      return (
                        <button
                          key={p.cluster_id}
                          type="button"
                          onClick={() =>
                            setHighlightedClusterId((prev) =>
                              prev === p.cluster_id ? null : p.cluster_id
                            )
                          }
                          className={`inline-flex items-center gap-1.5 cursor-pointer ${
                            active ? 'font-semibold text-[#141413] underline' : 'hover:text-[#141413]'
                          }`}
                        >
                          <span
                            className="w-2.5 h-2.5 rounded-xs inline-block"
                            style={{
                              backgroundColor: SEGMENT_PALETTE[idx % SEGMENT_PALETTE.length],
                            }}
                          />
                          <span>{p.persona_name}</span>
                        </button>
                      );
                    })}
                  </div>
                </section>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <section className="lg:col-span-6 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-[#141413]">
                      Ward Hierarchical Clustering Dendrogram
                    </h2>
                    <span className="text-xs font-mono-tabular text-[#BE123C]">
                      k={segments.target_k} cut distance{' '}
                      {segments.ward_dendrogram?.cut_height.toFixed(2)}
                    </span>
                  </div>
                  {segments.ward_dendrogram && (
                    <WardDendrogramChart
                      dendro={segments.ward_dendrogram}
                      targetK={segments.target_k}
                    />
                  )}
                </section>

                <section className="lg:col-span-6 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-[#141413]">
                      Model Selection Sweeps (k = 2..10)
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
          )}

          {/* 03. BENCHMARKS & SURROGATE RULES */}
          {section === 'benchmarks' && (
            <div className="space-y-6">
              <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
                <div className="px-5 py-4 border-b border-[#E4E2DC] flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-[#141413]">
                    Multi-Algorithm Clustering Benchmark
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
                    {Object.entries(segments.stability.per_cluster_jaccard).map(
                      ([cid, jaccard]) => {
                        const persona = segments.personas.find((p) => String(p.cluster_id) === cid);
                        return (
                          <div key={cid} className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="text-[#141413] font-medium">
                                Cluster {cid} · {persona?.persona_name}
                              </span>
                              <span className="font-mono-tabular text-[#5C5B56]">
                                {jaccard.toFixed(4)}
                              </span>
                            </div>
                            <div className="h-2 bg-[#F1EFE9] rounded-full overflow-hidden">
                              <div
                                className="h-full bg-[#1E40AF] rounded-full"
                                style={{ width: `${Math.min(100, jaccard * 100)}%` }}
                              />
                            </div>
                          </div>
                        );
                      }
                    )}
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
                          onClick={() => setCorrMode(m)}
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
                            <td className="py-2.5 px-2 text-right font-mono-tabular">
                              {s.std.toFixed(2)}
                            </td>
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
                        corrMode === 'pearson'
                          ? segments.eda.pearson_matrix
                          : segments.eda.spearman_matrix
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
                      <span className="font-mono-tabular text-[#141413]">
                        {rule.condition_text}
                      </span>
                      <span className="font-mono-tabular text-[#5C5B56] whitespace-nowrap shrink-0">
                        Support {rule.support_count} · Purity {(rule.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}

          {/* 04. LIVE DECISIONING WORKBENCH (WIRED TO FASTAPI POST /api/v1/score) */}
          {section === 'decisioning' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <section className="lg:col-span-5 bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-6 space-y-5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <SlidersHorizontal className="w-4 h-4 text-[#1E40AF]" />
                      <h2 className="text-sm font-semibold text-[#141413]">
                        Customer Profile Parameters
                      </h2>
                    </div>
                    <span className="text-xs font-mono-tabular text-[#8A8880]">
                      POST /api/v1/score
                    </span>
                  </div>

                  {/* Quick archetype presets */}
                  <div className="flex flex-wrap gap-1.5">
                    {SCENARIO_PRESETS.map((preset) => (
                      <button
                        key={preset.label}
                        type="button"
                        onClick={() => {
                          setGenderIn(preset.gender);
                          setAgeIn(preset.age);
                          setIncomeIn(preset.income);
                          setSpendingIn(preset.spending);
                          void submitLiveScore(
                            preset.gender,
                            preset.age,
                            preset.income,
                            preset.spending
                          );
                        }}
                        className="px-2.5 py-1 text-xs bg-[#F7F6F2] hover:bg-[#EAE8E1] border border-[#E4E2DC] rounded-md text-[#141413] cursor-pointer transition-colors"
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs text-[#5C5B56] mb-1">
                        Customer Reference
                      </label>
                      <input
                        type="text"
                        maxLength={64}
                        value={customerRefIn}
                        onChange={(e) => setCustomerRefIn(e.target.value)}
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
                            onClick={() => setGenderIn(g)}
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
                          onChange={(e) => setAgeIn(Math.min(100, Math.max(16, Number(e.target.value) || 16)))}
                          className="w-20 px-2 py-0.5 text-right text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded"
                        />
                      </div>
                      <input
                        type="range"
                        min={16}
                        max={85}
                        value={Math.min(85, ageIn)}
                        onChange={(e) => setAgeIn(Number(e.target.value))}
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
                          onChange={(e) => setIncomeIn(Math.min(500, Math.max(5, Number(e.target.value) || 5)))}
                          className="w-20 px-2 py-0.5 text-right text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded"
                        />
                      </div>
                      <input
                        type="range"
                        min={10}
                        max={maxIncomeSlider}
                        value={Math.min(incomeIn, maxIncomeSlider)}
                        onChange={(e) => setIncomeIn(Number(e.target.value))}
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
                          onChange={(e) => setSpendingIn(Math.min(100, Math.max(1, Number(e.target.value) || 1)))}
                          className="w-20 px-2 py-0.5 text-right text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded"
                        />
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={100}
                        value={spendingIn}
                        onChange={(e) => setSpendingIn(Number(e.target.value))}
                        className="w-full accent-[#1E40AF]"
                      />
                    </div>

                    <button
                      type="button"
                      disabled={scoringBusy}
                      onClick={() => void submitLiveScore()}
                      className="w-full py-2.5 px-4 text-xs font-semibold bg-[#1E40AF] text-white rounded-lg hover:bg-[#1E3A8A] disabled:opacity-60 transition-colors cursor-pointer"
                    >
                      {scoringBusy ? 'Scoring via FastAPI…' : 'Score & Persist Event'}
                    </button>

                    {scoreError && (
                      <p className="text-xs text-[#BE123C]">{scoreError}</p>
                    )}
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
                          <h2 className="text-2xl font-bold text-[#141413]">
                            {liveScore.persona_name}
                          </h2>
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
                                      backgroundColor:
                                        colorMap.get(personaName) ?? '#1D4ED8',
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
                    <span className="text-xs font-mono-tabular text-[#64635E]">
                      scoring_events table
                    </span>
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
                              No live scoring events recorded yet.
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
          )}

          {/* 05. BATCH DRIFT AUDIT & CUSTOMER LEDGER */}
          {section === 'drift' && (
            <div className="space-y-6">
              <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-5">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <h2 className="text-sm font-semibold text-[#141413]">
                      Batch Scoring & Population Stability Index (PSI) Audit
                    </h2>
                    <p className="text-xs text-[#64635E]">
                      Executes POST /api/v1/batch-score, computes per-feature PSI against training
                      quantiles, and persists drift reports
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      disabled={batchBusy}
                      onClick={() => void runScenarioBatch('nominal')}
                      className="px-3 py-1.5 text-xs font-medium bg-[#F7F6F2] hover:bg-[#EAE8E1] border border-[#E4E2DC] rounded-lg cursor-pointer"
                    >
                      Run Nominal Batch
                    </button>
                    <button
                      type="button"
                      disabled={batchBusy}
                      onClick={() => void runScenarioBatch('youth_wave')}
                      className="px-3 py-1.5 text-xs font-medium bg-[#F7F6F2] hover:bg-[#EAE8E1] border border-[#E4E2DC] rounded-lg cursor-pointer"
                    >
                      Test Youth Shift
                    </button>
                    <button
                      type="button"
                      disabled={batchBusy}
                      onClick={() => void runScenarioBatch('luxury_drift')}
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
                        onChange={(e) => void handleCsvUpload(e)}
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

                {activeModelQuery.data && (
                  <div className="p-4 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                    <div>
                      <span className="text-[#64635E] block">Active Model Version</span>
                      <span className="font-mono-tabular font-semibold text-[#141413]">
                        {activeModelQuery.data.version}
                      </span>
                    </div>
                    <div>
                      <span className="text-[#64635E] block">Feature Schema Hash</span>
                      <span className="font-mono-tabular font-semibold text-[#141413]">
                        {activeModelQuery.data.feature_schema_hash}
                      </span>
                    </div>
                    <div>
                      <span className="text-[#64635E] block">Registered Algorithm</span>
                      <span className="font-medium text-[#141413]">
                        {activeModelQuery.data.algorithm}
                      </span>
                    </div>
                    <div>
                      <span className="text-[#64635E] block">Artifact Path</span>
                      <span className="font-mono-tabular text-[#5C5B56] truncate block">
                        {activeModelQuery.data.artifact_path}
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
                              <td className="py-2.5 px-4 text-right font-mono-tabular">
                                {rep.batch_size}
                              </td>
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
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search ref or persona"
                        className="pl-8 pr-3 py-1.5 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[#141413] placeholder:text-[#8A8880] focus:outline-none focus:border-[#1E40AF]"
                      />
                    </div>

                    <div className="flex items-center gap-1.5">
                      <Filter className="w-3.5 h-3.5 text-[#8A8880]" />
                      <select
                        value={segmentFilter}
                        onChange={(e) => setSegmentFilter(e.target.value)}
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
                      onClick={() => setBoundaryOnly((prev) => !prev)}
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
                          <td className="py-2.5 px-3 font-mono-tabular text-[#64635E]">
                            {row.source}
                          </td>
                          <td className="py-2.5 px-3 text-[#5C5B56]">{row.gender}</td>
                          <td className="py-2.5 px-3 text-right font-mono-tabular">{row.age}</td>
                          <td className="py-2.5 px-3 text-right font-mono-tabular">
                            ${row.annual_income_k.toFixed(0)}k
                          </td>
                          <td className="py-2.5 px-3 text-right font-mono-tabular">
                            {row.spending_score.toFixed(0)}
                          </td>
                          <td className="py-2.5 px-3 font-medium text-[#141413]">
                            {row.persona_name}
                          </td>
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
          )}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
