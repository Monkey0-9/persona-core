import type {
  ActiveModelMetadata,
  AuthSession,
  BatchCustomersResponse,
  CounterfactualResult,
  CustomerRecord,
  DriftReportItem,
  HealthStatus,
  ModelComparisonResult,
  ModelVersionEntry,
  MonitoringDriftResponse,
  ObservabilitySummary,
  RollingDriftEvalResult,
  ScoreResult,
  ScoringEventRecord,
  SegmentsPayload,
} from '../types';

// In-memory JWT session store (never persisted in localStorage or sessionStorage)
let currentSession: AuthSession | null = null;
const sessionListeners = new Set<(session: AuthSession | null) => void>();

export function getAuthSession(): AuthSession | null {
  return currentSession;
}

export function setAuthSession(next: AuthSession | null): void {
  currentSession = next;
  sessionListeners.forEach((cb) => cb(next));
}

export function subscribeAuthSession(cb: (session: AuthSession | null) => void): () => void {
  sessionListeners.add(cb);
  return () => sessionListeners.delete(cb);
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function assertHealthStatus(raw: unknown): HealthStatus {
  if (
    !isObject(raw) ||
    typeof raw.database_ok !== 'boolean' ||
    typeof raw.model_loaded !== 'boolean' ||
    (raw.status !== 'ok' && raw.status !== 'degraded')
  ) {
    throw new Error('Malformed /api/health payload returned by backend.');
  }
  return raw as unknown as HealthStatus;
}

function assertSegmentsPayload(raw: unknown): SegmentsPayload {
  if (
    !isObject(raw) ||
    typeof raw.run_key !== 'string' ||
    !Array.isArray(raw.personas) ||
    !Array.isArray(raw.benchmarks) ||
    !isObject(raw.eda) ||
    !isObject(raw.features)
  ) {
    throw new Error('Malformed /api/v1/segments payload returned by backend.');
  }
  return raw as unknown as SegmentsPayload;
}

function assertScoreResult(raw: unknown): ScoreResult {
  if (
    !isObject(raw) ||
    typeof raw.persona_name !== 'string' ||
    typeof raw.normalized_entropy !== 'number' ||
    typeof raw.gmm_confidence !== 'number'
  ) {
    throw new Error('Malformed customer score payload returned by backend.');
  }
  return raw as unknown as ScoreResult;
}

function assertActiveModel(raw: unknown): ActiveModelMetadata {
  if (
    !isObject(raw) ||
    typeof raw.version !== 'string' ||
    typeof raw.feature_schema_hash !== 'string'
  ) {
    throw new Error('Malformed /api/v1/models/active payload returned by backend.');
  }
  return raw as unknown as ActiveModelMetadata;
}

function assertMonitoringDrift(raw: unknown): MonitoringDriftResponse {
  if (!isObject(raw) || typeof raw.latest_status !== 'string' || !Array.isArray(raw.recent_reports)) {
    throw new Error('Malformed /api/v1/monitoring/drift payload returned by backend.');
  }
  return raw as unknown as MonitoringDriftResponse;
}

async function parseErrorDetail(res: Response): Promise<string> {
  let message = `HTTP ${res.status}`;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') {
      message = body.detail;
    } else if (Array.isArray(body.detail) && body.detail.length > 0) {
      message = JSON.stringify(body.detail[0]);
    }
  } catch {
    // ignore unparseable error body
  }
  return message;
}

async function tryRefreshSession(): Promise<boolean> {
  if (!currentSession?.refresh_token) return false;
  try {
    const res = await fetch('/api/v1/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: currentSession.refresh_token }),
    });
    if (!res.ok) {
      setAuthSession(null);
      return false;
    }
    const next = (await res.json()) as AuthSession;
    setAuthSession(next);
    return true;
  } catch {
    setAuthSession(null);
    return false;
  }
}

export async function apiFetch<T>(
  url: string,
  init: RequestInit = {},
  validator?: (raw: unknown) => T
): Promise<T> {
  const buildHeaders = (): Headers => {
    const h = new Headers(init.headers);
    if (currentSession?.access_token && !h.has('Authorization')) {
      h.set('Authorization', `Bearer ${currentSession.access_token}`);
    }
    return h;
  };

  let res = await fetch(url, { ...init, headers: buildHeaders() });
  if (res.status === 401 && currentSession?.refresh_token) {
    const refreshed = await tryRefreshSession();
    if (refreshed) {
      res = await fetch(url, { ...init, headers: buildHeaders() });
    }
  }

  if (!res.ok) {
    if (res.status === 401 && currentSession) {
      setAuthSession(null);
    }
    throw new Error(await parseErrorDetail(res));
  }

  const raw: unknown = await res.json();
  return validator ? validator(raw) : (raw as T);
}

export const apiClient = {
  getHealth: () => apiFetch('/api/health', {}, assertHealthStatus),
  getSegments: () => apiFetch('/api/v1/segments', {}, assertSegmentsPayload),
  getActiveModel: () => apiFetch('/api/v1/models/active', {}, assertActiveModel),
  getMonitoringDrift: () => apiFetch('/api/v1/monitoring/drift', {}, assertMonitoringDrift),
  getCustomers: (limit = 500, offset = 0) =>
    apiFetch<{ total: number; items: CustomerRecord[] }>(
      `/api/v1/customers?limit=${limit}&offset=${offset}`
    ),
  getScoringEvents: (limit = 15, offset = 0) =>
    apiFetch<{ total: number; items: ScoringEventRecord[] }>(
      `/api/v1/scoring-events?limit=${limit}&offset=${offset}`
    ),
  getDriftReports: (limit = 10, offset = 0) =>
    apiFetch<{ total: number; items: DriftReportItem[] }>(
      `/api/v1/drift-reports?limit=${limit}&offset=${offset}`
    ),
  scoreCustomer: (payload: {
    customer_ref?: string;
    gender: 'Female' | 'Male';
    age: number;
    annual_income_k: number;
    spending_score: number;
  }) =>
    apiFetch(
      '/api/v1/customers/score',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      assertScoreResult
    ),
  getCounterfactual: (payload: {
    customer_ref?: string;
    gender: 'Female' | 'Male';
    age: number;
    annual_income_k: number;
    spending_score: number;
    target_persona_key?: string;
  }) =>
    apiFetch<CounterfactualResult>('/api/v1/customers/counterfactual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  batchScore: (customers: {
    customer_ref?: string;
    gender: 'Female' | 'Male';
    age: number;
    annual_income_k: number;
    spending_score: number;
  }[]) =>
    apiFetch<BatchCustomersResponse>('/api/v1/customers/batch-score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customers }),
    }),
  uploadBatchFile: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return apiFetch<BatchCustomersResponse>('/api/v1/customers/batch-score/upload', {
      method: 'POST',
      body: form,
    });
  },
  evaluateRollingDrift: (windowSize = 100, lookbackDays = 7) =>
    apiFetch<RollingDriftEvalResult>('/api/v1/monitoring/drift/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ window_size: windowSize, lookback_days: lookbackDays }),
    }),
  listModelVersions: () =>
    apiFetch<{ total: number; items: ModelVersionEntry[] }>('/api/v1/models'),
  registerCandidateModel: (payload: {
    version: string;
    target_k: number;
    include_gender: boolean;
    notes?: string;
  }) =>
    apiFetch<{ version: string; run_key: string; comparison: ModelComparisonResult }>(
      '/api/v1/models/candidates',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    ),
  compareModelVersion: (version: string) =>
    apiFetch<ModelComparisonResult>(`/api/v1/models/${encodeURIComponent(version)}/compare`),
  promoteModelVersion: (version: string) =>
    apiFetch<ModelVersionEntry>(`/api/v1/models/${encodeURIComponent(version)}/promote`, {
      method: 'POST',
    }),
  rollbackModelVersion: (targetVersion?: string) =>
    apiFetch<ModelVersionEntry>('/api/v1/models/rollback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(targetVersion ? { target_version: targetVersion } : {}),
    }),
  getObservabilitySummary: () =>
    apiFetch<ObservabilitySummary>('/api/v1/observability/summary'),
  getAuthStatus: () => apiFetch<{ has_users: boolean }>('/api/v1/auth/status'),
  bootstrapAdmin: async (email: string, password: string): Promise<AuthSession> => {
    const sess = await apiFetch<AuthSession>('/api/v1/auth/bootstrap-admin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    setAuthSession(sess);
    return sess;
  },
  login: async (email: string, password: string): Promise<AuthSession> => {
    const sess = await apiFetch<AuthSession>('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    setAuthSession(sess);
    return sess;
  },
  logout: async (): Promise<void> => {
    const rt = currentSession?.refresh_token;
    setAuthSession(null);
    if (rt) {
      await fetch('/api/v1/auth/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: rt }),
      }).catch(() => undefined);
    }
  },
};
