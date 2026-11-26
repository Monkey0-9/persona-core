export interface NumericSummary {
  feature: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  iqr_outlier_count: number;
}

export interface CorrelationPair {
  feature_a: string;
  feature_b: string;
  pearson: number;
  spearman: number;
}

export interface BenchmarkMetric {
  algorithm: string;
  n_clusters: number;
  silhouette: number | null;
  calinski_harabasz: number | null;
  davies_bouldin: number | null;
  noise_count: number;
  extra: Record<string, number | string>;
}

export interface SweepPoint {
  k: number;
  inertia?: number;
  silhouette?: number;
  bic?: number;
  aic?: number;
}

export interface PersonaProfile {
  cluster_id: number;
  persona_key: string;
  persona_name: string;
  tagline: string;
  recommended_action: string;
  target_channel: string;
  customer_count: number;
  share: number;
  mean_age: number;
  mean_income: number;
  mean_spending: number;
  female_share: number;
  male_share: number;
}

export interface SurrogateRule {
  rule_id: number;
  cluster_id: number;
  persona_name: string;
  support_count: number;
  confidence: number;
  condition_text: string;
}

export interface GmmEllipse {
  cluster_id: number;
  persona_name: string;
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle_deg: number;
}

export interface WardDendrogram {
  icoord: number[][];
  dcoord: number[][];
  ivl: string[];
  cut_height: number;
}

export interface AlgorithmPoint {
  customer_ref: string | null;
  kmeans_cluster_id: number;
  ward_cluster_id: number;
  gmm_cluster_id: number;
  dbscan_cluster_id: number;
}

export interface SegmentsPayload {
  run_key: string;
  dataset_source: string;
  row_count: number;
  target_k: number;
  include_gender: boolean;
  silhouette_score: number;
  mean_jaccard_stability: number;
  passed_stability_gate: boolean;
  surrogate_fidelity: number;
  created_at: string;
  entropy_threshold: number;
  boundary_customer_count: number;
  dbscan_eps?: number;
  eda: {
    row_count: number;
    numeric_summaries: NumericSummary[];
    gender_breakdown: {
      counts: Record<string, number>;
      shares: Record<string, number>;
    };
    pearson_matrix: Record<string, Record<string, number>>;
    spearman_matrix: Record<string, Record<string, number>>;
    correlation_highlights: CorrelationPair[];
  };
  features: {
    feature_names: string[];
    include_gender: boolean;
    explained_variance_ratio: number[];
    cumulative_variance_ratio: number;
    pca_loadings: Record<string, number[]>;
  };
  benchmarks: BenchmarkMetric[];
  kmeans_sweep: SweepPoint[];
  gmm_sweep: SweepPoint[];
  stability: {
    n_bootstrap: number;
    n_clusters: number;
    per_cluster_jaccard: Record<string, number>;
    mean_jaccard: number;
    min_jaccard: number;
    min_jaccard_threshold: number;
    passed_gate: boolean;
  };
  personas: PersonaProfile[];
  explainability: {
    fidelity_score: number;
    max_depth: number;
    feature_importances: Record<string, number>;
    rules: SurrogateRule[];
    tree_text: string;
  };
  gmm_ellipses?: GmmEllipse[];
  ward_dendrogram?: WardDendrogram;
  algorithm_points?: AlgorithmPoint[];
}

export interface CustomerRecord {
  id: number;
  customer_ref: string | null;
  gender: 'Male' | 'Female';
  age: number;
  annual_income_k: number;
  spending_score: number;
  cluster_id: number;
  persona_name: string;
  recommended_action: string;
  gmm_confidence: number;
  normalized_entropy: number;
  is_boundary_customer: boolean;
  pca_1: number;
  pca_2: number;
  source: string;
  created_at: string;
}

export interface ScoreResult {
  customer_ref: string | null;
  gender: 'Male' | 'Female';
  age: number;
  annual_income_k: number;
  spending_score: number;
  cluster_id: number;
  persona_key: string;
  persona_name: string;
  recommended_action: string;
  gmm_confidence: number;
  normalized_entropy: number;
  is_boundary_customer: boolean;
  distance_to_centroid: number;
  drift_flag: boolean;
  drifted_features: string[];
  pca_1: number;
  pca_2: number;
  posterior_probabilities: Record<string, number>;
}

export interface ScoringEventRecord {
  id: number;
  gender: string;
  age: number;
  annual_income_k: number;
  spending_score: number;
  cluster_id: number;
  persona_name: string;
  recommended_action: string;
  gmm_confidence: number;
  normalized_entropy: number;
  is_boundary_customer: boolean;
  distance_to_centroid?: number;
  drift_flag?: boolean;
  created_at: string;
}

export interface FeatureDrift {
  feature: string;
  psi: number;
  ks_statistic?: number;
  ks_pvalue?: number;
  severity: 'stable' | 'monitor' | 'drift';
  baseline_mean: number;
  current_mean: number;
  baseline_std: number;
  current_std: number;
}

export interface DriftReportItem {
  id: number;
  run_id: number | null;
  batch_size: number;
  overall_status: 'stable' | 'monitor' | 'drift';
  alert_status?: 'NORMAL' | 'WARNING' | 'CRITICAL';
  max_psi: number;
  max_ks_statistic?: number;
  window_label?: string;
  alert_dispatched?: boolean;
  features: FeatureDrift[];
  created_at: string;
}

export interface ActiveModelMetadata {
  version: string;
  algorithm: string;
  feature_schema_hash: string;
  artifact_path: string;
  metrics: Record<string, number | string>;
  reference_stats: Record<string, Record<string, number>>;
  created_at: string;
}

export interface ModelVersionEntry {
  id: number;
  version: string;
  algorithm: string;
  feature_schema_hash: string;
  artifact_path: string;
  is_active: boolean;
  status: string;
  parent_version: string | null;
  notes: string | null;
  metrics: Record<string, number | string | boolean>;
  promoted_at: string | null;
  created_at: string;
}

export interface ModelComparisonResult {
  candidate: ModelVersionEntry;
  baseline: ModelVersionEntry | null;
  deltas: Record<string, number | null>;
  centroid_shift_l2: number | null;
}

export interface BatchRowError {
  row_index: number;
  customer_ref: string | null;
  errors: string[];
}

export interface BatchCustomersResponse {
  batch_id: string;
  count: number;
  accepted_count: number;
  rejected_count: number;
  boundary_count: number;
  row_errors: BatchRowError[];
  drift_report: {
    overall_status: string;
    alert_status?: 'NORMAL' | 'WARNING' | 'CRITICAL';
    max_psi: number;
    max_ks_statistic?: number;
  };
}

export interface RollingDriftEvalResult {
  evaluated: boolean;
  reason?: string;
  report_id?: number;
  window_label: string;
  alert_status: 'NORMAL' | 'WARNING' | 'CRITICAL';
  sample_size: number;
  max_psi?: number;
  max_ks_statistic?: number;
  alert_dispatched?: boolean;
}

export interface ObservabilitySummary {
  total_requests: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  scoring_by_persona: Record<string, number>;
  drift_alerts: Record<string, number>;
  db_up: boolean;
  redis_up: boolean | null;
  model_loaded: boolean;
  active_model_version: string | null;
}

export interface MonitoringDriftResponse {
  total_reports: number;
  latest_status: 'stable' | 'monitor' | 'drift';
  latest_alert_status?: 'NORMAL' | 'WARNING' | 'CRITICAL';
  latest_max_psi: number;
  latest_max_ks?: number;
  reference_stats: Record<string, Record<string, number>>;
  recent_reports: Omit<DriftReportItem, 'run_id'>[];
}

export interface AuthUser {
  id: number;
  email: string;
  role: 'admin' | 'analyst' | 'viewer' | string;
}

export interface AuthSession {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
  user: AuthUser;
}

export interface HealthStatus {
  status: 'ok' | 'degraded';
  database_ok: boolean;
  model_loaded: boolean;
  active_run_key: string | null;
}

export interface ManualCustomerInput {
  customer_id: number;
  gender: 'Male' | 'Female';
  age: number;
  annual_income_k: number;
  spending_score: number;
}

export const SEGMENT_PALETTE = [
  '#1D4ED8', // Cobalt
  '#047857', // Deep Emerald
  '#B45309', // Copper Amber
  '#BE123C', // Crimson Rose
  '#6D28D9', // Royal Violet
  '#0E7490', // Deep Cyan
  '#4D7C0F', // Olive Green
  '#C2410C', // Terracotta
  '#9D174D', // Berry
  '#334155', // Slate Ink
];
