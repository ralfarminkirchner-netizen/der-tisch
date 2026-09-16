export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'interrupted';
export type MarkerKind = 'uebereinstimmung' | 'widerspruch' | 'einzigartig';

export interface ModelInfo {
  id: string;
  label: string;
  provider: string;
  model: string;
}

export interface AppConfig {
  models: ModelInfo[];
  max_prompt_chars: number;
  fake_providers_enabled: boolean;
  env: string;
}

export interface Session {
  id: string;
  title: string;
  status: 'offen' | 'abgeschlossen';
  created_at: number;
  closed_at: number | null;
  closing_note: string | null;
}

export interface Spark {
  id: string;
  session_id: string;
  seq: number;
  prompt: string;
  client_request_id: string;
  created_at: number;
}

export interface Job {
  id: string;
  session_id: string;
  spark_id: string;
  model_id: string;
  label: string;
  provider: string;
  model: string;
  status: JobStatus;
  text: string;
  error: string | null;
  partial: boolean;
  latency_ms: number | null;
}

export interface Marker {
  id: string;
  session_id: string;
  spark_id: string;
  job_id: string;
  related_job_id: string | null;
  kind: MarkerKind;
  start_offset: number;
  end_offset: number;
  quote: string;
  note: string;
}

export interface SummaryModelRow {
  job_id: string;
  model_id: string;
  label: string;
  provider: string;
  status: JobStatus;
  chars: number;
  sentences: number;
  latency_ms: number | null;
  partial: boolean;
  error: string | null;
  unique: number;
  agreements: number;
  contradictions: number;
}

export interface SummaryPair {
  a_job_id: string;
  b_job_id: string;
  a_model_id: string;
  b_model_id: string;
  a_label: string;
  b_label: string;
  agreements: number;
  contradictions: number;
  topics: string[];
}

export interface Summary {
  spark_id: string;
  computed_at: number;
  models: SummaryModelRow[];
  pairs: SummaryPair[];
  counts: Record<MarkerKind, number>;
  analysed_jobs: string[];
  method: string;
}

export interface SparkEntry {
  spark: Spark;
  jobs: Job[];
  markers: Marker[];
  summary: Summary | null;
}

export interface SessionBundle {
  session: Session;
  sparks: SparkEntry[];
  last_event_id: number;
  exported_at: number;
}

export interface HealthInfo {
  status: string;
  env: string;
  database: string;
  providers: Record<string, { ready: boolean; reason: string }>;
  fake_providers_enabled: boolean;
  missing_credentials: string[];
}
