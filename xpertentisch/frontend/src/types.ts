export type JobStatus =
  | 'not_requested'
  | 'queued'
  | 'running'
  | 'streaming'
  | 'done'
  | 'error'
  | 'interrupted'
  | 'cancelled';

/** Was für eine Art Eingabe — bestimmt die gesetzte Beziehung. */
export type SparkKind =
  | 'funke'
  | 'antwort'
  | 'weitergabe'
  | 'gegenposition'
  | 'vertiefung'
  | 'pingpong'
  | 'szenario'
  | 'kuratierung';

export interface PingPongRun {
  id: string;
  session_id: string;
  status: 'laeuft' | 'gestoppt' | 'beendet';
  turn: number;
  max_turns: number;
  participants: string[];
  labels: string[];
  prompt: string;
  refs: string[];
  stopped_reason: string;
}

/** Ein Gedanke, der noch nicht beim Server angekommen ist. */
export interface Entwurf {
  clientRequestId: string;
  prompt: string;
  refs: string[];
  kind: SparkKind;
  modelIds: string[] | null;
  curate: boolean;
  createdAt: number;
  lastError: string;
}

export interface Relation {
  id: string;
  session_id: string;
  from_id: string;
  to_id: string;
  type: string;
  origin: 'mensch' | 'maschine';
  status: 'vorschlag' | 'bestaetigt' | 'abgelehnt';
  note: string;
  created_at: number;
}

export interface ContextEntry {
  id: string;
  label: string;
  role: 'mensch' | 'modell';
  reason: string;
  shortened: boolean;
  chars: number;
}

export interface JobContext {
  job: { id: string; label: string; provider: string; model: string; status: JobStatus };
  entries: ContextEntry[];
  rendered: string;
  rule: string;
  truncated: boolean;
  created_at: number;
}

/** Ein gewählter Bezug in der Eingabe. */
export interface Bezug {
  id: string;
  label: string;
  kind: SparkKind;
  /** Nur dieses Modell ansprechen; leer heißt: alle am Tisch. */
  modelId?: string;
  hint: string;
}
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
  /** Sagt nur, ob ein Zugangswort eingerichtet ist — nie welches. */
  settings_available: boolean;
  /** Wer die Kuratierung übernimmt — null heißt: niemand. */
  curator: { id: string; label: string; model: string } | null;
}

export interface ProviderRow {
  id: string;
  label: string;
  /** Art der Schnittstelle: openai | anthropic | google | fake. */
  kind: string;
  base_url: string | null;
  model: string;
  enabled: boolean;
  is_preset: boolean;
  needs_key: boolean;
  key_source: 'einstellungen' | 'umgebung' | 'fehlt';
  /** Nur die letzten vier Zeichen, nie der Schlüssel selbst. */
  key_hint: string;
  key_env: string;
  ready: boolean;
  reason: string;
  key_url: string;
  models_url: string;
  /** Preis je Million Token. Null heißt unbekannt — es wird nichts geraten. */
  price_in: number | null;
  price_out: number | null;
}

export interface CustomHint {
  label: string;
  base_url: string;
  model: string;
}

export interface AdminSettings {
  providers: ProviderRow[];
  kinds: { id: string; label: string }[];
  custom_hints: CustomHint[];
  request_timeout_s: number;
  timeout_source: string;
  /** Anbieterkennung des Kurators, leer wenn keiner eingestellt ist. */
  curator: string;
  env: string;
  fake_providers_enabled: boolean;
  /** Im Test- und Entwicklungsbetrieb steht der Tisch fest im Quelltext. */
  editable: boolean;
}

export interface ProviderTestResult {
  ok: boolean;
  detail: string;
  latency_ms: number | null;
}

export interface NewProvider {
  label: string;
  kind: string;
  base_url: string;
  model: string;
  api_key: string;
}

export interface ProviderPatch {
  label?: string;
  base_url?: string;
  model?: string;
  api_key?: string;
  enabled?: boolean;
  price_in?: number;
  price_out?: number;
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
  kind: SparkKind;
  refs: string[];
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
  tokens_in: number | null;
  tokens_out: number | null;
  /** Kosten in Millionstel der eingetragenen Währung. Null heißt unbekannt. */
  cost_micro: number | null;
  cost_source: 'berechnet' | 'unbekannt';
  /** Gemessene Zeitpunkte in Sekunden seit Epoche. Null heißt: nie eingetreten. */
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
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
  /** Begriffe, an denen diese Fundstelle hängt — Grundlage der Themen-Linse. */
  topics: string[];
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
  /** Laufkennung der aktuellen Auswertung — verhindert unbemerkte Mischung. */
  analysis_run_id?: string;
  method_version?: string;
  claim_level?: string;
  epistemik?: string;
  zahlen?: unknown;
}

export interface SparkEntry {
  spark: Spark;
  jobs: Job[];
  markers: Marker[];
  summary: Summary | null;
}

/** Eine einzelne genannte Folge bzw. ein Themencluster in einer Szenario-Runde. */
export interface Folge {
  id: string;
  /**
   * Bei `gleiche_aussage`: Wortlaut der Nennung.
   * Bei `themencluster`: Beschriftung des Clusters — keine Behauptung „alle sagen A“.
   */
  text: string;
  /** Gleiche Aussage vs. Themencluster mit einzelnen Aussagen. */
  art?: 'gleiche_aussage' | 'themencluster';
  /** Was `anzahl` zählt: gemeinsame Nennung oder Clusterbeteiligung. */
  zaehlung?: 'gemeinsame_nenung' | 'clusterbeteiligung';
  themen: string[];
  nennungen: {
    job_id: string;
    label: string;
    quote: string;
    start_offset: number;
    end_offset: number;
  }[];
  /** Stimmen in diesem Cluster / bei dieser Nennung. Eine Häufigkeit, keine Zustimmung. */
  anzahl: number;
  /** Wie viele Stimmen überhaupt auswertbar geantwortet haben. */
  von: number;
  /** Folgen, die dieser bei gleichem Thema entgegenstehen. */
  gegensatz: string[];
}

/** Die Konsequenzkarte einer Szenario-Runde. Vom Server gerechnet. */
export interface Szenario {
  spark_id: string;
  seq: number;
  prompt: string;
  ausgang: { job_id: string; label: string; auszug: string; gekuerzt: boolean } | null;
  stimmen: { job_id: string; label: string }[];
  folgen: Folge[];
  /** Genannte Folgen, die nicht gezeigt werden — gezählt, nicht verschwiegen. */
  uebergangen: number;
  methode: string;
}

export interface SessionBundle {
  session: Session;
  sparks: SparkEntry[];
  /** Konsequenzkarten der Szenario-Runden dieser Sitzung. */
  szenarien: Szenario[];
  relations: Relation[];
  /** Läuft noch etwas? Ein Bericht wäre dann vorläufig. */
  pending: boolean;
  last_event_id: number;
  exported_at: number;
}

export interface HealthInfo {
  status: string;
  env: string;
  database: string;
  providers: Record<string, { ready: boolean; reason: string; label?: string }>;
  fake_providers_enabled: boolean;
  missing_credentials: string[];
}
