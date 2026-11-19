export type PipelineState =
  | "IDLE"
  | "SLEEPING"
  | "WAKE_DETECTED"
  | "LISTENING"
  | "PROCESSING"
  | "GENERATING"
  | "SPEAKING"
  | "INTERRUPTED"
  | "RETURNING_TO_STANDBY"
  | "ERROR"
  | "STOPPING";

export type NavTab = "companion" | "memory" | "history" | "system" | "settings";

export const MEMORY_CATEGORIES = [
  "personal_preference",
  "user_fact",
  "interest_hobby",
  "ongoing_project",
  "important_decision",
  "learning_progress",
  "conversation_event",
  "pending_task",
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export const CATEGORY_LABELS: Record<MemoryCategory, string> = {
  personal_preference: "Preferences",
  user_fact: "User Facts",
  interest_hobby: "Interests",
  ongoing_project: "Projects",
  important_decision: "Decisions",
  learning_progress: "Learning",
  conversation_event: "Events",
  pending_task: "Tasks",
};

export interface TurnMetricsSummary {
  turn_id?: number;
  tts_backend?: string;
  tts_language?: string;
  vad_ms?: number;
  stt_ms?: number;
  llm_ttft_ms?: number;
  llm_total_ms?: number;
  llm_tokens_per_sec?: number;
  tts_ms?: number;
  playback_ms?: number;
  response_latency_ms?: number;
  first_audio_latency_ms?: number;
  total_turn_ms?: number;
  memory_mb?: number;
  interrupted?: boolean;
  tts_chunks?: number;
  barge_in_triggered?: boolean;
  memory_retrieval_ms?: number;
  memories_retrieved?: number;
  memory_context_chars?: number;
  [key: string]: unknown;
}

export interface LiveMessageItem {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  language: string | null;
  created_at: string;
  metrics?: TurnMetricsSummary | null;
}

export interface WakeWordDetectionEvent {
  wake_phrase: string;
  model_name: string;
  confidence: number;
  threshold: number;
  inference_ms: number;
  audio_rms: number;
  matched_by?: "openwakeword_onnx" | "personalized_voice" | "hybrid";
  generic_confidence?: number | null;
  personal_confidence?: number | null;
  timestamp: string;
  detected_at_monotonic: number;
}

export interface WakeWordEngineInfo {
  engine: string;
  wake_phrase: string;
  model_name: string;
  model_path: string;
  loaded: boolean;
  models_present: boolean;
  disk_size_mb: number | null;
  execution_provider: string;
  last_inference_ms: number | null;
  last_score: number;
  last_personal_score?: number;
  total_detections: number;
  last_detection_at: string | null;
  personalized_enrolled?: boolean;
  personalized_samples?: number;
  personal_only?: boolean;
}

export interface VoiceProfileSummary {
  enrolled: boolean;
  samples_recorded: number;
  required_samples: number;
  in_progress_samples: number;
  wake_phrase: string;
  profile_path: string;
  updated_at: string | null;
  personal_enabled: boolean;
  personal_only: boolean;
  personal_threshold: number;
  consistency_score: number | null;
}

export interface EnrollmentSampleResult {
  sample_index: number;
  required_samples: number;
  completed: boolean;
  duration_s: number;
  audio_rms: number;
  generic_score: number;
  consistency_with_previous: number | null;
  message: string;
  profile: VoiceProfileSummary;
}

export interface StateSnapshotResponse {
  pipeline_state: PipelineState;
  mic_active: boolean;
  session_id: string;
  streaming_partial: string;
  messages: LiveMessageItem[];
  last_metrics: TurnMetricsSummary | null;
  wakeword_enabled?: boolean;
  wakeword_active?: boolean;
  wake_phrase?: string;
  last_wake_event?: WakeWordDetectionEvent | null;
  wakeword_engine?: WakeWordEngineInfo | null;
  voice_profile?: VoiceProfileSummary | null;
}

export interface ChatResponse {
  session_id: string;
  response_text: string;
  metrics: TurnMetricsSummary | null;
  pipeline_state: PipelineState;
}

export interface VoiceControlResponse {
  status: string;
  pipeline_state: PipelineState;
  mic_active: boolean;
  message: string;
}

export interface MemoryRecord {
  id: string;
  category: MemoryCategory;
  subject: string;
  content: string;
  keywords: string[];
  importance: number;
  confidence: number;
  source: "user_explicit" | "user_stated" | "assistant_inferred";
  confirmed_by_user: boolean;
  normalized_hash: string;
  session_id: string | null;
  access_count: number;
  created_at: string;
  updated_at: string;
  last_accessed_at: string | null;
}

export interface MemoryListResponse {
  total_count: number;
  category_counts: Record<string, number>;
  items: MemoryRecord[];
  match_scores: Record<string, number>;
}

export interface MemoryMutationResponse {
  status: string;
  action: string;
  record: MemoryRecord | null;
  deleted_count: number;
}

export interface MemoryCreatePayload {
  category: MemoryCategory;
  subject: string;
  content: string;
  keywords: string[];
  importance: number;
}

export interface MemoryUpdatePayload {
  category?: MemoryCategory;
  subject?: string;
  content?: string;
  keywords?: string[];
  importance?: number;
}

export interface ConversationSessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  preview: string;
}

export interface ConversationMessageRecord {
  id: number;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  language: string | null;
  metrics: TurnMetricsSummary | null;
  created_at: string;
}

export interface ConversationSessionDetail {
  session: ConversationSessionSummary;
  messages: ConversationMessageRecord[];
}

export interface ConversationListResponse {
  items: ConversationSessionSummary[];
  active_session_id: string;
}

export interface ConversationContinueResponse {
  detail: ConversationSessionDetail;
  loaded_turns: number;
  active_session_id: string;
}

export interface EngineStatus {
  name: string;
  status: "ready" | "loaded" | "offline" | "missing" | "error";
  detail: string;
  model_name: string | null;
  loaded: boolean;
  disk_size_mb: number | null;
  memory_mb: number | null;
}

export interface ExtractionWorkerSummary {
  completed_jobs: number;
  queue_size: number;
  idle: boolean;
  error_count: number;
  last_outcome: {
    candidates_parsed: number;
    actions: string[];
    skipped_privacy: number;
    skipped_low_value: number;
    extraction_ms: number;
    write_ms: number;
  } | null;
}

export interface SystemMonitorResponse {
  hardware: string;
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  process_rss_mb: number;
  ollama_connected: boolean;
  qwen_available: boolean;
  available_ollama_models: string[];
  engines: EngineStatus[];
  sqlite_integrity: Record<string, unknown>;
  sqlite_size_mb: number | null;
  extraction_worker: ExtractionWorkerSummary;
  recent_latencies: Array<Record<string, unknown>>;
  pipeline_state: PipelineState;
  mic_active: boolean;
}

export interface SettingsPayload {
  kokoro_voice: string;
  kokoro_speed: number;
  piper_length_scale: number;
  stt_default_language: string | null;
  tts_routing: "auto" | "kokoro" | "piper";
  streaming_tts_enabled: boolean;
  barge_in_enabled: boolean;
  barge_in_sensitivity: "low" | "medium" | "high";
  llm_model_name: string;
  llm_temperature: number;
  max_history_turns: number;
  memory_enabled: boolean;
  memory_auto_extract: boolean;
  memory_privacy_filter_enabled: boolean;
  wakeword_enabled: boolean;
  wake_phrase: string;
  wakeword_sensitivity: "low" | "medium" | "high";
  wakeword_threshold: number;
  wakeword_conversation_timeout_s: number;
  wakeword_ack_enabled: boolean;
  wakeword_activation_grace_ms: number;
  wakeword_personal_enabled?: boolean;
  wakeword_personal_only?: boolean;
  wakeword_personal_threshold?: number;
}

export interface SettingsResponse {
  settings: SettingsPayload;
  available_kokoro_voices: string[];
  available_ollama_models: string[];
  wakeword_engine?: WakeWordEngineInfo | null;
  voice_profile?: VoiceProfileSummary | null;
  mic_active?: boolean;
}

export interface AudioLevelState {
  source: "mic" | "speaker";
  rms: number;
  bars: number[];
  updatedAt: number;
  wakeScore?: number;
  genericScore?: number;
  personalScore?: number;
  wakeThreshold?: number;
}
