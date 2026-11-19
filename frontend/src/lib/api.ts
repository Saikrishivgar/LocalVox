import type {
  ChatResponse,
  ConversationContinueResponse,
  ConversationListResponse,
  ConversationSessionDetail,
  ConversationSessionSummary,
  EnrollmentSampleResult,
  MemoryCategory,
  MemoryCreatePayload,
  MemoryListResponse,
  MemoryMutationResponse,
  MemoryUpdatePayload,
  SettingsPayload,
  SettingsResponse,
  StateSnapshotResponse,
  SystemMonitorResponse,
  VoiceControlResponse,
  VoiceProfileSummary,
} from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export function getWebSocketUrl(): string {
  const wsBase = API_BASE.replace(/^http/i, "ws");
  return `${wsBase}/ws/events`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...(init?.headers || {}),
      },
    });
  } catch {
    throw new Error(
      `Cannot connect to LocalVox backend at ${API_BASE}. Start the API server in another terminal with "make api".`
    );
  }

  const text = await response.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { detail: text };
    }
  }

  if (!response.ok) {
    const detail =
      parsed && typeof parsed === "object" && "detail" in parsed
        ? String((parsed as { detail: unknown }).detail)
        : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return parsed as T;
}

export const companionApi = {
  getState: () => requestJson<StateSnapshotResponse>("/api/state"),

  sendChat: (text: string, speak: boolean, language?: string | null) =>
    requestJson<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ text, speak, language: language ?? null }),
    }),

  startVoice: (mode: "auto" | "direct" | "wakeword" = "auto") =>
    requestJson<VoiceControlResponse>(
      `/api/voice/start?mode=${encodeURIComponent(mode)}`,
      { method: "POST" }
    ),

  stopVoice: () =>
    requestJson<VoiceControlResponse>("/api/voice/stop", { method: "POST" }),

  stopSpeaking: () =>
    requestJson<VoiceControlResponse>("/api/voice/stop-speaking", {
      method: "POST",
    }),

  enableWakeWord: () =>
    requestJson<VoiceControlResponse>("/api/wakeword/enable", {
      method: "POST",
    }),

  disableWakeWord: () =>
    requestJson<VoiceControlResponse>("/api/wakeword/disable", {
      method: "POST",
    }),

  getVoiceProfile: () =>
    requestJson<VoiceProfileSummary>("/api/wakeword/profile"),

  startVoiceEnrollment: () =>
    requestJson<VoiceProfileSummary>("/api/wakeword/enroll/start", {
      method: "POST",
    }),

  recordVoiceEnrollmentSample: (timeoutS = 6.5) =>
    requestJson<EnrollmentSampleResult>(
      `/api/wakeword/enroll/record?timeout_s=${encodeURIComponent(String(timeoutS))}`,
      { method: "POST" }
    ),

  deleteVoiceProfile: () =>
    requestJson<VoiceProfileSummary>("/api/wakeword/profile", {
      method: "DELETE",
    }),

  listMemories: (query = "", category: MemoryCategory | null = null) => {
    const params = new URLSearchParams();
    if (query.trim()) params.set("query", query.trim());
    if (category) params.set("category", category);
    params.set("limit", "200");
    return requestJson<MemoryListResponse>(`/api/memories?${params.toString()}`);
  },

  createMemory: (payload: MemoryCreatePayload) =>
    requestJson<MemoryMutationResponse>("/api/memories", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateMemory: (id: string, payload: MemoryUpdatePayload) =>
    requestJson<MemoryMutationResponse>(
      `/api/memories/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        body: JSON.stringify(payload),
      }
    ),

  deleteMemory: (id: string) =>
    requestJson<MemoryMutationResponse>(
      `/api/memories/${encodeURIComponent(id)}`,
      { method: "DELETE" }
    ),

  clearAllMemories: (confirmPhrase: string) =>
    requestJson<MemoryMutationResponse>("/api/memories/clear", {
      method: "POST",
      body: JSON.stringify({ confirm_phrase: confirmPhrase }),
    }),

  listConversations: (query = "") => {
    const params = new URLSearchParams();
    if (query.trim()) params.set("query", query.trim());
    params.set("limit", "100");
    return requestJson<ConversationListResponse>(
      `/api/conversations?${params.toString()}`
    );
  },

  getConversation: (sessionId: string) =>
    requestJson<ConversationSessionDetail>(
      `/api/conversations/${encodeURIComponent(sessionId)}`
    ),

  newConversation: () =>
    requestJson<ConversationSessionSummary>("/api/conversations/new", {
      method: "POST",
    }),

  continueConversation: (sessionId: string) =>
    requestJson<ConversationContinueResponse>(
      `/api/conversations/${encodeURIComponent(sessionId)}/continue`,
      { method: "POST" }
    ),

  deleteConversation: (sessionId: string) =>
    requestJson<{ status: string; deleted: boolean }>(
      `/api/conversations/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    ),

  getSystemMonitor: () => requestJson<SystemMonitorResponse>("/api/system"),

  getSettings: () => requestJson<SettingsResponse>("/api/settings"),

  updateSettings: (payload: SettingsPayload) =>
    requestJson<SettingsResponse>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
};
