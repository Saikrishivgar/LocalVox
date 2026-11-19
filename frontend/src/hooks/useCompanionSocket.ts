"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { companionApi, getWebSocketUrl } from "@/lib/api";
import type {
  AudioLevelState,
  LiveMessageItem,
  PipelineState,
  StateSnapshotResponse,
  TurnMetricsSummary,
  VoiceProfileSummary,
  WakeWordDetectionEvent,
  WakeWordEngineInfo,
} from "@/lib/types";

const ZERO_BARS = new Array<number>(24).fill(0);

export interface CompanionLiveState {
  connected: boolean;
  pipelineState: PipelineState;
  micActive: boolean;
  sessionId: string;
  messages: LiveMessageItem[];
  streamingPartial: string;
  lastMetrics: TurnMetricsSummary | null;
  audioLevel: AudioLevelState;
  errorBanner: string | null;
  memoryVersion: number;
  historyVersion: number;
  settingsVersion: number;
  wakewordEnabled: boolean;
  wakewordActive: boolean;
  wakePhrase: string;
  lastWakeEvent: WakeWordDetectionEvent | null;
  wakewordEngine: WakeWordEngineInfo | null;
  voiceProfile: VoiceProfileSummary | null;
  clearError: () => void;
  refreshSnapshot: () => Promise<void>;
}

export function useCompanionSocket(): CompanionLiveState {
  const [connected, setConnected] = useState(false);
  const [pipelineState, setPipelineState] = useState<PipelineState>("IDLE");
  const [micActive, setMicActive] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<LiveMessageItem[]>([]);
  const [streamingPartial, setStreamingPartial] = useState("");
  const [lastMetrics, setLastMetrics] = useState<TurnMetricsSummary | null>(
    null
  );
  const [audioLevel, setAudioLevel] = useState<AudioLevelState>({
    source: "mic",
    rms: 0,
    bars: ZERO_BARS,
    updatedAt: 0,
  });
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [memoryVersion, setMemoryVersion] = useState(0);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [settingsVersion, setSettingsVersion] = useState(0);
  const [wakewordEnabled, setWakewordEnabled] = useState(true);
  const [wakewordActive, setWakewordActive] = useState(false);
  const [wakePhrase, setWakePhrase] = useState("Hey Jarvis");
  const [lastWakeEvent, setLastWakeEvent] =
    useState<WakeWordDetectionEvent | null>(null);
  const [wakewordEngine, setWakewordEngine] =
    useState<WakeWordEngineInfo | null>(null);
  const [voiceProfile, setVoiceProfile] =
    useState<VoiceProfileSummary | null>(null);

  const mountedRef = useRef(true);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const applySnapshot = useCallback((snap: StateSnapshotResponse) => {
    setPipelineState(snap.pipeline_state);
    setMicActive(snap.mic_active);
    setSessionId(snap.session_id);
    setMessages(snap.messages || []);
    setStreamingPartial(snap.streaming_partial || "");
    setLastMetrics(snap.last_metrics || null);
    if (typeof snap.wakeword_enabled === "boolean") {
      setWakewordEnabled(snap.wakeword_enabled);
    }
    if (typeof snap.wakeword_active === "boolean") {
      setWakewordActive(snap.wakeword_active);
    }
    if (typeof snap.wake_phrase === "string" && snap.wake_phrase) {
      setWakePhrase(snap.wake_phrase);
    }
    if (snap.last_wake_event !== undefined) {
      setLastWakeEvent(snap.last_wake_event ?? null);
    }
    if (snap.wakeword_engine !== undefined) {
      setWakewordEngine(snap.wakeword_engine ?? null);
    }
    if (snap.voice_profile !== undefined) {
      setVoiceProfile(snap.voice_profile ?? null);
    }
  }, []);

  const refreshSnapshot = useCallback(async () => {
    try {
      const snap = await companionApi.getState();
      if (mountedRef.current) {
        applySnapshot(snap);
      }
    } catch {
      // Backend may still be booting; reconnect handles sync.
    }
  }, [applySnapshot]);

  // Smoothly decay visual audio bars if no real audio_level frames arrive recently.
  useEffect(() => {
    const timer = setInterval(() => {
      setAudioLevel((prev) => {
        if (prev.rms === 0 || Date.now() - prev.updatedAt < 180) {
          return prev;
        }
        return {
          source: prev.source,
          rms: 0,
          bars: ZERO_BARS,
          updatedAt: Date.now(),
        };
      });
    }, 150);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refreshSnapshot();

    function connect() {
      if (!mountedRef.current) return;
      try {
        const ws = new WebSocket(getWebSocketUrl());
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mountedRef.current) return;
          setConnected(true);
          setErrorBanner(null);
        };

        ws.onmessage = (event) => {
          if (!mountedRef.current) return;
          try {
            const payload = JSON.parse(String(event.data));
            switch (payload.type) {
              case "snapshot":
                applySnapshot(payload as StateSnapshotResponse);
                break;
              case "state_changed":
                setPipelineState(payload.state as PipelineState);
                if (typeof payload.mic_active === "boolean") {
                  setMicActive(payload.mic_active);
                }
                break;
              case "mic_status":
                setMicActive(Boolean(payload.mic_active));
                if (typeof payload.wakeword_active === "boolean") {
                  setWakewordActive(payload.wakeword_active);
                }
                break;
              case "wake_detected":
                if (payload.event) {
                  setLastWakeEvent(payload.event as WakeWordDetectionEvent);
                }
                break;
              case "transcript": {
                const msg = payload.message as LiveMessageItem | undefined;
                if (msg) {
                  setStreamingPartial("");
                  setMessages((prev) =>
                    prev.some((m) => m.id === msg.id) ? prev : [...prev, msg]
                  );
                }
                break;
              }
              case "response_token":
                if (typeof payload.partial === "string") {
                  setStreamingPartial(payload.partial);
                } else if (typeof payload.token === "string") {
                  setStreamingPartial((prev) => prev + payload.token);
                }
                break;
              case "response_complete": {
                const msg = payload.message as LiveMessageItem | undefined;
                setStreamingPartial("");
                if (msg) {
                  setMessages((prev) =>
                    prev.some((m) => m.id === msg.id) ? prev : [...prev, msg]
                  );
                }
                break;
              }
              case "turn_complete": {
                const metrics = payload.metrics as
                  | TurnMetricsSummary
                  | undefined;
                if (metrics) {
                  setLastMetrics(metrics);
                  setMessages((prev) => {
                    const next = [...prev];
                    for (let i = next.length - 1; i >= 0; i -= 1) {
                      if (next[i].role === "assistant" && !next[i].metrics) {
                        next[i] = {
                          ...next[i],
                          language:
                            (metrics.tts_language as string | undefined) ??
                            next[i].language,
                          metrics,
                        };
                        break;
                      }
                    }
                    return next;
                  });
                }
                setHistoryVersion((v) => v + 1);
                setMemoryVersion((v) => v + 1);
                break;
              }
              case "audio_level":
                setAudioLevel({
                  source: payload.source === "speaker" ? "speaker" : "mic",
                  rms: Number(payload.rms || 0),
                  bars: Array.isArray(payload.bars)
                    ? (payload.bars as number[])
                    : ZERO_BARS,
                  updatedAt: Date.now(),
                  wakeScore:
                    typeof payload.wake_score === "number"
                      ? payload.wake_score
                      : undefined,
                  genericScore:
                    typeof payload.generic_score === "number"
                      ? payload.generic_score
                      : undefined,
                  personalScore:
                    typeof payload.personal_score === "number"
                      ? payload.personal_score
                      : undefined,
                  wakeThreshold:
                    typeof payload.wake_threshold === "number"
                      ? payload.wake_threshold
                      : undefined,
                });
                break;
              case "voice_profile_updated":
                if (payload.profile) {
                  setVoiceProfile(payload.profile as VoiceProfileSummary);
                }
                break;
              case "barge_in":
                setStreamingPartial("");
                break;
              case "session_changed":
              case "session_deleted":
                void refreshSnapshot();
                setHistoryVersion((v) => v + 1);
                break;
              case "memory_changed":
                setMemoryVersion((v) => v + 1);
                break;
              case "settings_updated":
                setSettingsVersion((v) => v + 1);
                void refreshSnapshot();
                break;
              case "error":
                if (typeof payload.message === "string") {
                  setErrorBanner(payload.message);
                }
                break;
              default:
                break;
            }
          } catch {
            // Ignore malformed socket message.
          }
        };

        ws.onclose = () => {
          if (!mountedRef.current) return;
          setConnected(false);
          reconnectTimerRef.current = setTimeout(connect, 1800);
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch {
        setConnected(false);
        reconnectTimerRef.current = setTimeout(connect, 2000);
      }
    }

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [applySnapshot, refreshSnapshot]);

  const clearError = useCallback(() => setErrorBanner(null), []);

  return {
    connected,
    pipelineState,
    micActive,
    sessionId,
    messages,
    streamingPartial,
    lastMetrics,
    audioLevel,
    errorBanner,
    memoryVersion,
    historyVersion,
    settingsVersion,
    wakewordEnabled,
    wakewordActive,
    wakePhrase,
    lastWakeEvent,
    wakewordEngine,
    voiceProfile,
    clearError,
    refreshSnapshot,
  };
}
