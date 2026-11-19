"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Fingerprint,
  Mic,
  MicOff,
  PlusCircle,
  Radio,
  Send,
  Square,
  Volume2,
  VolumeX,
} from "lucide-react";
import { companionApi } from "@/lib/api";
import { STATE_THEMES } from "@/lib/stateTheme";
import type {
  AudioLevelState,
  LiveMessageItem,
  PipelineState,
  TurnMetricsSummary,
  VoiceProfileSummary,
  WakeWordDetectionEvent,
  WakeWordEngineInfo,
} from "@/lib/types";
import { AudioWaveform } from "./AudioWaveform";
import { CompanionOrb } from "./CompanionOrb";
import { VoiceEnrollmentModal } from "./VoiceEnrollmentModal";

interface CompanionViewProps {
  pipelineState: PipelineState;
  micActive: boolean;
  messages: LiveMessageItem[];
  streamingPartial: string;
  lastMetrics: TurnMetricsSummary | null;
  audioLevel: AudioLevelState;
  wakewordEnabled?: boolean;
  wakewordActive?: boolean;
  wakePhrase?: string;
  lastWakeEvent?: WakeWordDetectionEvent | null;
  wakewordEngine?: WakeWordEngineInfo | null;
  voiceProfile?: VoiceProfileSummary | null;
  onRefreshState: () => Promise<void>;
}

export function CompanionView({
  pipelineState,
  micActive,
  messages,
  streamingPartial,
  lastMetrics,
  audioLevel,
  wakewordEnabled = true,
  wakewordActive = false,
  wakePhrase = "Hey Jarvis",
  lastWakeEvent = null,
  wakewordEngine = null,
  voiceProfile = null,
  onRefreshState,
}: CompanionViewProps) {
  const [input, setInput] = useState("");
  const [speakReplies, setSpeakReplies] = useState(true);
  const [sending, setSending] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [enrollOpen, setEnrollOpen] = useState(false);

  const feedEndRef = useRef<HTMLDivElement>(null);
  const theme = STATE_THEMES[pipelineState];
  const canStopSpeaking =
    pipelineState === "SPEAKING" || pipelineState === "GENERATING";

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streamingPartial]);

  async function handleToggleMic() {
    setActionError(null);
    try {
      if (micActive) {
        await companionApi.stopVoice();
      } else {
        await companionApi.startVoice();
      }
      await onRefreshState();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleStopSpeaking() {
    setActionError(null);
    try {
      await companionApi.stopSpeaking();
      await onRefreshState();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleNewSession() {
    setActionError(null);
    try {
      await companionApi.newConversation();
      await onRefreshState();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSendText(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || sending) return;
    setSending(true);
    setActionError(null);
    setInput("");
    try {
      await companionApi.sendChat(trimmed, speakReplies);
      await onRefreshState();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-12">
      {/* Left Interactive Presence Stage (Centerpiece 3D Core) */}
      <section className="glass-panel flex flex-col items-center justify-between rounded-2xl p-5 lg:col-span-7">
        <div className="flex w-full flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-widest text-slate-400">
              Acoustic Core
            </span>
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${
                wakewordActive
                  ? "border-indigo-400/40 bg-indigo-500/15 text-indigo-200"
                  : wakewordEnabled
                    ? "border-white/10 bg-white/[0.04] text-slate-300"
                    : "border-white/10 bg-white/[0.02] text-slate-500"
              }`}
              data-testid="wakeword-status-badge"
            >
              <Radio
                className={`h-3 w-3 ${wakewordActive ? "animate-pulse text-cyan-300" : "text-slate-400"}`}
              />
              <span>Wake Word: {wakePhrase}</span>
              <span className="font-mono text-[10px] opacity-80">
                ·{" "}
                {wakewordActive
                  ? pipelineState === "SLEEPING"
                    ? "Standby"
                    : pipelineState === "WAKE_DETECTED"
                      ? "Triggered"
                      : "Active"
                  : !wakewordEnabled
                    ? "Disabled"
                    : wakewordEngine?.models_present === false
                      ? "Missing"
                      : "Ready"}
              </span>
            </span>
          </div>
          <button
            type="button"
            onClick={() => void handleNewSession()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1 text-xs text-slate-300 transition hover:bg-white/[0.09] hover:text-white"
          >
            <PlusCircle className="h-3.5 w-3.5" />
            New Chat
          </button>
        </div>

        <div className="flex flex-1 flex-col items-center justify-center gap-3 py-1">
          <CompanionOrb
            pipelineState={pipelineState}
            rms={audioLevel.rms}
            bars={audioLevel.bars}
            onClick={() => void handleToggleMic()}
          />

          <motion.div
            key={pipelineState}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22 }}
            className="text-center"
          >
            <div className="text-xl font-semibold tracking-tight text-white">
              {theme.label}
            </div>
            <p className="mt-0.5 text-xs text-slate-400">{theme.subtitle}</p>
          </motion.div>

          <AudioWaveform
            audioLevel={audioLevel}
            pipelineState={pipelineState}
          />

          {wakewordActive && pipelineState === "SLEEPING" && (
            <div
              className="w-full max-w-xs space-y-1 pt-0.5"
              data-testid="standby-wake-meter"
            >
              <div className="flex items-center justify-between font-mono text-[10px] text-slate-400">
                <span>
                  Say &ldquo;{wakePhrase}&rdquo;
                  {voiceProfile?.enrolled ? " · My Voice" : ""}
                </span>
                <span>
                  Match {Math.round((audioLevel.wakeScore ?? 0) * 100)}%
                </span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded-full bg-white/[0.07]">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-cyan-300 transition-all duration-100"
                  style={{
                    width: `${Math.min(100, Math.round((audioLevel.wakeScore ?? 0) * 100))}%`,
                  }}
                />
              </div>
            </div>
          )}
        </div>

        <div className="w-full max-w-md space-y-2.5">
          <div className="flex flex-wrap items-center justify-center gap-2.5">
            <button
              type="button"
              onClick={() => void handleToggleMic()}
              className={`inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition-all ${
                micActive
                  ? "border border-rose-400/40 bg-rose-500/20 text-rose-100 hover:bg-rose-500/30"
                  : "border border-sky-400/40 bg-sky-500/20 text-sky-100 hover:bg-sky-500/30"
              }`}
            >
              {micActive ? (
                <>
                  <MicOff className="h-4 w-4" />
                  Stop Microphone
                </>
              ) : (
                <>
                  <Mic className="h-4 w-4" />
                  Start Microphone
                </>
              )}
            </button>

            <button
              type="button"
              onClick={() => void handleStopSpeaking()}
              disabled={!canStopSpeaking}
              className="inline-flex items-center gap-2 rounded-full border border-amber-400/35 bg-amber-500/15 px-4 py-2.5 text-sm font-medium text-amber-100 transition hover:bg-amber-500/25 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/[0.03] disabled:text-slate-500"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              Stop Speaking
            </button>
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-2 text-xs">
              <div className="flex items-center gap-1.5 text-slate-300">
                <Fingerprint
                  className={`h-3.5 w-3.5 shrink-0 ${
                    voiceProfile?.enrolled ? "text-emerald-300" : "text-cyan-300"
                  }`}
                />
                <span className="truncate">
                  {voiceProfile?.enrolled
                    ? `My Voice (${voiceProfile.samples_recorded}/3)`
                    : "My Voice"}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setEnrollOpen(true)}
                className={`shrink-0 rounded-full px-2.5 py-1 font-medium transition ${
                  voiceProfile?.enrolled
                    ? "bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-400/30 hover:bg-emerald-500/25"
                    : "bg-cyan-500/20 text-cyan-100 ring-1 ring-cyan-400/35 hover:bg-cyan-500/30"
                }`}
              >
                {voiceProfile?.enrolled ? "Manage" : "Personalize Voice"}
              </button>
            </div>

            <div className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-2 text-xs">
              <span className="text-slate-400">Speak Replies</span>
              <button
                type="button"
                onClick={() => setSpeakReplies((v) => !v)}
                aria-pressed={speakReplies}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium transition ${
                  speakReplies
                    ? "bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-400/30"
                    : "bg-white/[0.06] text-slate-400"
                }`}
              >
                {speakReplies ? (
                  <>
                    <Volume2 className="h-3.5 w-3.5" />
                    Audio On
                  </>
                ) : (
                  <>
                    <VolumeX className="h-3.5 w-3.5" />
                    Silent Text
                  </>
                )}
              </button>
            </div>
          </div>

          {lastWakeEvent && (
            <div className="flex items-center justify-between rounded-xl border border-cyan-400/20 bg-cyan-500/[0.07] px-3 py-1.5 font-mono text-[11px] text-cyan-200">
              <span>
                Last Wake: &ldquo;{lastWakeEvent.wake_phrase}&rdquo; (
                {(lastWakeEvent.confidence * 100).toFixed(1)}%
                {lastWakeEvent.matched_by === "personalized_voice"
                  ? " · My Voice"
                  : lastWakeEvent.matched_by === "hybrid"
                    ? " · Hybrid"
                    : ""}
                )
              </span>
              <span>{lastWakeEvent.inference_ms.toFixed(2)} ms</span>
            </div>
          )}

          {lastMetrics && (
            <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
              <div className="glass-subtle rounded-xl p-2">
                <div className="text-[10px] text-slate-500">STT</div>
                <div className="mt-0.5 text-slate-200">
                  {Math.round(lastMetrics.stt_ms || 0)} ms
                </div>
              </div>
              <div className="glass-subtle rounded-xl p-2">
                <div className="text-[10px] text-slate-500">LLM TTFT</div>
                <div className="mt-0.5 text-slate-200">
                  {Math.round(lastMetrics.llm_ttft_ms || 0)} ms
                </div>
              </div>
              <div className="glass-subtle rounded-xl p-2">
                <div className="text-[10px] text-slate-500">Engine</div>
                <div className="mt-0.5 uppercase text-slate-200">
                  {lastMetrics.tts_backend || "local"}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Right Compact Transcript & Composer Panel */}
      <section className="glass-panel flex min-h-0 flex-col rounded-2xl lg:col-span-5">
        <div className="flex items-center justify-between border-b border-white/[0.07] px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">
              Live Conversation
            </h2>
            <p className="text-[11px] text-slate-400">
              English · Tamil · Tanglish · Memory-grounded
            </p>
          </div>
          <span className="font-mono text-[11px] text-slate-500">
            {messages.length} messages
          </span>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          {messages.length === 0 && !streamingPartial ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] px-6 py-5 max-w-md">
                <p className="text-sm font-medium text-slate-200">
                  Start speaking or type below
                </p>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                  Every voice turn runs locally on Silero VAD, MLX Whisper,
                  Qwen3:8B, and Kokoro / Piper with persistent SQLite memory.
                </p>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => {
                const isUser = msg.role === "user";
                return (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.18 }}
                    className={`flex flex-col ${
                      isUser ? "items-end" : "items-start"
                    }`}
                  >
                    <div
                      className={`max-w-[90%] rounded-2xl px-3.5 py-2.5 text-[13px] leading-relaxed ${
                        isUser
                          ? "border border-sky-400/25 bg-sky-500/15 text-slate-100"
                          : "border border-white/10 bg-white/[0.045] text-slate-200"
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                      {(msg.language || msg.metrics) && (
                        <div className="mt-2 flex flex-wrap items-center gap-2 font-mono text-[10px] text-slate-400">
                          {msg.language && (
                            <span className="rounded bg-white/[0.07] px-1.5 py-0.5 uppercase">
                              {msg.language}
                            </span>
                          )}
                          {msg.metrics && (
                            <>
                              <span>
                                {(msg.metrics.llm_tokens_per_sec || 0).toFixed(
                                  1
                                )}{" "}
                                tok/s
                              </span>
                              <span>·</span>
                              <span>
                                {Math.round(msg.metrics.total_turn_ms || 0)}ms
                              </span>
                              {typeof msg.metrics.memories_retrieved ===
                                "number" &&
                                msg.metrics.memories_retrieved > 0 && (
                                  <>
                                    <span>·</span>
                                    <span className="text-violet-300">
                                      {msg.metrics.memories_retrieved} memories
                                    </span>
                                  </>
                                )}
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                );
              })}

              {streamingPartial && (
                <div className="flex flex-col items-start">
                  <div className="max-w-[85%] rounded-2xl border border-indigo-400/25 bg-indigo-500/10 px-4 py-3 text-sm leading-relaxed text-slate-100">
                    <span className="whitespace-pre-wrap">
                      {streamingPartial}
                    </span>
                    <span className="ml-1 inline-block h-3.5 w-1.5 animate-pulse bg-indigo-300 align-middle" />
                  </div>
                </div>
              )}
              <div ref={feedEndRef} />
            </>
          )}
        </div>

        <form
          onSubmit={(e) => void handleSendText(e)}
          className="border-t border-white/[0.07] p-4"
        >
          {actionError && (
            <div className="mb-2.5 rounded-xl border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
              {actionError}
            </div>
          )}
          <div className="flex items-center gap-2.5">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder='Message LocalVox or say "Remember..."'
              className="flex-1 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none transition focus:border-sky-400/40 focus:bg-white/[0.06]"
            />
            <button
              type="submit"
              disabled={!input.trim() || sending}
              aria-label="Send message"
              className="inline-flex h-10 items-center gap-2 rounded-xl bg-sky-500/90 px-4 text-sm font-medium text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-slate-500"
            >
              <Send className="h-4 w-4" />
              <span>{sending ? "Thinking" : "Send"}</span>
            </button>
          </div>
        </form>
      </section>

      <VoiceEnrollmentModal
        open={enrollOpen}
        wakePhrase={wakePhrase}
        initialProfile={voiceProfile}
        audioLevel={audioLevel}
        onClose={() => setEnrollOpen(false)}
        onProfileChanged={onRefreshState}
      />
    </div>
  );
}
