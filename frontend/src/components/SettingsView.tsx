"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  CheckCircle2,
  Fingerprint,
  Mic,
  MicOff,
  Radio,
  RefreshCw,
  Save,
  Trash2,
} from "lucide-react";
import { companionApi } from "@/lib/api";
import type {
  AudioLevelState,
  SettingsPayload,
  VoiceProfileSummary,
  WakeWordEngineInfo,
} from "@/lib/types";
import { VoiceEnrollmentModal } from "./VoiceEnrollmentModal";

interface SettingsViewProps {
  settingsVersion: number;
  micActive?: boolean;
  audioLevel?: AudioLevelState;
  onRefreshState?: () => Promise<void>;
}

export function SettingsView({
  settingsVersion,
  micActive: liveMicActive,
  audioLevel,
  onRefreshState,
}: SettingsViewProps) {
  const [draft, setDraft] = useState<SettingsPayload | null>(null);
  const [voices, setVoices] = useState<string[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [wwEngine, setWwEngine] = useState<WakeWordEngineInfo | null>(null);
  const [voiceProfile, setVoiceProfile] = useState<VoiceProfileSummary | null>(
    null
  );
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [apiMicActive, setApiMicActive] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedNotice, setSavedNotice] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSettings = useCallback(async () => {
    setError(null);
    try {
      const res = await companionApi.getSettings();
      setDraft(res.settings);
      setVoices(res.available_kokoro_voices);
      setModels(res.available_ollama_models);
      setWwEngine(res.wakeword_engine ?? null);
      setVoiceProfile(res.voice_profile ?? null);
      setApiMicActive(Boolean(res.mic_active));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings, settingsVersion]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!draft) return;
    setSaving(true);
    setError(null);
    setSavedNotice(false);
    try {
      const res = await companionApi.updateSettings(draft);
      setDraft(res.settings);
      setWwEngine(res.wakeword_engine ?? null);
      setVoiceProfile(res.voice_profile ?? null);
      setApiMicActive(Boolean(res.mic_active));
      setSavedNotice(true);
      setTimeout(() => setSavedNotice(false), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteVoiceProfile() {
    setError(null);
    try {
      const cleared = await companionApi.deleteVoiceProfile();
      setVoiceProfile(cleared);
      if (onRefreshState) await onRefreshState();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const effectiveMicActive = liveMicActive ?? apiMicActive;

  if (!draft) {
    return (
      <div className="glass-panel flex h-full items-center justify-center rounded-2xl p-8 text-sm text-slate-400">
        {error ? `Failed to load settings: ${error}` : "Loading preferences..."}
      </div>
    );
  }

  return (
    <form
      onSubmit={(e) => void handleSubmit(e)}
      className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto pr-1"
    >
      <div className="glass-panel flex flex-wrap items-center justify-between gap-3 rounded-2xl p-5">
        <div>
          <h2 className="text-base font-semibold text-white">
            Companion Preferences
          </h2>
          <p className="text-xs text-slate-400">
            Persisted cleanly to local config.yaml · Filesystem paths locked
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={() => void loadSettings()}
            className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.09]"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Reload
          </button>
          <button
            type="submit"
            disabled={saving}
            className="inline-flex items-center gap-1.5 rounded-xl bg-sky-500/90 px-4 py-2 text-xs font-semibold text-slate-950 transition hover:bg-sky-400 disabled:opacity-50"
          >
            <Save className="h-3.5 w-3.5" />
            {saving ? "Saving..." : "Save Settings"}
          </button>
        </div>
      </div>

      {savedNotice && (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-400/35 bg-emerald-500/15 px-4 py-2.5 text-xs text-emerald-200">
          <CheckCircle2 className="h-4 w-4" />
          Settings validated and saved to config.yaml.
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-rose-400/35 bg-rose-500/15 px-4 py-2.5 text-xs text-rose-200">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Voice & Speech */}
        <section className="glass-panel space-y-4 rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-white">Voice & Speech</h3>

          <label className="block space-y-1.5">
            <span className="text-xs text-slate-300">Kokoro English Voice</span>
            <select
              value={draft.kokoro_voice}
              onChange={(e) =>
                setDraft({ ...draft, kokoro_voice: e.target.value })
              }
              className="w-full rounded-xl border border-white/10 bg-obsidian-900 px-3 py-2 text-sm text-slate-100 outline-none"
            >
              {Array.from(new Set([draft.kokoro_voice, ...voices])).map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>

          <label className="block space-y-1.5">
            <div className="flex justify-between text-xs text-slate-300">
              <span>Kokoro Speed</span>
              <span className="font-mono">{draft.kokoro_speed.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min={0.6}
              max={1.6}
              step={0.05}
              value={draft.kokoro_speed}
              onChange={(e) =>
                setDraft({ ...draft, kokoro_speed: Number(e.target.value) })
              }
              className="w-full"
            />
          </label>

          <label className="block space-y-1.5">
            <div className="flex justify-between text-xs text-slate-300">
              <span>Piper Tamil Length Scale</span>
              <span className="font-mono">
                {draft.piper_length_scale.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.6}
              max={1.6}
              step={0.05}
              value={draft.piper_length_scale}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  piper_length_scale: Number(e.target.value),
                })
              }
              className="w-full"
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block space-y-1.5">
              <span className="text-xs text-slate-300">STT Language</span>
              <select
                value={draft.stt_default_language ?? "auto"}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    stt_default_language:
                      e.target.value === "auto" ? null : e.target.value,
                  })
                }
                className="w-full rounded-xl border border-white/10 bg-obsidian-900 px-3 py-2 text-xs text-slate-100 outline-none"
              >
                <option value="auto">Auto Detect (EN / TA)</option>
                <option value="en">English (en)</option>
                <option value="ta">Tamil (ta)</option>
              </select>
            </label>

            <label className="block space-y-1.5">
              <span className="text-xs text-slate-300">TTS Routing</span>
              <select
                value={draft.tts_routing}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    tts_routing: e.target.value as "auto" | "kokoro" | "piper",
                  })
                }
                className="w-full rounded-xl border border-white/10 bg-obsidian-900 px-3 py-2 text-xs text-slate-100 outline-none"
              >
                <option value="auto">Auto (EN → Kokoro · TA → Piper)</option>
                <option value="kokoro">Kokoro Only</option>
                <option value="piper">Piper Only</option>
              </select>
            </label>
          </div>

          <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2.5 text-xs">
            <span className="text-slate-200">Sentence-Level Streaming TTS</span>
            <input
              type="checkbox"
              checked={draft.streaming_tts_enabled}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  streaming_tts_enabled: e.target.checked,
                })
              }
            />
          </label>
        </section>

        {/* Barge-In, LLM & Memory */}
        <section className="glass-panel space-y-4 rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-white">
            Barge-In, LLM & Memory
          </h3>

          <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2.5 text-xs">
            <span className="text-slate-200">
              Natural Voice Barge-In Interruption
            </span>
            <input
              type="checkbox"
              checked={draft.barge_in_enabled}
              onChange={(e) =>
                setDraft({ ...draft, barge_in_enabled: e.target.checked })
              }
            />
          </label>

          <div className="space-y-1.5">
            <span className="text-xs text-slate-300">Barge-In Sensitivity</span>
            <div className="grid grid-cols-3 gap-2">
              {(["low", "medium", "high"] as const).map((level) => (
                <button
                  key={level}
                  type="button"
                  onClick={() =>
                    setDraft({ ...draft, barge_in_sensitivity: level })
                  }
                  className={`rounded-xl border py-2 text-xs font-medium capitalize transition ${
                    draft.barge_in_sensitivity === level
                      ? "border-sky-400/45 bg-sky-500/20 text-sky-100"
                      : "border-white/10 bg-white/[0.03] text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="block space-y-1.5">
              <span className="text-xs text-slate-300">Local LLM Model</span>
              <select
                value={draft.llm_model_name}
                onChange={(e) =>
                  setDraft({ ...draft, llm_model_name: e.target.value })
                }
                className="w-full rounded-xl border border-white/10 bg-obsidian-900 px-3 py-2 text-xs text-slate-100 outline-none"
              >
                {Array.from(new Set([draft.llm_model_name, ...models])).map(
                  (m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  )
                )}
              </select>
            </label>

            <label className="block space-y-1.5">
              <span className="text-xs text-slate-300">
                Max Context Turns ({draft.max_history_turns})
              </span>
              <input
                type="number"
                min={1}
                max={50}
                value={draft.max_history_turns}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    max_history_turns: Math.max(
                      1,
                      Math.min(50, Number(e.target.value) || 1)
                    ),
                  })
                }
                className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-100 outline-none"
              />
            </label>
          </div>

          <label className="block space-y-1.5">
            <div className="flex justify-between text-xs text-slate-300">
              <span>LLM Temperature</span>
              <span className="font-mono">
                {draft.llm_temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.0}
              max={1.2}
              step={0.05}
              value={draft.llm_temperature}
              onChange={(e) =>
                setDraft({
                  ...draft,
                  llm_temperature: Number(e.target.value),
                })
              }
              className="w-full"
            />
          </label>

          <div className="space-y-2 pt-1">
            <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2 text-xs">
              <span className="text-slate-200">Persistent SQLite Memory</span>
              <input
                type="checkbox"
                checked={draft.memory_enabled}
                onChange={(e) =>
                  setDraft({ ...draft, memory_enabled: e.target.checked })
                }
              />
            </label>
            <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2 text-xs">
              <span className="text-slate-200">
                Automatic Background Memory Extraction
              </span>
              <input
                type="checkbox"
                checked={draft.memory_auto_extract}
                onChange={(e) =>
                  setDraft({ ...draft, memory_auto_extract: e.target.checked })
                }
              />
            </label>
            <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2 text-xs">
              <span className="text-slate-200">
                Secret & Credential Privacy Filter
              </span>
              <input
                type="checkbox"
                checked={draft.memory_privacy_filter_enabled}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    memory_privacy_filter_enabled: e.target.checked,
                  })
                }
              />
            </label>
          </div>
        </section>

        {/* Hands-Free Wake Word (Phase 6) */}
        <section className="glass-panel space-y-4 rounded-2xl p-5 lg:col-span-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Hands-Free Wake Word (Offline openWakeWord ONNX)
              </h3>
              <p className="text-xs text-slate-400">
                100% local standby inference · Zero background audio recording
                or external transmission
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[11px] ${
                  wwEngine?.loaded
                    ? "border-emerald-400/35 bg-emerald-500/15 text-emerald-200"
                    : wwEngine?.models_present
                      ? "border-sky-400/35 bg-sky-500/15 text-sky-200"
                      : "border-amber-400/35 bg-amber-500/15 text-amber-200"
                }`}
              >
                <Radio className="h-3 w-3" />
                {wwEngine
                  ? `${wwEngine.model_name} · ${wwEngine.loaded ? "Loaded" : wwEngine.models_present ? "Ready" : "Missing"}`
                  : "Checking Engine"}
              </span>

              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${
                  effectiveMicActive
                    ? "border-sky-400/35 bg-sky-500/15 text-sky-200"
                    : "border-white/10 bg-white/[0.03] text-slate-400"
                }`}
              >
                {effectiveMicActive ? (
                  <Mic className="h-3 w-3 text-sky-300" />
                ) : (
                  <MicOff className="h-3 w-3 text-slate-500" />
                )}
                {effectiveMicActive
                  ? "Mic Monitoring Active"
                  : "Mic Monitoring Off"}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-3">
              <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2.5 text-xs">
                <span className="text-slate-200">
                  Enable Wake-Word Standby Activation
                </span>
                <input
                  type="checkbox"
                  aria-label="Enable Wake-Word Standby Activation"
                  checked={draft.wakeword_enabled}
                  onChange={(e) =>
                    setDraft({ ...draft, wakeword_enabled: e.target.checked })
                  }
                />
              </label>

              <label className="block space-y-1.5">
                <span className="text-xs text-slate-300">Wake Phrase</span>
                <input
                  type="text"
                  aria-label="Wake Phrase"
                  value={draft.wake_phrase}
                  onChange={(e) =>
                    setDraft({ ...draft, wake_phrase: e.target.value })
                  }
                  className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-slate-100 outline-none focus:border-sky-400/40"
                />
              </label>

              <div className="space-y-1.5">
                <span className="text-xs text-slate-300">
                  Wake Detection Sensitivity
                </span>
                <div className="grid grid-cols-3 gap-2">
                  {(["low", "medium", "high"] as const).map((level) => (
                    <button
                      key={level}
                      type="button"
                      aria-label={`Wake sensitivity ${level}`}
                      onClick={() =>
                        setDraft({ ...draft, wakeword_sensitivity: level })
                      }
                      className={`rounded-xl border py-2 text-xs font-medium capitalize transition ${
                        draft.wakeword_sensitivity === level
                          ? "border-indigo-400/50 bg-indigo-500/20 text-indigo-100"
                          : "border-white/10 bg-white/[0.03] text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <label className="block space-y-1.5">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Base Confidence Threshold</span>
                  <span className="font-mono">
                    {draft.wakeword_threshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.15}
                  max={0.9}
                  step={0.05}
                  value={draft.wakeword_threshold}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_threshold: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </label>

              <label className="block space-y-1.5">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Conversation Timeout (Return to Standby)</span>
                  <span className="font-mono">
                    {Math.round(draft.wakeword_conversation_timeout_s)} s
                  </span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={60}
                  step={1}
                  value={draft.wakeword_conversation_timeout_s}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_conversation_timeout_s: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </label>

              <label className="block space-y-1.5">
                <div className="flex justify-between text-xs text-slate-300">
                  <span>Post-Wake Activation Grace Period</span>
                  <span className="font-mono">
                    {draft.wakeword_activation_grace_ms} ms
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1000}
                  step={50}
                  value={draft.wakeword_activation_grace_ms}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_activation_grace_ms: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </label>

              <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3.5 py-2 text-xs">
                <span className="text-slate-200">
                  Speak Activation Acknowledgment (&ldquo;Yes, I&apos;m
                  listening.&rdquo;)
                </span>
                <input
                  type="checkbox"
                  checked={draft.wakeword_ack_enabled}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_ack_enabled: e.target.checked,
                    })
                  }
                />
              </label>
            </div>
          </div>

          {/* Personalized 3-Sample Voice Enrollment Panel */}
          <div className="space-y-3 rounded-xl border border-cyan-400/20 bg-cyan-500/[0.04] p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2.5">
                <Fingerprint className="h-4 w-4 text-cyan-300" />
                <div>
                  <div className="text-xs font-semibold text-white">
                    Personalized Voice Recognition (3 Samples)
                  </div>
                  <p className="text-[11px] text-slate-400">
                    {voiceProfile?.enrolled
                      ? `Enrolled (${voiceProfile.samples_recorded}/3 samples for "${voiceProfile.wake_phrase}"${
                          typeof voiceProfile.consistency_score === "number"
                            ? ` · ${Math.round(voiceProfile.consistency_score * 100)}% consistency`
                            : ""
                        })`
                      : "Record 3 short voice samples so LocalVox recognizes your voice reliably offline."}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setEnrollOpen(true)}
                  className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-400/40 bg-cyan-500/20 px-3 py-1.5 text-xs font-medium text-cyan-100 transition hover:bg-cyan-500/30"
                >
                  <Fingerprint className="h-3.5 w-3.5" />
                  {voiceProfile?.enrolled
                    ? "Re-record 3 Samples"
                    : "Personalize My Voice (3 Samples)"}
                </button>
                {voiceProfile?.enrolled && (
                  <button
                    type="button"
                    onClick={() => void handleDeleteVoiceProfile()}
                    className="inline-flex items-center gap-1 rounded-xl border border-rose-400/25 bg-rose-500/10 px-2.5 py-1.5 text-xs text-rose-200 transition hover:bg-rose-500/20"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Delete
                  </button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-2 text-xs">
                <span className="text-slate-200">
                  Use Personalized Profile
                </span>
                <input
                  type="checkbox"
                  checked={draft.wakeword_personal_enabled ?? true}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_personal_enabled: e.target.checked,
                    })
                  }
                />
              </label>

              <label className="flex items-center justify-between rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-2 text-xs">
                <span className="text-slate-200">My Voice Only Mode</span>
                <input
                  type="checkbox"
                  aria-label="My Voice Only Mode"
                  checked={draft.wakeword_personal_only ?? false}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_personal_only: e.target.checked,
                    })
                  }
                />
              </label>

              <label className="block space-y-1 rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-2">
                <div className="flex justify-between text-[11px] text-slate-300">
                  <span>Voice Match Threshold</span>
                  <span className="font-mono">
                    {(draft.wakeword_personal_threshold ?? 0.68).toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={0.9}
                  step={0.01}
                  value={draft.wakeword_personal_threshold ?? 0.68}
                  onChange={(e) =>
                    setDraft({
                      ...draft,
                      wakeword_personal_threshold: Number(e.target.value),
                    })
                  }
                  className="w-full"
                />
              </label>
            </div>
          </div>
        </section>
      </div>

      <VoiceEnrollmentModal
        open={enrollOpen}
        wakePhrase={draft.wake_phrase}
        initialProfile={voiceProfile}
        audioLevel={audioLevel}
        onClose={() => setEnrollOpen(false)}
        onProfileChanged={async () => {
          await loadSettings();
          if (onRefreshState) await onRefreshState();
        }}
      />
    </form>
  );
}
