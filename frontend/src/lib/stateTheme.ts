import type { PipelineState } from "./types";

export interface StateTheme {
  label: string;
  subtitle: string;
  dotClass: string;
  pillClass: string;
  accentHex: string;
  secondaryHex: string;
  glowRgba: string;
}

export const STATE_THEMES: Record<PipelineState, StateTheme> = {
  IDLE: {
    label: "Ready",
    subtitle: "Offline · Awaiting voice or text input",
    dotClass: "bg-slate-400",
    pillClass: "border-slate-500/30 bg-slate-500/10 text-slate-200",
    accentHex: "#7c94b8",
    secondaryHex: "#475569",
    glowRgba: "rgba(100, 128, 175, 0.20)",
  },
  SLEEPING: {
    label: "Standby",
    subtitle: 'Local wake monitor active · Say "Hey Jarvis"',
    dotClass: "bg-indigo-400/90 animate-pulse",
    pillClass: "border-indigo-400/35 bg-indigo-500/12 text-indigo-200",
    accentHex: "#6366f1",
    secondaryHex: "#312e81",
    glowRgba: "rgba(99, 102, 241, 0.22)",
  },
  WAKE_DETECTED: {
    label: "Wake Detected",
    subtitle: "Hey Jarvis recognized · Activating conversation",
    dotClass: "bg-cyan-300 animate-ping",
    pillClass: "border-cyan-300/50 bg-cyan-400/20 text-cyan-100",
    accentHex: "#22d3ee",
    secondaryHex: "#0284c7",
    glowRgba: "rgba(34, 211, 238, 0.44)",
  },
  LISTENING: {
    label: "Listening",
    subtitle: "Silero VAD active · Speak naturally",
    dotClass: "bg-sky-400 animate-pulse",
    pillClass: "border-sky-400/40 bg-sky-500/15 text-sky-200",
    accentHex: "#38bdf8",
    secondaryHex: "#0284c7",
    glowRgba: "rgba(56, 189, 248, 0.30)",
  },
  PROCESSING: {
    label: "Thinking",
    subtitle: "Transcribing speech & retrieving local memory",
    dotClass: "bg-violet-400 animate-pulse",
    pillClass: "border-violet-400/40 bg-violet-500/15 text-violet-200",
    accentHex: "#a78bfa",
    secondaryHex: "#7c3aed",
    glowRgba: "rgba(167, 139, 250, 0.30)",
  },
  GENERATING: {
    label: "Thinking",
    subtitle: "Streaming Qwen3:8B response locally",
    dotClass: "bg-indigo-400 animate-pulse",
    pillClass: "border-indigo-400/40 bg-indigo-500/15 text-indigo-200",
    accentHex: "#818cf8",
    secondaryHex: "#4f46e5",
    glowRgba: "rgba(129, 140, 248, 0.32)",
  },
  SPEAKING: {
    label: "Speaking",
    subtitle: "Streaming neural speech · Interrupt anytime",
    dotClass: "bg-emerald-400 animate-pulse",
    pillClass: "border-emerald-400/40 bg-emerald-500/15 text-emerald-200",
    accentHex: "#34d399",
    secondaryHex: "#0d9488",
    glowRgba: "rgba(52, 211, 153, 0.30)",
  },
  INTERRUPTED: {
    label: "Interrupted",
    subtitle: "Barge-in detected · Capturing your follow-up",
    dotClass: "bg-amber-400 animate-pulse",
    pillClass: "border-amber-400/40 bg-amber-500/15 text-amber-200",
    accentHex: "#fbbf24",
    secondaryHex: "#d97706",
    glowRgba: "rgba(251, 191, 36, 0.28)",
  },
  RETURNING_TO_STANDBY: {
    label: "Returning to Standby",
    subtitle: "Conversation idle · Resuming local wake monitor",
    dotClass: "bg-slate-300",
    pillClass: "border-slate-400/35 bg-slate-500/15 text-slate-200",
    accentHex: "#818cf8",
    secondaryHex: "#334155",
    glowRgba: "rgba(129, 140, 248, 0.22)",
  },
  ERROR: {
    label: "Error",
    subtitle: "Check System Monitor for engine status",
    dotClass: "bg-rose-400",
    pillClass: "border-rose-400/40 bg-rose-500/15 text-rose-200",
    accentHex: "#fb7185",
    secondaryHex: "#e11d48",
    glowRgba: "rgba(251, 113, 133, 0.26)",
  },
  STOPPING: {
    label: "Stopping",
    subtitle: "Closing audio streams safely",
    dotClass: "bg-slate-400",
    pillClass: "border-slate-500/30 bg-slate-500/10 text-slate-300",
    accentHex: "#64748b",
    secondaryHex: "#334155",
    glowRgba: "rgba(100, 116, 139, 0.18)",
  },
};
