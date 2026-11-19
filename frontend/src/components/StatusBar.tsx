"use client";

import React from "react";
import { Mic, MicOff, ShieldCheck, Wifi, WifiOff } from "lucide-react";
import { STATE_THEMES } from "@/lib/stateTheme";
import type { PipelineState, TurnMetricsSummary } from "@/lib/types";

interface StatusBarProps {
  connected: boolean;
  pipelineState: PipelineState;
  micActive: boolean;
  wakewordActive?: boolean;
  wakePhrase?: string;
  lastMetrics: TurnMetricsSummary | null;
}

export function StatusBar({
  connected,
  pipelineState,
  micActive,
  wakewordActive = false,
  wakePhrase = "Hey Jarvis",
  lastMetrics,
}: StatusBarProps) {
  const theme = STATE_THEMES[pipelineState];

  return (
    <header className="glass-panel flex h-12 shrink-0 items-center justify-between rounded-2xl px-4">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5" aria-hidden="true">
          <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
          <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
          <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
        </div>
        <div className="h-4 w-px bg-white/10" />
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold tracking-tight text-slate-100">
            LocalVox
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/20 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
            <ShieldCheck className="h-3 w-3" />
            100% Offline
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2.5">
        {lastMetrics && (
          <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] text-slate-300 md:flex">
            <span>TTFT {Math.round(lastMetrics.llm_ttft_ms || 0)}ms</span>
            <span className="text-white/20">·</span>
            <span>
              1st Audio{" "}
              {Math.round(
                lastMetrics.first_audio_latency_ms ||
                  lastMetrics.response_latency_ms ||
                  0
              )}
              ms
            </span>
            {typeof lastMetrics.memories_retrieved === "number" &&
              lastMetrics.memories_retrieved > 0 && (
                <>
                  <span className="text-white/20">·</span>
                  <span className="text-violet-300">
                    {lastMetrics.memories_retrieved} mem
                  </span>
                </>
              )}
          </div>
        )}

        <div
          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${
            micActive
              ? "border-sky-400/35 bg-sky-500/15 text-sky-200"
              : "border-white/10 bg-white/[0.03] text-slate-400"
          }`}
          data-testid="mic-monitor-indicator"
        >
          {micActive ? (
            <Mic className="h-3 w-3 animate-pulse text-sky-300" />
          ) : (
            <MicOff className="h-3 w-3 text-slate-500" />
          )}
          <span>
            {micActive
              ? wakewordActive && pipelineState === "SLEEPING"
                ? `Mic Standby · ${wakePhrase}`
                : "Mic Monitoring · Local"
              : "Mic Off"}
          </span>
        </div>

        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition-colors ${theme.pillClass}`}
          data-testid="status-pill"
        >
          <span className={`h-2 w-2 rounded-full ${theme.dotClass}`} />
          <span>{theme.label}</span>
        </div>

        <div
          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${
            connected
              ? "border-white/10 bg-white/[0.04] text-slate-300"
              : "border-rose-400/30 bg-rose-500/15 text-rose-200"
          }`}
        >
          {connected ? (
            <Wifi className="h-3 w-3 text-emerald-400" />
          ) : (
            <WifiOff className="h-3 w-3 text-rose-400" />
          )}
          <span>{connected ? "Live Sync" : "Reconnecting"}</span>
        </div>
      </div>
    </header>
  );
}
