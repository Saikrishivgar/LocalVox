"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Cpu,
  Database,
  HardDrive,
  RefreshCw,
  ServerCrash,
} from "lucide-react";
import { companionApi } from "@/lib/api";
import type { EngineStatus, SystemMonitorResponse } from "@/lib/types";

const STATUS_STYLES: Record<EngineStatus["status"], string> = {
  loaded: "border-emerald-400/35 bg-emerald-500/15 text-emerald-200",
  ready: "border-sky-400/35 bg-sky-500/15 text-sky-200",
  offline: "border-amber-400/35 bg-amber-500/15 text-amber-200",
  missing: "border-rose-400/35 bg-rose-500/15 text-rose-200",
  error: "border-rose-400/40 bg-rose-500/20 text-rose-100",
};

export function SystemMonitorView() {
  const [data, setData] = useState<SystemMonitorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSystem = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await companionApi.getSystemMonitor();
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchSystem();
    const timer = setInterval(() => {
      void fetchSystem();
    }, 5000);
    return () => clearInterval(timer);
  }, [fetchSystem]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto pr-1">
      {/* Header & Host Resources */}
      <div className="glass-panel rounded-2xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-white">
              Local System Monitor
            </h2>
            <p className="text-xs text-slate-400">
              {data ? data.hardware : "Inspecting local Apple Silicon host..."}
            </p>
          </div>
          <button
            type="button"
            onClick={() => void fetchSystem()}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-white/[0.1]"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
            />
            Refresh Telemetry
          </button>
        </div>

        {error && (
          <div className="mt-3 rounded-xl border border-rose-400/30 bg-rose-500/15 px-3.5 py-2 text-xs text-rose-200">
            {error}
          </div>
        )}

        {data && (
          <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="glass-subtle rounded-xl p-3.5">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Cpu className="h-3.5 w-3.5 text-sky-400" />
                <span>Host CPU</span>
              </div>
              <div className="mt-1.5 font-mono text-xl font-semibold text-white">
                {data.cpu_percent.toFixed(1)}%
              </div>
              <div className="mt-1 text-[11px] text-slate-500">
                Pipeline {data.pipeline_state}
              </div>
            </div>

            <div className="glass-subtle rounded-xl p-3.5">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <HardDrive className="h-3.5 w-3.5 text-violet-400" />
                <span>Unified RAM</span>
              </div>
              <div className="mt-1.5 font-mono text-xl font-semibold text-white">
                {data.ram_used_gb.toFixed(1)} / {data.ram_total_gb.toFixed(1)} GB
              </div>
              <div className="mt-1 text-[11px] text-slate-500">
                {data.ram_percent.toFixed(1)}% utilized
              </div>
            </div>

            <div className="glass-subtle rounded-xl p-3.5">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Activity className="h-3.5 w-3.5 text-emerald-400" />
                <span>Backend Process RSS</span>
              </div>
              <div className="mt-1.5 font-mono text-xl font-semibold text-white">
                {data.process_rss_mb.toFixed(1)} MiB
              </div>
              <div className="mt-1 text-[11px] text-slate-500">
                Extraction queue: {data.extraction_worker.queue_size}
              </div>
            </div>

            <div className="glass-subtle rounded-xl p-3.5">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Database className="h-3.5 w-3.5 text-amber-400" />
                <span>SQLite Integrity</span>
              </div>
              <div className="mt-1.5 flex items-center gap-1.5 font-mono text-xl font-semibold text-white">
                {data.sqlite_integrity.ok ? (
                  <>
                    <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                    <span>Healthy</span>
                  </>
                ) : (
                  <>
                    <ServerCrash className="h-5 w-5 text-rose-400" />
                    <span>Error</span>
                  </>
                )}
              </div>
              <div className="mt-1 text-[11px] text-slate-500">
                {data.sqlite_size_mb !== null
                  ? `${data.sqlite_size_mb.toFixed(2)} MiB on disk`
                  : "Local WAL"}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Local AI Engines Grid */}
      {data && (
        <div className="glass-panel rounded-2xl p-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-white">
              Local Engine Readiness
            </h3>
            <span className="font-mono text-[11px] text-slate-400">
              Ollama models:{" "}
              {data.available_ollama_models.length > 0
                ? data.available_ollama_models.join(", ")
                : "none reachable"}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {data.engines.map((eng) => (
              <div
                key={eng.name}
                className="glass-subtle flex flex-col justify-between rounded-xl p-4"
              >
                <div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-medium text-white">
                      {eng.name}
                    </span>
                    <span
                      className={`rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase ${STATUS_STYLES[eng.status]}`}
                    >
                      {eng.status}
                    </span>
                  </div>
                  {eng.model_name && (
                    <div className="mt-1 font-mono text-[11px] text-slate-400">
                      {eng.model_name}
                    </div>
                  )}
                  <p className="mt-2 text-xs text-slate-300">{eng.detail}</p>
                </div>

                <div className="mt-3 flex items-center justify-between border-t border-white/[0.06] pt-2 font-mono text-[10px] text-slate-500">
                  <span>
                    {eng.disk_size_mb !== null
                      ? `Disk ${eng.disk_size_mb.toFixed(1)} MiB`
                      : "Streaming/Daemon"}
                  </span>
                  <span>
                    {eng.memory_mb !== null
                      ? `Active ${eng.memory_mb.toFixed(1)} MiB`
                      : eng.loaded
                        ? "In Memory"
                        : "Lazy Load"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Extraction Worker & Recent Latency Log */}
      {data && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="glass-panel rounded-2xl p-5 lg:col-span-5">
            <h3 className="text-sm font-semibold text-white">
              Background Memory Worker
            </h3>
            <div className="mt-3 space-y-2 text-xs text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-400">Completed Extractions</span>
                <span className="font-mono">
                  {data.extraction_worker.completed_jobs}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Worker Status</span>
                <span className="font-mono">
                  {data.extraction_worker.idle ? "Idle" : "Extracting"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Error Count</span>
                <span className="font-mono">
                  {data.extraction_worker.error_count}
                </span>
              </div>
              {data.extraction_worker.last_outcome && (
                <div className="mt-2 rounded-xl bg-white/[0.03] p-3 font-mono text-[11px] text-slate-300">
                  Last pass:{" "}
                  {data.extraction_worker.last_outcome.candidates_parsed}{" "}
                  candidates (
                  {data.extraction_worker.last_outcome.extraction_ms} ms)
                </div>
              )}
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-5 lg:col-span-7">
            <h3 className="text-sm font-semibold text-white">
              Recent Local Stage Latencies
            </h3>
            {data.recent_latencies.length === 0 ? (
              <p className="mt-3 text-xs text-slate-500">
                No entries recorded in logs/latency.jsonl yet.
              </p>
            ) : (
              <div className="mt-3 max-h-48 space-y-1.5 overflow-y-auto pr-1 font-mono text-[11px]">
                {data.recent_latencies.slice(0, 12).map((row, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between rounded-lg bg-white/[0.03] px-3 py-1.5 text-slate-300"
                  >
                    <span>{String(row.stage ?? "stage")}</span>
                    <span className="text-sky-300">
                      {typeof row.latency_ms === "number"
                        ? `${row.latency_ms.toFixed(1)} ms`
                        : "recorded"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
