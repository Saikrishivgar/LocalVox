"use client";

import React, { useCallback, useEffect, useState } from "react";
import { X } from "lucide-react";
import { CompanionView } from "@/components/CompanionView";
import { ConversationHistoryView } from "@/components/ConversationHistoryView";
import { MemoryDashboard } from "@/components/MemoryDashboard";
import { SettingsView } from "@/components/SettingsView";
import { SidebarNav } from "@/components/SidebarNav";
import { StatusBar } from "@/components/StatusBar";
import { SystemMonitorView } from "@/components/SystemMonitorView";
import { useCompanionSocket } from "@/hooks/useCompanionSocket";
import { companionApi } from "@/lib/api";
import type { NavTab } from "@/lib/types";

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<NavTab>("companion");
  const [memoryCount, setMemoryCount] = useState<number | null>(null);
  const [ollamaConnected, setOllamaConnected] = useState<boolean | null>(null);

  const live = useCompanionSocket();

  const syncSidebarBadges = useCallback(async () => {
    try {
      const [memRes, sysRes] = await Promise.all([
        companionApi.listMemories(),
        companionApi.getSystemMonitor(),
      ]);
      setMemoryCount(memRes.total_count);
      setOllamaConnected(sysRes.ollama_connected);
    } catch {
      // Backend might be restarting cleanly.
    }
  }, []);

  useEffect(() => {
    void syncSidebarBadges();
  }, [syncSidebarBadges, live.memoryVersion, live.connected]);

  async function handleContinueSession() {
    await live.refreshSnapshot();
    setActiveTab("companion");
  }

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-obsidian-950 p-4 text-slate-100">
      {/* Subtle studio ambient backdrop */}
      <div
        className="pointer-events-none fixed inset-0 opacity-60"
        aria-hidden="true"
        style={{
          background:
            "radial-gradient(circle at 22% 18%, rgba(56, 189, 248, 0.08), transparent 42%), radial-gradient(circle at 78% 82%, rgba(139, 92, 246, 0.08), transparent 45%)",
        }}
      />

      <div className="relative z-10 mx-auto flex h-full w-full max-w-[1440px] flex-col gap-3.5">
        <StatusBar
          connected={live.connected}
          pipelineState={live.pipelineState}
          micActive={live.micActive}
          wakewordActive={live.wakewordActive}
          wakePhrase={live.wakePhrase}
          lastMetrics={live.lastMetrics}
        />

        {live.errorBanner && (
          <div className="flex items-center justify-between rounded-xl border border-rose-400/35 bg-rose-500/15 px-4 py-2 text-xs text-rose-100">
            <span>{live.errorBanner}</span>
            <button
              type="button"
              onClick={live.clearError}
              aria-label="Dismiss notification"
              className="rounded p-1 hover:bg-white/10"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        <div className="flex min-h-0 flex-1 gap-3.5">
          <SidebarNav
            activeTab={activeTab}
            onSelectTab={setActiveTab}
            memoryCount={memoryCount}
            micActive={live.micActive}
            ollamaConnected={ollamaConnected}
            wakewordEnabled={live.wakewordEnabled}
            wakewordActive={live.wakewordActive}
            wakePhrase={live.wakePhrase}
          />

          <main className="min-h-0 min-w-0 flex-1">
            {activeTab === "companion" && (
              <CompanionView
                pipelineState={live.pipelineState}
                micActive={live.micActive}
                messages={live.messages}
                streamingPartial={live.streamingPartial}
                lastMetrics={live.lastMetrics}
                audioLevel={live.audioLevel}
                wakewordEnabled={live.wakewordEnabled}
                wakewordActive={live.wakewordActive}
                wakePhrase={live.wakePhrase}
                lastWakeEvent={live.lastWakeEvent}
                wakewordEngine={live.wakewordEngine}
                voiceProfile={live.voiceProfile}
                onRefreshState={live.refreshSnapshot}
              />
            )}
            {activeTab === "memory" && (
              <MemoryDashboard
                memoryVersion={live.memoryVersion}
                onCountChange={setMemoryCount}
              />
            )}
            {activeTab === "history" && (
              <ConversationHistoryView
                historyVersion={live.historyVersion}
                onContinueSession={handleContinueSession}
              />
            )}
            {activeTab === "system" && <SystemMonitorView />}
            {activeTab === "settings" && (
              <SettingsView
                settingsVersion={live.settingsVersion}
                micActive={live.micActive}
                audioLevel={live.audioLevel}
                onRefreshState={live.refreshSnapshot}
              />
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
