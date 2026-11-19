"use client";

import React from "react";
import {
  Brain,
  Cpu,
  History,
  Mic,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import type { NavTab } from "@/lib/types";

interface SidebarNavProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  memoryCount: number | null;
  micActive: boolean;
  ollamaConnected: boolean | null;
  wakewordEnabled?: boolean;
  wakewordActive?: boolean;
  wakePhrase?: string;
}

const NAV_ITEMS: Array<{
  id: NavTab;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  { id: "companion", label: "Companion", icon: Sparkles },
  { id: "memory", label: "Memory", icon: Brain },
  { id: "history", label: "History", icon: History },
  { id: "system", label: "System", icon: Cpu },
  { id: "settings", label: "Settings", icon: SlidersHorizontal },
];

export function SidebarNav({
  activeTab,
  onSelectTab,
  memoryCount,
  micActive,
  ollamaConnected,
  wakewordEnabled = true,
  wakewordActive = false,
  wakePhrase = "Hey Jarvis",
}: SidebarNavProps) {
  return (
    <aside className="glass-panel flex w-56 shrink-0 flex-col justify-between rounded-2xl p-3">
      <div className="space-y-1.5">
        <div className="px-2.5 pb-2 pt-1 text-[11px] font-medium uppercase tracking-widest text-slate-500">
          Workspace
        </div>
        <nav className="space-y-1" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-sm font-medium transition-all ${
                  active
                    ? "bg-white/[0.11] text-white shadow-sm ring-1 ring-white/15"
                    : "text-slate-400 hover:bg-white/[0.05] hover:text-slate-200"
                }`}
              >
                <span className="flex items-center gap-2.5">
                  <Icon
                    className={`h-4 w-4 ${active ? "text-sky-300" : "text-slate-400"}`}
                  />
                  <span>{item.label}</span>
                </span>
                {item.id === "memory" && memoryCount !== null && (
                  <span className="rounded-full bg-white/[0.08] px-2 py-0.5 font-mono text-[11px] text-slate-300">
                    {memoryCount}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      <div className="glass-subtle space-y-2.5 rounded-xl p-3">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Local Engine</span>
          <span className="inline-flex items-center gap-1.5 font-medium text-slate-200">
            <span
              className={`h-2 w-2 rounded-full ${
                ollamaConnected === null
                  ? "bg-slate-400"
                  : ollamaConnected
                    ? "bg-emerald-400"
                    : "bg-amber-400"
              }`}
            />
            {ollamaConnected === null
              ? "Checking"
              : ollamaConnected
                ? "Ollama Ready"
                : "Ollama Offline"}
          </span>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Audio Input</span>
          <span className="inline-flex items-center gap-1 text-slate-300">
            <Mic
              className={`h-3 w-3 ${micActive ? "text-sky-400" : "text-slate-500"}`}
            />
            {micActive ? "Monitoring" : "Off"}
          </span>
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Wake Word</span>
          <span className="truncate font-mono text-[11px] text-slate-300">
            {wakewordActive
              ? `${wakePhrase} · On`
              : wakewordEnabled
                ? wakePhrase
                : "Disabled"}
          </span>
        </div>
      </div>
    </aside>
  );
}
