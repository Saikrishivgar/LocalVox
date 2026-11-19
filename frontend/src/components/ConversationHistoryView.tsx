"use client";

import React, { useCallback, useEffect, useState } from "react";
import { ArrowRight, MessageSquare, Search, Trash2 } from "lucide-react";
import { companionApi } from "@/lib/api";
import type {
  ConversationSessionDetail,
  ConversationSessionSummary,
} from "@/lib/types";

interface ConversationHistoryViewProps {
  historyVersion: number;
  onContinueSession: () => Promise<void>;
}

export function ConversationHistoryView({
  historyVersion,
  onContinueSession,
}: ConversationHistoryViewProps) {
  const [sessions, setSessions] = useState<ConversationSessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationSessionDetail | null>(null);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  const loadList = useCallback(async () => {
    try {
      const res = await companionApi.listConversations(query);
      setSessions(res.items);
      setActiveSessionId(res.active_session_id);
      if (res.items.length > 0 && !selectedId) {
        setSelectedId(res.items[0].id);
      } else if (
        selectedId &&
        !res.items.some((s) => s.id === selectedId)
      ) {
        setSelectedId(res.items[0]?.id ?? null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [query, selectedId]);

  useEffect(() => {
    const t = setTimeout(() => {
      void loadList();
    }, 150);
    return () => clearTimeout(t);
  }, [loadList, historyVersion]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    companionApi
      .getConversation(selectedId)
      .then((res) => {
        if (!cancelled) setDetail(res);
      })
      .catch(() => {
        if (!cancelled) setDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  async function handleContinue() {
    if (!selectedId) return;
    setError(null);
    try {
      await companionApi.continueConversation(selectedId);
      await onContinueSession();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await companionApi.deleteConversation(id);
      if (selectedId === id) {
        setSelectedId(null);
        setDetail(null);
      }
      await loadList();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-12">
      {/* Sessions List */}
      <section className="glass-panel flex min-h-0 flex-col rounded-2xl p-4 lg:col-span-5">
        <div className="mb-3">
          <h2 className="text-base font-semibold text-white">
            Conversation History
          </h2>
          <p className="text-xs text-slate-400">
            Persisted locally in SQLite · Zero fabricated entries
          </p>
        </div>

        <div className="relative mb-3">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search saved transcripts..."
            className="w-full rounded-xl border border-white/10 bg-white/[0.04] py-2 pl-9 pr-3 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-sky-400/40"
          />
        </div>

        {error && (
          <div className="mb-2 rounded-xl border border-rose-400/30 bg-rose-500/15 px-3 py-2 text-xs text-rose-200">
            {error}
          </div>
        )}

        <div className="flex-1 space-y-2 overflow-y-auto pr-1">
          {sessions.length === 0 ? (
            <div className="flex h-40 flex-col items-center justify-center text-center">
              <MessageSquare className="mb-2 h-6 w-6 text-slate-600" />
              <p className="text-sm text-slate-300">
                No conversations recorded yet
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Real voice or text sessions appear here automatically.
              </p>
            </div>
          ) : (
            sessions.map((s) => {
              const selected = s.id === selectedId;
              const isActive = s.id === activeSessionId;
              return (
                <div
                  key={s.id}
                  onClick={() => setSelectedId(s.id)}
                  className={`cursor-pointer rounded-xl border p-3.5 transition ${
                    selected
                      ? "border-sky-400/40 bg-sky-500/10"
                      : "border-white/[0.07] bg-white/[0.025] hover:bg-white/[0.05]"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="line-clamp-1 text-sm font-medium text-white">
                      {s.title}
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleDelete(s.id);
                      }}
                      aria-label={`Delete conversation ${s.title}`}
                      className="rounded p-1 text-slate-500 hover:bg-rose-500/20 hover:text-rose-200"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  {s.preview && (
                    <p className="mt-1 line-clamp-2 text-xs text-slate-400">
                      {s.preview}
                    </p>
                  )}
                  <div className="mt-2 flex items-center justify-between font-mono text-[10px] text-slate-500">
                    <span>
                      {s.turn_count} turns
                      {isActive ? " · Active" : ""}
                    </span>
                    <span>{new Date(s.updated_at).toLocaleString()}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </section>

      {/* Transcript Inspector */}
      <section className="glass-panel flex min-h-0 flex-col rounded-2xl p-5 lg:col-span-7">
        {detail ? (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] pb-4">
              <div>
                <h3 className="text-sm font-semibold text-white">
                  {detail.session.title}
                </h3>
                <p className="mt-0.5 font-mono text-[11px] text-slate-400">
                  Session {detail.session.id} · {detail.messages.length}{" "}
                  messages
                </p>
              </div>
              <button
                type="button"
                onClick={() => void handleContinue()}
                className="inline-flex items-center gap-1.5 rounded-xl bg-sky-500/90 px-4 py-2 text-xs font-medium text-slate-950 transition hover:bg-sky-400"
              >
                Continue Conversation
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="mt-4 flex-1 space-y-3 overflow-y-auto pr-1">
              {detail.messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex flex-col ${
                    m.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                      m.role === "user"
                        ? "border border-sky-400/25 bg-sky-500/15 text-slate-100"
                        : "border border-white/10 bg-white/[0.045] text-slate-200"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{m.content}</p>
                    <div className="mt-1.5 font-mono text-[10px] text-slate-500">
                      {new Date(m.created_at).toLocaleTimeString()}
                      {m.language ? ` · ${m.language.toUpperCase()}` : ""}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-slate-500">
            Select a persisted conversation to inspect its transcript.
          </div>
        )}
      </section>
    </div>
  );
}
