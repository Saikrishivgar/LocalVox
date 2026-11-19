"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  Check,
  Edit3,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { companionApi } from "@/lib/api";
import {
  CATEGORY_LABELS,
  MEMORY_CATEGORIES,
  type MemoryCategory,
  type MemoryListResponse,
  type MemoryRecord,
} from "@/lib/types";

interface MemoryDashboardProps {
  memoryVersion: number;
  onCountChange: (count: number) => void;
}

export function MemoryDashboard({
  memoryVersion,
  onCountChange,
}: MemoryDashboardProps) {
  const [data, setData] = useState<MemoryListResponse | null>(null);
  const [query, setQuery] = useState("");
  const [selectedCategory, setSelectedCategory] =
    useState<MemoryCategory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [editing, setEditing] = useState<MemoryRecord | null>(null);
  const [creating, setCreating] = useState(false);
  const [formCategory, setFormCategory] =
    useState<MemoryCategory>("personal_preference");
  const [formSubject, setFormSubject] = useState("");
  const [formContent, setFormContent] = useState("");
  const [formKeywords, setFormKeywords] = useState("");
  const [formImportance, setFormImportance] = useState(0.8);

  const [showClearModal, setShowClearModal] = useState(false);
  const [confirmPhrase, setConfirmPhrase] = useState("");

  const fetchMemories = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await companionApi.listMemories(query, selectedCategory);
      setData(res);
      onCountChange(res.total_count);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [query, selectedCategory, onCountChange]);

  useEffect(() => {
    const t = setTimeout(() => {
      void fetchMemories();
    }, 150);
    return () => clearTimeout(t);
  }, [fetchMemories, memoryVersion]);

  function openCreate() {
    setEditing(null);
    setFormCategory("personal_preference");
    setFormSubject("");
    setFormContent("");
    setFormKeywords("");
    setFormImportance(0.8);
    setCreating(true);
  }

  function openEdit(item: MemoryRecord) {
    setCreating(false);
    setEditing(item);
    setFormCategory(item.category);
    setFormSubject(item.subject);
    setFormContent(item.content);
    setFormKeywords(item.keywords.join(", "));
    setFormImportance(item.importance);
  }

  async function handleSaveForm(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const keywords = formKeywords
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    try {
      if (editing) {
        await companionApi.updateMemory(editing.id, {
          category: formCategory,
          subject: formSubject,
          content: formContent,
          keywords,
          importance: formImportance,
        });
        setEditing(null);
      } else {
        await companionApi.createMemory({
          category: formCategory,
          subject: formSubject,
          content: formContent,
          keywords,
          importance: formImportance,
        });
        setCreating(false);
      }
      await fetchMemories();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDelete(id: string) {
    setError(null);
    try {
      await companionApi.deleteMemory(id);
      await fetchMemories();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleClearAll(e: React.FormEvent) {
    e.preventDefault();
    if (confirmPhrase !== "CLEAR_ALL_MEMORIES") return;
    setError(null);
    try {
      await companionApi.clearAllMemories(confirmPhrase);
      setShowClearModal(false);
      setConfirmPhrase("");
      await fetchMemories();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      {/* Header & Search */}
      <div className="glass-panel flex flex-col gap-4 rounded-2xl p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-white">
              Persistent SQLite Memory
            </h2>
            <p className="text-xs text-slate-400">
              {data ? data.total_count : 0} stored memories across 8 categories ·
              FTS5 indexed locally
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={openCreate}
              className="inline-flex items-center gap-1.5 rounded-xl bg-sky-500/90 px-3.5 py-2 text-xs font-medium text-slate-950 transition hover:bg-sky-400"
            >
              <Plus className="h-3.5 w-3.5" />
              Add Memory
            </button>
            <button
              type="button"
              onClick={() => {
                setConfirmPhrase("");
                setShowClearModal(true);
              }}
              disabled={!data || data.total_count === 0}
              className="inline-flex items-center gap-1.5 rounded-xl border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-xs font-medium text-rose-200 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear All
            </button>
          </div>
        </div>

        <div className="relative">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search memories via SQLite FTS5..."
            className="w-full rounded-xl border border-white/10 bg-white/[0.04] py-2.5 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-sky-400/40"
          />
        </div>

        {/* Category Filter Pills */}
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setSelectedCategory(null)}
            className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
              selectedCategory === null
                ? "border-sky-400/40 bg-sky-500/20 text-sky-100"
                : "border-white/10 bg-white/[0.03] text-slate-400 hover:text-slate-200"
            }`}
          >
            All ({data ? data.total_count : 0})
          </button>
          {MEMORY_CATEGORIES.map((cat) => {
            const count = data?.category_counts?.[cat] ?? 0;
            const active = selectedCategory === cat;
            return (
              <button
                key={cat}
                type="button"
                onClick={() =>
                  setSelectedCategory(active ? null : cat)
                }
                className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                  active
                    ? "border-violet-400/40 bg-violet-500/20 text-violet-100"
                    : "border-white/10 bg-white/[0.03] text-slate-400 hover:text-slate-200"
                }`}
              >
                {CATEGORY_LABELS[cat]} ({count})
              </button>
            );
          })}
        </div>

        {error && (
          <div className="rounded-xl border border-rose-400/30 bg-rose-500/15 px-3.5 py-2 text-xs text-rose-200">
            {error}
          </div>
        )}
      </div>

      {/* Create / Edit Modal Drawer */}
      {(creating || editing) && (
        <form
          onSubmit={(e) => void handleSaveForm(e)}
          className="glass-panel space-y-3 rounded-2xl border border-sky-400/30 p-5"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">
              {editing ? "Edit Memory" : "Store New Memory"}
            </h3>
            <button
              type="button"
              onClick={() => {
                setCreating(false);
                setEditing(null);
              }}
              className="rounded-lg p-1 text-slate-400 hover:bg-white/10 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <select
              value={formCategory}
              onChange={(e) =>
                setFormCategory(e.target.value as MemoryCategory)
              }
              className="rounded-xl border border-white/10 bg-obsidian-900 px-3 py-2 text-xs text-slate-100 outline-none"
            >
              {MEMORY_CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {CATEGORY_LABELS[cat]} ({cat})
                </option>
              ))}
            </select>
            <input
              type="text"
              value={formSubject}
              onChange={(e) => setFormSubject(e.target.value)}
              placeholder="Subject key (e.g. voice, python)"
              className="rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none"
            />
            <input
              type="text"
              value={formKeywords}
              onChange={(e) => setFormKeywords(e.target.value)}
              placeholder="Keywords (comma separated)"
              className="rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none"
            />
          </div>
          <textarea
            required
            rows={2}
            value={formContent}
            onChange={(e) => setFormContent(e.target.value)}
            placeholder="Memory fact or preference..."
            className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none"
          />
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-3 text-xs text-slate-300">
              <span>Importance: {formImportance.toFixed(2)}</span>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={formImportance}
                onChange={(e) => setFormImportance(Number(e.target.value))}
              />
            </label>
            <button
              type="submit"
              className="inline-flex items-center gap-1.5 rounded-xl bg-sky-500/90 px-4 py-1.5 text-xs font-medium text-slate-950 hover:bg-sky-400"
            >
              <Check className="h-3.5 w-3.5" />
              Save Memory
            </button>
          </div>
        </form>
      )}

      {/* Memory Records List */}
      <div className="glass-panel flex-1 overflow-y-auto rounded-2xl p-5">
        {loading && !data ? (
          <p className="text-sm text-slate-400">Loading memories...</p>
        ) : !data || data.items.length === 0 ? (
          <div className="flex h-48 flex-col items-center justify-center text-center">
            <p className="text-sm font-medium text-slate-300">
              No matching memories in SQLite
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Talk naturally or say &ldquo;Remember that...&rdquo; to build
              persistent local knowledge.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {data.items.map((item) => {
              const score = data.match_scores[item.id];
              return (
                <div
                  key={item.id}
                  className="glass-subtle flex flex-col justify-between rounded-xl p-4 transition hover:border-white/20"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="rounded-full border border-violet-400/30 bg-violet-500/15 px-2.5 py-0.5 text-[11px] font-medium text-violet-200">
                          {CATEGORY_LABELS[item.category]}
                        </span>
                        {item.subject && (
                          <span className="rounded-full bg-white/[0.06] px-2 py-0.5 font-mono text-[11px] text-slate-300">
                            {item.subject}
                          </span>
                        )}
                        {score !== undefined && (
                          <span className="rounded-full bg-sky-500/20 px-2 py-0.5 font-mono text-[10px] text-sky-200">
                            FTS {score.toFixed(2)}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => openEdit(item)}
                          aria-label={`Edit memory ${item.subject || item.id}`}
                          className="rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white"
                        >
                          <Edit3 className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleDelete(item.id)}
                          aria-label={`Delete memory ${item.subject || item.id}`}
                          className="rounded-lg p-1.5 text-slate-400 hover:bg-rose-500/20 hover:text-rose-200"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                    <p className="mt-2.5 text-sm leading-relaxed text-slate-100">
                      {item.content}
                    </p>
                  </div>

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-2.5 font-mono text-[10px] text-slate-400">
                    <span>
                      imp {item.importance.toFixed(2)} · {item.source} ·{" "}
                      {item.access_count} hits
                    </span>
                    <span>
                      {new Date(item.updated_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Confirmed Bulk Wipe Modal */}
      {showClearModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <form
            onSubmit={(e) => void handleClearAll(e)}
            className="glass-panel w-full max-w-md space-y-4 rounded-2xl border border-rose-400/35 p-6"
          >
            <div className="flex items-center gap-2.5 text-rose-300">
              <AlertTriangle className="h-5 w-5" />
              <h3 className="text-base font-semibold">
                Confirm Permanent Memory Wipe
              </h3>
            </div>
            <p className="text-xs leading-relaxed text-slate-300">
              This will permanently delete all {data?.total_count ?? 0} stored
              SQLite memories and FTS5 index entries. Type{" "}
              <code className="rounded bg-white/10 px-1.5 py-0.5 text-rose-200">
                CLEAR_ALL_MEMORIES
              </code>{" "}
              below to confirm.
            </p>
            <input
              type="text"
              value={confirmPhrase}
              onChange={(e) => setConfirmPhrase(e.target.value)}
              placeholder="CLEAR_ALL_MEMORIES"
              className="w-full rounded-xl border border-white/15 bg-white/[0.05] px-3.5 py-2 font-mono text-sm text-white outline-none focus:border-rose-400"
            />
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowClearModal(false)}
                className="rounded-xl border border-white/10 px-4 py-2 text-xs text-slate-300 hover:bg-white/[0.06]"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={confirmPhrase !== "CLEAR_ALL_MEMORIES"}
                className="rounded-xl bg-rose-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-rose-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Wipe All Memories
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
