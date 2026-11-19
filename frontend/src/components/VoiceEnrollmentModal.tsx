"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  Fingerprint,
  Mic,
  Radio,
  RotateCcw,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";
import { companionApi } from "@/lib/api";
import type {
  AudioLevelState,
  EnrollmentSampleResult,
  VoiceProfileSummary,
} from "@/lib/types";

interface VoiceEnrollmentModalProps {
  open: boolean;
  wakePhrase: string;
  initialProfile: VoiceProfileSummary | null;
  audioLevel?: AudioLevelState;
  onClose: () => void;
  onProfileChanged: () => Promise<void>;
}

const TOTAL_SAMPLES = 3;

export function VoiceEnrollmentModal({
  open,
  wakePhrase,
  initialProfile,
  audioLevel,
  onClose,
  onProfileChanged,
}: VoiceEnrollmentModalProps) {
  const [profile, setProfile] = useState<VoiceProfileSummary | null>(
    initialProfile
  );
  const [samples, setSamples] = useState<EnrollmentSampleResult[]>([]);
  const [recording, setRecording] = useState(false);
  const [activeStep, setActiveStep] = useState<number>(1);
  const [activating, setActivating] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingMode, setTrainingMode] = useState(false);

  const wasOpenRef = useRef(false);
  const cancelledRef = useRef(false);

  // Only reset local wizard state when the modal transitions from closed -> open.
  // Never wipe captured samples when initialProfile updates mid-enrollment!
  useEffect(() => {
    if (open && !wasOpenRef.current) {
      cancelledRef.current = false;
      setProfile(initialProfile);
      setError(null);
      setStatusMessage(null);
      setSamples([]);
      setActiveStep(1);
      setRecording(false);
      setTrainingMode(!initialProfile?.enrolled);
    } else if (!open && wasOpenRef.current) {
      cancelledRef.current = true;
    }
    wasOpenRef.current = open;
  }, [open, initialProfile]);

  if (!open) return null;

  const currentCount = trainingMode
    ? samples.length
    : profile?.enrolled
      ? profile.samples_recorded
      : samples.length;
  const isComplete = trainingMode
    ? samples.length >= TOTAL_SAMPLES
    : Boolean(profile?.enrolled);
  const nextSampleNum = Math.min(TOTAL_SAMPLES, currentCount + 1);
  const bars = audioLevel?.bars ?? [];

  async function handleStartFresh() {
    cancelledRef.current = false;
    setError(null);
    setStatusMessage(null);
    try {
      const reset = await companionApi.startVoiceEnrollment();
      setProfile(reset);
      setSamples([]);
      setActiveStep(1);
      setTrainingMode(true);
      await onProfileChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRunGuidedRecording() {
    if (recording) return;
    cancelledRef.current = false;
    setError(null);
    setRecording(true);
    setTrainingMode(true);

    let capturedList = [...samples];
    try {
      if (capturedList.length === 0) {
        await companionApi.startVoiceEnrollment();
      }

      while (capturedList.length < TOTAL_SAMPLES && !cancelledRef.current) {
        const stepNum = capturedList.length + 1;
        setActiveStep(stepNum);
        setStatusMessage(
          `Listening for Sample ${stepNum} of ${TOTAL_SAMPLES} — say "${wakePhrase}" now...`
        );

        const result = await companionApi.recordVoiceEnrollmentSample(7.0);
        if (cancelledRef.current) break;

        capturedList = [...capturedList, result];
        setSamples(capturedList);
        setProfile(result.profile);
        setStatusMessage(result.message);

        // Sync snapshot in background without interrupting wizard state.
        void onProfileChanged();

        if (result.completed || capturedList.length >= TOTAL_SAMPLES) {
          setTrainingMode(false);
          // Automatically ensure hands-free wake standby is active right away.
          try {
            await companionApi.enableWakeWord();
            void onProfileChanged();
          } catch {
            // Non-fatal if already active.
          }
          break;
        }

        // Brief natural beat so user sees Sample N turn green before speaking Sample N+1.
        setStatusMessage(
          `Sample ${stepNum} saved! Get ready to say "${wakePhrase}" again...`
        );
        await new Promise((resolve) => setTimeout(resolve, 650));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatusMessage(null);
    } finally {
      setRecording(false);
    }
  }

  async function handleActivateHandsFree() {
    setActivating(true);
    setError(null);
    try {
      await companionApi.enableWakeWord();
      await onProfileChanged();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setActivating(false);
    }
  }

  async function handleDeleteProfile() {
    setError(null);
    try {
      const cleared = await companionApi.deleteVoiceProfile();
      setProfile(cleared);
      setSamples([]);
      setActiveStep(1);
      setTrainingMode(true);
      setStatusMessage("Personalized voice profile removed.");
      await onProfileChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="Personalize Voice Wake Word"
    >
      <div className="glass-panel relative w-full max-w-lg rounded-2xl border border-white/15 bg-obsidian-950/95 p-6 shadow-2xl">
        <button
          type="button"
          onClick={onClose}
          disabled={recording}
          aria-label="Close personalization dialog"
          className="absolute right-4 top-4 rounded-lg p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-white disabled:opacity-40"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/35 bg-cyan-500/15 text-cyan-200">
            <Fingerprint className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">
              Personalize Your Voice
            </h2>
            <p className="text-xs text-slate-400">
              Say &ldquo;{wakePhrase}&rdquo; 3 times so LocalVox learns your
              voice locally
            </p>
          </div>
        </div>

        {/* 3-Step Visual Progress Tracker */}
        <div className="mt-5 grid grid-cols-3 gap-2.5">
          {[1, 2, 3].map((step) => {
            const captured =
              step <= currentCount ||
              (!trainingMode && Boolean(profile?.enrolled));
            const isCurrent =
              recording && step === activeStep && !captured;
            const detail = samples[step - 1];
            return (
              <div
                key={step}
                data-testid={`enroll-step-${step}`}
                className={`flex flex-col items-center rounded-xl border px-3 py-3 text-center transition ${
                  captured
                    ? "border-emerald-400/40 bg-emerald-500/12 text-emerald-100"
                    : isCurrent
                      ? "border-cyan-400/60 bg-cyan-500/15 text-cyan-100 ring-1 ring-cyan-400/30"
                      : "border-white/10 bg-white/[0.03] text-slate-400"
                }`}
              >
                {captured ? (
                  <CheckCircle2 className="h-5 w-5 text-emerald-300" />
                ) : (
                  <div
                    className={`flex h-5 w-5 items-center justify-center rounded-full border text-[11px] font-semibold ${
                      isCurrent
                        ? "animate-pulse border-cyan-300 text-cyan-200"
                        : "border-slate-600 text-slate-400"
                    }`}
                  >
                    {step}
                  </div>
                )}
                <div className="mt-1.5 text-xs font-medium">Sample {step}</div>
                <div className="mt-0.5 font-mono text-[10px] opacity-75">
                  {detail
                    ? `Saved · ${detail.duration_s.toFixed(1)}s`
                    : captured
                      ? "Saved"
                      : isCurrent
                        ? "Listening..."
                        : "Pending"}
                </div>
              </div>
            );
          })}
        </div>

        {/* Live prompt + mic bar feedback */}
        <div className="mt-4 rounded-xl border border-white/[0.08] bg-white/[0.025] p-4 text-center">
          {isComplete && !trainingMode ? (
            <div className="space-y-1">
              <div className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/35 bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-200">
                <CheckCircle2 className="h-3.5 w-3.5" />
                Personalized Voice Active ({profile?.samples_recorded ?? 3}/3)
              </div>
              <p className="text-xs text-slate-300">
                LocalVox is tuned to recognize your voice for &ldquo;
                {profile?.wake_phrase || wakePhrase}&rdquo;
                {typeof profile?.consistency_score === "number"
                  ? ` · ${Math.round(profile.consistency_score * 100)}% voice match consistency`
                  : ""}
                .
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wider text-slate-400">
                {recording
                  ? `Listening — Sample ${activeStep} of ${TOTAL_SAMPLES}`
                  : `Ready for Sample ${nextSampleNum} of ${TOTAL_SAMPLES}`}
              </p>
              <p className="text-lg font-semibold text-white">
                &ldquo;{wakePhrase}&rdquo;
              </p>
              {recording && (
                <div
                  className="mx-auto flex h-7 max-w-[220px] items-center justify-center gap-1 pt-1"
                  aria-label="Live recording microphone level"
                >
                  {(bars.length
                    ? bars.slice(0, 16)
                    : new Array(16).fill(0.1)
                  ).map((val, idx) => (
                    <span
                      key={idx}
                      className="w-1 rounded-full bg-cyan-400/90 transition-all duration-75"
                      style={{
                        height: `${Math.max(4, Math.min(26, Math.round(val * 26)))}px`,
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {statusMessage && (
          <div className="mt-3 rounded-xl border border-sky-400/25 bg-sky-500/10 px-3.5 py-2 text-xs text-sky-100">
            {statusMessage}
          </div>
        )}

        {error && (
          <div className="mt-3 rounded-xl border border-rose-400/35 bg-rose-500/15 px-3.5 py-2 text-xs text-rose-200">
            {error}
          </div>
        )}

        {/* Action buttons */}
        <div className="mt-5 flex flex-wrap items-center justify-between gap-2.5">
          {!isComplete || trainingMode ? (
            <button
              type="button"
              onClick={() => void handleRunGuidedRecording()}
              disabled={recording}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl border border-cyan-400/45 bg-cyan-500/20 px-4 py-2.5 text-sm font-medium text-cyan-100 transition hover:bg-cyan-500/30 disabled:cursor-wait disabled:opacity-70"
            >
              <Mic className={`h-4 w-4 ${recording ? "animate-pulse" : ""}`} />
              {recording
                ? `Listening (${activeStep}/${TOTAL_SAMPLES})... Say "${wakePhrase}"`
                : currentCount > 0
                  ? `Continue from Sample ${nextSampleNum} of ${TOTAL_SAMPLES}`
                  : `Record Sample 1 of ${TOTAL_SAMPLES}`}
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={() => void handleActivateHandsFree()}
                disabled={activating}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-sky-500/90 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:opacity-60"
              >
                <Radio className="h-4 w-4" />
                {activating
                  ? "Starting Standby..."
                  : `Done — Say "${wakePhrase}"`}
              </button>
              <button
                type="button"
                onClick={() => void handleStartFresh()}
                className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2.5 text-xs text-slate-300 transition hover:bg-white/[0.09] hover:text-white"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Re-record
              </button>
              <button
                type="button"
                onClick={() => void handleDeleteProfile()}
                aria-label="Delete personalized voice profile"
                className="inline-flex items-center gap-1.5 rounded-xl border border-rose-400/25 bg-rose-500/10 px-3 py-2.5 text-xs text-rose-200 transition hover:bg-rose-500/20"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </button>
            </>
          )}
        </div>

        <div className="mt-4 flex items-center gap-2 border-t border-white/[0.07] pt-3 text-[11px] text-slate-400">
          <ShieldCheck className="h-3.5 w-3.5 shrink-0 text-emerald-400/80" />
          <span>
            100% offline · Only mathematical voice embeddings are saved
            (voice_profile.npz); raw microphone audio is immediately discarded.
          </span>
        </div>
      </div>
    </div>
  );
}
