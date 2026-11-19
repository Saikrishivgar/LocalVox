"use client";

import React from "react";
import { STATE_THEMES } from "@/lib/stateTheme";
import type { AudioLevelState, PipelineState } from "@/lib/types";

interface AudioWaveformProps {
  audioLevel: AudioLevelState;
  pipelineState: PipelineState;
}

export function AudioWaveform({
  audioLevel,
  pipelineState,
}: AudioWaveformProps) {
  const theme = STATE_THEMES[pipelineState];
  const bars =
    audioLevel.bars && audioLevel.bars.length > 0
      ? audioLevel.bars
      : new Array<number>(24).fill(0);

  const isActive = audioLevel.rms > 0.004;

  return (
    <div
      className="flex flex-col items-center gap-1.5"
      aria-label={`Live ${audioLevel.source} waveform`}
    >
      <div className="flex h-10 items-center justify-center gap-1 px-3">
        {bars.map((val, idx) => {
          const clamped = Math.min(1, Math.max(0.06, val));
          const heightPx = Math.round(clamped * 34);
          return (
            <span
              key={idx}
              className="w-1 rounded-full transition-all duration-75"
              style={{
                height: `${isActive ? Math.max(4, heightPx) : 3}px`,
                backgroundColor: isActive ? theme.accentHex : "rgba(148, 163, 184, 0.25)",
                opacity: isActive ? 0.55 + clamped * 0.45 : 0.45,
              }}
            />
          );
        })}
      </div>
      <div className="flex items-center gap-2 font-mono text-[10px] tracking-wider text-slate-400">
        <span className="uppercase">
          {isActive
            ? audioLevel.source === "speaker"
              ? "Playback Output"
              : "Mic Input"
            : "Audio Standby"}
        </span>
        <span>•</span>
        <span>RMS {audioLevel.rms.toFixed(3)}</span>
      </div>
    </div>
  );
}
