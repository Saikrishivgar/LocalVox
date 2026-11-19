import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CompanionView } from "@/components/CompanionView";
import { ConversationHistoryView } from "@/components/ConversationHistoryView";
import { MemoryDashboard } from "@/components/MemoryDashboard";
import { SettingsView } from "@/components/SettingsView";
import { StatusBar } from "@/components/StatusBar";
import { SystemMonitorView } from "@/components/SystemMonitorView";
import { companionApi } from "@/lib/api";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("LocalVox Phase 5 Desktop Companion UI", () => {
  it("renders companion state transitions, live audio waveform, and chat controls", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    const startSpy = vi.spyOn(companionApi, "startVoice").mockResolvedValue({
      status: "started",
      pipeline_state: "LISTENING",
      mic_active: true,
      message: "started",
    });
    const chatSpy = vi.spyOn(companionApi, "sendChat").mockResolvedValue({
      session_id: "sess1",
      response_text: "I am running locally on Qwen3:8B.",
      metrics: null,
      pipeline_state: "IDLE",
    });

    const { rerender } = render(
      <CompanionView
        pipelineState="LISTENING"
        micActive={false}
        messages={[]}
        streamingPartial=""
        lastMetrics={null}
        audioLevel={{
          source: "mic",
          rms: 0.142,
          bars: new Array(24).fill(0.4),
          updatedAt: Date.now(),
        }}
        onRefreshState={refresh}
      />
    );

    expect(screen.getByText("Listening")).toBeInTheDocument();
    expect(screen.getByText(/RMS 0\.142/)).toBeInTheDocument();

    // Stop Speaking disabled when not speaking/generating
    const stopSpeakBtn = screen.getByRole("button", { name: /Stop Speaking/i });
    expect(stopSpeakBtn).toBeDisabled();

    // Start microphone button triggers API
    fireEvent.click(screen.getByRole("button", { name: /Start Microphone/i }));
    await waitFor(() => expect(startSpy).toHaveBeenCalledTimes(1));

    // Send chat text
    const input = screen.getByPlaceholderText(/Message LocalVox/i);
    fireEvent.change(input, { target: { value: "Hello LocalVox" } });
    fireEvent.click(screen.getByRole("button", { name: /Send message/i }));
    await waitFor(() =>
      expect(chatSpy).toHaveBeenCalledWith("Hello LocalVox", true)
    );

    // State indicator updates to Speaking and enables Stop Speaking
    rerender(
      <CompanionView
        pipelineState="SPEAKING"
        micActive={true}
        messages={[
          {
            id: "m1",
            role: "assistant",
            content: "Hello!",
            language: "en",
            created_at: new Date().toISOString(),
          },
        ]}
        streamingPartial=""
        lastMetrics={null}
        audioLevel={{
          source: "speaker",
          rms: 0.25,
          bars: new Array(24).fill(0.6),
          updatedAt: Date.now(),
        }}
        onRefreshState={refresh}
      />
    );
    expect(screen.getByText("Speaking")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Stop Speaking/i })
    ).toBeEnabled();
  });

  it("enforces confirmed bulk wipe and renders SQLite memory search/filters", async () => {
    const listSpy = vi
      .spyOn(companionApi, "listMemories")
      .mockResolvedValue({
        total_count: 1,
        category_counts: {
          personal_preference: 1,
          user_fact: 0,
          interest_hobby: 0,
          ongoing_project: 0,
          important_decision: 0,
          learning_progress: 0,
          conversation_event: 0,
          pending_task: 0,
        },
        items: [
          {
            id: "mem-1",
            category: "personal_preference",
            subject: "editor",
            content: "User prefers Neovim for Rust.",
            keywords: ["neovim"],
            importance: 0.9,
            confidence: 1.0,
            source: "user_explicit",
            confirmed_by_user: true,
            normalized_hash: "abc",
            session_id: "s1",
            access_count: 2,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            last_accessed_at: null,
          },
        ],
        match_scores: { "mem-1": 0.91 },
      });
    const clearSpy = vi
      .spyOn(companionApi, "clearAllMemories")
      .mockResolvedValue({
        status: "ok",
        action: "cleared_all",
        record: null,
        deleted_count: 1,
      });
    const onCount = vi.fn();

    render(<MemoryDashboard memoryVersion={0} onCountChange={onCount} />);

    await waitFor(() =>
      expect(
        screen.getByText("User prefers Neovim for Rust.")
      ).toBeInTheDocument()
    );
    expect(listSpy).toHaveBeenCalled();
    expect(onCount).toHaveBeenCalledWith(1);

    // Open Clear All confirmation modal
    fireEvent.click(screen.getByRole("button", { name: /Clear All/i }));
    const wipeBtn = screen.getByRole("button", { name: /Wipe All Memories/i });
    expect(wipeBtn).toBeDisabled();

    const confirmInput = screen.getByPlaceholderText("CLEAR_ALL_MEMORIES");
    fireEvent.change(confirmInput, { target: { value: "CLEAR_ALL_MEMORIES" } });
    expect(wipeBtn).toBeEnabled();

    fireEvent.click(wipeBtn);
    await waitFor(() =>
      expect(clearSpy).toHaveBeenCalledWith("CLEAR_ALL_MEMORIES")
    );
  });

  it("displays empty history honestly and triggers Continue Conversation", async () => {
    const listSpy = vi
      .spyOn(companionApi, "listConversations")
      .mockResolvedValue({
        active_session_id: "active-1",
        items: [
          {
            id: "hist-1",
            title: "SQLite FTS5 study",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            turn_count: 2,
            preview: "SQLite FTS5 uses BM25 ranking locally.",
          },
        ],
      });
    vi.spyOn(companionApi, "getConversation").mockResolvedValue({
      session: {
        id: "hist-1",
        title: "SQLite FTS5 study",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        turn_count: 2,
        preview: "SQLite FTS5 uses BM25 ranking locally.",
      },
      messages: [
        {
          id: 1,
          session_id: "hist-1",
          role: "user",
          content: "Explain SQLite FTS5",
          language: "en",
          metrics: null,
          created_at: new Date().toISOString(),
        },
        {
          id: 2,
          session_id: "hist-1",
          role: "assistant",
          content: "SQLite FTS5 provides fast local full-text indexing.",
          language: "en",
          metrics: null,
          created_at: new Date().toISOString(),
        },
      ],
    });
    const continueSpy = vi
      .spyOn(companionApi, "continueConversation")
      .mockResolvedValue({
        loaded_turns: 1,
        active_session_id: "hist-1",
        detail: {
          session: {
            id: "hist-1",
            title: "SQLite FTS5 study",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            turn_count: 2,
            preview: "",
          },
          messages: [],
        },
      });
    const onContinue = vi.fn().mockResolvedValue(undefined);

    render(
      <ConversationHistoryView
        historyVersion={0}
        onContinueSession={onContinue}
      />
    );

    await waitFor(() =>
      expect(
        screen.getByText(
          "SQLite FTS5 provides fast local full-text indexing."
        )
      ).toBeInTheDocument()
    );
    expect(listSpy).toHaveBeenCalled();

    fireEvent.click(
      screen.getByRole("button", { name: /Continue Conversation/i })
    );
    await waitFor(() => expect(continueSpy).toHaveBeenCalledWith("hist-1"));
    expect(onContinue).toHaveBeenCalledTimes(1);
  });

  it("renders real system monitor hardware metrics and honest Ollama offline status", async () => {
    vi.spyOn(companionApi, "getSystemMonitor").mockResolvedValue({
      hardware: "Apple M5 (24 GB Unified Memory)",
      cpu_percent: 18.4,
      ram_used_gb: 11.2,
      ram_total_gb: 24.0,
      ram_percent: 46.7,
      process_rss_mb: 245.5,
      ollama_connected: false,
      qwen_available: false,
      available_ollama_models: [],
      engines: [
        {
          name: "Ollama Qwen LLM",
          status: "offline",
          detail: "Ollama daemon unreachable on local host.",
          model_name: "qwen3:8b",
          loaded: false,
          disk_size_mb: null,
          memory_mb: null,
        },
      ],
      sqlite_integrity: { ok: true, fts_ok: true },
      sqlite_size_mb: 0.12,
      extraction_worker: {
        completed_jobs: 3,
        queue_size: 0,
        idle: true,
        error_count: 0,
        last_outcome: null,
      },
      recent_latencies: [{ stage: "ollama_chat", latency_ms: 142.3 }],
      pipeline_state: "IDLE",
      mic_active: false,
    });

    render(<SystemMonitorView />);

    await waitFor(() =>
      expect(
        screen.getByText("Apple M5 (24 GB Unified Memory)")
      ).toBeInTheDocument()
    );
    expect(screen.getByText("18.4%")).toBeInTheDocument();
    expect(screen.getByText("offline")).toBeInTheDocument();
    expect(screen.getByText("Healthy")).toBeInTheDocument();
    expect(screen.getByText("142.3 ms")).toBeInTheDocument();
  });

  it("loads and persists desktop settings cleanly", async () => {
    vi.spyOn(companionApi, "getSettings").mockResolvedValue({
      settings: {
        kokoro_voice: "af_heart",
        kokoro_speed: 1.0,
        piper_length_scale: 1.0,
        stt_default_language: null,
        tts_routing: "auto",
        streaming_tts_enabled: true,
        barge_in_enabled: true,
        barge_in_sensitivity: "medium",
        llm_model_name: "qwen3:8b",
        llm_temperature: 0.25,
        max_history_turns: 10,
        memory_enabled: true,
        memory_auto_extract: true,
        memory_privacy_filter_enabled: true,
        wakeword_enabled: true,
        wake_phrase: "Hey Jarvis",
        wakeword_sensitivity: "medium",
        wakeword_threshold: 0.5,
        wakeword_conversation_timeout_s: 15,
        wakeword_ack_enabled: true,
        wakeword_activation_grace_ms: 350,
      },
      available_kokoro_voices: ["af_heart", "af_bella"],
      available_ollama_models: ["qwen3:8b"],
      wakeword_engine: {
        engine: "openwakeword-onnx",
        wake_phrase: "Hey Jarvis",
        model_name: "hey_jarvis_v0.1",
        model_path: "data/models/wakeword/hey_jarvis_v0.1.onnx",
        loaded: true,
        models_present: true,
        disk_size_mb: 3.51,
        execution_provider: "CPUExecutionProvider",
        last_inference_ms: 1.02,
        last_score: 0.01,
        total_detections: 2,
        last_detection_at: null,
      },
      mic_active: true,
    });
    const updateSpy = vi
      .spyOn(companionApi, "updateSettings")
      .mockImplementation(async (payload) => ({
        settings: payload,
        available_kokoro_voices: ["af_heart", "af_bella"],
        available_ollama_models: ["qwen3:8b"],
        wakeword_engine: null,
        mic_active: false,
      }));

    render(<SettingsView settingsVersion={0} micActive={true} />);

    await waitFor(() =>
      expect(screen.getByText("Companion Preferences")).toBeInTheDocument()
    );
    expect(screen.getByText(/Mic Monitoring Active/i)).toBeInTheDocument();
    expect(screen.getByText(/hey_jarvis_v0\.1 · Loaded/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "high" }));
    fireEvent.click(
      screen.getByRole("button", { name: /Wake sensitivity high/i })
    );
    fireEvent.click(screen.getByRole("button", { name: /Save Settings/i }));

    await waitFor(() =>
      expect(updateSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          barge_in_sensitivity: "high",
          wakeword_sensitivity: "high",
        })
      )
    );
    expect(
      screen.getByText(/Settings validated and saved to config\.yaml/i)
    ).toBeInTheDocument();
  });

  it("renders automatic wake-word standby, wake detection, and privacy mic indicator states", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    const stopSpy = vi.spyOn(companionApi, "stopVoice").mockResolvedValue({
      status: "stopped",
      pipeline_state: "IDLE",
      mic_active: false,
      message: "stopped",
    });

    const { rerender } = render(
      <>
        <StatusBar
          connected={true}
          pipelineState="SLEEPING"
          micActive={true}
          wakewordActive={true}
          wakePhrase="Hey Jarvis"
          lastMetrics={null}
        />
        <CompanionView
          pipelineState="SLEEPING"
          micActive={true}
          messages={[]}
          streamingPartial=""
          lastMetrics={null}
          audioLevel={{
            source: "mic",
            rms: 0.002,
            bars: new Array(24).fill(0),
            updatedAt: Date.now(),
          }}
          wakewordEnabled={true}
          wakewordActive={true}
          wakePhrase="Hey Jarvis"
          lastWakeEvent={null}
          onRefreshState={refresh}
        />
      </>
    );

    expect(screen.getByTestId("mic-monitor-indicator")).toHaveTextContent(
      /Mic Standby · Hey Jarvis/i
    );
    expect(screen.getByTestId("wakeword-status-badge")).toHaveTextContent(
      /Wake Word: Hey Jarvis/i
    );
    // Manual "Hands-Free Wake" Enable button removed — wake standby is automatic.
    expect(screen.queryByText(/Hands-Free Wake/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Stop Microphone/i }));
    await waitFor(() => expect(stopSpy).toHaveBeenCalledTimes(1));

    rerender(
      <>
        <StatusBar
          connected={true}
          pipelineState="WAKE_DETECTED"
          micActive={true}
          wakewordActive={true}
          wakePhrase="Hey Jarvis"
          lastMetrics={null}
        />
        <CompanionView
          pipelineState="WAKE_DETECTED"
          micActive={true}
          messages={[]}
          streamingPartial=""
          lastMetrics={null}
          audioLevel={{
            source: "mic",
            rms: 0.08,
            bars: new Array(24).fill(0.2),
            updatedAt: Date.now(),
          }}
          wakewordEnabled={true}
          wakewordActive={true}
          wakePhrase="Hey Jarvis"
          lastWakeEvent={{
            wake_phrase: "Hey Jarvis",
            model_name: "hey_jarvis_v0.1",
            confidence: 0.9964,
            threshold: 0.5,
            inference_ms: 1.03,
            audio_rms: 0.06,
            timestamp: new Date().toISOString(),
            detected_at_monotonic: 12.0,
          }}
          onRefreshState={refresh}
        />
      </>
    );

    expect(screen.getByTestId("status-pill")).toHaveTextContent(
      "Wake Detected"
    );
    expect(screen.getByText(/99\.6%/)).toBeInTheDocument();
  });

  it("guides the user through 3-sample voice personalization and activates hands-free", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    const startEnrollSpy = vi
      .spyOn(companionApi, "startVoiceEnrollment")
      .mockResolvedValue({
        enrolled: false,
        samples_recorded: 0,
        required_samples: 3,
        in_progress_samples: 0,
        wake_phrase: "Hey Jarvis",
        profile_path: "data/models/wakeword/voice_profile.npz",
        updated_at: null,
        personal_enabled: true,
        personal_only: false,
        personal_threshold: 0.72,
        consistency_score: null,
      });

    let sampleNum = 0;
    const recordSpy = vi
      .spyOn(companionApi, "recordVoiceEnrollmentSample")
      .mockImplementation(async () => {
        sampleNum += 1;
        const done = sampleNum === 3;
        return {
          sample_index: sampleNum,
          required_samples: 3,
          completed: done,
          duration_s: 0.95,
          audio_rms: 0.08,
          generic_score: 0.32,
          consistency_with_previous: sampleNum > 1 ? 0.93 : null,
          message: done
            ? "Voice profile saved! LocalVox now recognizes your voice."
            : `Sample ${sampleNum} of 3 captured.`,
          profile: {
            enrolled: done,
            samples_recorded: done ? 3 : 0,
            required_samples: 3,
            in_progress_samples: done ? 0 : sampleNum,
            wake_phrase: "Hey Jarvis",
            profile_path: "data/models/wakeword/voice_profile.npz",
            updated_at: done ? new Date().toISOString() : null,
            personal_enabled: true,
            personal_only: false,
            personal_threshold: 0.72,
            consistency_score: done ? 0.94 : null,
          },
        };
      });

    const enableWakeSpy = vi
      .spyOn(companionApi, "enableWakeWord")
      .mockResolvedValue({
        status: "started",
        pipeline_state: "SLEEPING",
        mic_active: true,
        message: "Hands-free standby active",
      });

    render(
      <CompanionView
        pipelineState="SLEEPING"
        micActive={true}
        messages={[]}
        streamingPartial=""
        lastMetrics={null}
        audioLevel={{
          source: "mic",
          rms: 0.04,
          bars: new Array(24).fill(0.25),
          updatedAt: Date.now(),
          wakeScore: 0.42,
          wakeThreshold: 0.72,
        }}
        wakewordEnabled={true}
        wakewordActive={true}
        wakePhrase="Hey Jarvis"
        voiceProfile={null}
        onRefreshState={refresh}
      />
    );

    expect(screen.getByTestId("standby-wake-meter")).toHaveTextContent(
      /Match 42%/i
    );

    // Open 3-sample voice personalization modal
    fireEvent.click(screen.getByRole("button", { name: /Personalize Voice/i }));
    expect(screen.getByText(/Personalize Your Voice/i)).toBeInTheDocument();

    // Clicking Record Sample 1 of 3 runs the guided 1 -> 2 -> 3 flow automatically
    // without resetting captured samples when profile snapshots refresh.
    fireEvent.click(
      screen.getByRole("button", { name: /Record Sample 1 of 3/i })
    );
    await waitFor(
      () =>
        expect(
          screen.getByText(/Personalized Voice Active \(3\/3\)/i)
        ).toBeInTheDocument(),
      { timeout: 4000 }
    );
    // Hands-free standby activates automatically on Sample 3 completion.
    expect(startEnrollSpy).toHaveBeenCalledTimes(1);
    expect(recordSpy).toHaveBeenCalledTimes(3);
    await waitFor(() => expect(enableWakeSpy).toHaveBeenCalledTimes(1));

    fireEvent.click(
      screen.getByRole("button", { name: /Done — Say "Hey Jarvis"/i })
    );
    await waitFor(() => expect(enableWakeSpy).toHaveBeenCalledTimes(2));
  });
});
