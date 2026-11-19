# LocalVox Phase 3 Architecture & Implementation Plan

## Objective

Extend the offline Apple Silicon LocalVox voice pipeline into a natural, interruptible, streaming conversational companion supporting:
1. Full-duplex barge-in / user interruption handling with reference-aware echo suppression
2. Sentence-level incremental TTS streaming (`Ollama -> SentenceBuffer -> TTS Worker -> Audio Queue -> Speaker`)
3. Explicit, thread-safe conversation state machine with clean cancellation (`INTERRUPTED`, `STOPPING`) and zero leaked background threads
4. Natural turn-taking, silence timeout handling, and rapid follow-up capture
5. End-to-end Phase 3 latency instrumentation (`speech_end_to_stt_ms`, `tts_ttfa_ms`, `first_audio_latency_ms`, `interruption_detect_ms`, `playback_cancel_ms`)

## Honest Hardware & Acoustic Echo Analysis (macOS Apple Silicon)

- PortAudio (`sounddevice`) opens `MacBook Pro Microphone` and `MacBook Pro Speakers` as independent standard CoreAudio HAL streams and does **not** invoke Apple's kernel/AudioUnit `kAudioUnitSubType_VoiceProcessingIO` hardware AEC.
- Therefore, LocalVox implements **software reference-aware echo suppression + double-talk gating** (`backend/pipeline/echo.py`), NOT true hardware/OS AEC:
  1. `SpeakerAudioPlayer` feeds every played block (resampled cleanly to `16 kHz`) into a thread-safe `PlaybackReferenceBuffer`.
  2. During `SPEAKING`, `EchoSuppressor` locates the best lag in the recent playback reference (`0..echo_max_delay_ms`) via `scipy.signal.fftconvolve` (`~0.07 ms` per `32 ms` frame on Apple M5), removes the correlated scalar projection when correlation passes `echo_min_corr`, and tracks online speaker-to-mic acoustic coupling (`coupling_ema`).
  3. Silero VAD (`barge_in_vad_threshold`, higher than idle `vad.threshold`) evaluates the echo-suppressed residual frame and requires `barge_in_min_speech_ms` consecutive speech frames plus sufficient residual/uncorrelated energy before firing barge-in.
  4. Because `coupling_ema` adapts toward ~0 when headphones are worn (minimal acoustic leak) and adapts upward when built-in speakers leak into the mic, headphone barge-in works even at low voice volume while speaker echo is suppressed according to `barge_in_sensitivity` (`low`, `medium`, `high`).
- **Honest Limitation**: Without CoreAudio `VoiceProcessingIO` or headphones, loud speaker volume or strong room reverberation can still couple non-linear acoustic distortion into the built-in mic. Sensitivity and disable fallback (`barge_in_enabled: false`) remain configurable in `config.yaml`.

## Module Plan

1. **`config.yaml` & `backend/config.py`**
   - Extend `PipelineConfig` with barge-in sensitivity (`low`/`medium`/`high`), barge-in VAD/RMS/ratio thresholds, echo suppression settings, streaming TTS chunking/queue limits, `vad_hangover_ms: 832`, and `listen_timeout_s`.
2. **`backend/pipeline/state.py`**
   - Add `PipelineState.INTERRUPTED` and `PipelineState.STOPPING`.
   - Enforce `VALID_TRANSITIONS` atomically under lock.
   - Add monotonic `turn_generation_id` and per-turn `cancel_event` generation helpers so stale LLM/TTS tasks cannot leak across turns.
   - Extend `TurnMetrics` with `speech_end_to_stt_ms`, `tts_ttfa_ms`, `first_audio_latency_ms`, `interruption_detect_ms`, `playback_cancel_ms`, `tts_chunks`, and `barge_in_triggered`.
3. **`backend/llm/ollama_client.py`**
   - Add `cancel_event: threading.Event | None` support to `OllamaClient.chat(...)` so interrupted generation aborts immediately and closes the streaming HTTP connection.
   - Add `cancelled: bool = False` on `LlmResult`.
4. **`backend/pipeline/echo.py` (New)**
   - Implement `PlaybackReferenceBuffer`, `EchoSuppressionResult`, and `EchoSuppressor`.
5. **`backend/tts/streaming.py` (New)**
   - Implement incremental `SentenceBuffer` (preserving decimals/abbreviations, respecting `tts_min_sentence_chars` / `tts_max_chunk_chars`, filtering non-speakable fragments) and `AudioChunk` queue record.
6. **`backend/pipeline/audio_io.py`**
   - Update `SpeakerAudioPlayer` with `play_stream(...)` using small output blocks (`~25 ms`) wired to `PlaybackReferenceBuffer` for sub-`50 ms` cancellation and continuous multi-sentence audio without reopening CoreAudio streams.
   - Update `SpeechEndpointDetector.capture_utterance(...)` to accept seeded `initial_chunks` / `already_in_speech` (preserving barge-in speech onset frames so zero initial syllables are lost) and fix `in_speech` timeout safety.
7. **`backend/pipeline/voice_pipeline.py` & `scripts/voice_assistant.py`**
   - Orchestrate per-turn `LLM streaming thread -> SentenceBuffer -> sequential TTS worker -> SpeakerAudioPlayer.play_stream` with concurrent `BargeInMonitor` when `barge_in_enabled=True`, clean `INTERRUPTED` handoff, bounded `join()` cleanup guaranteeing zero zombie threads, and CLI/event callbacks for interruptions and Phase 3 metrics.
8. **Tests & Documentation**
   - Keep `tests/test_pipeline.py` Phase 2/2.1 half-duplex gating regression tests intact and passing.
   - Create `tests/test_phase3.py` covering all required interruption, streaming TTS, echo suppression, queue ordering, error recovery, and thread cleanup scenarios.
   - Verify with offline `make test`, `make lint`, real CoreAudio hardware checks, and update `README.md` and benchmark docs.

