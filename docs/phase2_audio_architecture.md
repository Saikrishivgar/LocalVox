# LocalVox — Phase 2.1 Audio Interaction & Architecture Guide

This document explains the Phase 2.1 audio pipeline architecture, the root cause and fix for the audio feedback loop, VAD endpointing configuration, future barge-in and Acoustic Echo Cancellation (AEC) design, sentence-level streaming TTS analysis, and the manual testing procedure.

---

## 1. Root Cause of the Phase 2 Audio Feedback Loop

During manual testing of Phase 2, LocalVox transcribed its own previous spoken reply as the user's next utterance.

**Exact root cause in the original code:**

1. `VoicePipeline.run()` kept `MicrophoneAudioCapture`'s PortAudio `InputStream` callback running continuously across `PROCESSING`, `GENERATING`, and `SPEAKING` without gating.
2. While `SpeakerAudioPlayer.play()` played 4–13 seconds of Kokoro/Piper speech through `MacBook Pro Speakers`, the active `InputStream` callback captured the speaker output and pushed those frames into `MicrophoneAudioCapture._queue` (`maxsize=256`, ~8.2 s capacity) and `_pending`.
3. When playback finished and `VoicePipeline.run_once()` re-entered `SpeechEndpointDetector.capture_utterance()`, `read_chunk()` immediately consumed the buffered playback audio from `_queue`/`_pending`—causing Silero VAD to trigger on LocalVox's own voice and MLX Whisper to transcribe it as Turn `N+1`.

---

## 2. Implemented Solution: Half-Duplex Capture Gating & Buffer Lifecycle

Rather than closing and reopening CoreAudio streams on every turn or adding fragile `time.sleep()` delays, LocalVox coordinates microphone capture and speaker playback via `PipelineStateMachine` and `MicrophoneAudioCapture`:

```text
  [LISTENING]  ──(VAD speech start -> end)──▶ capture.pause() + clear_buffer()
       ▲                                                │
       │                                                ▼
  capture.resume()                             [PROCESSING] (STT)
  (after clear_buffer() + vad.reset_states())           │
       ▲                                                ▼
       │                                       [GENERATING] (Ollama)
       │                                                │
       └─────── Playback completes / errors ◀── [SPEAKING] (TTS + Speaker)
                (mic callback drops all frames while _accepting is cleared)
```

1. **Callback Gating (`MicrophoneAudioCapture.pause()` / `resume()`)**:
   - `MicrophoneAudioCapture` maintains a `threading.Event` (`_accepting`).
   - When `pause()` is called before `PROCESSING` / `GENERATING` / `SPEAKING`, `_accepting` is cleared and `clear_buffer()` drains `_queue` and resets `_pending` under `_buffer_lock`.
   - Inside PortAudio's real-time `_callback()`, if `not self._accepting.is_set()`, the callback returns immediately without resampling or enqueueing audio—keeping the `sd.InputStream` open with zero hardware reopen latency.
2. **State Machine Enforcement (`PipelineStateMachine`)**:
   - States: `IDLE`, `LISTENING`, `PROCESSING`, `GENERATING`, `SPEAKING`, `ERROR`.
   - `SpeechEndpointDetector.capture_utterance()` checks `state_machine.can_accept_speech()` (true only when `state == LISTENING` and `not stop_event.is_set()`).
3. **Post-Playback Reset**:
   - After `SpeakerAudioPlayer.play()` returns (or on error in `process_utterance()`'s `finally:` block), `VoicePipeline` calls `capture.clear_buffer()` and `vad.reset_states()`, transitions back to `LISTENING`, and calls `capture.resume()`.
4. **Error & `Ctrl+C` Cleanup**:
   - STT, LLM, and TTS failures transition to `ERROR`, notify `on_error`, flush stale buffers/VAD state, and recover cleanly to `LISTENING`.
   - `Ctrl+C` sets `stop_event` + `barge_in_event`, stops playback, closes the input stream once in `finally`, resets VAD, and transitions to `IDLE`.

---

## 3. VAD & Speech Endpointing Settings (`config.yaml`)

All endpointing parameters are exposed under `vad:` and `pipeline:` in [`config.yaml`](../config.yaml) and validated by [`backend/config.py`](../backend/config.py):

| Setting | Value | Purpose |
| --- | ---: | --- |
| `vad.threshold` | `0.5` | Silero VAD speech probability threshold per `512`-sample (`32 ms`) frame |
| `vad.min_speech_duration_ms` | `250` | Minimum consecutive speech duration before confirming speech onset |
| `pipeline.pre_speech_pad_ms` | `288` | Pre-roll ring buffer preserving initial consonant/word onsets |
| `pipeline.vad_hangover_ms` | `768` | Trailing silence required to finalize an utterance (handles short mid-sentence pauses) |
| `pipeline.post_speech_pad_ms` | `160` | Trailing padding kept at utterance end to avoid clipping final syllables |
| `pipeline.min_utterance_ms` | `256` | Minimum utterance length before sending audio to Whisper |
| `pipeline.max_utterance_s` | `20.0` | Hard cap preventing unbounded buffering on continuous noise |
| `pipeline.min_audio_rms` | `0.002` | RMS energy floor rejecting near-zero ambient hum before onset and submission |

---

## 4. Future Barge-In & Acoustic Echo Cancellation (AEC) Roadmap

- **Currently Implemented**:
  - `PipelineStateMachine.cancel_playback()` and `barge_in_event` (`threading.Event`).
  - `SpeakerAudioPlayer.cancel()`, `reset_cancel()`, and `40 ms`-chunked playback checking cancellation flags and calling `stream.abort()` immediately on interrupt.
  - `MicrophoneAudioCapture(..., frame_processor=...)` hook for future per-frame audio DSP.
- **Not Yet Implemented (Deferred)**:
  - **Full Microphone Barge-In**: Simultaneously listening to the microphone while `SPEAKING` requires **Acoustic Echo Cancellation (AEC)** so the speaker output played into the room does not falsely trigger barge-in or corrupt Whisper transcripts.
  - **Planned AEC Integration**: Pass speaker reference frames (`play` output) and mic capture frames through an AEC processor (e.g., WebRTC AEC / SpeexDSP) plugged into `MicrophoneAudioCapture(frame_processor=...)` before enabling concurrent `SPEAKING` + VAD barge-in monitoring.

---

## 5. Sentence-Level Streaming TTS Investigation

**Current behavior**: LocalVox buffers the full `OllamaClient.chat()` response before running `TtsRouter.synthesize()` and `SpeakerAudioPlayer.play()`.

**Proposed Streaming TTS Architecture (Deferred to Phase 2.2 / Phase 3)**:

1. `OllamaClient.chat(on_token=...)` feeds tokens into a sentence boundary splitter (splitting on `.`, `!`, `?`, `\n`, and Tamil sentence terminators, respecting decimal numbers and abbreviations).
2. Complete sentences push onto a bounded `sentence_queue` consumed by a dedicated TTS synthesis worker thread (`tts_worker`) producing `(waveform, sample_rate)` chunks into `audio_queue`.
3. `SpeakerAudioPlayer` drains `audio_queue` continuously and flushes both queues on `cancel_playback()`.

**Compatibility Limitations Observed in Kokoro & Piper**:

- **`kokoro-onnx` Global eSpeak Lock**: `kokoro_onnx.tokenizer` guards `phonemizer` with a process-global `_espeak_lock` because `libespeak-ng` uses global C state. While Kokoro can synthesize individual sentences sequentially inside one worker thread, interleaving Kokoro and Piper phonemization across multiple threads risks eSpeak state contention unless serialized in one TTS worker thread.
- **Cross-Chunk Prosody & Clicks**: Synthesizing short independent fragments with Kokoro (`24 kHz`) or Piper (`22.05 kHz`) can produce boundary clicks or pitch resets without short silence/cross-fade stitching and consistent sample-rate conversion (`24 kHz` vs `22.05 kHz` on mixed-language turns).
- **Decision**: Deferred sentence-level TTS to keep Phase 2.1 half-duplex feedback isolation completely deterministic and stable.

---

## 6. Manual End-to-End Test Procedure (Live Mic & Speaker)

Launch LocalVox in a normal macOS terminal (with Ollama running and Microphone permission granted):

```bash
cd ~/Desktop/LocalVox
HF_HUB_OFFLINE=1 make voice
```

Verify each scenario and confirm LocalVox returns to `[LISTENING]` without ever transcribing its own previous spoken reply:

1. **Single English sentence**: Say `"What is photosynthesis?"` → confirm accurate transcript, natural spoken reply without Markdown/emojis, and silent wait afterward.
2. **Several consecutive English sentences**: Speak a 2–3 sentence study question with a brief `~0.4 s` pause mid-question → confirm `vad_hangover_ms` (`768 ms`) preserves the complete question and does not clip final words.
3. **Tamil**: Say `"வணக்கம், நியூட்டனின் முதல் விதியை எளிய தமிழில் சொல்லுங்கள்."` → confirm Whisper recognizes Tamil, `qwen3:8b` responds in Tamil, and Piper `ta_IN` plays Tamil speech.
4. **Tanglish**: Say `"Binary search algorithm எப்படி work ஆகுது?"` → confirm LocalVox responds naturally without claiming unsupported accent capabilities.
5. **Follow-up questions**: Immediately ask `"Give me one simple example of that."` → confirm `ConversationManager` retains session context without repeating stock greetings.
6. **Wait silently**: Stay quiet for 10+ seconds after playback finishes → confirm zero false triggers on room noise or playback echo.
7. **Speak again after playback**: Ask another question after silence → confirm immediate response without needing any manual restart.
8. **10+ consecutive turns & shutdown**: Complete at least 10 turns back-to-back, test `"clear conversation"`, and press `Ctrl+C` (or say `"goodbye localvox"`) to confirm clean shutdown.

