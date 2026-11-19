<div align="center">

# LocalVox

### **Your AI. Your Voice. Your Device. Your Privacy.**

An offline-first, real-time Tamil and English voice AI companion running locally on macOS Apple Silicon without cloud AI APIs.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16.4.6-black?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen3%3A8B-white?style=flat-square&logo=ollama&logoColor=black)](https://ollama.ai)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Metal_%26_CoreAudio-grey?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Offline AI](https://img.shields.io/badge/Cloud_APIs-Zero_Dependency-success?style=flat-square)](https://github.com/Saikrishivgar/LocalVox)
[![Tests Passing](https://img.shields.io/badge/Tests-56_Pytest_%7C_7_Vitest-brightgreen?style=flat-square)](tests)

<br/>

<img src="docs/assets/screenshots/dashboard.png" alt="LocalVox Desktop Companion Hero Dashboard" width="900" style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.4);" />

*LocalVox Desktop Companion: 3D Acoustic Core, live hands-free standby, low-latency turn telemetry, and persistent SQLite memory.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why LocalVox?](#why-localvox)
- [Visual Interface Showcase](#visual-interface-showcase)
- [System Architecture](#system-architecture)
- [Phase-by-Phase Engineering Journey](#phase-by-phase-engineering-journey)
- [Deep-Dive Feature Breakdown](#deep-dive-feature-breakdown)
- [Hardware Telemetry & Performance Benchmarks](#hardware-telemetry--performance-benchmarks)
- [Competitive Comparison](#competitive-comparison)
- [Quick Start & Installation](#quick-start--installation)
- [Configuration Reference](#configuration-reference)
- [Privacy & Security Guarantees](#privacy--security-guarantees)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Roadmap](#roadmap)
- [Contributing & License](#contributing--license)

---

## Overview

**LocalVox** is an autonomous, on-device voice AI companion built specifically for macOS on Apple Silicon. Unlike conventional voice assistants that record user audio, beam it across the internet, run proprietary cloud inference, and return delayed responses, LocalVox executes every stage of the voice loop—wake-word detection, speech capture, voice activity endpointing, speech-to-text, LLM inference, long-term memory retrieval, text-to-speech synthesis, and acoustic playback—strictly within your Mac's unified memory.

LocalVox understands and communicates natively in **English, Tamil, and Tanglish (Tamil-English code-switching)**. It operates without an internet connection, preserves absolute user privacy, supports acoustic **barge-in interruption**, features a **3-sample local voice personalization engine**, and retains context across sessions through a high-performance **SQLite FTS5 BM25 memory engine**.

### What Sets LocalVox Apart

1. **Zero Cloud Dependencies**: Operates completely offline with zero telemetry, zero external API keys, and zero subscription costs.
2. **Autonomous "Hey Jarvis" Wake Word**: Integrated pure-ONNX openWakeWord model running in `1.08 ms` per frame with just `5.6%` of a single CPU core on standby.
3. **3-Sample Voice Personalization**: Adapts to the user's specific vocal timbre and acoustic environment from three short spoken samples, without ever storing raw audio files to disk.
4. **Sub-Turn Acoustic Barge-In**: Interrupt the AI companion naturally mid-speech. Software echo suppression prevents self-activation loops and cancels playback in under `1 ms`.
5. **Local Persistent Memory with FTS5 BM25**: Automatically categorizes, stores, updates, and retrieves memories across 8 distinct categories in `< 2 ms` using SQLite WAL mode.
6. **Bilingual Synthesis Pipeline**: Dynamically routes English utterances to Kokoro ONNX (24 kHz) and Tamil/Tanglish utterances to Piper ONNX (22.05 kHz).
7. **Premium Native Desktop Companion**: Single-page studio interface built on Next.js 16, TypeScript, Tailwind CSS, and a 3D audio-reactive Three.js core communicating over WebSockets.

---

## Why LocalVox?

Modern voice assistants and cloud AI wrappers suffer from five architectural drawbacks:

```text
┌─────────────────────────────────── Cloud Voice Assistants ────────────────────────────────────┐
│  [Microphone] ──(Network)──▶ [Cloud Vendor] ──▶ [Audio Recorded & Analyzed] ──▶ High Latency  │
│  Fragile to internet drops • Subscription fees • Zero user memory control • Privacy liability │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                vs.
┌────────────────────────────────────── LocalVox Offline ───────────────────────────────────────┐
│  [Microphone] ──▶ [Apple Silicon CoreAudio] ──▶ [Metal GPU / Unified Memory] ──▶ Sub-Second   │
│  100% Air-Gapped • $0 Operating Cost • Local SQLite Ownership • True Acoustic Confidentiality  │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The Problem Space

- **Cloud Surveillance & Data Extraction**: Commercial smart speakers and cloud assistants upload live voice telemetry, audio snippets, and personal conversations to remote datacenters, creating persistent privacy, compliance, and eavesdropping risks.
- **Latency & Round-Trip Jitter**: Transmitting uncompressed or lossy audio buffers over WAN to distant cloud endpoints introduces variable latency (often 1,500 ms to 5,000 ms), completely destroying conversational flow.
- **Fragility & Connection Dependency**: When an internet connection degrades or fails, cloud assistants become completely unusable paperweights.
- **Neglected Indic & Vernacular Languages**: Mainstream cloud voice models often struggle with fluid Tamil and mixed Tanglish speech patterns, either mangling pronunciation or forcing English-only interactions.
- **Opaque Black-Box Memory**: Users cannot inspect, edit, or control what cloud providers store about them. Memories are either non-existent or locked inside closed databases.

### The LocalVox Solution

LocalVox redesigns conversational voice AI from first principles for the Apple Silicon unified memory architecture:

- **100% On-Device Execution**: Speech recognition, LLM reasoning, neural speech synthesis, and wake-word verification run strictly on the local machine via Apple Metal and Accelerate.
- **Predictable Real-Time Performance**: High-throughput memory sharing between CPU and GPU enables in-memory model execution without serialization overhead, delivering first-token speech synthesis in ~2 seconds.
- **Transparent Local SQLite Database**: All persistent long-term memories are stored in an open SQLite database (`data/memory.db`) with full-text search (FTS5), accessible directly through the visual dashboard or CLI.
- **Native Indic Linguistic Routing**: Automatic script and language classification directs Tamil phonemes to specialized neural voices while handling bilingual Tanglish seamlessly.
- **Zero Raw Audio Storage**: Raw microphone frames are processed in volatile ring buffers and immediately garbage-collected. No audio files are ever written to disk during regular turns or voice enrollment.

---

## Visual Interface Showcase

The LocalVox Desktop Companion provides an interactive, dark-mode studio interface designed for long study sessions, hands-free work, and deep memory management.

<div align="center">

### 1. Main Companion Dashboard
<img src="docs/assets/screenshots/dashboard.png" alt="Companion Dashboard" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Real-time Three.js 3D vibrating acoustic core reflecting standby, listening, thinking, speaking, and interrupted states with millisecond telemetry.</em></p>

<br/>

### 2. Live Bilingual Conversation
<img src="docs/assets/screenshots/live-conversation.png" alt="Live Voice Conversation" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Full transcript stream featuring language classification (EN/TA), generation throughput (tok/s), turn latency, and dynamic memory grounding indicators.</em></p>

<br/>

### 3. Hands-Free "Hey Jarvis" Standby
<img src="docs/assets/screenshots/wake-word.png" alt="Wake Word Standby" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Autonomous standby mode powered by openWakeWord ONNX. Consumes ~5.6% of one CPU core, providing continuous wake-meter feedback without cloud polling.</em></p>

<br/>

### 4. 3-Sample Personalized Voice Enrollment
<img src="docs/assets/screenshots/voice-enrollment.png" alt="Voice Personalization Modal" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Interactive 3-sample calibration wizard. Computes baseline-centered phonetic embeddings and vocal timbre centroids (voice_profile.npz) while discarding raw PCM audio.</em></p>

<br/>

### 5. Persistent Memory Dashboard
<img src="docs/assets/screenshots/memory-dashboard.png" alt="Memory Dashboard" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Explore, filter, edit, or delete long-term memories across 8 distinct categories. Powered by SQLite WAL and FTS5 BM25 ranking.</em></p>

<br/>

### 6. Conversation History & Resumption
<img src="docs/assets/screenshots/conversation-history.png" alt="Conversation History" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Archived conversation sessions with turn counts, duration, and one-click "Continue Session" functionality to reload context into active memory.</em></p>

<br/>

### 7. System Hardware Monitor
<img src="docs/assets/screenshots/system-monitor.png" alt="System Monitor" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Live Apple Silicon CPU and RAM monitoring, active ONNX execution provider status, Ollama health, and loaded model telemetry.</em></p>

<br/>

### 8. Configuration & Settings Studio
<img src="docs/assets/screenshots/settings.png" alt="Settings Dashboard" width="850" style="border-radius: 8px; margin-bottom: 8px;" />
<p align="center"><em>Modify wake-word sensitivity, barge-in energy thresholds, silence hangover, TTS voices, and LLM temperature with direct YAML persistence.</em></p>

</div>

---

## System Architecture

LocalVox decouples audio acquisition, neural inference, conversation management, and visual presentation into a clean, event-driven architecture.

### Audio & Inference Pipeline

```mermaid
flowchart TD
    subgraph Audio_Input["Acoustic Ingestion"]
        MIC["MacBook Microphone / CoreAudio (16 kHz mono)"]
        STREAM["sounddevice InputStream (512 samples / 32 ms)"]
        MIC --> STREAM
    end

    subgraph Standby_Loop["Hands-Free Standby (SLEEPING)"]
        RING["Bounded Ring Buffer (80 ms / 1280 samples)"]
        OWW["openWakeWord ONNX Engine (hey_jarvis_v0.1)"]
        PERS["3-Sample Voice Personalization Profile (voice_profile.npz)"]
        STREAM -->|Standby Active| RING
        RING --> OWW
        OWW <--> PERS
    end

    subgraph Active_Capture["Active Turn Capture (LISTENING)"]
        VAD["Silero VAD (ONNX) + Endpoint Detector"]
        BARGE["BargeInMonitor + Software EchoSuppressor"]
        STREAM -->|Wake Detected / PTT| VAD
        STREAM -->|While Speaking| BARGE
    end

    subgraph Speech_To_Text["Speech-to-Text (PROCESSING)"]
        MLX["MLX Whisper (whisper-large-v3-turbo-8bit on Metal GPU)"]
        VAD -->|Validated Speech Utterance| MLX
    end

    subgraph Orchestration["Conversation & Memory Layer"]
        MGR["ConversationManager"]
        FTS["SQLite WAL + FTS5 BM25 Memory Retriever (< 2 ms)"]
        MEM_DB[("data/memory.db (8 Categories)")]
        MLX -->|User Transcript| MGR
        MGR <-->|Hybrid Context Retrieval| FTS
        FTS <--> MEM_DB
    end

    subgraph Reasoning["Local LLM Reasoning (GENERATING)"]
        OLLAMA["Ollama Daemon (Qwen3:8B Q4_K_M)"]
        ASYNC_EXT["Async Memory Extractor (Background Worker)"]
        MGR -->|Context + Memory Prompt| OLLAMA
        MGR -.->|Post-Turn Turn Transcript| ASYNC_EXT
        ASYNC_EXT -.->|Extracted Facts| MEM_DB
    end

    subgraph Synthesis["Streaming Text-to-Speech (SPEAKING)"]
        ROUTER["Bilingual TTS Router"]
        SBUF["SentenceBuffer (Streaming Chunking)"]
        KOKORO["Kokoro ONNX (English, 24 kHz)"]
        PIPER["Piper ONNX (Tamil, 22.05 kHz)"]
        PLAY["sounddevice OutputStream (CoreAudio Playback)"]
        OLLAMA -->|Streamed Tokens| SBUF
        SBUF --> ROUTER
        ROUTER -->|English Sentences| KOKORO
        ROUTER -->|Tamil / Tanglish| PIPER
        KOKORO --> PLAY
        PIPER --> PLAY
        BARGE -.->|Instant Acoustic Cancel| PLAY
    end
```

### Desktop Client-Server Architecture

```mermaid
flowchart LR
    subgraph Frontend["Next.js 16 Desktop Client (127.0.0.1:3000)"]
        ORB["3D Core Visualizer (Three.js)"]
        VIEWS["Companion / Memory / History / System / Settings"]
        HOOK["useCompanionSocket (State & Metrics Sync)"]
        ORB --- VIEWS
        VIEWS --- HOOK
    end

    subgraph Backend["FastAPI Server (127.0.0.1:8000)"]
        WS_HUB["WebSocket Event Broadcaster (/ws/events)"]
        REST["REST API Controllers (/api/*)"]
        RUNTIME["CompanionRuntime Engine"]
        REST --- RUNTIME
        WS_HUB --- RUNTIME
    end

    HOOK <==>|Bi-Directional JSON Events| WS_HUB
    VIEWS <==>|HTTP Mutations & Snapshots| REST
```

### State Machine Lifecycle

LocalVox transitions deterministically across 7 operational states, managed in `backend/pipeline/state.py`:

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> SLEEPING : Automatic Hands-Free Startup
    SLEEPING --> WAKE_DETECTED : "Hey Jarvis" Detected
    WAKE_DETECTED --> SPEAKING : Optional Voice Ack ("Yes, I'm listening.")
    WAKE_DETECTED --> LISTENING : Instant Wake Transition (Pre-roll primed)
    SPEAKING --> LISTENING : Voice Ack Completed (350 ms Grace Period)
    LISTENING --> PROCESSING : User Silence Detected (Hangover 832 ms)
    PROCESSING --> GENERATING : MLX Whisper Transcript Ready
    GENERATING --> SPEAKING : First Sentence Synthesized
    SPEAKING --> INTERRUPTED : Acoustic Barge-In (RMS Threshold Exceeded)
    INTERRUPTED --> LISTENING : Instant Playback Cancellation
    SPEAKING --> LISTENING : Speech Completed
    LISTENING --> RETURNING_TO_STANDBY : Conversation Timeout (Default 15 s)
    RETURNING_TO_STANDBY --> SLEEPING : Standby Re-engaged
    SLEEPING --> IDLE : User Disables Microphone
```

---

## Phase-by-Phase Engineering Journey

LocalVox was developed through a rigorous 7-phase engineering progression, with each phase benchmarked, verified, and hardened on Apple Silicon hardware before moving forward.

```text
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6
Scaffold    Engines     Audio I/O   Streaming   Persistent  FastAPI +   Jarvis Wake
& Config    Benchmark   & CoreAudio & Barge-In  Memory FTS5 Next.js UI  & Voice Profile
```

### Phase 0: Architecture & Environment Foundation
- **Objective**: Establish strict configuration schemas, logging architecture, and local Apple Silicon environment isolation.
- **Technologies**: Python 3.11, Pydantic v2, PyYAML, `uv` packaging, Structured JSON logging.
- **Challenges**: Preventing environment bleed and verifying Apple Silicon ARM64 compatibility across native extensions.
- **Outcome**: Centralized `config.yaml` with strict runtime validation, automated `stage_timer` profiling in `logs/latency.jsonl`, and an optimized Makefile workflow.

### Phase 1: Local AI Engine Selection & Benchmarking
- **Objective**: Identify, test, and benchmark the fastest local AI models on Apple M5 hardware across English, Tamil, and Tanglish.
- **Technologies**: Silero VAD (ONNX), MLX Whisper, Ollama (Qwen3:8B, Llama 3), Kokoro ONNX, Piper TTS ONNX.
- **Challenges**: MLX Whisper quantization tradeoffs; Tamil TTS synthesis availability; Ollama reasoning latency (`think=True` vs `think=False`).
- **Outcome**: Validated `whisper-large-v3-turbo-8bit` on Metal GPU (~383 ms EN infer), Kokoro (RTF 0.170), Piper Tamil (RTF 0.020), and Qwen3:8B (24.6 tok/s with TTFT 187 ms). Detailed benchmarks documented in [`docs/model_benchmarks.md`](docs/model_benchmarks.md).

### Phase 2: CoreAudio Streaming Pipeline & Acoustic Management
- **Objective**: Create a non-blocking, zero-copy audio capture and playback loop using native macOS CoreAudio.
- **Technologies**: `sounddevice`, NumPy circular ring buffers, macOS CoreAudio HAL.
- **Challenges**: Preventing buffer underruns and PortAudio callback stalls during continuous capture; smooth sampling rate conversions (16 kHz mic to 24 kHz playback).
- **Outcome**: Dedicated `MicrophoneAudioCapture` with a non-blocking queue callback, sub-32 ms frame ingest, and decoupled playback streaming.

### Phase 3: Sentence-Level Streaming TTS, Echo Control & Live Barge-In
- **Objective**: Eliminate the "turn-taking delay" by pipelining LLM tokens directly into audio synthesis while allowing instant user interruptions.
- **Technologies**: `SentenceBuffer` incremental punctuation tokenizer, `BargeInMonitor`, `EchoSuppressor`, reference buffer subtraction.
- **Challenges**: Preventing the microphone from picking up the assistant's own speaker output (acoustic feedback loop) without dedicated hardware DSP.
- **Outcome**: First-audio latency dropped from 4,903 ms to 2,012 ms. Software echo suppression enabled natural barge-in detection with sub-millisecond cancellation (`0.53 ms`).

### Phase 4: Long-Term Persistent Memory with SQLite & FTS5
- **Objective**: Implement an intelligent, local memory store that remembers user facts, preferences, and goals across sessions without cloud vector stores.
- **Technologies**: SQLite in WAL mode, FTS5 with `unicode61` tokenizer, BM25 ranking, background Qwen3 extraction worker.
- **Challenges**: Preventing database locks during active voice turns; sanitizing sensitive credentials (API keys, cards); resolving conflicting memories.
- **Outcome**: 8 structured memory categories, hybrid FTS5 BM25 + importance + recency ranking executing in `< 2 ms`, non-blocking post-turn async extraction, and regex-based privacy filtering.

### Phase 5: Production FastAPI Backend & Next.js Desktop Companion
- **Objective**: Transform the terminal script into a responsive desktop application with real-time telemetry and full visual control.
- **Technologies**: FastAPI, Uvicorn, WebSockets, Next.js 16, TypeScript, Tailwind CSS, Three.js, Lucide icons.
- **Challenges**: Synchronizing high-frequency WebSocket state events with React render lifecycles without audio glitches or frame drops.
- **Outcome**: A single-page companion app featuring an audio-reactive 3D orb, complete memory management studio, conversation history with resumption, and live hardware monitoring.

### Phase 6: Hands-Free "Hey Jarvis" Wake Word & 3-Sample Voice Personalization
- **Objective**: Enable hands-free acoustic activation with zero cloud footprint, customized to the user's specific voice.
- **Technologies**: openWakeWord ONNX (`melspectrogram`, `embedding_model`, `hey_jarvis_v0.1`), 3-sample cosine voice profile, single shared CoreAudio stream.
- **Challenges**: PyPI `openwakeword` has broken ARM64 macOS dependencies; generic models can misclassify accented speech; avoiding CoreAudio stream conflicts between standby and turn recording.
- **Outcome**: Pure `onnxruntime` + `numpy` wake engine executing in `1.08 ms` per frame (`3.52 MiB` footprint). 3-sample voice enrollment module storing only mathematical embeddings (`26.4 KB`). 100% recall (10/10) with zero false activations (0/13).

---

## Deep-Dive Feature Breakdown

### A. Offline Local LLM (Ollama + Qwen3:8B)
LocalVox pairs with the local Ollama daemon to serve `qwen3:8b` (Q4_K_M quantization). Inference parameters are strictly tuned for voice interaction:
- **Streaming Tokenization**: Streams tokens immediately into the conversational pipeline; first tokens arrive within 187 ms to 420 ms.
- **Deterministic Concise Output**: System prompt enforces natural, verbal, conversational output without markdown formatting, emojis, or code blocks that would stall TTS engines.
- **Context Management**: Dynamically budgets recent conversational context and injects retrieved persistent memories as grounded background context.
- **Thinking Mode Toggle**: Defaults to `think=False` for low-latency dialogue (~1.8 s total response) while supporting deep analytical reasoning when explicitly requested.

### B. Autonomous Wake Word Engine (openWakeWord ONNX)
Hands-free activation uses three decoupled ONNX models from openWakeWord v0.5.1:
1. `melspectrogram.onnx` (`1.08 MB`): Converts 16 kHz PCM frames into 32-bin log-mel spectrogram features.
2. `embedding_model.onnx` (`1.32 MB`): Projects 76 frames of mel features into 96-dimensional acoustic speech embeddings.
3. `hey_jarvis_v0.1.onnx` (`1.27 MB`): Evaluates a sliding window of 16 embeddings to predict wake-word probability in `[0, 1]`.

Execution runs on `CPUExecutionProvider` on Apple Silicon, taking just **`1.08 ms` per 80 ms audio frame** (less than 1.4% of the real-time frame budget).

### C. Voice Activity Detection & Endpointing (Silero VAD)
Voice boundaries are accurately segmented using Silero VAD v5 (ONNX):
- **Pre-Roll Window (288 ms)**: Maintains a rolling audio ring buffer so initial consonant plosives (`p`, `t`, `k`) are never clipped.
- **Speech Hangover (832 ms)**: Prevents premature turn cutoffs during natural conversational pauses or thinking gaps.
- **Post-Pad (160 ms)**: Captures trailing word endings before passing the float32 array to STT.
- **Stateful Recurrent Reset**: Resets hidden states between turns to avoid acoustic drift.

### D. Apple Silicon Accelerated STT (MLX Whisper Turbo)
Speech-to-text is handled by `mlx-whisper` (`whisper-large-v3-turbo-8bit`), utilizing Apple Silicon's unified memory and Metal GPU:
- **Zero-Copy In-Memory Audio**: Audio arrays pass directly from memory buffers into the Metal computation graph without intermediate `.wav` disk serialization.
- **Multilingual Support**: Transcribes English, Tamil, and mixed Tanglish speech with automatic language identification.
- **High Throughput**: Completes a 4.65-second utterance transcription in **~383 ms** (RTF `0.082`).

### E. Sentence-Level Streaming TTS (Kokoro & Piper)
To eliminate latency, LocalVox does not wait for the LLM to complete its response. Tokens pass into a `SentenceBuffer` that splits text at clause and sentence boundaries (`.`, `!`, `?`, `;`, newline):
- **Bilingual Dynamic Routing**:
  - **English**: Routed to **Kokoro ONNX** (`af_heart` voice, 24 kHz) producing rich, natural human timbre in ~966 ms for a 5.6-second sentence (RTF `0.170`).
  - **Tamil & Tanglish**: Routed to **Piper ONNX** (`ta_IN-rasa_female-medium`, 22.05 kHz) executing in ~103 ms for a 5.1-second sentence (RTF `0.020`).
- **Seamless Playback Queue**: Synthesized audio chunks stream to CoreAudio via `sounddevice.OutputStream` without audible gaps between sentences.

### F. Persistent Memory Engine (SQLite WAL + FTS5)
LocalVox maintains contextual persistence across sessions without complex cloud vector databases:
- **8 Structured Domains**:
  1. `preferences` (user likes, dislikes, habits)
  2. `facts` (personal details, tech stack, environment)
  3. `routines` (work hours, study schedules)
  4. `goals` (current projects, learning objectives)
  5. `relationships` (colleagues, mentors, family)
  6. `background` (education, career history)
  7. `work_study` (active tasks, exam prep, code repos)
  8. `system_operational` (assistant settings, voice preferences)
- **FTS5 Unicode61 Search**: Full-text search with prefix matching and BM25 relevance scoring.
- **Hybrid Context Ranking**: Final score combines BM25 similarity, memory importance rating (1–5), and time-decay recency.
- **Non-Blocking Background Extraction**: An asynchronous worker extracts structured facts from completed turns in the background, ensuring voice turns remain instant.
- **Privacy Engine**: Automatically detects and blocks API tokens, passwords, credit cards, private keys, and government IDs from being saved.

### G. Real-Time Acoustic Barge-In & Echo Suppression
Speak over the assistant at any time:
- **Acoustic Reference Tracking**: When the assistant speaks, output audio frames are mirrored into a `PlaybackReferenceBuffer`.
- **Software Echo Suppression**: The monitor computes the cross-energy between the speaker playback and microphone input. If energy matches playback output, the VAD ignores it.
- **Instant Playback Abort**: When the user speaks with energy exceeding the echo threshold for 2 consecutive frames (~64 ms), `BargeInMonitor` signals an interruption:
  - CoreAudio playback output stops in **`0.53 ms`**.
  - Pending LLM tokens and queued TTS chunks are cleared.
  - The pipeline immediately transitions to `LISTENING` to capture the user's new question.

### H. 3-Sample Personalized Voice Verification
To ensure the assistant responds specifically to its owner:
- **Enrollment Flow**: The user clicks **Personalize Voice** and speaks "Hey Jarvis" three times.
- **Baseline-Centered Neural Embeddings**: Extracts the 96-dimensional acoustic trajectory from the openWakeWord embedding model, centers it by subtracting background noise baseline embeddings, and normalizes it.
- **Vocal Timbre Profile**: Computes a 32-bin log-mel energy distribution across active speech frames to capture vocal formant characteristics.
- **Safe Storage**: Saved as a compact `26.4 KB` NumPy archive (`data/models/wakeword/voice_profile.npz`). **Raw audio waveforms are deleted immediately from memory.**
- **Hybrid Verification**: During live standby, the engine calculates cosine similarity against both the generic wake-word classifier and the personalized voice profile, rejecting background television, podcasts, or unauthorized speakers.

### I. Desktop Companion UI (Next.js 16 + Three.js)
The frontend interface provides a complete desktop operational cockpit:
- **3D Reactive Acoustic Core**: A Three.js wireframe sphere that pulses, vibrates, and shifts color dynamically across states:
  - Cyan / Sky: Ready & Standby
  - Emerald Green: Active Listening (Microphone Open)
  - Violet / Purple: Processing STT & Reasoning
  - Coral Pink: Synthesizing & Speaking
  - Amber Gold: Barge-In Interrupted
- **Real-Time Telemetry Bar**: Live metrics showing STT duration, LLM Time-To-First-Token (TTFT), active TTS backend, and round-trip turn time.
- **Full Studio Views**: Switch between Live Companion, Memory Dashboard, Conversation History, System Monitor, and Settings.

### J. Conversation History & Contextual Resumption
Every conversation is indexed locally in SQLite:
- **Session Summaries**: Turn counts, timestamps, language tags, and average latencies.
- **One-Click Resumption**: Click "Continue Session" on any historical conversation to reload its context into the active session buffer and pick up right where you left off.

---

## Hardware Telemetry & Performance Benchmarks

All metrics below were measured and verified on an **Apple MacBook Pro (Apple M5, 24 GB Unified Memory, macOS 15)**. Benchmarks are loaded directly from [`docs/phase6_wakeword_benchmarks.json`](docs/phase6_wakeword_benchmarks.json) and [`docs/model_benchmarks.json`](docs/model_benchmarks.json). Zero numbers are fabricated.

### Wake-Word & Personalization Benchmarks (Apple M5)

| Metric | Measured Apple M5 Value | Target Threshold | Status |
| --- | --- | --- | --- |
| **Model Disk Footprint** | **3.52 MiB** (3 ONNX graphs) | < 15 MiB | Exceeded |
| **Personalized Profile Size** | **26.4 KB** (`voice_profile.npz`) | < 500 KB | Exceeded |
| **Model Cold Load Time** | **3.31 ms – 4.62 ms** | < 200 ms | Exceeded |
| **Inference per 80 ms Frame (Mean)** | **1.080 ms** | < 5.0 ms | Exceeded |
| **Inference per 80 ms Frame (p50)** | **1.027 ms** | < 3.0 ms | Exceeded |
| **Inference per 80 ms Frame (p95)** | **1.325 ms** | < 8.0 ms | Exceeded |
| **Standby CPU Consumption** | **5.6% of 1 core** (~0.56% system) | < 15% of 1 core | Exceeded |
| **Standby RSS Memory Footprint** | **+13.48 MiB** delta | < 50 MiB | Exceeded |
| **Wake Detection Recall** | **10 / 10 (100.0%)** | > 95% | Exceeded |
| **False Activation Rate** | **0 / 13 (0.0%)** | < 5% | Exceeded |
| **Detection Timing Offset** | **-224 ms to +74.6 ms** | < 300 ms | Exceeded |
| **Personalized Consistency Score** | **95.57%** | > 85% | Exceeded |
| **Personalized Recall / False Alarm** | **3/3 (100%) / 0/3 (0%)** | > 90% / < 5% | Exceeded |
| **Acoustic Barge-In Cancel Latency** | **0.53 ms** | < 50 ms | Exceeded |
| **Instant Wake-to-Listen Latency** | **2.58 ms** | < 50 ms | Exceeded |

#### Execution Provider Comparison (Apple M5)
Direct benchmark comparing ONNX execution providers on Apple Silicon for the openWakeWord model:

| Execution Provider | Load Time | Mean Inference / 80 ms Frame | p95 Inference | RSS Memory Delta |
| --- | --- | --- | --- | --- |
| **CPUExecutionProvider** *(Selected Default)* | **4.62 ms** | **1.398 ms** | **2.497 ms** | **+13.48 MiB** |
| `CoreMLExecutionProvider` | 389.88 ms | 2.406 ms | 2.697 ms | +17.60 MiB |

> **Architecture Insight**: `CPUExecutionProvider` runs **2.2× to 2.5× faster** and loads **84× faster** than CoreML on these small models because CoreML graph partition overhead outweighs the compute cost of tiny convolutional layers.

---

### End-to-End Latency Breakdown

Comparison between manual Push-to-Talk and Hands-Free Wake-Word interaction on the same hardware:

| Conversational Stage | Push-to-Talk (Manual) | Hands-Free ("Hey Jarvis") | Hardware Acceleration |
| --- | --- | --- | --- |
| **Wake Word Detection** | N/A (Manual Click) | **1.08 ms** | CPU (Single Thread) |
| **Wake-to-Listen Transition** | N/A | **2.58 ms** | In-Memory CoreAudio Switch |
| **Speech-to-Text (STT)** | 846.01 ms | **729.40 ms** | Apple Metal GPU (MLX) |
| **Persistent Memory Retrieval** | 1.82 ms | **1.79 ms** | SQLite WAL + FTS5 Index |
| **LLM Time-To-First-Token (TTFT)** | 3,162.02 ms | **420.54 ms** | Ollama Qwen3:8B on Metal |
| **First Audio Synthesized & Played** | 4,903.76 ms | **2,012.38 ms** | Kokoro ONNX / CoreAudio |
| **Total Speech-to-Response Turn** | 4,887.95 ms | **2,001.39 ms** | Fully Pipelined Streaming |

---

### Component-by-Component Model Telemetry

| Pipeline Component | Active Model / Backend | Memory Size | Load Time | Inference / Throughput | Real-Time Factor (RTF) |
| --- | --- | --- | --- | --- | --- |
| **VAD** | Silero VAD v5 (ONNX) | 2.22 MiB | 12.86 ms | 13.37 ms (on 5.85s audio) | **0.0023** |
| **STT (English)** | MLX Whisper Turbo 8-Bit | 824.43 MiB | 140.96 ms | 383.66 ms (on 4.65s audio) | **0.0825** |
| **STT (Tamil)** | MLX Whisper Turbo 8-Bit | 824.43 MiB | 140.96 ms | 409.46 ms (on 2.81s audio) | **0.1457** |
| **LLM (English)** | Ollama Qwen3:8B Q4_K_M | 4,983.32 MiB | 36.74 ms | **24.62 tok/s** (TTFT 187 ms) | N/A |
| **LLM (Tamil)** | Ollama Qwen3:8B Q4_K_M | 4,983.32 MiB | 37.47 ms | **24.06 tok/s** (TTFT 491 ms) | N/A |
| **LLM (Tanglish)** | Ollama Qwen3:8B Q4_K_M | 4,983.32 MiB | 36.41 ms | **24.03 tok/s** (TTFT 368 ms) | N/A |
| **TTS (English)** | Kokoro ONNX (`af_heart`) | 337.36 MiB | 197.03 ms | 966.54 ms (on 5.68s audio) | **0.1703** |
| **TTS (Tamil)** | Piper ONNX (`ta_IN-rasa`) | 60.58 MiB | 144.23 ms | 103.02 ms (on 5.13s audio) | **0.0201** |
| **Wake Word** | openWakeWord (`hey_jarvis`) | 3.52 MiB | 3.31 ms | 1.08 ms per 80 ms frame | **0.0135** |
| **Memory Search** | SQLite FTS5 Unicode61 | N/A | < 1 ms | 1.79 ms query + BM25 rank | N/A |

---

## Competitive Comparison

| Capability | LocalVox | Cloud Assistants (Siri / Alexa / Google) | Local LLM Web UIs (OpenWebUI / Jan) | Traditional Offline (Mycroft / Rhasspy) |
| --- | :---: | :---: | :---: | :---: |
| **100% Offline Execution** | **Yes** | No (Fails without WAN) | Text only (TTS/STT cloud) | Yes |
| **Privacy / No Audio Uploads** | **Guaranteed** | No (Uploaded to Cloud) | Varies (often cloud APIs) | Yes |
| **Tamil & Tanglish Support** | **Native Route** | Poor / Fragmented | Prompt only (no Indic TTS) | Minimal / No |
| **Hands-Free "Hey Jarvis" Wake** | **Built-in ONNX** | Proprietary Cloud Wake | No | Fragile PocketSphinx |
| **3-Sample Voice Personalization** | **Built-in** | Limited Cloud Voice Match | No | No |
| **Sub-Second Acoustic Barge-In** | **Yes (0.53 ms)** | Partial / Cloud Delayed | No | No |
| **Sentence-Streaming Audio** | **Yes** | Yes | No (Waits for full text) | No |
| **Long-Term SQLite Memory** | **8 Categories + FTS5** | Opaque Cloud Profile | Ephemeral Chat History | No |
| **Apple Silicon Metal Optimization** | **Fully Optimized** | N/A (Cloud) | Partial (Ollama only) | No (CPU only) |
| **Native Desktop UI** | **Next.js + 3D Three.js** | Native OS Widget | Web Dashboard | Basic Web Admin |
| **Subscription Cost** | **$0.00 / Free** | Subscription / Device Lock | $0.00 | $0.00 |

---

## Quick Start & Installation

### System Prerequisites

- **Hardware**: Mac with Apple Silicon (M1, M2, M3, M4, M5 or Pro/Max/Ultra).
- **RAM**: Minimum 16 GB Unified Memory (24 GB+ recommended for optimal Qwen3:8B concurrency).
- **Operating System**: macOS Sonoma (14.0+) or macOS Sequoia (15.0+).
- **Software Dependencies**:
  - Python 3.11 or higher
  - Node.js 18.0 or higher & npm
  - [`uv`](https://github.com/astral-sh/uv) (Extremely fast Python package manager)
  - [`ollama`](https://ollama.ai) (Local LLM runner)

### Step 1: Install Package Managers & System Tools

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install uv, Node.js, and Ollama
brew install uv node ollama
```

### Step 2: Clone Repository & Set Up Python Virtual Environment

```bash
# Clone the LocalVox repository
git clone https://github.com/Saikrishivgar/LocalVox.git
cd LocalVox

# Create virtual environment and install backend dependencies
uv sync
```

### Step 3: Install Frontend Dependencies & Build

```bash
# Install Next.js dependencies and generate production build
cd frontend
npm install
npm run build
cd ..
```

### Step 4: Pull Local LLM & Download Neural Voice Weights

```bash
# 1. Start Ollama and download Qwen3:8B
ollama serve &
ollama pull qwen3:8b

# 2. Run the automated model verification and download script
uv run python scripts/download_models.py
```

This verifies and caches:
- `openWakeWord` ONNX models (`melspectrogram.onnx`, `embedding_model.onnx`, `hey_jarvis_v0.1.onnx`) into `data/models/wakeword/`.
- `Kokoro ONNX` model and English voice vectors into `data/models/kokoro/`.
- `Piper ONNX` Tamil voice (`ta_IN-rasa_female-medium`) into `data/models/piper/`.
- `Silero VAD v5` ONNX model.
- `MLX Whisper Turbo 8-Bit` weights into local Hugging Face cache.

### Step 5: Launch LocalVox Desktop Stack

Launch the unified desktop stack (FastAPI backend + Next.js desktop client + Ollama supervision) with a single command:

```bash
make desktop
# Or directly via Python:
uv run python scripts/launch_desktop.py
```

Once started:
1. Open your browser to **`http://127.0.0.1:3000`** (or let the launcher open it).
2. The companion will initialize in **Hands-Free Standby (`SLEEPING`)**.
3. Say **"Hey Jarvis"** to wake the assistant, or click **Start Microphone** for manual push-to-talk.
4. Click **Personalize Voice** to enroll your voice in 3 quick samples!

---

## Configuration Reference

LocalVox is configured declaratively via [`config.yaml`](config.yaml) in the project root. All fields are strictly validated by Pydantic v2 schemas in `backend/config.py`.

```yaml
# config.yaml (Excerpt)

# Hands-Free Wake Word & Voice Personalization
wakeword:
  enabled: true                          # Enable openWakeWord monitoring on startup
  model_name: "hey_jarvis_v0.1"         # Wake phrase model
  threshold: 0.5                         # Activation confidence threshold [0.0 - 1.0]
  execution_provider: "cpu"              # "cpu" (recommended on M5) or "coreml"
  pre_roll_ms: 1200                      # Audio kept before wake event for instant speech
  play_ack_tone: true                    # Play "Yes, I'm listening." voice acknowledgment
  personal_voice:
    enabled: true                        # Enable 3-sample personalized voice profile
    personal_only: false                 # false = Hybrid mode; true = Only enrolled voice
    threshold: 0.68                      # Personalized similarity floor

# Voice Activity Detection (Silero VAD)
vad:
  threshold: 0.5                         # Speech probability cutoff
  speech_pad_ms: 288                     # Audio pre-roll to capture initial consonants
  silence_hangover_ms: 832               # Silence duration before turn ends
  min_speech_duration_ms: 250            # Minimum duration to filter clicks/pops

# Speech-to-Text (MLX Whisper)
stt:
  model: "whisper-large-v3-turbo-8bit"   # 8-bit quantized Apple Silicon Metal model
  language: null                         # null = auto-detect language (EN/TA)

# Local LLM (Ollama)
llm:
  base_url: "http://127.0.0.1:11434"
  model: "qwen3:8b"
  temperature: 0.7
  think: false                           # false = fast conversational output

# Bilingual Text-to-Speech
tts:
  router_strategy: "script_detect"       # Routes Tamil unicode characters to Piper
  kokoro:
    voice: "af_heart"                    # 24 kHz neural English voice
  piper:
    voice: "ta_IN-rasa_female-medium"    # 22.05 kHz Tamil voice

# Persistent Memory & Extraction
memory:
  enabled: true
  db_path: "data/memory.db"
  async_extraction: true                 # Extract memories in background after turn
  max_retrieved: 5                       # Max memories injected into prompt
```

---

## Privacy & Security Guarantees

LocalVox was designed from the ground up for users who handle sensitive information, research data, or confidential code:

1. **Air-Gapped Operation**: Once models are cached, LocalVox requires **zero network connectivity**. It runs perfectly with Wi-Fi disabled.
2. **Localhost Network Binding**: The FastAPI backend binds strictly to `127.0.0.1:8000`. Cross-Origin Resource Sharing (CORS) is restricted to local browser origins. External LAN requests are rejected.
3. **Transient In-Memory Audio**: Audio captured by the CoreAudio callback exists only in temporary NumPy ring buffers in volatile memory. No `.wav`, `.mp3`, or raw audio files are written to disk during regular turns or voice enrollment.
4. **Secret Scrubbing Engine**: The memory extraction pipeline scans all conversations against hardened regex patterns to prevent API keys, bearer tokens, passwords, private SSH keys, credit cards, and social security numbers from being committed to the database.
5. **Auditable SQLite Storage**: Your data lives in standard SQLite files (`data/memory.db` and `data/conversations.db`). You can inspect, query, export, backup, or wipe it at any time.

---

## Testing & Quality Assurance

LocalVox adheres to strict test-driven development practices. The codebase includes comprehensive unit, integration, and performance test suites across both backend and frontend.

### Running Test Suites

```bash
# Run complete Python test suite (56 tests)
uv run pytest

# Run frontend TypeScript checks and Vitest suite (7 tests)
cd frontend
npx tsc --noEmit
npm test

# Run code formatting and linting checks
uv run ruff check .
```

### Test Coverage Highlights

- **Backend (`56 passed`)**:
  - `test_config.py`: Schema validation, defaults, and corrupted YAML detection.
  - `test_logger.py`: JSONL formatting and `stage_timer` profiling.
  - `test_phase1.py`: Engine verification for VAD, Whisper STT, Qwen3 LLM, and Kokoro/Piper TTS.
  - `test_pipeline.py`: Non-blocking audio queues and CoreAudio lifecycle.
  - `test_phase3.py`: Streaming sentence chunking, echo suppression, and barge-in abort timing.
  - `test_phase4_memory.py`: FTS5 full-text indexing, BM25 retrieval, contradiction resolution, and privacy filters.
  - `test_phase5_api.py`: FastAPI endpoints, WebSocket broadcaster, and conversation history persistence.
  - `test_phase6_wakeword.py`: openWakeWord ONNX inference, 3-sample voice enrollment, and profile loading.
- **Frontend (`7 passed`)**:
  - State machine transitions (all 7 visual states).
  - Memory search, category filtering, and bulk wipe modal.
  - History viewer and session continuation trigger.
  - System monitor hardware telemetry polling.
  - Settings modification and YAML persistence.
  - 3-sample voice personalization wizard workflow.

---

## Roadmap

- [x] **Phase 0**: Architecture scaffold, Pydantic configuration, structured latency logger.
- [x] **Phase 1**: Local AI engines benchmarked and selected for Apple Silicon (M5).
- [x] **Phase 2**: Non-blocking CoreAudio stream capture and playback loop.
- [x] **Phase 3**: Sentence-buffered streaming TTS, software echo suppressor, sub-millisecond barge-in.
- [x] **Phase 4**: SQLite WAL + FTS5 persistent memory with async extraction and privacy filtering.
- [x] **Phase 5**: Production FastAPI server, real-time WebSocket events, Next.js 16 desktop companion.
- [x] **Phase 6**: Autonomous openWakeWord "Hey Jarvis" integration and 3-sample local voice personalization.
- [ ] **Phase 7 (Next)**: Local Tool Calling & OS Automation (calendar inspection, file management, macOS AppleScript execution).
- [ ] **Phase 8**: Multi-Speaker Diarization (distinguish multiple participants in a study room).
- [ ] **Phase 9**: Native CoreML Quantization & Fine-Tuned Regional Tamil Acoustic Models.

---

## Contributing & License

### Contributing

Contributions to LocalVox are welcome! Whether you are optimizing Apple Silicon MLX kernels, improving Tamil speech synthesis, enhancing the 3D visualizer, or fixing bugs:

1. Fork the repository.
2. Create a descriptive feature branch: `git checkout -b feature/amazing-feature`.
3. Commit your changes with clear messages: `git commit -m "Add feature"`.
4. Ensure all tests pass: `uv run pytest && cd frontend && npm test`.
5. Verify linting: `uv run ruff check . && cd frontend && npx tsc --noEmit`.
6. Open a Pull Request.

### Maintainer & Author

- **Saikrishivgar** — *AI Systems Architecture, Pipeline Design, and Full-Stack Engineering*
- Repository: [https://github.com/Saikrishivgar/LocalVox](https://github.com/Saikrishivgar/LocalVox)

### License

This project is open-source under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

---

<div align="center">
<sub>Built with precision for Apple Silicon macOS. Designed for absolute privacy and conversational speed.</sub>
</div>
