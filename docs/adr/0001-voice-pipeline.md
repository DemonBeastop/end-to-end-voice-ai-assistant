# ADR-0001: Pipecat pipeline for a local voice assistant on a 4 GB GPU

**Status:** Accepted
**Date:** 2026-10-12
**Deciders:** Project author

## Context

The assignment is a voice assistant built only from open-weight models running locally: STT, LLM and TTS, a web or desktop UI, memory across turns, and low latency. Bonus points go to streaming, voice activity detection (VAD), interruptions, and Hindi + English.

Hard constraints from the target machine:

| Resource | Available |
|---|---|
| GPU | RTX 3050 Laptop, **4 GB VRAM** (about 3.7 GB usable) |
| CPU | Ryzen 7 5800H, 8 cores / 16 threads |
| RAM | 12 GB |

Everything must fit together on the GPU and CPU at once. A turn has a latency budget of about 2 s from when the user stops speaking to the first bot audio.

Two decisions follow: **(1)** what orchestrates the audio pipeline, and **(2)** where each model runs.

## Decision

1. Use **Pipecat 1.12** as the pipeline, with `SmallWebRTCTransport` to the browser and a single hand-written `index.html` that uses Pipecat's JS client.
2. Place the models as follows:

| Stage | Model | Device | Memory |
|---|---|---|---|
| STT | faster-whisper `large-v3-turbo`, int8_float16 | GPU | ~1.1 GB VRAM |
| LLM | `llama3.2:3b` Q4_K_M via Ollama | GPU, all layers | ~2.3 GB VRAM |
| TTS | Kokoro 82M (kokoro-onnx, fp32) | CPU | RAM, inside the bot process (3.3 GB in total) |
| VAD + turn detection | Silero + Smart Turn v3 | CPU | small |

## Options considered (orchestration)

### Option A: Pipecat + SmallWebRTC + custom page (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | Low. About 350 lines of our own Python plus one HTML file |
| Cost | Free, everything local |
| Scalability | One session per GPU, which is fine for this scope |
| Familiarity | New framework, but has local services for every stage |

**Pros:**
- VAD, Smart Turn end-of-turn detection, barge-in interruptions, sentence-streamed TTS, conversation context, RTVI events and metrics to the browser all come built in.
- WebRTC gives echo cancellation and jitter buffering in the browser for free.

**Cons:**
- Framework internals leak in a few places: we replace the TTS text aggregator through a private attribute and override `WhisperSTTService.run_stt`.

### Option B: Hand-rolled FastAPI + WebSocket pipeline
| Dimension | Assessment |
|---|---|
| Complexity | High. VAD gating, barge-in cancellation, sentence chunking and audio jitter handling would all be ours |
| Cost | Free |
| Scalability | Same |
| Familiarity | No framework to learn |

**Pros:** full control, no framework quirks.
**Cons:** roughly 5× the code, concentrated in the parts that are hardest to get right (cancellation and timing).

### Option C: Pipecat + LocalAudioTransport (terminal mic/speaker)
**Pros:** simplest pipeline.
**Cons:** no UI, which fails the interface requirement. A desktop GUI would cost more than option A's HTML page.

## Trade-off analysis (model placement, measured)

- **The LLM must be 100% on the GPU, and it must load first.** If Whisper is already resident, Ollama's offload estimate puts a third of the layers on the CPU: **10 tok/s and a 22 s cold prompt eval**. With all layers on the GPU: **71 tok/s and a 13 ms prompt eval**. `bot.py` loads the LLM before Whisper and warns if any layers end up on the CPU.
- **Whisper uses a single encoder pass.** Calling `detect_language()` and then `transcribe()` runs the encoder twice (0.39 s + 0.43 s). Encoding once and reusing it for detection and decoding takes **0.42 s**, with identical transcripts in English and Hindi.
- **TTS on the CPU: Kokoro, not Piper.** We tried four ways to run TTS:

  | TTS option | Speed (RTF) | Note |
  |---|---|---|
  | Kokoro fp32, CPU | 0.32–0.37 | chosen |
  | Kokoro int8, CPU | 1.5–1.9 | slower; this CPU lacks fast int8 paths |
  | Kokoro, GPU | — | out of memory next to Whisper and the LLM |
  | Piper, CPU | 0.04–0.06 | much faster, audibly more robotic |

  RTF (real-time factor) is synthesis time divided by audio length; below 1 means faster than real time. We kept Kokoro for naturalness and hid its speed instead: a custom aggregator caps each reply's **first** chunk at 7 words (4 in Hindi) or the first clause, and later chunks synthesize while earlier ones play.
- **Language routing.** Whisper detects the language of each turn, restricted to en/hi; Urdu probability is folded into Hindi because spoken Hindi and Urdu are near-identical. When the language changes, the STT pushes an LLM system-prompt update ("Reply only in English" / "Reply only in Hindi, in Devanagari") and a Kokoro voice change, both ahead of the transcript. Without the explicit instruction, the 3B model drifted into romanized Hindi; with it, its Hindi was clean.

## Consequences

**Easier:**
- Swapping any model is an environment variable (`LLM_MODEL`, `WHISPER_MODEL`, …).
- Pipecat metrics provide per-stage TTFB and voice-to-voice latency for free.

**Harder:**
- VRAM headroom is about 0.5 GB (3.5 of 4 GB used during calls), and even an ONNX Runtime CUDA session for Kokoro didn't fit in it. A bigger LLM such as Gemma 3 4B won't fit next to Whisper.
- Two Pipecat internals are patched: `run_stt` and the TTS `_text_aggregator`. Upgrading Pipecat needs the end-to-end run repeated.
- One user at a time; the Whisper model is shared across sessions.

**Revisit:**
- If the GPU grows to 8 GB or more, move Kokoro to the GPU (about 10× faster TTS) and try a 7B LLM.
- If sessions get long, trim the LLM context; currently the full history is kept.
- large-v3-turbo's `no_speech_prob` is always about 0, so hallucinations on noise are filtered by average log-prob instead (`MIN_AVG_LOGPROB`). Re-tune that threshold on the real microphone.

## Action items
1. [x] Pipeline, UI, en/hi routing, checks (`test_voice.py`)
2. [x] End-to-end runs with a scripted fake microphone; median voice-to-voice latency 1.71 s (15 turns, range 1.15–2.18 s)
3. [ ] Re-tune `MIN_AVG_LOGPROB` and VAD on the demo microphone and room
