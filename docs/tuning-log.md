# Tuning log: running a voice assistant on a 4 GB GPU

These are the problems found while building and tuning the assistant, in the order they came up. Each entry gives the cause, the fix and the measured result.

**Machine:**
- RTX 3050 Laptop GPU with 4 GB of VRAM (about 3.7 GB usable)
- Ryzen 7 5800H (8 cores, 16 threads)
- 12 GB of RAM (about 5 GB free)
- System Python 3.14, no sudo access

**Latency** below always means voice-to-voice: the time from when the user stops speaking to the bot's first audio.

## 1. Getting everything to run

1. **Python 3.14 was too new.** The ML libraries (CTranslate2, ONNX Runtime) might not have builds for it. **Fix:** pinned Python 3.12 for the project with `uv` (`.python-version`).
2. **Ollama couldn't be installed:** the official installer needs sudo. **Fix:** extracted the official release archive into `~/.local/ollama` and ran it as a normal user.
3. **Ollama saw 0 VRAM** and said "llama-server not found". **Cause:** it was started before the archive finished extracting, and it checks for a GPU only once, at startup. **Fix:** restarted it after extraction. It then found the GPU (3.7 GB).
4. **Whisper crashed on its first transcription** with "libcublas.so.12 not found". **Cause:** its CTranslate2 backend needs CUDA 12 cuBLAS and cuDNN 9, which weren't installed, and a system CUDA install needs sudo. **Fix:** installed NVIDIA's pip packages for those libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) and loaded them into memory before Whisper starts. No system CUDA is needed.

## 2. Fitting three models into 4 GB of VRAM

5. **Model sizes chosen for 4 GB:**
   - **Whisper `large-v3-turbo`, int8_float16:** about 1 GB, with good Hindi. Smaller Whisper models have poor Hindi.
   - **Llama 3.2 3B, Q4_K_M:** about 2.3 GB, with official Hindi support.
   - **Gemma 3 4B was rejected:** at 3.3 GB it can't fit next to Whisper.
   - **Kokoro TTS runs on the CPU** to save VRAM.
6. **The LLM was one-third on the CPU: 10 tokens/s, and 22 s for the first prompt.** **Cause:** Whisper loaded first, and Ollama's cautious memory estimate pushed some LLM layers to the CPU. **Fix:** load the LLM first, then Whisper, and log a warning if the LLM is ever split. **Result:** 100% on the GPU, **71 tokens/s, 13 ms prompt processing.** Total VRAM is 3.5 of 4 GB during calls.
7. **Ollama tuning:** flash attention on, 8-bit KV cache (`q8_0`), model kept loaded permanently (`keep_alive -1`), and a warm-up at startup so the first reply isn't slow.

## 3. Speech-to-text (Whisper)

8. **Pipecat's Whisper service threw away the detected language,** and Whisper often labels Hindi as Urdu. **Fix:** a custom STT class keeps only English vs Hindi, counts Urdu as Hindi, and switches the TTS voice and the LLM's reply language to match.
9. **The stock Whisper settings were slow:**
   - beam search width 5;
   - decoding ran inside the main audio loop, blocking audio while it worked;
   - the model reloaded on every connection.

   **Fix:** beam width 1, decoding in a background thread, and one model shared across sessions.
10. **The Whisper encoder ran twice per turn:** once to detect the language (0.39 s) and once to transcribe (0.43 s). **Fix:** run the encoder once and reuse its output for both. **Result: 0.82 s → 0.42 s,** with identical transcripts in English and Hindi.
11. **Whisper hallucinated "you" or "Thank you." on silence.** **Cause:** large-v3-turbo's "no speech" score is always about 0, so Pipecat's filter never triggers. **Fix:** filter on the model's average confidence (log-probability) instead. Real speech scored −0.01 to −0.18 and hallucinations −0.6 to −3.0, so the cutoff is −1.0.

## 4. Turn-taking: the bot replying too early

12. **The bot answered before it had heard the user, then answered again.** **Cause:** Whisper's transcript arrives about 0.95 s after the user stops, but Pipecat waits only 1.0 s by default, so the two raced. When the transcript lost, the LLM ran on empty text, and the late transcript then started a second turn that cut off the first. **Fix:** raised the wait to 2 s as a safety net only. A finished transcript still ends the turn at once, so this adds no delay.
13. **It still happened when the user paused mid-sentence.** **Cause:** voice detection split one sentence into two segments, and the first segment's transcript arrived while the second was still being spoken, so the turn ended on half a sentence. **Fix:** while the user is still talking, hold that transcript and join it with the next one. **Result:** zero double turns in every run since.

## 5. Text-to-speech: Kokoro on the CPU, the biggest bottleneck

14. **Kokoro synthesizes a whole chunk before any of it plays,** and on this CPU it runs at about 0.35× real time (RTF 0.32–0.37; RTF is synthesis time divided by audio length). A 20-word sentence meant about 2.5 s of silence first. Four options were tried:

    | Option | Result | Kept? |
    |---|---|---|
    | Kokoro int8 (quantized) | **5× slower** (RTF 1.5–1.9); this CPU lacks fast int8 instructions | No |
    | Kokoro on the GPU | **Out of memory:** a 13 MB allocation failed next to the LLM and Whisper. It also needed CUDA 13 libraries while Whisper uses CUDA 12 | No |
    | Piper TTS | 8× faster (RTF 0.04–0.06), but sounds more robotic | No |
    | Kokoro, with replies split into shorter chunks | Natural voice; later chunks synthesize while earlier ones play | **Yes** |

15. **How replies are split:** at commas once at least 4 words have built up, and the first chunk of each reply is capped at **7 words**. Only that first chunk is on the critical path.
16. **Hindi turns were slower (2.8–3.0 s),** because Hindi words take about twice as long to say. **Fix:** the first chunk is capped at **4 words** in Hindi. **Result: 1.7–2.2 s.**
17. **The greeting took 2.4 s to start.** **Cause:** spoken greetings skip the splitter, so the whole greeting was synthesized at once. **Fix:** queue the greeting one sentence at a time.
18. **Accepted as is:** during a live call Kokoro runs about 0.3 s slower than in isolated benchmarks, because the browser, voice detection and audio processing share the CPU.

## 6. Making a 3B model behave

19. **It answered English with romanized Hindi, and its Hindi was garbled.** **Cause:** "reply in the user's language" is too vague for a 3B model. **Fix:** when the detected language changes, swap in an explicit instruction: "Reply only in English" or "Reply only in Hindi, written in Devanagari". **Result:** clean Hindi (`भारत की राजधानी नई दिल्ली है।`).
20. **It said "I'm a text-to-speech assistant, I don't have a voice".** **Cause:** it took the prompt's mention of a TTS engine literally. **Fix:** reworded the prompt.
21. **It kept telling a story after "Stop, what is 2 plus 2?"** **Fix:** a prompt rule to always answer the latest message. Shorter first chunks also mean interruptions land sooner. **Result:** "It's 4."
22. **The role-play scenarios skipped steps and produced nonsense in Hindi.** **Fix:**
    - each scenario's job is written as **numbered steps**;
    - **temperature lowered from 0.8 to 0.3**;
    - the dentist's emergency rule moved into step 1, because in Hindi the model missed it as a separate rule.

## 7. UI and connection

23. **Each bot sentence appeared twice** (once when queued, once when spoken). **Fix:** show only spoken sentences, so an interrupted reply shows exactly what the user heard.
24. **The latency line was missing,** because it arrived before the reply bubble existed. **Fix:** create the bubble when either one arrives first.
25. **One spoken sentence showed as two chat bubbles.** **Fix:** merged them.
26. **The scenario and language choices were silently lost.** **Cause:** the browser sends them as `requestData`, and the server's form parser dropped that field. **Fix:** read the raw request with Pipecat's own parser and validate the values.

## Summary

**Final hardware split:**

| Component | Runs on | Memory |
|---|---|---|
| Llama 3.2 3B (Q4_K_M) | GPU | 2.3 GB VRAM |
| Whisper large-v3-turbo (int8_float16) | GPU | 1.1 GB VRAM |
| Kokoro TTS, Silero voice detection, Smart Turn | CPU | RAM, inside the bot process (3.3 GB in total) |
| **Total GPU** | | **3.5 of 4 GB** (measured during calls) |

**Latency progress:**

| Stage | Per turn |
|---|---|
| First measured run, before most fixes | 2.0–5.6 s, some turns answered twice |
| After segment merging (#13) | 1.45–2.53 s, no double turns |
| After the 7-word first chunk (#15) | 1.15–1.82 s |
| **Final, general assistant (15 turns)** | **median 1.71 s, range 1.15–2.18 s** |
| Scenario calls, English / Hindi | 1.5–2.6 s / 1.7–2.2 s |

**Still limited by the hardware:**
- TTS on the CPU is the largest part of each turn's latency.
- The 3B model makes occasional reasoning slips.
- Only about 0.5 GB of VRAM is left free.

An 8 GB GPU would allow Kokoro on the GPU (about 10× faster TTS) and a 7B LLM.
