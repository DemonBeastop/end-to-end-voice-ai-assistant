"""Whisper STT that picks English or Hindi per turn and switches the LLM reply language and
Kokoro voice to match."""

import asyncio
from functools import cache

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from pipecat.frames.frames import LLMUpdateSettingsFrame, TranscriptionFrame, TTSUpdateSettingsFrame
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from roles import system_prompt

VOICES = {"en": "af_heart", "hi": "hf_alpha"}

# large-v3-turbo reports no_speech_prob ~0 even for silence, so hallucinations ("you",
# "Thank you.") are caught by their low average token log-prob instead. Measured: real
# speech -0.01..-0.18, hallucinations -0.6..-3.0. Raise toward -0.5 if a noisy mic leaks junk.
MIN_AVG_LOGPROB = -1.0

def tts_settings(lang: str) -> KokoroTTSService.Settings:
    return KokoroTTSService.Settings(voice=VOICES[lang], language=Language(lang))


def pick_language(probs: list[tuple[str, float]]) -> str:
    """Whisper's language probabilities -> 'en' or 'hi'.

    Spoken Hindi and Urdu are near-identical, and Whisper often labels Hindi
    speech as Urdu, so Urdu's probability counts toward Hindi.
    """
    p = dict(probs)
    return "hi" if p.get("hi", 0) + p.get("ur", 0) > p.get("en", 0) else "en"


def _preload_cuda_libs():
    """CTranslate2 dlopens CUDA 12 cuBLAS/cuDNN 9 by soname; load the pip wheels' copies first
    so no system CUDA install or LD_LIBRARY_PATH is needed."""
    import ctypes
    import glob

    import nvidia

    lib = nvidia.__path__[0]
    for f in [
        f"{lib}/cublas/lib/libcublasLt.so.12",
        f"{lib}/cublas/lib/libcublas.so.12",
        *sorted(glob.glob(f"{lib}/cudnn/lib/libcudnn*.so.9")),
    ]:
        ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)


@cache
def load_whisper(name: str, device: str, compute_type: str) -> WhisperModel:
    # ponytail: one model shared by all sessions; fine for one user per GPU.
    if device != "cpu":
        _preload_cuda_libs()
    return WhisperModel(name, device=device, compute_type=compute_type)


class LangWhisperSTT(WhisperSTTService):
    def __init__(self, *, role: str = "assistant", language: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.role = role  # scenario from roles.ROLES; picks the system prompt
        self.forced = language  # "en"/"hi" from the UI selector; None = auto-detect
        self._lang = language or "en"  # language the LLM prompt and TTS voice are set to
        self._held = ""  # transcript of earlier segments of a turn still in progress

    def _load(self):
        # Base class reloads the model per session; reuse it so connecting is instant.
        self._model = load_whisper(self._settings.model, self._device, self._compute_type)

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        m = self._model
        if len(audio) > m.feature_extractor.n_samples:  # > 30 s: rare, use the public API
            return self._transcribe_long(audio)
        # One encoder pass shared by language detection and decoding. detect_language() +
        # transcribe() would run the encoder twice (~0.4 s each on an RTX 3050).
        enc = m.encode(pad_or_trim(m.feature_extractor(audio)))
        probs = [(token[2:-2], p) for token, p in m.model.detect_language(enc)[0]]
        lang = self.forced or pick_language(probs)
        tokenizer = Tokenizer(m.hf_tokenizer, True, task="transcribe", language=lang)
        prompt = m.get_prompt(tokenizer, [], without_timestamps=True)
        r = m.model.generate(
            enc,
            [prompt],
            beam_size=1,
            max_length=m.max_length,
            return_scores=True,
            return_no_speech_prob=True,
        )[0]
        if r.scores[0] < MIN_AVG_LOGPROB or r.no_speech_prob >= self._settings.no_speech_prob:
            return "", lang
        return tokenizer.decode([t for t in r.sequences_ids[0] if t < tokenizer.eot]).strip(), lang

    def _transcribe_long(self, audio: np.ndarray) -> tuple[str, str]:
        lang = self.forced or pick_language(self._model.detect_language(audio)[2])
        segments, _ = self._model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0,  # no slow fallback re-decodes
            # ponytail: the log-prob filter is weaker here (faster-whisper averages over n+1
            # tokens); fine since VAD-gated turns over 30 s always contain speech.
        )
        # Iterate here (worker thread): faster-whisper decodes lazily while segments are read.
        limit = self._settings.no_speech_prob
        text = " ".join(
            s.text.strip()
            for s in segments
            if s.no_speech_prob < limit and s.avg_logprob >= MIN_AVG_LOGPROB
        )
        return text.strip(), lang

    async def run_stt(self, audio: bytes):
        await self.start_processing_metrics()
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        text, lang = await asyncio.to_thread(self._transcribe, pcm)
        await self.stop_processing_metrics()
        text = f"{self._held} {text}".strip()
        if self._user_speaking or not self._segment_queue.empty():
            # The user is still talking (VAD split the turn at a short pause). Every segment
            # transcript counts as final, so emitting this one would end the turn on partial
            # text; merge it into the next segment's transcript instead.
            self._held = text
            return
        self._held = ""
        if not text:
            return
        if lang != self._lang:
            self._lang = lang
            # Sent ahead of the transcript, so they reach the LLM and TTS before this turn's reply.
            yield LLMUpdateSettingsFrame(
                delta=OLLamaLLMService.Settings(system_instruction=system_prompt(self.role, lang))
            )
            yield TTSUpdateSettingsFrame(delta=tts_settings(lang))
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language(lang))
