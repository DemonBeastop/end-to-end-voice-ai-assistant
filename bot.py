"""Local voice assistant: browser mic -> WebRTC -> Whisper -> Ollama -> Kokoro -> browser speaker."""

import json
import os
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.workers.runner import WorkerRunner

from lang import VOICES, LangWhisperSTT, load_whisper, tts_settings
from roles import ROLES, system_prompt

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8_float16")

class ClauseAggregator(SimpleTextAggregator):
    """Sentence aggregator tuned for time-to-first-audio with Kokoro on CPU.

    Kokoro runs ~0.35x realtime and synthesizes a whole chunk before any of it plays, so the
    first chunk of a reply sets the latency (a 12-word sentence ~1.2 s). Later chunks are
    synthesized while earlier ones play, so splitting them costs nothing. Besides sentence
    ends, text is released:
    - at ", " or ": " once 4+ words are buffered (short "Sure," chunks sound choppy);
    - for a reply's first chunk only, at any word boundary once 7 words are buffered (4 in
      Hindi, whose words take about twice as long to say).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._first = True  # next chunk opens a reply

    async def _check_sentence_with_lookahead(self, char: str) -> Aggregation | None:
        result = await super()._check_sentence_with_lookahead(char)
        if result is None and char == " " and self._lookahead_state.name == "IDLE":
            words = len(self._text.split())
            clause_end = self._text[-2:-1] in (",", ":") and words >= 4
            first_cap = 4 if self.language == "hi" else 7
            if clause_end or (self._first and words >= first_cap):
                result = Aggregation(text=self._text.strip(" "), type=AggregationType.SENTENCE)
                self._text = ""
        if result:
            self._first = False
        return result

    async def flush(self) -> Aggregation | None:  # end of reply
        self._first = True
        return await super().flush()

    async def handle_interruption(self):
        self._first = True
        await super().handle_interruption()


async def run_bot(connection: SmallWebRTCConnection, role: str, language: str | None):
    """One call: `role` picks the scenario, `language` is "en"/"hi" or None for auto-detect."""
    lang = language or "en"  # language of the greeting and first reply
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = LangWhisperSTT(
        role=role,
        language=language,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        settings=LangWhisperSTT.Settings(model=WHISPER_MODEL, language=None),
        # Safety net only: each segment's transcript is final, so the turn ends as soon as it
        # arrives. The 1.0 s default raced our ~0.95 s transcript and fired the LLM early.
        ttfs_p99_latency=2.0,
    )
    llm = OLLamaLLMService(
        base_url=f"{OLLAMA_URL}/v1",
        settings=OLLamaLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=system_prompt(role, lang),
            # 0.3, not Ollama's 0.8: the 3B model then follows the scenario steps instead of
            # wandering, and its Hindi stops producing nonsense words.
            temperature=0.3,
        ),
    )
    tts = KokoroTTSService(settings=tts_settings(lang))
    tts._text_aggregator = ClauseAggregator(language=lang)  # no public hook for this

    # ponytail: full history, no trimming; add a turn window if sessions outgrow num_ctx.
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    pipeline = Pipeline(
        [transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg]
    )
    latency = UserBotLatencyObserver()
    worker = PipelineWorker(
        pipeline, params=PipelineParams(enable_metrics=True), observers=[latency]
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        # One frame per sentence: spoken text skips the clause aggregator, so a single frame
        # would synthesize the whole greeting (~2.4 s) before any audio. An interruption also
        # keeps the sentences already spoken in the LLM context.
        greeting = ROLES[role]["greeting"][lang]
        await worker.queue_frames([TTSSpeakFrame(s) for s in re.split(r"(?<=[.?!।])\s+", greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await worker.cancel()

    @worker.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, msg):
        if msg.type == "set-language":
            stt.forced = msg.data if msg.data in VOICES else None

    @latency.event_handler("on_latency_measured")
    async def on_latency(observer, secs):
        await worker.rtvi.send_server_message({"latency": round(secs, 2)})

    @latency.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):
        logger.info("Turn latency:\n  " + "\n  ".join(breakdown.turn_contribution_lines()))

    await WorkerRunner(handle_sigint=False).run(worker)


def preflight():
    """Fail fast if Ollama or the model is missing; load both models before the first call."""
    try:
        tags = json.load(urlopen(f"{OLLAMA_URL}/api/tags", timeout=3))
    except OSError:
        sys.exit(f"Ollama is not reachable at {OLLAMA_URL}. Start it with: ollama serve")
    if not any(m["name"] in (LLM_MODEL, f"{LLM_MODEL}:latest") for m in tags["models"]):
        sys.exit(f"Model {LLM_MODEL} is not pulled. Run: ollama pull {LLM_MODEL}")
    # LLM first: Ollama sizes its GPU offload from free VRAM at load time, and with
    # Whisper already resident it puts ~1/3 of the layers on CPU (10 vs 70 tok/s).
    logger.info(f"Loading {LLM_MODEL} into Ollama...")
    body = json.dumps({"model": LLM_MODEL, "keep_alive": -1}).encode()
    urlopen(Request(f"{OLLAMA_URL}/api/generate", data=body), timeout=120).read()
    for m in json.load(urlopen(f"{OLLAMA_URL}/api/ps", timeout=3))["models"]:
        if m["size_vram"] < m["size"]:
            logger.warning(f"{m['name']} is partly on CPU (slow). Free VRAM, then restart Ollama.")
    logger.info("Loading Whisper...")
    load_whisper(WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)


app = FastAPI()
webrtc = SmallWebRTCRequestHandler()


@app.get("/")
async def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/api/offer")
async def offer(body: dict, background_tasks: BackgroundTasks):
    # The JS client sends camelCase "requestData"; from_dict maps it to request_data.
    request = SmallWebRTCRequest.from_dict(body)
    options = request.request_data if isinstance(request.request_data, dict) else {}
    role, language = str(options.get("role")), str(options.get("language"))
    role = role if role in ROLES else "assistant"
    language = language if language in VOICES else None  # None = auto-detect

    async def start(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection, role, language)

    return await webrtc.handle_web_request(request, start)


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await webrtc.handle_patch_request(request)
    return {"status": "success"}


if __name__ == "__main__":
    preflight()
    uvicorn.run(app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", "7860")))
