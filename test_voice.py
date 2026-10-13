import asyncio
from datetime import date

from pipecat.frames.frames import LLMUpdateSettingsFrame, TranscriptionFrame, TTSUpdateSettingsFrame

from bot import ClauseAggregator
from lang import LangWhisperSTT, pick_language
from roles import REPLY_IN, ROLES, open_slots, system_prompt


def test_pick_language():
    assert pick_language([("en", 0.7), ("hi", 0.2)]) == "en"
    assert pick_language([("hi", 0.6), ("en", 0.3)]) == "hi"
    assert pick_language([("ur", 0.5), ("hi", 0.2), ("en", 0.3)]) == "hi"  # Urdu label -> Hindi
    assert pick_language([("fr", 0.9), ("en", 0.06), ("hi", 0.04)]) == "en"  # other langs ignored
    assert pick_language([]) == "en"


def test_voice_switches_only_on_language_change():
    stt = LangWhisperSTT.__new__(LangWhisperSTT)  # skip model loading
    stt._lang, stt.forced, stt._user_id, stt._held = "en", None, "user", ""
    stt._user_speaking, stt._segment_queue, stt.role = False, asyncio.Queue(), "dentist"

    async def noop():
        pass

    stt.start_processing_metrics = stt.stop_processing_metrics = noop
    turns = iter([("hello", "en"), ("नमस्ते", "hi"), ("कैसे हो", "hi"), ("bye", "en"), ("", "hi")])
    stt._transcribe = lambda audio: next(turns)

    async def turn():
        return [f async for f in stt.run_stt(b"\0\0" * 160)]

    async def main():
        return [await turn() for _ in range(5)]

    t1, t2, t3, t4, t5 = asyncio.run(main())
    assert [type(f) for f in t1] == [TranscriptionFrame]
    assert [type(f) for f in t2] == [LLMUpdateSettingsFrame, TTSUpdateSettingsFrame, TranscriptionFrame]
    assert "Hindi" in t2[0].delta.system_instruction and "Bright Smile" in t2[0].delta.system_instruction
    assert t2[1].delta.voice == "hf_alpha" and t2[2].language == "hi"
    assert [type(f) for f in t3] == [TranscriptionFrame]
    assert "only in English" in t4[0].delta.system_instruction and t4[1].delta.voice == "af_heart"
    assert t5 == []  # silence/hallucination filtered -> nothing pushed, voice unchanged
    assert stt._lang == "en"


def test_segments_of_one_turn_merge():
    stt = LangWhisperSTT.__new__(LangWhisperSTT)
    stt._lang, stt.forced, stt._user_id, stt._held = "en", None, "user", ""
    stt._segment_queue = asyncio.Queue()

    async def noop():
        pass

    stt.start_processing_metrics = stt.stop_processing_metrics = noop
    segments = iter([("Hi, my name is Purin.", "en"), ("And I live in Delhi.", "en")])
    stt._transcribe = lambda audio: next(segments)

    async def main():
        stt._user_speaking = True  # user resumed before segment 1 finished transcribing
        first = [f async for f in stt.run_stt(b"\0\0")]
        stt._user_speaking = False
        second = [f async for f in stt.run_stt(b"\0\0")]
        return first, second

    first, second = asyncio.run(main())
    assert first == []
    assert [f.text for f in second] == ["Hi, my name is Purin. And I live in Delhi."]
    assert stt._held == ""


def test_clause_aggregator():
    async def chunks(text, language="en"):
        agg = ClauseAggregator(language=language)
        out = [a.text async for a in agg.aggregate(text)]
        last = await agg.flush()
        return out + ([last.text] if last else [])

    def run(text, language="en"):
        return asyncio.run(chunks(text, language))

    assert run("Nice to meet you Purin, where in Delhi do you live? Great.") == [
        "Nice to meet you Purin,",
        "where in Delhi do you live?",
        "Great.",
    ]
    # A reply's first chunk is capped at 7 words; later sentences stay whole.
    assert run("Let me tell you a story about a cat named Kito. It was a very brave little cat.") == [
        "Let me tell you a story about",
        "a cat named Kito.",
        "It was a very brave little cat.",
    ]
    assert run("Sure, I can help.") == ["Sure, I can help."]  # too short to split
    assert run("It costs about 1,000 rupees today.") == ["It costs about 1,000 rupees today."]
    assert run("नमस्ते पुरिन जी आप कैसे हैं, मैं ठीक हूँ।", "hi") == ["नमस्ते पुरिन जी आप", "कैसे हैं, मैं ठीक हूँ।"]
    assert run("आपके दांत में दर्द और सूजन के कारण का पता चलेगा।", "hi") == [
        "आपके दांत में दर्द",
        "और सूजन के कारण का पता चलेगा।",
    ]


def test_roles():
    # 10 Oct 2026 is a Saturday; the clinic is closed Sunday, so slots start Monday.
    assert open_slots([["9 AM"], ["10 AM", "4 PM"]], date(2026, 10, 10)) == (
        "Monday 12 October: 9 AM; Tuesday 13 October: 10 AM, 4 PM"
    )
    for role in ROLES:
        for lang in REPLY_IN:
            assert ROLES[role]["greeting"][lang]
            assert system_prompt(role, lang).endswith(REPLY_IN[lang])


if __name__ == "__main__":
    test_pick_language()
    test_voice_switches_only_on_language_change()
    test_segments_of_one_turn_merge()
    test_clause_aggregator()
    test_roles()
    print("ok")
