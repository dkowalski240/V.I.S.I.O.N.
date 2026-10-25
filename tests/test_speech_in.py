import numpy as np
import pytest

from assistant import events as ev
from assistant.bus import Bus
from assistant.clock import FakeClock
from assistant.config import SpeechInConfig, STTConfig
from assistant.inputs import parse_script
from assistant.speech.audio_in import BLOCK, SAMPLE_RATE, ArraySource, load_audio
from assistant.speech.input import SpeechInput
from assistant.speech.ptt import NoPTT
from assistant.speech.stt import FakeSTT, STTWorker, looks_like_hallucination
from assistant.speech.vad import EnergyVAD, Segmenter

BLK_S = BLOCK / SAMPLE_RATE


def _blocks(pattern, rng):
    """pattern: list of (seconds, loud?) -> list of blocks."""
    out = []
    for secs, loud in pattern:
        for _ in range(int(round(secs / BLK_S))):
            out.append((rng.standard_normal(BLOCK) * (0.2 if loud else 0.001)).astype(np.float32))
    return out


def test_ambient_segment_with_pre_roll(rng):
    seg = Segmenter(SpeechInConfig(), EnergyVAD())
    segs = []
    for k, b in enumerate(_blocks([(1.0, False), (1.0, True), (1.5, False)], rng)):
        segs += seg.feed(b, 100.0 + k * BLK_S, ptt_down=False)
    assert len(segs) == 1 and segs[0].source == "ambient"
    s = segs[0]
    assert 0.25 < 100.0 + 1.0 - s.t_start < 0.40    # starts ~pre_roll_ms (300) before the sound
    assert 1.4 < len(s.audio) / SAMPLE_RATE < 2.2


def test_push_to_talk_records_even_quiet_speech(rng):
    cfg = SpeechInConfig()
    seg = Segmenter(cfg, EnergyVAD())
    blocks = _blocks([(3.0, False)], rng)
    segs = []
    for k, b in enumerate(blocks):
        ts = k * BLK_S
        segs += seg.feed(b, ts, ptt_down=0.5 <= ts <= 1.5)
    assert [s.source for s in segs] == ["ptt"]
    assert 1.0 < len(segs[0].audio) / SAMPLE_RATE < 1.8


def test_ambient_off_only_listens_with_ptt(rng):
    seg = Segmenter(SpeechInConfig(ambient=False), EnergyVAD())
    segs = []
    for k, b in enumerate(_blocks([(0.5, False), (1.0, True), (1.0, False)], rng)):
        segs += seg.feed(b, k * BLK_S, ptt_down=False)
    assert segs == []


def test_stt_worker_publishes_and_filters_hallucinations():
    bus = Bus(FakeClock())
    sub = bus.subscribe(ev.SPEECH_HEARD)
    worker = STTWorker(STTConfig(), bus, FakeSTT(["Hi, I'm Sarah.", "Thank you.", "Thank you."]))
    from assistant.speech.vad import Segment

    audio = np.zeros(16000, dtype=np.float32)
    worker.process(Segment(audio, 1.0, 2.0, "ambient"))
    worker.process(Segment(audio, 3.0, 4.0, "ambient"))   # dropped: classic Whisper hallucination
    worker.process(Segment(audio, 5.0, 6.0, "ptt"))       # kept: the user meant it
    got = sub.drain()
    assert [(m["text"], m["source"], m["timestamp"]) for m in got] == [
        ("Hi, I'm Sarah.", "ambient", 1.0), ("Thank you.", "ptt", 5.0)]
    assert looks_like_hallucination("Thanks for watching!", {})


def test_speech_input_loop_end_to_end(rng):
    bus = Bus(FakeClock())
    sub = bus.subscribe(ev.SPEECH_HEARD)
    src = ArraySource()
    worker = STTWorker(STTConfig(), bus, FakeSTT(["where's my wallet"]))
    si = SpeechInput(SpeechInConfig(), src, EnergyVAD(), NoPTT(), worker)
    for k, b in enumerate(_blocks([(0.5, False), (1.0, True), (1.0, False)], rng)):
        src.push(b, k * BLK_S)
    while si.feed_one(0):
        pass
    seg = worker._q.get_nowait()
    worker.process(seg)
    assert sub.drain()[0]["text"] == "where's my wallet"


def test_load_audio_resamples_to_16k(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(44100, dtype=np.float32), 44100)
    audio = load_audio(str(path))
    assert abs(len(audio) - 16000) < 50 and audio.dtype == np.float32


def test_replay_script_parsing():
    acts = parse_script("""
        # demo
        12.0        ambient   Hi, I'm Sarah
        40.2-42.0   ptt
        31          type      where's my wallet   # comment
    """)
    assert [(a.t0, a.t1, a.action, a.text) for a in acts] == [
        (12.0, 12.0, "ambient", "Hi, I'm Sarah"), (31.0, 31.0, "type", "where's my wallet"),
        (40.2, 42.0, "ptt", "")]
    with pytest.raises(ValueError):
        parse_script("12 shout hello")


def test_bundled_silero_vad_streams():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_whisper")
    from assistant.speech.vad import SileroVAD

    vad = SileroVAD()
    rng = np.random.default_rng(0)
    probs = [vad.prob((rng.standard_normal(BLOCK) * 0.001).astype(np.float32)) for _ in range(30)]
    assert max(probs) < 0.2  # near-silence is not speech
