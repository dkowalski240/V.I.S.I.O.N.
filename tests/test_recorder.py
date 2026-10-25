"""--record: debug window + microphone + the assistant's voice in one video."""

import os
import sys
import threading
import time
import types

import cv2
import numpy as np
import pytest

from assistant.clock import FakeClock
from assistant.config import DebugConfig, load_config
from assistant.recorder import OUT_RATE, Recorder, RecordingSink, _MicTrack
from assistant.speech.output import NullSink

av = pytest.importorskip("av")


def _tone(freq, seconds, sr, amp=0.2):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def _read(path):
    with av.open(path) as c:
        v = c.streams.video[0]
        frames = sum(1 for _ in c.decode(v))
    with av.open(path) as c:
        a = c.streams.audio[0]
        rate, chans = a.codec_context.sample_rate, a.codec_context.channels
        chunks = [f.to_ndarray() for f in c.decode(a)]
    audio = np.concatenate(chunks, axis=1)  # (2, n) planar
    return frames, rate, chans, audio


def test_one_video_with_picture_microphone_and_panned_voice(tmp_path):
    clock = FakeClock()
    rec = Recorder(str(tmp_path / "demo.mp4"), DebugConfig(record_fps=10), clock).start()
    t0 = clock.now()
    # A window loop that is slow for the first second (5 fps) then fast (20 fps).
    times = [t0 + k * 0.2 for k in range(5)] + [t0 + 1.0 + k * 0.05 for k in range(20)]
    for k, t in enumerate(times):
        rec.add_frame(np.full((120, 160, 3), (k * 9) % 255, np.uint8), t)
        time.sleep(0.01)
    sr = 16000
    mic = _tone(1000, 2.0, sr)
    for i in range(0, len(mic), 512):  # what the phone hears, in both ears
        rec.add_mic(mic[i:i + 512], t0 + i / sr, sr)
    voice = np.zeros((int(0.3 * 22050), 2), np.float32)  # the assistant, panned hard left
    voice[:, 0] = _tone(440, 0.3, 22050, amp=0.5)
    rec.add_voice(voice, 22050, t0 + 1.0)

    path = rec.finish(t0 + 2.0)
    assert path == str(tmp_path / "demo.mp4") and os.path.exists(path)
    assert not os.path.exists(rec.video_part) and not os.path.exists(rec.audio_part)  # parts cleaned up
    frames, rate, chans, audio = _read(path)
    assert abs(frames - 20) <= 1          # 2 s at 10 fps: the slow second was filled in
    assert rate == OUT_RATE and chans == 2
    assert abs(audio.shape[1] / OUT_RATE - 2.0) < 0.15   # + AAC encoder priming/padding
    left, right = audio[0], audio[1]
    during = slice(int(1.05 * OUT_RATE), int(1.25 * OUT_RATE))
    before = slice(int(0.3 * OUT_RATE), int(0.8 * OUT_RATE))
    assert _rms(left[during]) > 2 * _rms(right[during])           # the voice kept its pan
    assert _rms(right[before]) > 0.08                               # the microphone is there...
    assert abs(_rms(left[before]) - _rms(right[before])) < 0.02    # ...in both ears


def test_an_interrupted_voice_is_cut_where_it_stopped():
    clock = FakeClock()
    rec = Recorder("unused.mp4", DebugConfig(), clock)
    t0 = clock.now()
    v = rec.add_voice(np.full((22050, 2), 0.3, np.float32), 22050, t0)  # 1 s long...
    rec.cut_voice(v, t0 + 0.4)                                          # ...cut after 0.4 s
    mix = rec.mix(t0, 1.0)
    assert _rms(mix[: int(0.35 * OUT_RATE)]) > 0.2
    assert _rms(mix[int(0.45 * OUT_RATE):]) == 0.0


def test_recording_sink_records_what_is_played_and_where_it_stopped():
    clock = FakeClock()
    rec = Recorder("unused.mp4", DebugConfig(), clock)
    sink = RecordingSink(NullSink(realtime=True), rec, clock)
    stereo = np.full((22050, 2), 0.3, np.float32)
    done = []
    th = threading.Thread(target=lambda: done.append(sink.play(stereo, 22050)))
    th.start()
    time.sleep(0.1)
    clock.advance(0.25)
    sink.stop()
    th.join(2)
    assert done == [False]
    assert len(rec._voices) == 1 and rec._voices[0].end is not None
    assert sink.played  # attributes of the wrapped sink still reachable


def test_microphone_blocks_stay_on_the_clock_after_a_gap():
    track = _MicTrack()
    track.add(np.ones(1600, np.float32), 100.0, 16000)   # 0.1 s
    track.add(np.ones(1600, np.float32), 100.1, 16000)   # continuous
    track.add(np.ones(1600, np.float32), 101.0, 16000)   # after a 0.8 s stall
    audio, t0, sr = track.render()
    assert t0 == 100.0 and sr == 16000
    assert len(audio) == 16000 + 1600
    assert audio[3200:16000].max() == 0.0 and audio[16000:].min() == 1.0


def test_microphone_by_name_picks_one_entry(monkeypatch):
    from assistant.speech.audio_in import resolve_input_device

    devices = [
        {"name": "Microphone Array (Realtek)", "max_input_channels": 2, "hostapi": 0},
        {"name": "Microphone (DroidCam Audio)", "max_input_channels": 1, "hostapi": 0},
        {"name": "Speakers (Realtek)", "max_input_channels": 0, "hostapi": 0},
        {"name": "Microphone (DroidCam Audio)", "max_input_channels": 1, "hostapi": 2},
    ]
    fake = types.ModuleType("sounddevice")
    fake.query_devices = lambda device=None, kind=None: {"hostapi": 2} if kind == "input" else devices
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    assert resolve_input_device("droidcam") == 3          # same audio system as the default mic
    assert resolve_input_device("2") == 2 and resolve_input_device(None) is None
    with pytest.raises(RuntimeError, match="DroidCam Audio"):
        resolve_input_device("Camo")


def test_record_a_replay_end_to_end(tmp_path):
    """The whole app (no models) replaying a short clip with --record."""
    from assistant.app import App, RunOptions

    clip = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
    for k in range(23):  # 1.5 s
        w.write(np.full((240, 320, 3), k * 10, np.uint8))
    w.release()
    cfg = load_config(None)
    cfg.storage.db_path = str(tmp_path / "a.db")
    cfg.storage.snapshot_dir = str(tmp_path / "snaps")
    cfg.storage.log_dir = str(tmp_path / "logs")
    cfg.debug.console_input = False
    cfg.debug.record_fps = 10
    out = str(tmp_path / "out.mp4")
    app = App(cfg, RunOptions(stub=True, replay_video=clip, window=False, record_path=out))
    app.run()
    frames, rate, chans, audio = _read(out)
    assert frames >= 14 and rate == OUT_RATE and chans == 2   # >= 1.5 s of picture, a sound track
