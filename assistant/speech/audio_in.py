"""Audio sources: 16 kHz mono float32 blocks of 512 samples (32 ms), each
with the shared-clock time of its first sample.

* MicSource: the microphone (sounddevice).
* FileAudioSource: a WAV/MP3/MP4 file for replay, released in real time on
  the replay clock, so audio at 12.4 s lines up with the video frame at 12.4 s.
  (An .mp4 recorded by scripts/record_session.py works for both video and audio.)
* ArraySource: blocks pushed by tests.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..clock import Clock

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK = 512  # Silero VAD's native block size at 16 kHz

# A tap gets every block as it arrives: tap(block, shared_clock_time_of_first_sample, sample_rate).
# The recorder (--record) uses one to put what the assistant hears into the video.
Tap = Callable[[np.ndarray, float, int], None]


def resolve_input_device(spec):
    """A microphone by number, or by (part of) its name, e.g. "DroidCam".

    Windows lists each device once per audio system (MME, DirectSound, WASAPI), which
    makes sounddevice refuse a bare name as ambiguous; this picks the entry in the same
    audio system as the default microphone."""
    if spec is None or isinstance(spec, int):
        return spec
    text = str(spec).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    import sounddevice as sd

    devices = list(sd.query_devices())
    matches = [i for i, d in enumerate(devices)
               if d.get("max_input_channels", 0) > 0 and text.lower() in str(d.get("name", "")).lower()]
    if not matches:
        names = sorted({str(d.get("name")) for d in devices if d.get("max_input_channels", 0) > 0})
        raise RuntimeError(f"No microphone named like {text!r}. Microphones: {names}")
    try:
        preferred = sd.query_devices(kind="input")["hostapi"]
    except Exception:
        preferred = None
    for i in matches:
        if devices[i].get("hostapi") == preferred:
            return i
    return matches[0]


def _feed_taps(taps: List[Tap], block: np.ndarray, ts: float) -> None:
    for tap in taps:
        try:
            tap(block, ts, SAMPLE_RATE)
        except Exception:
            log.exception("Audio tap failed")


class MicSource:
    def __init__(self, device=None, clock: Optional[Clock] = None):
        self.device = device
        self.clock = clock or Clock()
        self._q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue(maxsize=400)
        self._stream = None
        self.ended = False
        self.level = 0.0  # RMS of the latest block, for the debug window
        self.taps: List[Tap] = []

    def start(self) -> "MicSource":
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            block = indata[:, 0].copy()
            ts = self.clock.now() - frames / SAMPLE_RATE
            self.level = float(np.sqrt(np.mean(block ** 2)))
            if self.taps:
                _feed_taps(self.taps, block, ts)
            try:
                self._q.put_nowait((block, ts))
            except queue.Full:
                pass

        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                                      device=resolve_input_device(self.device), callback=callback)
        self._stream.start()
        log.info("Microphone started (%s)", self.device if self.device is not None else "default")
        return self

    def read(self, timeout: float = 0.2) -> Optional[Tuple[np.ndarray, float]]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def _decode_with_av(path: str) -> np.ndarray:
    """PyAV decode + resample (wav, mp3, mp4 audio tracks...). Written against
    the stable PyAV API rather than faster_whisper.decode_audio, which breaks
    on PyAV 19."""
    import av

    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)


def load_audio(path: str) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32."""
    try:
        return _decode_with_av(path)
    except ImportError:
        pass
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio.astype(np.float32)


class FileAudioSource:
    def __init__(self, path: str, clock, offset_s: float = 0.0):
        self.audio = load_audio(path)
        self.clock = clock
        self.offset_s = offset_s
        self._q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue()
        self._stop = threading.Event()
        self.ended = False
        self.level = 0.0
        self.taps: List[Tap] = []
        log.info("Replay audio: %s (%.1f s)", path, len(self.audio) / SAMPLE_RATE)

    def start(self) -> "FileAudioSource":
        threading.Thread(target=self._run, name="audio-file", daemon=True).start()
        return self

    def _run(self) -> None:
        at = getattr(self.clock, "at", None)
        base = self.clock.now()
        n = len(self.audio) // BLOCK
        for k in range(n):
            if self._stop.is_set():
                return
            media_s = self.offset_s + k * BLOCK / SAMPLE_RATE
            ts = at(media_s) if at is not None else base + media_s
            self.clock.sleep_until(ts + BLOCK / SAMPLE_RATE)  # block is complete at its end time
            block = self.audio[k * BLOCK:(k + 1) * BLOCK]
            self.level = float(np.sqrt(np.mean(block ** 2)))
            if self.taps:
                _feed_taps(self.taps, block, ts)
            self._q.put((block, ts))
        self.ended = True

    def read(self, timeout: float = 0.2) -> Optional[Tuple[np.ndarray, float]]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()


class ArraySource:
    def __init__(self):
        self._q: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue()
        self.ended = False
        self.level = 0.0

    def push(self, block: np.ndarray, ts: float) -> None:
        self._q.put((block.astype(np.float32), ts))

    def read(self, timeout: float = 0.0) -> Optional[Tuple[np.ndarray, float]]:
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            return None

    def start(self):
        return self

    def stop(self):
        pass
