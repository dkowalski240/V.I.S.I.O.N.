"""Offline text-to-speech engines and short tones.

All engines return mono float32 audio in [-1, 1] plus a sample rate, so the
output module can pan it left/right before playing.

* PiperTTS (default): fast neural voice, ~50-150 ms for a short phrase on
  the CPU. Needs a voice file (scripts/download_models.py fetches
  en_US-lessac-medium into models/piper/).
* Pyttsx3TTS: Windows' built-in SAPI voices. No download, robotic, slower
  (it renders to a temporary WAV so we can pan it).
* PrintTTS: no audio, prints the text. Used by tests and when nothing else loads.

Repeated phrases ("Sarah, 2 o'clock") are cached, so they play instantly.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from collections import OrderedDict
from typing import Protocol, Tuple

import numpy as np

from ..config import TTSConfig

log = logging.getLogger(__name__)


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def synth(self, text: str) -> np.ndarray: ...


class PiperTTS:
    name = "piper"

    def __init__(self, cfg: TTSConfig):
        from piper import PiperVoice, SynthesisConfig

        if not os.path.exists(cfg.piper_voice):
            raise FileNotFoundError(
                f"Piper voice not found at {cfg.piper_voice}. Run: python scripts/download_models.py --only piper"
            )
        self.voice = PiperVoice.load(cfg.piper_voice)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.syn = SynthesisConfig(length_scale=1.0 / max(cfg.speed, 0.1), volume=cfg.volume, normalize_audio=True)

    def synth(self, text: str) -> np.ndarray:
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=self.syn)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)


class Pyttsx3TTS:
    """SAPI5 via pyttsx3. The engine must be used from the thread that created
    it, so it's created lazily on first use (inside the speech thread)."""

    name = "pyttsx3"

    def __init__(self, cfg: TTSConfig):
        import pyttsx3  # noqa: F401  (fail early if missing)

        self.cfg = cfg
        self._engine = None
        self._owner = None
        self.sample_rate = 22050  # updated after the first render

    def _get(self):
        if self._engine is None or self._owner is not threading.current_thread():
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.cfg.pyttsx3_rate)
            self._engine.setProperty("volume", self.cfg.volume)
            self._owner = threading.current_thread()
        return self._engine

    def synth(self, text: str) -> np.ndarray:
        import soundfile as sf

        eng = self._get()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            eng.save_to_file(text, path)
            eng.runAndWait()
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.sample_rate = int(sr)
        return audio.astype(np.float32)


class PrintTTS:
    name = "print"
    sample_rate = 16000

    def __init__(self, cfg: TTSConfig | None = None, seconds_per_word: float = 0.0):
        self.seconds_per_word = seconds_per_word

    def synth(self, text: str) -> np.ndarray:
        # The output module logs every "SAY ..." line, so nothing to print here.
        n = int(self.sample_rate * self.seconds_per_word * max(1, len(text.split())))
        return np.zeros(n, dtype=np.float32)


def make_tts(cfg: TTSConfig) -> TTSEngine:
    """Build the configured engine, falling back piper -> pyttsx3 -> print."""
    order = {"piper": ["piper", "pyttsx3", "print"], "pyttsx3": ["pyttsx3", "print"], "print": ["print"]}[cfg.engine]
    for name in order:
        try:
            if name == "piper":
                return PiperTTS(cfg)
            if name == "pyttsx3":
                return Pyttsx3TTS(cfg)
            return PrintTTS(cfg)
        except Exception as e:  # missing package, missing voice file...
            log.warning("TTS engine %s unavailable (%s); trying the next one", name, e)
    return PrintTTS(cfg)


class CachedTTS:
    """LRU cache in front of an engine. Thread-safe."""

    def __init__(self, engine: TTSEngine, size: int = 128):
        self.engine = engine
        self.name = engine.name
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    def synth(self, text: str) -> np.ndarray:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
        audio = self.engine.synth(text)
        with self._lock:
            self._cache[text] = audio
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return audio


# ------------------------------------------------------------------------ tones
_TONES = {
    # name: list of (frequency Hz, duration s); 0 Hz = silence
    "listen": [(880, 0.07)],
    "done": [(660, 0.06), (0, 0.02), (990, 0.08)],
    "error": [(330, 0.08), (0, 0.02), (220, 0.12)],
    "person": [(1040, 0.04), (0, 0.04), (1040, 0.04)],
    "capture": [(520, 0.12)],
}


def tone(name: str, sample_rate: int, volume: float = 0.35) -> np.ndarray:
    parts = []
    for freq, dur in _TONES.get(name, _TONES["listen"]):
        n = int(sample_rate * dur)
        t = np.arange(n, dtype=np.float32) / sample_rate
        if freq <= 0:
            parts.append(np.zeros(n, dtype=np.float32))
            continue
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
        fade = min(int(0.005 * sample_rate), n // 2)
        if fade > 0:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            wave[:fade] *= ramp
            wave[-fade:] *= ramp[::-1]
        parts.append(wave * volume)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def to_stereo(mono: np.ndarray, gains: Tuple[float, float]) -> np.ndarray:
    left, right = gains
    return np.stack([mono * left, mono * right], axis=1).astype(np.float32)
