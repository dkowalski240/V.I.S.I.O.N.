"""Speech-to-text with faster-whisper (local), on its own thread.

Segments from the VAD/push-to-talk segmenter are queued here; each one is
transcribed and published as

    {"type": "speech_heard", "text": "...", "timestamp": t_start, "t_end": ..., "source": "ambient"|"ptt"}

Speed: small.en on the RTX 4060 with int8_float16 takes roughly 0.1-0.3 s for
a 2-second utterance. The GPU is only busy while someone is talking.

Names of enrolled people and items are passed as `hotwords`, which nudges
Whisper toward spelling them the same way every time.

Windows note: ctranslate2 needs cublas64_12.dll. The CUDA 12.6 PyTorch build
ships it in torch/lib, so we import torch first and add that folder to the
DLL search path. If the GPU still fails, set stt.device: cpu (small.en int8 on
the i7 is ~1 s per utterance).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .. import events as ev
from ..bus import Bus
from ..config import STTConfig
from .vad import Segment

log = logging.getLogger(__name__)

_HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "bye bye",
    "subtitles by the amara org community", "please subscribe", "so", "okay", "uh", "um", "hmm",
}


def prepare_cuda_dlls() -> None:
    """Make CUDA libraries from the PyTorch wheel (and nvidia-* wheels) findable on Windows."""
    if not sys.platform.startswith("win"):
        return
    dirs: List[Path] = []
    try:
        import torch  # noqa: F401  (loads torch's own CUDA DLLs into the process)

        dirs.append(Path(torch.__file__).parent / "lib")
    except Exception:
        pass
    try:
        import nvidia  # pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

        for base in getattr(nvidia, "__path__", []):
            dirs += [p for p in Path(base).glob("*/bin") if p.is_dir()]
    except Exception:
        pass
    for d in dirs:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


class WhisperSTT:
    def __init__(self, cfg: STTConfig):
        if cfg.device == "cuda":
            prepare_cuda_dlls()
        from faster_whisper import WhisperModel

        self.cfg = cfg
        log.info("Loading faster-whisper %s on %s (%s)", cfg.model, cfg.device, cfg.compute_type)
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type,
                                  cpu_threads=cfg.cpu_threads)

    def transcribe(self, audio: np.ndarray, hotwords: Optional[str] = None) -> Tuple[str, dict]:
        segments, _info = self.model.transcribe(
            audio, language=self.cfg.language, beam_size=self.cfg.beam_size, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True, hotwords=hotwords or None,
            no_speech_threshold=self.cfg.no_speech_threshold,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        meta = {
            "no_speech_prob": max((s.no_speech_prob for s in segs), default=1.0),
            "avg_logprob": float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0,
        }
        return text, meta


class FakeSTT:
    """Tests: returns queued transcripts in order."""

    def __init__(self, texts: Optional[List[str]] = None):
        self.texts = list(texts or [])

    def transcribe(self, audio, hotwords=None):
        return (self.texts.pop(0) if self.texts else ""), {"no_speech_prob": 0.0, "avg_logprob": -0.2}


def looks_like_hallucination(text: str, meta: dict) -> bool:
    t = "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()
    if t in _HALLUCINATIONS:
        return True
    return meta.get("no_speech_prob", 0) > 0.6 and meta.get("avg_logprob", 0) < -1.0


class STTWorker:
    def __init__(self, cfg: STTConfig, bus: Bus, stt, hotwords_fn: Optional[Callable[[], str]] = None):
        self.cfg = cfg
        self.bus = bus
        self.stt = stt
        self.hotwords_fn = hotwords_fn
        self._q: "queue.Queue[Segment]" = queue.Queue(maxsize=20)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.busy = False

    def submit(self, seg: Segment) -> None:
        if len(seg.audio) < 1600:  # < 0.1 s
            return
        try:
            self._q.put_nowait(seg)
        except queue.Full:
            log.warning("STT queue full; dropping a segment")

    def start(self) -> "STTWorker":
        self._thread = threading.Thread(target=self._run, name="stt", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def process(self, seg: Segment) -> Optional[str]:
        hot = self.hotwords_fn() if self.hotwords_fn else None
        self.busy = True
        try:
            text, meta = self.stt.transcribe(seg.audio, hotwords=hot)
        finally:
            self.busy = False
        if self.cfg.drop_hallucinations and seg.source == "ambient" and looks_like_hallucination(text, meta):
            log.debug("Dropped likely hallucination: %r %s", text, meta)
            return None
        if not text:
            if seg.source == "ptt":
                self.bus.publish(ev.SAY, text="", priority=ev.PRI_ANSWER, kind=ev.KIND_ANSWER, tone=ev.TONE_ERROR,
                                 max_age_s=2.0)
            return None
        log.info("Heard (%s): %s", seg.source, text)
        self.bus.publish(ev.SPEECH_HEARD, ts=seg.t_end, text=text, timestamp=seg.t_start, t_end=seg.t_end,
                         source=seg.source)
        return text

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                seg = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process(seg)
            except Exception:
                log.exception("STT failed on a segment")
