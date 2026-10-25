"""Speech input loop: audio blocks -> VAD/push-to-talk segments -> STT worker.

Runs on its own thread and never waits for Whisper: finished segments are
handed to the STT worker's queue.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..config import SpeechInConfig
from .stt import STTWorker
from .vad import Segmenter

log = logging.getLogger(__name__)


class SpeechInput:
    def __init__(self, cfg: SpeechInConfig, source, vad, ptt, stt_worker: STTWorker):
        self.cfg = cfg
        self.source = source
        self.segmenter = Segmenter(cfg, vad)
        self.ptt = ptt
        self.stt_worker = stt_worker
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def level(self) -> float:
        return getattr(self.source, "level", 0.0)

    @property
    def vad_prob(self) -> float:
        return self.segmenter.last_prob

    @property
    def mode(self) -> str:
        return self.segmenter.mode

    def start(self) -> "SpeechInput":
        self.source.start()
        try:
            self.ptt.start()
        except Exception as e:  # e.g. pynput missing or a bad key name: keep listening anyway
            from .ptt import NoPTT

            log.warning("Push-to-talk unavailable (%s); type commands in the console instead", e)
            self.ptt = NoPTT()
        self.stt_worker.start()
        self._thread = threading.Thread(target=self._run, name="speech-in", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.source.stop()
        self.ptt.stop()
        self.stt_worker.stop()

    def feed_one(self, timeout: float = 0.2) -> bool:
        item = self.source.read(timeout)
        if item is None:
            return False
        block, ts = item
        for seg in self.segmenter.feed(block, ts, self.ptt.is_down(ts)):
            log.debug("Segment %s %.2fs", seg.source, len(seg.audio) / 16000)
            self.stt_worker.submit(seg)
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.feed_one(0.2)
            except Exception:
                log.exception("Speech input error")
