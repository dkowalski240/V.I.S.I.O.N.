"""Speech output: one priority queue, one speaker, never two voices at once.

Anything that wants to talk publishes a `say` event (or calls `say()`):

    {"type": "say", "text": "Sarah, approaching, 10 o'clock", "priority": 0,
     "pan": -0.9, "kind": "approach", "max_age_s": 2.5}

Rules:
* Lower priority number first: approaching people (0) > answers (1) >
  people in view (2) > everything else (3). Equal priorities play in order.
* One message at a time. If an approaching person comes up while a
  lower-priority message is playing, that message is cut off (and an answer
  is put back in the queue to be said after). Enrollment prompts are never cut off.
* Messages carry a max age: "Sarah, 2 o'clock" said 5 seconds late is wrong,
  so it's dropped instead.
* `tone` plays a short beep instead of (or before) words.
* Quiet mode: only answers and prompts are spoken; an approaching known
  person becomes a short panned tone.
* `repeat()` says the last spoken sentence again.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .. import events as ev
from ..bus import Bus
from ..clock import Clock
from ..config import TTSConfig
from ..geometry import pan_gains
from .tts import CachedTTS, PrintTTS, TTSEngine, to_stereo, tone

log = logging.getLogger(__name__)


class SoundDeviceSink:
    """Plays stereo float32 audio on an output device; stop() interrupts."""

    def __init__(self, device=None):
        import sounddevice as sd

        self.sd = sd
        self.device = device
        self._stop = threading.Event()

    def play(self, stereo: np.ndarray, sample_rate: int) -> bool:
        """Blocking. Returns False if interrupted."""
        self._stop.clear()
        if len(stereo) == 0:
            return True
        self.sd.play(stereo, sample_rate, device=self.device, blocking=False)
        end = time.monotonic() + len(stereo) / sample_rate + 0.05
        while time.monotonic() < end:
            if self._stop.is_set():
                self.sd.stop()
                return False
            time.sleep(0.01)
        return True

    def stop(self) -> None:
        self._stop.set()


class NullSink:
    """Records what would have been played. `realtime=True` sleeps for the
    audio's duration so timing behaves like a real speaker."""

    def __init__(self, realtime: bool = False):
        self.realtime = realtime
        self.played: List[Tuple[int, float]] = []  # (samples, pan sign) for tests
        self._stop = threading.Event()

    def play(self, stereo: np.ndarray, sample_rate: int) -> bool:
        self._stop.clear()
        self.played.append((len(stereo), float(stereo[:, 1].sum() - stereo[:, 0].sum()) if len(stereo) else 0.0))
        if self.realtime and len(stereo):
            end = time.monotonic() + len(stereo) / sample_rate
            while time.monotonic() < end:
                if self._stop.is_set():
                    return False
                time.sleep(0.005)
        return True

    def stop(self) -> None:
        self._stop.set()


@dataclass(order=True)
class _Item:
    priority: int
    seq: int
    msg: dict = field(compare=False)


@dataclass
class Spoken:
    ts: float
    text: str
    pan: float
    kind: str
    tone: Optional[str]
    interrupted: bool = False


_DEFAULT_KIND = {ev.PRI_APPROACH: ev.KIND_APPROACH, ev.PRI_ANSWER: ev.KIND_ANSWER,
                 ev.PRI_PERSON: ev.KIND_PERSON, ev.PRI_INFO: ev.KIND_INFO}


class SpeechOutput:
    def __init__(self, cfg: TTSConfig, bus: Optional[Bus] = None, clock: Optional[Clock] = None,
                 tts: Optional[TTSEngine] = None, sink=None):
        self.cfg = cfg
        self.bus = bus
        self.clock = clock or Clock()
        self.tts = CachedTTS(tts or PrintTTS(cfg))
        self.sink = sink if sink is not None else NullSink()
        self._heap: List[_Item] = []
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._current: Optional[_Item] = None
        self._interrupted = False
        self.quiet = False
        self.last_text: Optional[str] = None
        self.history: Deque[Spoken] = deque(maxlen=200)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._sub = bus.subscribe(ev.SAY, maxsize=64) if bus is not None else None

    # ------------------------------------------------------------------ public API
    def say(self, text: str = "", priority: int = ev.PRI_INFO, pan: float = 0.0, kind: Optional[str] = None,
            tone: Optional[str] = None, max_age_s: Optional[float] = None, ts: Optional[float] = None,
            **_ignored) -> None:
        kind = kind or _DEFAULT_KIND.get(priority, ev.KIND_INFO)
        if max_age_s is None:
            max_age_s = {ev.KIND_ANSWER: self.cfg.answer_max_age_s, ev.KIND_PROMPT: 3600.0}.get(
                kind, self.cfg.info_max_age_s
            )
        msg = {"text": text, "priority": int(priority), "pan": float(pan), "kind": kind, "tone": tone,
               "max_age_s": float(max_age_s), "ts": ts if ts is not None else self.clock.now()}
        with self._cv:
            if text and any(i.msg["text"] == text for i in self._heap):
                return  # same sentence already waiting
            heapq.heappush(self._heap, _Item(msg["priority"], next(self._seq), msg))
            cur = self._current
            if (cur is not None and self.cfg.interrupt_for_approaching and kind == ev.KIND_APPROACH
                    and msg["priority"] < cur.priority and cur.msg["kind"] != ev.KIND_PROMPT):
                self._interrupted = True
                self.sink.stop()
            self._cv.notify_all()

    def repeat(self) -> None:
        if self.last_text:
            self.say(self.last_text, priority=ev.PRI_ANSWER, kind=ev.KIND_ANSWER)
        else:
            self.say("Nothing to repeat.", priority=ev.PRI_ANSWER, kind=ev.KIND_ANSWER)

    def set_quiet(self, on: bool) -> None:
        self.quiet = on
        if on:  # drop queued chatter right away
            with self._cv:
                self._heap = [i for i in self._heap if i.msg["kind"] in (ev.KIND_ANSWER, ev.KIND_PROMPT)]
                heapq.heapify(self._heap)

    def pending(self) -> int:
        with self._cv:
            return len(self._heap)

    def idle(self) -> bool:
        with self._cv:
            return not self._heap and self._current is None

    # ------------------------------------------------------------------ threads
    def start(self) -> "SpeechOutput":
        t = threading.Thread(target=self._run, name="speech-out", daemon=True)
        t.start()
        self._threads.append(t)
        if self._sub is not None:
            t2 = threading.Thread(target=self._intake, name="speech-intake", daemon=True)
            t2.start()
            self._threads.append(t2)
        return self

    def stop(self) -> None:
        self._stop.set()
        self.sink.stop()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=2)

    def _intake(self) -> None:
        while not self._stop.is_set():
            msg = self._sub.get(timeout=0.1)
            if msg is not None:
                self.say(**{k: v for k, v in msg.items() if k != "type"})

    def _next(self, timeout: float = 0.1) -> Optional[_Item]:
        with self._cv:
            if not self._heap:
                self._cv.wait(timeout)
            if not self._heap:
                return None
            item = heapq.heappop(self._heap)
            self._current = item
            self._interrupted = False
            return item

    def process_one(self, timeout: float = 0.0) -> bool:
        """Play the next queued message (used by the thread and by tests)."""
        item = self._next(timeout)
        if item is None:
            return False
        try:
            self._play(item)
        finally:
            with self._cv:
                self._current = None
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.process_one(timeout=0.1)
            except Exception:
                log.exception("Speech output error")

    def _play(self, item: _Item) -> None:
        msg = item.msg
        now = self.clock.now()
        if now - msg["ts"] > msg["max_age_s"]:
            log.debug("Dropped stale message: %s", msg["text"])
            return
        text, kind, tone_name = msg["text"], msg["kind"], msg["tone"]
        if self.quiet and kind in (ev.KIND_PERSON, ev.KIND_INFO):
            return
        if self.quiet and kind == ev.KIND_APPROACH:
            text, tone_name = "", ev.TONE_PERSON
        sr = self.tts.sample_rate
        parts = []
        if tone_name and self.cfg.tones:
            parts.append(tone(tone_name, sr))
            if text:
                parts.append(np.zeros(int(0.05 * sr), dtype=np.float32))
        if text:
            audio = self.tts.synth(text)
            sr = self.tts.sample_rate  # pyttsx3 learns its rate on the first render
            parts.append(audio)
        if not parts:
            return
        mono = np.concatenate(parts).astype(np.float32)
        stereo = to_stereo(mono, pan_gains(msg["pan"]))
        log.info("SAY [%s p%d pan %+.2f] %s", kind, msg["priority"], msg["pan"], text or f"<{tone_name}>")
        finished = self.sink.play(stereo, sr)
        interrupted = (not finished) or self._interrupted
        self.history.append(Spoken(now, text, msg["pan"], kind, tone_name, interrupted))
        if text and not interrupted:
            self.last_text = text
        if interrupted and kind == ev.KIND_ANSWER:
            # Put the answer back; it'll be said after the urgent message.
            with self._cv:
                msg = dict(msg, ts=self.clock.now())
                heapq.heappush(self._heap, _Item(msg["priority"], next(self._seq), msg))
                self._cv.notify_all()


def make_sink(cfg: TTSConfig, engine_name: str):
    if engine_name == "print":
        return NullSink()
    try:
        return SoundDeviceSink(cfg.output_device)
    except Exception as e:
        log.warning("No audio output (%s); speech will only be printed", e)
        return NullSink()
