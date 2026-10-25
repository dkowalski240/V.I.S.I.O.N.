"""Clocks. Every module asks a clock for the time instead of calling time.time().

* Clock: the wall clock (live mode).
* ReplayClock: wall-clock-like timestamps that start when the replay starts,
  so video frames, audio blocks and "14 minutes ago" all agree on one timeline.
  `at(media_s)` converts a position in the recording to a timestamp.
* FakeClock: advanced by hand in tests.
"""

from __future__ import annotations

import threading
import time


class Clock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def sleep_until(self, t: float) -> None:
        self.sleep(t - self.now())


class ReplayClock(Clock):
    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.origin = time.time()
        self._m0 = time.monotonic()

    def now(self) -> float:
        return self.origin + (time.monotonic() - self._m0) * self.speed

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds / self.speed)

    def at(self, media_s: float) -> float:
        """Timestamp of a position (seconds) in the recording."""
        return self.origin + media_s

    def media(self, t: float) -> float:
        return t - self.origin


class FakeClock(Clock):
    def __init__(self, start: float = 1_000_000.0):
        self.t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self.t

    def advance(self, dt: float) -> float:
        with self._lock:
            self.t += dt
            return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.advance(seconds)

    def at(self, media_s: float) -> float:
        return 1_000_000.0 + media_s
