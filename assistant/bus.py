"""The shared event bus: modules talk only through small dict messages.

    bus.publish("person_seen", name="Sarah", clock_direction=10, ...)
    sub = bus.subscribe("person_seen", "speech_heard")
    msg = sub.get(timeout=0.1)     # -> dict or None

Each subscriber gets its own bounded queue, so a slow consumer can't block a
fast producer; when a queue is full the oldest message is dropped (stale
"Sarah at 2 o'clock" is useless anyway). Camera frames never travel on the
bus. Modules read the newest frame from the camera directly.

Every message can also be appended to a JSONL log (data/logs/), which makes
replay debugging easy: you can see exactly what was heard, decided and said.
Embeddings and images are never put on the bus, so they never reach the log.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np

from .clock import Clock

log = logging.getLogger(__name__)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


class Subscription:
    def __init__(self, types: Iterable[str], maxsize: int):
        self.types: Optional[Set[str]] = set(types) or None  # None = everything
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def wants(self, msg_type: str) -> bool:
        return self.types is None or msg_type in self.types

    def put(self, msg: dict) -> None:
        while True:
            try:
                self._q.put_nowait(msg)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def get(self, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> List[dict]:
        out = []
        while True:
            m = self.get()
            if m is None:
                return out
            out.append(m)


class Bus:
    def __init__(self, clock: Optional[Clock] = None, log_path: Optional[str] = None):
        self.clock = clock or Clock()
        self._subs: List[Subscription] = []
        self._lock = threading.Lock()
        self._log_fh = None
        self._log_lock = threading.Lock()
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_path, "a", encoding="utf-8")
            log.info("Logging events to %s", log_path)

    def subscribe(self, *types: str, maxsize: int = 256) -> Subscription:
        sub = Subscription(types, maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def publish(self, msg_type: str, **fields) -> dict:
        msg = {"type": msg_type, "ts": fields.pop("ts", None) or self.clock.now(), **fields}
        with self._lock:
            subs = [s for s in self._subs if s.wants(msg_type)]
        for s in subs:
            s.put(msg)
        if self._log_fh is not None:
            line = json.dumps(msg, default=_json_default)
            with self._log_lock:
                try:
                    self._log_fh.write(line + "\n")
                    self._log_fh.flush()
                except ValueError:  # closed during shutdown
                    pass
        return msg

    def close(self) -> None:
        with self._log_lock:
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None
