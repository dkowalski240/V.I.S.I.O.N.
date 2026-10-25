"""Non-microphone inputs: typed console commands and replay scripts.

Console (on by default) - type a line and press Enter:

    where's my wallet          a command, exactly like push-to-talk
    !ambient Hi, I'm Sarah     pretend this was overheard (introductions, answers)
    !yes / !no                 answer a consent question as the other person
    !reset-people              forget every enrolled person at once (no confirmation; for testing)

Replay script (--script file.txt), one action per line, times in seconds from
the start of the recording:

    # time        action      text
    12.0          ambient     Hi, I'm Sarah
    15.5          ambient     Sure
    31.0          type        where's my wallet
    40.2-42.0     ptt                      # hold push-to-talk during this window
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import events as ev
from .bus import Bus
from .clock import Clock

log = logging.getLogger(__name__)


class ConsoleInput:
    def __init__(self, bus: Bus, clock: Clock):
        self.bus = bus
        self.clock = clock
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ConsoleInput":
        if not sys.stdin or not sys.stdin.isatty():
            log.info("Console input disabled (stdin is not a terminal)")
            return self
        self._thread = threading.Thread(target=self._run, name="console", daemon=True)
        self._thread.start()
        print("Type commands here (e.g. \"who's here\", \"!ambient Hi, I'm Sarah\", \"!yes\", \"!reset-people\").",
              flush=True)
        return self

    def _run(self) -> None:
        for line in sys.stdin:
            publish_line(self.bus, self.clock, line)

    def stop(self) -> None:
        pass


def publish_line(bus: Bus, clock: Clock, line: str) -> None:
    line = line.strip()
    if not line:
        return
    now = clock.now()
    if line.lower() in ("!yes", "!no"):
        bus.publish(ev.SPEECH_HEARD, text=line[1:], timestamp=now, t_end=now, source="ambient")
    elif line.lower().startswith("!ambient "):
        bus.publish(ev.SPEECH_HEARD, text=line[9:], timestamp=now - 1.0, t_end=now, source="ambient")
    else:
        bus.publish(ev.SPEECH_HEARD, text=line, timestamp=now, t_end=now, source="typed")


@dataclass
class ScriptAction:
    t0: float
    t1: float
    action: str   # ambient | type | ptt
    text: str = ""


_LINE = re.compile(r"^\s*(?P<t0>\d+(?:\.\d+)?)(?:\s*-\s*(?P<t1>\d+(?:\.\d+)?))?\s+(?P<action>\w+)\s*(?P<text>.*)$")


def parse_script(text: str) -> List[ScriptAction]:
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m or m["action"] not in ("ambient", "type", "ptt"):
            raise ValueError(f"Script line {n} not understood: {raw!r}")
        t0 = float(m["t0"])
        t1 = float(m["t1"]) if m["t1"] else t0
        out.append(ScriptAction(t0, t1, m["action"], m["text"].strip()))
    return sorted(out, key=lambda a: a.t0)


class ScriptPlayer:
    """Publishes scripted speech at the right replay time and answers
    'is push-to-talk held at time t?' for the speech input."""

    def __init__(self, actions: List[ScriptAction], bus: Bus, clock):
        self.actions = actions
        self.bus = bus
        self.clock = clock
        self.ptt_windows: List[Tuple[float, float]] = [
            (clock.at(a.t0), clock.at(a.t1)) for a in actions if a.action == "ptt"
        ]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def ptt_down(self, ts: float) -> bool:
        return any(t0 <= ts <= t1 for t0, t1 in self.ptt_windows)

    def start(self) -> "ScriptPlayer":
        self._thread = threading.Thread(target=self._run, name="script", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        for a in self.actions:
            if a.action == "ptt":
                continue
            due = self.clock.at(a.t0)
            while not self._stop.is_set() and self.clock.now() < due:
                self.clock.sleep(min(0.05, due - self.clock.now()))
            if self._stop.is_set():
                return
            src = "ambient" if a.action == "ambient" else "typed"
            now = self.clock.now()
            log.info("Script @%.1fs %s: %s", a.t0, a.action, a.text)
            self.bus.publish(ev.SPEECH_HEARD, text=a.text, timestamp=now - 1.0, t_end=now, source=src)

    def stop(self) -> None:
        self._stop.set()
