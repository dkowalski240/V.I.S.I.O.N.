"""Push-to-talk: hold a key to give a command.

The key is global (works while the debug window or another app has focus),
via pynput. `speech_in.ptt_key` takes a pynput key name (ctrl_r, alt_r, f8,
space, shift_r, scroll_lock, pause, media_volume_up...) or one character.
A cheap Bluetooth selfie button usually sends media_volume_up, which makes a
handy wearable push-to-talk.

A short beep plays when the key goes down, so the user knows it's listening.

TODO (stretch goal): a wake word ("hey assistant") as an alternative to the key,
e.g. openWakeWord on the same 16 kHz stream in speech/input.py.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .. import events as ev
from ..bus import Bus
from ..clock import Clock

log = logging.getLogger(__name__)


class NoPTT:
    def is_down(self, ts: Optional[float] = None) -> bool:
        return False

    def start(self):
        return self

    def stop(self):
        pass


class KeyboardPTT:
    def __init__(self, key_name: str, bus: Bus, clock: Clock):
        self.key_name = key_name
        self.bus = bus
        self.clock = clock
        self._down = False
        self._lock = threading.Lock()
        self._listener = None

    def _target(self):
        from pynput import keyboard

        name = self.key_name.strip()
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name.lower())
        try:
            return getattr(keyboard.Key, name)
        except AttributeError:
            raise ValueError(f"speech_in.ptt_key {name!r} is not a pynput key name (e.g. ctrl_r, f8, space)")

    def start(self) -> "KeyboardPTT":
        from pynput import keyboard

        target = self._target()

        def matches(key) -> bool:
            if key == target:
                return True
            ch = getattr(key, "char", None)
            return isinstance(target, keyboard.KeyCode) and ch is not None and ch.lower() == target.char

        def on_press(key):
            if matches(key):
                with self._lock:
                    first = not self._down
                    self._down = True
                if first:
                    self.bus.publish(ev.PTT, state="pressed")
                    # kind=answer so the beep still plays in quiet mode
                    self.bus.publish(ev.SAY, text="", priority=ev.PRI_APPROACH, kind=ev.KIND_ANSWER,
                                     tone=ev.TONE_LISTEN, max_age_s=0.5)

        def on_release(key):
            if matches(key):
                with self._lock:
                    self._down = False
                self.bus.publish(ev.PTT, state="released")

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()
        log.info("Push-to-talk: hold %s", self.key_name)
        return self

    def is_down(self, ts: Optional[float] = None) -> bool:
        with self._lock:
            return self._down

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


class ScriptedPTT:
    """Replay: push-to-talk windows come from the --script file."""

    def __init__(self, player):
        self.player = player

    def is_down(self, ts: Optional[float] = None) -> bool:
        return self.player.ptt_down(ts if ts is not None else self.player.clock.now())

    def start(self):
        return self

    def stop(self):
        pass
