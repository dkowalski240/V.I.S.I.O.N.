"""A full assistant wired with fakes, stepped by hand on a FakeClock."""

from __future__ import annotations

from typing import List

from assistant import events as ev
from assistant.bus import Bus
from assistant.camera import StaticCamera
from assistant.clock import FakeClock
from assistant.faces.enrollment import Enrollment
from assistant.faces.module import FacesModule
from assistant.intents.router import Router
from assistant.objects.module import ObjectsModule

from conftest import (FakeDetector, FakeEmbedder, FakeImageEmbedder, FakeObjectDetector, Scene, make_output,
                      noisy, test_config)

FPS = 15.0


class Rig:
    def __init__(self, storage, seed: int = 0, objects: bool = True, radio=None):
        self.cfg = test_config()
        self.clock = FakeClock()
        self.bus = Bus(self.clock)
        self.events = self.bus.subscribe()
        self.storage = storage
        self.scene = Scene(seed)
        self.camera = StaticCamera()
        self.fdet, self.femb = FakeDetector(), FakeEmbedder(self.scene)
        self.faces = FacesModule(self.cfg, self.camera, self.bus, storage, self.clock, self.fdet, self.femb)
        self.enrollment = Enrollment(self.cfg, self.bus, storage, self.faces, self.clock)
        self.faces.enrollment = self.enrollment
        self.objects = None
        if objects:
            self.odet, self.oemb = FakeObjectDetector(), FakeImageEmbedder(self.scene)
            self.objects = ObjectsModule(self.cfg, self.camera, self.bus, storage, self.clock, self.odet,
                                         self.oemb, self.faces, radio=radio)
        self.output = make_output(self.cfg, self.bus, self.clock)
        self.router = Router(self.cfg, self.bus, storage, self.clock, faces=self.faces, objects=self.objects,
                             enrollment=self.enrollment, output=self.output)
        self.background = 10  # scene colour: different "rooms" in the place tests
        self.said: List[str] = []
        self.said_events: List[dict] = []
        self._next_obj = 0.0

    def enroll_directly(self, name: str, grey: int, n: int = 6) -> int:
        base = self.scene.identity(grey)
        pid = self.storage.add_person(name, [noisy(base, self.scene.rng) for _ in range(n)], [0.9] * n)
        self.faces.reload_gallery()
        return pid

    def step(self, faces=(), objects=(), dt: float = 1.0 / FPS) -> None:
        """One camera frame: faces every frame, objects at their own rate, then routing + speech."""
        img, dets, odets = self.scene.frame(faces, objects, self.background)
        now = self.clock.now()
        self.camera.push(img, now)
        self.fdet.next = dets
        self.faces.process_frame(img, now)
        if self.objects is not None:
            self.odet.next = odets
            rate = self.cfg.objects.enroll_fps if self.objects.enrolling() else self.cfg.objects.scan_fps
            if now >= self._next_obj:
                self.objects.process_frame(img, now)
                self._next_obj = now + 1.0 / rate
        self.router.process_pending()
        self._speak()
        self.clock.advance(dt)

    def _speak(self) -> None:
        for msg in self.output._sub.drain():
            self.said_events.append(msg)
            self.output.say(**{k: v for k, v in msg.items() if k != "type"})
        while self.output.process_one():
            s = self.output.history[-1]
            self.said.append(s.text or f"<{s.tone}>")

    def hear(self, text: str, source: str = "ambient", spoken_for: float = 1.0) -> None:
        now = self.clock.now()
        self.bus.publish(ev.SPEECH_HEARD, text=text, timestamp=now - spoken_for, t_end=now, source=source)
        self.router.process_pending()
        self._speak()

    def run(self, seconds: float, faces=(), objects=()) -> None:
        for _ in range(int(round(seconds * FPS))):
            self.step(faces, objects)

    def words(self) -> List[str]:
        return [s for s in self.said if not s.startswith("<")]

    def person_events(self) -> List[dict]:
        return [m for m in self.events.drain() if m["type"] == ev.PERSON_SEEN]
