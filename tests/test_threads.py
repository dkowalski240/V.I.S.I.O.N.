"""The real threads together (camera feeder, faces, router, speech output) on the
wall clock: an enrollment must complete without deadlocks or races."""

import threading
import time

from assistant import events as ev
from assistant.bus import Bus
from assistant.camera import StaticCamera
from assistant.clock import Clock
from assistant.faces.enrollment import Enrollment
from assistant.faces.module import FacesModule
from assistant.intents.router import Router

from conftest import FakeDetector, FakeEmbedder, Scene, make_output, test_config


def test_enrollment_with_live_threads(storage):
    cfg = test_config()
    cfg.enrollment.capture_s = 1.0
    clock = Clock()
    bus = Bus(clock)
    scene = Scene(3)
    cam = StaticCamera()
    det, emb = FakeDetector(), FakeEmbedder(scene)
    faces = FacesModule(cfg, cam, bus, storage, clock, det, emb)
    enrollment = Enrollment(cfg, bus, storage, faces, clock)
    faces.enrollment = enrollment
    output = make_output(cfg, bus, clock)
    router = Router(cfg, bus, storage, clock, faces=faces, enrollment=enrollment, output=output)

    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            img, dets, _ = scene.frame([(200, 560, 200, 180)])
            det.next = dets
            cam.push(img, clock.now())
            time.sleep(1 / 30)

    t = threading.Thread(target=feeder, daemon=True)
    t.start()
    faces.start()
    router.start()
    output.start()
    try:
        time.sleep(0.8)
        now = clock.now()
        bus.publish(ev.SPEECH_HEARD, text="Hi, I'm Sarah.", timestamp=now - 0.8, t_end=now, source="ambient")
        deadline = time.time() + 3
        while time.time() < deadline and not enrollment.awaiting_answer():
            time.sleep(0.05)
        assert enrollment.awaiting_answer()
        bus.publish(ev.SPEECH_HEARD, text="Sure.", timestamp=clock.now(), t_end=clock.now(), source="ambient")
        deadline = time.time() + 6
        while time.time() < deadline and not storage.list_people():
            time.sleep(0.05)
        assert [p["name"] for p in storage.list_people()] == ["Sarah"]
        deadline = time.time() + 2
        while time.time() < deadline and not any(s.text == "Saved Sarah." for s in output.history):
            time.sleep(0.05)
        assert any(s.text == "Saved Sarah." for s in output.history)
    finally:
        stop.set()
        router.stop()
        faces.stop()
        output.stop()
    assert not faces._thread.is_alive() and not router._thread.is_alive()
