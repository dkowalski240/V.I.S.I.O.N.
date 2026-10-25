"""Shared test helpers (the FaceLock trick, extended).

No PyTorch, camera, microphone or speaker needed:

* Fake faces are solid grey squares. The grey level encodes *who* it is; the
  FakeEmbedder reads it back off the aligned crop and returns that identity's
  vector plus noise. The real alignment, quality, tracking, matching,
  enrollment and announcement code all run unchanged.
* Fake objects are solid coloured rectangles; the FakeImageEmbedder maps the
  crop's median colour to a vector the same way.
* A FakeClock drives time, so "approaching within 1 s" is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.bus import Bus  # noqa: E402
from assistant.clock import FakeClock  # noqa: E402
from assistant.config import Config, QualityConfig  # noqa: E402
from assistant.faces.detection import Detection  # noqa: E402
from assistant.faces.embedding import l2_normalize  # noqa: E402
from assistant.objects.detector import ObjDet  # noqa: E402
from assistant.speech.output import NullSink, SpeechOutput  # noqa: E402
from assistant.speech.tts import PrintTTS  # noqa: E402
from assistant.storage import Storage  # noqa: E402

DIM = 512
W, H = 1280, 720


def random_identity(rng) -> np.ndarray:
    return l2_normalize(rng.standard_normal(DIM))


def noisy(base: np.ndarray, rng, amount: float = 0.4) -> np.ndarray:
    """A new 'photo' of the same person; amount 0.4 gives cos ~0.93 to base."""
    return l2_normalize(base + amount * l2_normalize(rng.standard_normal(DIM)))


def face_detection(x: float, y: float, size: float = 160, score: float = 0.99) -> Detection:
    rel = np.array([[0.3, 0.4], [0.7, 0.4], [0.5, 0.6], [0.35, 0.8], [0.65, 0.8]])
    return Detection(box=np.array([x, y, x + size, y + size], dtype=np.float32), score=score,
                     landmarks=(np.array([x, y]) + rel * size).astype(np.float32))


def permissive_quality() -> QualityConfig:
    return QualityConfig(min_face_px=40, min_det_confidence=0.5, min_sharpness=0.0, brightness_range=(0.0, 256.0),
                         max_saturated_fraction=1.0, edge_margin_frac=0.0)


def test_config() -> Config:
    cfg = Config()
    cfg.faces.quality = permissive_quality()
    cfg.faces.enroll_quality = permissive_quality()
    cfg.storage.log_events = False
    cfg.tts.engine = "print"
    return cfg


test_config.__test__ = False


class Scene:
    """Builds frames + detections. faces = [(grey, x, y, size[, talking])], objects = [(bgr, label, box)]."""

    def __init__(self, seed: int = 0, noise: float = 0.4):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.identities: Dict[int, np.ndarray] = {}
        self.colors: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.n_frames = 0

    def identity(self, grey: int) -> np.ndarray:
        if grey not in self.identities:
            self.identities[grey] = random_identity(self.rng)
        return self.identities[grey]

    def color_vec(self, bgr) -> np.ndarray:
        key = tuple(int(round(c / 10.0)) * 10 for c in bgr)
        if key not in self.colors:
            self.colors[key] = random_identity(self.rng)
        return self.colors[key]

    def frame(self, faces=(), objects=(), background=10) -> Tuple[np.ndarray, List[Detection], List[ObjDet]]:
        img = np.empty((H, W, 3), dtype=np.uint8)
        img[:] = background
        dets, odets = [], []
        for bgr, label, box in objects:
            x1, y1, x2, y2 = [int(v) for v in box]
            img[y1:y2, x1:x2] = bgr
            odets.append(ObjDet(np.array(box, dtype=np.float32), 0.5, label))
        self.n_frames += 1
        for face in faces:
            grey, x, y, size = face[:4]
            talking = len(face) > 4 and face[4]
            self.identity(grey)
            img[int(y):int(y + size), int(x):int(x + size)] = grey
            if talking:  # a "mouth" that opens and closes every frame
                shade = int(np.clip(grey + (60 if self.n_frames % 2 else -60), 0, 255))
                img[int(y + 0.74 * size):int(y + 0.88 * size), int(x + 0.38 * size):int(x + 0.62 * size)] = shade
            dets.append(face_detection(x, y, size))
        return img, dets, odets


class FakeDetector:
    def __init__(self):
        self.next: List[Detection] = []

    def detect(self, frame_bgr):
        return list(self.next)


class FakeEmbedder:
    dim = DIM

    def __init__(self, scene: Scene):
        self.scene = scene

    def embed(self, crops):
        out = []
        for crop in crops:
            c = crop.shape[0] // 2
            grey = int(np.median(crop[c - 20:c + 20, c - 20:c + 20]))
            out.append(noisy(self.scene.identity(grey), self.scene.rng, self.scene.noise))
        return np.array(out, dtype=np.float32).reshape(-1, DIM)


class FakeObjectDetector:
    """Returns the scripted object boxes whose label is among the queries."""

    def __init__(self):
        self.next: List[ObjDet] = []

    def detect(self, image, queries):
        qs = {q.lower() for q in queries}
        return [d for d in self.next if d.label.lower() in qs]


class FakeImageEmbedder:
    dim = DIM

    def __init__(self, scene: Scene, noise: float = 0.25):
        self.scene = scene
        self.noise = noise

    def embed(self, crops):
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            patch = crop[h // 3: 2 * h // 3 + 1, w // 3: 2 * w // 3 + 1].reshape(-1, 3)
            bgr = np.median(patch, axis=0)
            out.append(noisy(self.scene.color_vec(bgr), self.scene.rng, self.noise))
        return np.array(out, dtype=np.float32).reshape(-1, DIM)


def make_output(cfg: Config, bus: Bus, clock) -> SpeechOutput:
    return SpeechOutput(cfg.tts, bus, clock, tts=PrintTTS(cfg.tts), sink=NullSink())


def drain_speech(output: SpeechOutput) -> List[str]:
    """Move queued `say` events into the output and play them all; return what was said."""
    said_before = len(output.history)
    for msg in output._sub.drain():
        output.say(**{k: v for k, v in msg.items() if k != "type"})
    while output.process_one():
        pass
    return [s.text or f"<{s.tone}>" for s in list(output.history)[said_before:]]


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"), str(tmp_path / "snaps"))
    yield s
    s.close()
