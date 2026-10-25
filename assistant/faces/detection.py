"""[Vendored from FaceLock, imports adapted.] Face detection: find face boxes and 5 facial landmarks in a frame.

Two backends share one interface so you can swap them in the config:

* MTCNN (facenet-pytorch): three small CNNs in a cascade. Accurate, ships
  with facenet-pytorch, but slowish on a CPU.
* YuNet (OpenCV): a single tiny CNN. Several times faster, a good choice for a
  Raspberry Pi. Needs a ~230 KB .onnx file (see README).

Both return landmarks in the same canonical order, in *image* coordinates:

    0 = eye on the image-left        1 = eye on the image-right
    2 = nose tip
    3 = mouth corner on image-left   4 = mouth corner on image-right

Different libraries name these from the subject's point of view or the
viewer's, so instead of trusting names we sort each pair by x. That makes the
rest of the code independent of the detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

import cv2
import numpy as np

from ..config import FaceDetectionConfig as DetectionConfig

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: np.ndarray        # (4,) float: x1, y1, x2, y2 in full-resolution pixels
    score: float           # detector confidence, 0..1
    landmarks: np.ndarray  # (5, 2) float, canonical order (see module docstring)

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]: ...


def canonicalize_landmarks(points: np.ndarray) -> np.ndarray:
    """Sort eye pair and mouth pair left-to-right in image coordinates."""
    p = np.asarray(points, dtype=np.float32).reshape(5, 2).copy()
    if p[0, 0] > p[1, 0]:
        p[[0, 1]] = p[[1, 0]]
    if p[3, 0] > p[4, 0]:
        p[[3, 4]] = p[[4, 3]]
    return p


def _downscale(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class MTCNNDetector:
    def __init__(self, cfg: DetectionConfig):
        # Imported lazily so the rest of the package (and the tests) work
        # without PyTorch installed.
        from facenet_pytorch import MTCNN

        self.cfg = cfg
        # MTCNN's min_face_size is measured in the *downscaled* image.
        min_size = max(12, int(cfg.min_face_px * cfg.scale))
        self.model = MTCNN(keep_all=True, min_face_size=min_size, device=cfg.device, post_process=False)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        small = _downscale(frame_bgr, self.cfg.scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes, probs, points = self.model.detect(rgb, landmarks=True)
        if boxes is None:
            return []
        inv = 1.0 / self.cfg.scale
        out = []
        for box, prob, pts in zip(boxes, probs, points):
            if prob is None or prob < self.cfg.min_confidence:
                continue
            out.append(
                Detection(
                    box=np.asarray(box, dtype=np.float32) * inv,
                    score=float(prob),
                    landmarks=canonicalize_landmarks(np.asarray(pts) * inv),
                )
            )
        return out


class YuNetDetector:
    def __init__(self, cfg: DetectionConfig):
        path = Path(cfg.yunet_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {path}. Download face_detection_yunet_2023mar.onnx "
                "from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
            )
        self.cfg = cfg
        self.model = cv2.FaceDetectorYN.create(str(path), "", (320, 320), cfg.min_confidence, 0.3, 5000)
        self._size = (320, 320)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        small = _downscale(frame_bgr, self.cfg.scale)
        h, w = small.shape[:2]
        if (w, h) != self._size:
            self.model.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.model.detect(small)
        if faces is None:
            return []
        inv = 1.0 / self.cfg.scale
        out = []
        for f in faces:
            x, y, bw, bh = f[:4]
            if bw * inv < self.cfg.min_face_px:
                continue
            out.append(
                Detection(
                    box=np.array([x, y, x + bw, y + bh], dtype=np.float32) * inv,
                    score=float(f[14]),
                    landmarks=canonicalize_landmarks(f[4:14].reshape(5, 2) * inv),
                )
            )
        return out


def create_detector(cfg: DetectionConfig) -> Detector:
    if cfg.backend == "mtcnn":
        return MTCNNDetector(cfg)
    if cfg.backend == "yunet":
        return YuNetDetector(cfg)
    raise ValueError(f"Unknown detection backend {cfg.backend!r} (use 'mtcnn' or 'yunet')")
