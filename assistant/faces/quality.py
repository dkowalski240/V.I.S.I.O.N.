"""[Vendored from FaceLock, imports adapted.] Frame quality gate: only embed faces that are worth embedding.

A bad crop (blurry, turned away, half out of frame) produces an embedding that
sits somewhere between people. Fed into the matcher, those cause false "new
person" profiles; fed into the gallery, they slowly corrupt it. So we filter
hard and early. The checks run cheapest-first:

1. Geometry from the detector alone (no pixels touched):
   face size, detector confidence, distance from frame edge, head tilt,
   head turn (yaw) and nod (pitch) estimated from the 5 landmarks.
2. Pixels of the aligned 160x160 crop:
   brightness, blown-out highlights, sharpness.

Pose from 5 points (a cheap trick)
----------------------------------
After levelling the eyes, look at where the nose tip sits:

    yaw_ratio   = (nose_x - eye_midpoint_x) / eye_distance
    pitch_ratio = (nose_y - eye_y) / (mouth_y - eye_y)

Facing the camera, yaw_ratio ~ 0 and pitch_ratio ~ 0.5-0.6. Turning the head
pushes the nose sideways (yaw_ratio around 0.3 is roughly 25-30 degrees).
Not precise, but more than good enough to say "turn toward the camera".

Sharpness
---------
The variance of the Laplacian (a second-derivative edge filter) is high when
there are crisp edges and low when the image is smooth/blurred. We measure it
on the centre of the aligned crop, so the number does not depend on how far
away the person is or on background clutter. The right threshold depends on
your camera: the capture script prints it so you can calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .alignment import AlignedFace, map_points_to_crop
from ..config import QualityConfig
from .detection import Detection

# Short reason codes -> hints you can show to the person at the door.
HINTS = {
    "too_small": "Step closer",
    "low_confidence": "Face not clear",
    "at_edge": "Move to the centre",
    "tilted": "Straighten your head",
    "turned": "Look at the camera",
    "pitched": "Look straight ahead",
    "too_dark": "Too dark",
    "too_bright": "Too bright / backlit",
    "overexposed": "Glare on face",
    "blurry": "Hold still",
}


@dataclass
class QualityReport:
    ok: bool
    score: float = 0.0  # 0..1, higher = better; used to keep the best frames per track
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def hint(self) -> Optional[str]:
        return HINTS.get(self.reasons[0]) if self.reasons else None


def pose_ratios(landmarks_in_crop: np.ndarray) -> Tuple[float, float]:
    le, re, nose, ml, mr = landmarks_in_crop
    eye_mid = (le + re) / 2.0
    mouth_mid = (ml + mr) / 2.0
    iod = max(float(np.linalg.norm(re - le)), 1e-6)
    yaw = float((nose[0] - eye_mid[0]) / iod)
    span = float(mouth_mid[1] - eye_mid[1])
    pitch = float((nose[1] - eye_mid[1]) / span) if abs(span) > 1e-6 else 0.0
    return yaw, pitch


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _centre(img: np.ndarray, frac: float = 0.6) -> np.ndarray:
    h, w = img.shape[:2]
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return img[dh : h - dh, dw : w - dw]


class QualityGate:
    def __init__(self, cfg: QualityConfig):
        self.cfg = cfg

    def check_geometry(self, det: Detection, frame_shape: Tuple[int, ...]) -> Tuple[List[str], Dict[str, float]]:
        """Checks that need only the detection. Cheap; run before aligning."""
        c = self.cfg
        reasons: List[str] = []
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = det.box
        pad = c.edge_margin_frac * det.width
        metrics = {"face_px": det.width, "det_conf": det.score}

        if det.width < c.min_face_px:
            reasons.append("too_small")
        if det.score < c.min_det_confidence:
            reasons.append("low_confidence")
        if x1 < pad or y1 < pad or x2 > w - pad or y2 > h - pad:
            reasons.append("at_edge")

        (lx, ly), (rx, ry) = det.landmarks[0], det.landmarks[1]
        roll = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
        metrics["roll_deg"] = roll
        if abs(roll) > c.max_roll_deg:
            reasons.append("tilted")
        return reasons, metrics

    def assess(self, det: Detection, frame_shape: Tuple[int, ...], aligned: AlignedFace) -> QualityReport:
        c = self.cfg
        reasons, metrics = self.check_geometry(det, frame_shape)

        pts = map_points_to_crop(det.landmarks, aligned.dst_to_src)
        yaw, pitch = pose_ratios(pts)
        metrics.update(yaw_ratio=yaw, pitch_ratio=pitch)
        if abs(yaw) > c.max_yaw_ratio:
            reasons.append("turned")
        if not (c.pitch_ratio_range[0] <= pitch <= c.pitch_ratio_range[1]):
            reasons.append("pitched")

        gray = cv2.cvtColor(_centre(aligned.image), cv2.COLOR_BGR2GRAY)
        bright = float(gray.mean())
        saturated = float((gray >= 250).mean())
        sharp = sharpness(gray)
        metrics.update(brightness=bright, saturated=saturated, sharpness=sharp)
        if bright < c.brightness_range[0]:
            reasons.append("too_dark")
        elif bright > c.brightness_range[1]:
            reasons.append("too_bright")
        if saturated > c.max_saturated_fraction:
            reasons.append("overexposed")
        if sharp < c.min_sharpness:
            reasons.append("blurry")

        # A single 0..1 score so a track can keep its best few frames.
        sharp_n = min(1.0, sharp / max(c.min_sharpness * 4, 1.0))
        size_n = min(1.0, det.width / (c.min_face_px * 2))
        frontal = 1.0 - min(1.0, abs(yaw) / max(c.max_yaw_ratio, 1e-6))
        score = 0.4 * sharp_n + 0.3 * frontal + 0.2 * size_n + 0.1 * det.score
        metrics["score"] = score
        return QualityReport(ok=not reasons, score=score, reasons=reasons, metrics=metrics)
