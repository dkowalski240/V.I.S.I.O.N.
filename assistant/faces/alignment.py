"""[Vendored from FaceLock, imports adapted.] Face alignment: level the eyes and crop the face to a fixed-size square.

A note on matching the model's training data
--------------------------------------------
Embedding models work best when you feed them crops that look like the crops
they were trained on. The FaceNet weights in facenet-pytorch were trained on
MTCNN *bounding-box crops with a margin*, resized to 160x160. They were not
trained on ArcFace-style crops where 5 landmarks are warped onto a fixed
template.

So here we do the FaceNet-compatible thing: take the detector's box, add the
same kind of margin, and resize to 160x160, with one improvement: we rotate
around the box centre so the eyes are horizontal. That removes head tilt
("roll"), which the model handles poorly, without changing scale or framing.

If you later switch to an ArcFace/InsightFace model, replace this with a
5-point similarity transform onto that model's template
(cv2.estimateAffinePartial2D does the job).

How the warp works
------------------
We build a 2x3 matrix that maps each *output* pixel (u, v) back to a *source*
pixel, then let cv2.warpAffine sample the source there (WARP_INVERSE_MAP).

    source = centre + ex * (u - N/2) * (W / N) + ey * (v - N/2) * (H / N)

where ex is the unit vector along the eye line, ey is perpendicular to it,
W x H is the box size including margin, and N is the output size. If the eyes
lie along ex in the source image, they end up on a horizontal row in the output.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import AlignmentConfig
from .detection import Detection


@dataclass
class AlignedFace:
    image: np.ndarray        # (N, N, 3) uint8 BGR
    dst_to_src: np.ndarray   # (2, 3) affine matrix: output pixel -> source pixel
    roll_deg: float          # head tilt that was removed


def eye_angle(landmarks: np.ndarray) -> float:
    """Angle of the line from image-left eye to image-right eye, in radians.

    Image y points down, so a positive angle means the right-hand eye is lower
    (the head is tilted clockwise as you look at the screen).
    """
    (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
    return float(np.arctan2(ry - ly, rx - lx))


def alignment_matrix(det: Detection, image_size: int = 160, margin: int = 20) -> np.ndarray:
    """2x3 matrix mapping output pixels to source pixels (see module docstring)."""
    n = float(image_size)
    x1, y1, x2, y2 = [float(v) for v in det.box]
    w, h = x2 - x1, y2 - y1
    # facenet-pytorch expresses margin in *output* pixels; convert to a scale
    # factor on the box: the face occupies (N - margin) of the N output pixels.
    grow = n / max(n - margin, 1.0)
    W, H = w * grow, h * grow
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    theta = eye_angle(det.landmarks)
    ex = np.array([np.cos(theta), np.sin(theta)])   # along the eye line
    ey = np.array([-np.sin(theta), np.cos(theta)])  # perpendicular, pointing "down the face"

    A = np.column_stack([ex * (W / n), ey * (H / n)])  # 2x2
    b = np.array([cx, cy]) - A @ np.array([n / 2.0, n / 2.0])
    return np.hstack([A, b[:, None]]).astype(np.float64)


def map_points_to_crop(points: np.ndarray, dst_to_src: np.ndarray) -> np.ndarray:
    """Map source-image points into aligned-crop coordinates (inverse of the warp)."""
    A, b = dst_to_src[:, :2], dst_to_src[:, 2]
    A_inv = np.linalg.inv(A)
    return (np.asarray(points, dtype=np.float64) - b) @ A_inv.T


def align_face(frame_bgr: np.ndarray, det: Detection, cfg: AlignmentConfig) -> AlignedFace:
    M = alignment_matrix(det, cfg.image_size, cfg.margin)
    crop = cv2.warpAffine(
        frame_bgr,
        M,
        (cfg.image_size, cfg.image_size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return AlignedFace(image=crop, dst_to_src=M, roll_deg=float(np.degrees(eye_angle(det.landmarks))))
