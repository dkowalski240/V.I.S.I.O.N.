"""Is this face talking? A cheap visual voice-activity cue from mouth motion.

When several people are in view and someone says "Hi, I'm Sarah", the face
whose mouth moves during that speech is the one introducing herself, even if
a silent bystander is bigger or more central.

Per tracked face and frame, on the eye-levelled 160x160 crop:

    mouth patch: between the mouth corners, from just under the nose to the chin
    eye patch:   across both eyes (a reference that moves with the head)
    activity = mean |mouth - previous mouth|  -  mean |eyes - previous eyes|

Each patch is resized to a fixed size, scaled to 0..1 and has its own mean
removed, so lighting changes and face size cancel out. Subtracting the eye
motion removes most of the head-motion component: nodding moves both
patches, talking mostly moves the mouth.

It's a heuristic: smiling or chewing also counts. That's why the speaker
picker only trusts it when one face clearly talks more than the others
(enrollment.talk_min / talk_ratio), and otherwise falls back to "largest,
most central face". The debug window shows each face's value ("talk 0.03")
so the thresholds can be tuned on site.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

MOUTH_SIZE = (32, 16)  # (w, h)
EYES_SIZE = (32, 12)
MAX_GAP_S = 0.3        # frames further apart than this aren't compared


def _rect(x1, y1, x2, y2, w, h) -> Optional[Tuple[int, int, int, int]]:
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(w, x2)), int(min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def regions(pts: np.ndarray, size: int) -> Tuple[Optional[tuple], Optional[tuple]]:
    """Mouth and eye rectangles in crop coordinates from the 5 landmarks
    (left eye, right eye, nose, left mouth corner, right mouth corner)."""
    le, re, nose, ml, mr = [np.asarray(p, dtype=np.float64) for p in pts]
    mw = max(mr[0] - ml[0], 1.0)
    my = (ml[1] + mr[1]) / 2.0
    nh = max(my - nose[1], 1.0)
    mouth = _rect(ml[0] - 0.25 * mw, nose[1] + 0.4 * nh, mr[0] + 0.25 * mw, my + 0.9 * nh, size, size)
    ew = max(re[0] - le[0], 1.0)
    ey = (le[1] + re[1]) / 2.0
    eyes = _rect(le[0] - 0.3 * ew, ey - 0.35 * ew, re[0] + 0.3 * ew, ey + 0.25 * ew, size, size)
    return mouth, eyes


def _patch(gray: np.ndarray, rect, out_size) -> np.ndarray:
    x1, y1, x2, y2 = rect
    p = cv2.resize(gray[y1:y2, x1:x2], out_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return p - p.mean()


def update(track, crop_bgr: np.ndarray, pts_in_crop: np.ndarray, t: float) -> Optional[float]:
    """Measure this frame's mouth activity for `track` and append (t, activity)."""
    mouth_r, eyes_r = regions(pts_in_crop, crop_bgr.shape[0])
    if mouth_r is None or eyes_r is None:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mouth, eyes = _patch(gray, mouth_r, MOUTH_SIZE), _patch(gray, eyes_r, EYES_SIZE)
    act = None
    if track.prev_mouth is not None and t - track.prev_talk_t <= MAX_GAP_S:
        dm = float(np.abs(mouth - track.prev_mouth).mean())
        de = float(np.abs(eyes - track.prev_eyes).mean())
        act = max(0.0, dm - de)
        track.talk.append((t, act))
    track.prev_mouth, track.prev_eyes, track.prev_talk_t = mouth, eyes, t
    return act


def activity(track, t0: float, t1: float) -> float:
    """Mean mouth activity of a track between t0 and t1 (0 if never measured)."""
    vals = [a for t, a in track.talk if t0 <= t <= t1]
    return float(np.mean(vals)) if vals else 0.0
