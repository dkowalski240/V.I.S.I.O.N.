"""Pure geometry helpers: clock directions, stereo panning, approach speed, time phrases.

Clock directions
----------------
12 o'clock is straight ahead; each hour is 30 degrees. A normal webcam sees
about 70 degrees across, so its frame edges are only about 11 and 1 o'clock.
Two mappings (config `direction.mode`):

* zones: the frame is cut into equal columns, one per entry of
  `direction.zones` (default 10, 11, 12, 1, 2). Someone at the left edge is
  "10 o'clock" even though they're really about 35 degrees left. Good for a
  webcam demo; it overstates angles.
* angle: the true angle from a pinhole model with `direction.hfov_deg`,
  rounded to the nearest hour. Honest; use it with a wide lens.

`direction.style` decides how the hour is *said*: "2 o'clock" (clock), "to your
right" (words) or both. Clock positions are a standard orientation & mobility
convention but have to be learned; plain words need no learning but are coarser.

Panning uses the (possibly virtual) angle: -60 degrees or further left is
fully left, +60 or further right is fully right, with constant-power gains so
loudness doesn't dip in the middle.

Approach detection
------------------
A face that is walking toward the camera gets bigger. Its apparent height h
grows like 1/distance, so the slope of ln(h) over time is the closing speed
divided by the distance, independent of how big the face is. A slope of 0.3/s
means the face grows ~35% per second. Someone walking past keeps a roughly
constant size, so their slope stays near zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .config import DirectionConfig


@dataclass
class Direction:
    hour: int          # 1..12
    angle_deg: float   # negative = left (virtual angle in zones mode)
    pan: float         # -1 (left) .. +1 (right)


def _hour_from_angle(angle_deg: float) -> int:
    offset = int(round(angle_deg / 30.0))
    return ((12 + offset - 1) % 12) + 1


def direction_for_x(cx_norm: float, cfg: DirectionConfig) -> Direction:
    """Clock direction for a horizontal position in the frame (0 = left edge, 1 = right)."""
    cx = min(max(float(cx_norm), 0.0), 1.0)
    if cfg.mode == "zones":
        zones = list(cfg.zones)
        n = len(zones)
        i = min(n - 1, int(cx * n))
        hour = int(zones[i])
        angle = (cx - 0.5) * n * 30.0
    else:
        half = math.radians(cfg.hfov_deg / 2.0)
        angle = math.degrees(math.atan((2.0 * cx - 1.0) * math.tan(half)))
        hour = _hour_from_angle(angle)
    pan = max(-1.0, min(1.0, angle / 60.0)) * cfg.pan_strength
    return Direction(hour=hour, angle_deg=angle, pan=pan)


_WORDS = {
    12: "straight ahead", 11: "slightly to your left", 10: "to your left", 9: "far to your left",
    1: "slightly to your right", 2: "to your right", 3: "far to your right",
    8: "behind you, to the left", 7: "behind you, to the left", 6: "behind you",
    5: "behind you, to the right", 4: "behind you, to the right",
}


def direction_words(hour: int) -> str:
    """Plain words for a clock hour: 2 -> "to your right"."""
    return _WORDS[((int(hour) - 1) % 12) + 1]


def clock_phrase(hour: int, style: str = "clock") -> str:
    """A direction on its own, after a comma: "That's Sarah, 2 o'clock" / "..., to your right"."""
    if style == "words":
        return direction_words(hour)
    if style == "both":
        return f"{hour} o'clock, {direction_words(hour)}"
    return f"{hour} o'clock"


def at_phrase(hour: int, style: str = "clock") -> str:
    """A direction after a noun: "Sarah at 2 o'clock" / "Sarah to your right"."""
    if style == "words":
        return direction_words(hour)
    if style == "both":
        return f"at {hour} o'clock, {direction_words(hour)}"
    return f"at {hour} o'clock"


def pan_gains(pan: float) -> Tuple[float, float]:
    """Constant-power (left, right) gains for pan in [-1, 1]."""
    p = max(-1.0, min(1.0, float(pan)))
    theta = (p + 1.0) * math.pi / 4.0
    return math.cos(theta), math.sin(theta)


def box_center_x_norm(box: Sequence[float], frame_width: int) -> float:
    return ((float(box[0]) + float(box[2])) / 2.0) / max(float(frame_width), 1.0)


def approach_rate(points: Sequence[Tuple[float, float]], now: float, window_s: float,
                  min_points: int = 4) -> Optional[float]:
    """Least-squares slope of ln(height) vs time over the last `window_s` seconds.

    points = [(timestamp, face_height_px), ...]. Returns None if there's too
    little history to say.
    """
    pts = [(t, h) for t, h in points if now - t <= window_s and h > 0]
    if len(pts) < max(2, min_points):
        return None
    ts = [t for t, _ in pts]
    if max(ts) - min(ts) < window_s * 0.4:
        return None
    ys = [math.log(h) for _, h in pts]
    tm = sum(ts) / len(ts)
    ym = sum(ys) / len(ys)
    var = sum((t - tm) ** 2 for t in ts)
    if var <= 1e-9:
        return None
    return sum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / var


def time_ago_phrase(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 45:
        return "just now"
    m = int(round(s / 60.0))
    if m < 60:
        return "1 minute ago" if m <= 1 else f"{m} minutes ago"
    h = int(round(s / 3600.0))
    if h < 24:
        return "1 hour ago" if h <= 1 else f"{h} hours ago"
    d = int(round(s / 86400.0))
    return "yesterday" if d <= 1 else f"{d} days ago"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
