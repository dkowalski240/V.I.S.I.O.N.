"""Turn nearby detections into a short location phrase:
"on the table, next to the laptop".

* "on the X": a surface (table, desk, shelf...) whose box contains the item's
  bottom-centre point. The smallest such surface wins (the desk, not the floor).
* "next to the Y": the closest other context object, if it's within about
  1.5 item-sizes (+ a bit of the frame) of the item.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


def _gap(a, b) -> float:
    """Distance between two boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def describe(item_box: Sequence[float], context_dets: Iterable, surface_labels: Iterable[str],
             frame_w: int, frame_h: int) -> str:
    surfaces_set = set(surface_labels)
    x1, y1, x2, y2 = [float(v) for v in item_box]
    iw, ih = x2 - x1, y2 - y1
    cx, by = (x1 + x2) / 2, y2

    on: Optional[object] = None
    near: Optional[object] = None
    near_gap = 1.5 * max(iw, ih) + 0.08 * frame_w
    dets: List = list(context_dets)
    for d in dets:
        b = d.box
        if d.label in surfaces_set:
            inside_x = b[0] <= cx <= b[2]
            inside_y = b[1] - 0.15 * ih <= by <= b[3] + 0.05 * ih
            bigger = (b[2] - b[0]) * (b[3] - b[1]) > iw * ih
            if inside_x and inside_y and bigger:
                if on is None or (b[2] - b[0]) * (b[3] - b[1]) < (on.box[2] - on.box[0]) * (on.box[3] - on.box[1]):
                    on = d
    for d in dets:
        if d.label in surfaces_set or d is on:
            continue
        g = _gap(item_box, d.box)
        if g < near_gap:
            near, near_gap = d, g
    parts = []
    if on is not None:
        parts.append(f"on the {on.label}")
    if near is not None:
        parts.append(f"next to the {near.label}")
    return ", ".join(parts)
