"""Lock onto one face among several, for the capture scripts.

Starts on the largest face (ties: the most central), then follows it frame to
frame by box overlap. If it's lost, it can be found again by appearance (an
embedding similarity check supplied by the caller). `next()` switches to the
next face, left to right, so the operator can pick a different person.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from ..geometry import iou


class FaceFollower:
    def __init__(self, min_iou: float = 0.2):
        self.min_iou = min_iou
        self.box: Optional[np.ndarray] = None
        self._switch = False

    def next(self) -> None:
        self._switch = True

    def reset(self) -> None:
        self.box = None

    def update(self, dets: List, frame_w: int, same_person: Optional[Callable[[object], bool]] = None):
        """Return the followed detection in this frame (or None)."""
        if not dets:
            return None
        if self._switch and self.box is not None:
            self._switch = False
            order = sorted(dets, key=lambda d: (d.box[0] + d.box[2]) / 2)
            cur = max(order, key=lambda d: iou(d.box, self.box))
            i = next(k for k, d in enumerate(order) if d is cur)  # identity, not ==: boxes are arrays
            nxt = order[(i + 1) % len(order)]
            self.box = nxt.box
            return nxt
        if self.box is None:
            pick = max(dets, key=lambda d: d.width * (1.0 - 0.5 * abs((d.box[0] + d.box[2]) / 2 / frame_w - 0.5)))
            self.box = pick.box
            return pick
        best = max(dets, key=lambda d: iou(d.box, self.box))
        if iou(best.box, self.box) >= self.min_iou:
            self.box = best.box
            return best
        if same_person is not None:
            for d in dets:
                if same_person(d):
                    self.box = d.box
                    return d
        return None
