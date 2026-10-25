"""[Adapted from FaceLock.] Track faces between frames so each person is one
"encounter", announced once, instead of 15 guesses per second.

Detections that overlap the previous box (IoU) are the same person. Additions
for the assistant:

* `history`: (time, box height, centre x, area) per detection, used for
  approach speed and for picking who was in front of the user while they spoke.
* per-frame identity votes, so recognition needs several *consecutive*
  agreeing frames, not one lucky frame.
* announcement state (announced, announced_approach).

A track's embeddings live only in memory and disappear when the track ends.
That is how unknown faces are "discarded immediately".
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .detection import Detection
from .embedding import l2_normalize
from .quality import QualityReport
from ..geometry import iou


@dataclass
class Sample:
    embedding: np.ndarray  # (512,) unit vector
    quality: float         # QualityReport.score
    timestamp: float


@dataclass
class Track:
    id: int
    box: np.ndarray
    first_seen: float
    last_seen: float
    hits: int = 1
    samples: List[Sample] = field(default_factory=list)
    last_quality: Optional[QualityReport] = None
    # (t, height_px, cx_norm, area_frac) for every detection
    history: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=120))
    # (t, mouth activity) per frame, see talking.py; plus the previous patches it compares against
    talk: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=120))
    prev_mouth: Optional[np.ndarray] = None
    prev_eyes: Optional[np.ndarray] = None
    prev_talk_t: float = 0.0

    # identity votes: person_id or None per embedded frame
    vote_pid: Optional[int] = None
    vote_streak: int = 0
    # recognition result
    person_id: Optional[int] = None
    name: Optional[str] = None
    confidence: float = 0.0
    best_score: Optional[float] = None
    identified_at: Optional[float] = None
    # announcement state
    announced: bool = False
    announced_approach: bool = False
    continuation: bool = False       # same encounter as an earlier track of this person
    approaching: bool = False
    maybe_approaching: bool = False  # growing fast, but not enough total growth yet
    approach_rate: Optional[float] = None

    def add_sample(self, sample: Sample, max_samples: int) -> None:
        """Keep the `max_samples` best-quality samples."""
        self.samples.append(sample)
        self.samples.sort(key=lambda s: s.quality, reverse=True)
        del self.samples[max_samples:]

    def query_embedding(self) -> Optional[np.ndarray]:
        """Mean of the track's embeddings, re-normalised to unit length."""
        if not self.samples:
            return None
        return l2_normalize(np.mean([s.embedding for s in self.samples], axis=0))

    def heights(self) -> List[Tuple[float, float]]:
        return [(t, h) for t, h, _, _ in self.history]

    def reset_identity(self) -> None:
        self.person_id = None
        self.name = None
        self.confidence = 0.0
        self.identified_at = None
        self.vote_pid, self.vote_streak = None, 0
        self.announced = self.announced_approach = self.continuation = False
        self.samples.clear()

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.2, timeout_s: float = 1.0):
        self.iou_threshold = iou_threshold
        self.timeout_s = timeout_s
        self.tracks: List[Track] = []
        self._ids = itertools.count(1)

    def update(self, detections: List[Detection], now: float) -> Tuple[List[Tuple[Track, Detection]], List[Track]]:
        """Pair every detection with a track (new or existing).

        Returns (assignments, ended): ended tracks haven't been seen for
        `timeout_s` and are dropped (with their embeddings).
        """
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                score = iou(track.box, det.box)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        assignments: List[Tuple[Track, Detection]] = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            track, det = self.tracks[ti], detections[di]
            track.box = det.box
            track.last_seen = now
            track.hits += 1
            assignments.append((track, det))

        for di, det in enumerate(detections):
            if di not in used_d:
                track = Track(id=next(self._ids), box=det.box, first_seen=now, last_seen=now)
                self.tracks.append(track)
                assignments.append((track, det))

        ended = [t for t in self.tracks if now - t.last_seen > self.timeout_s]
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.timeout_s]
        return assignments, ended

    def flush(self) -> List[Track]:
        ended, self.tracks = self.tracks, []
        return ended
