"""[Adapted from FaceLock's open-set matcher.] Known person, or nobody we know?

Open-set: the answer can be "nobody". Two tests must both pass:

    best similarity >= threshold      (close enough to someone)
    best - runner-up >= margin        (and clearly *that* someone)

Used twice:
* per frame (`vote`) with t_frame: each embedded frame casts a vote for one
  person or for nobody. A track needs `agree_frames` consecutive votes for the
  same person.
* per track (`decide`) with t_match on the averaged embedding of the track's
  best frames. Both must agree before anything is announced. A stranger who
  looks a bit like Sarah in one frame won't get through both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import MatchingConfig
from .gallery import Gallery


@dataclass
class MatchResult:
    person_id: Optional[int]   # None = nobody we know
    name: Optional[str]
    best: Optional[float]      # best similarity (None if the gallery is empty)
    second: Optional[float]
    reason: str


def _match(query: np.ndarray, gallery: Gallery, threshold: float, margin: float) -> MatchResult:
    scores = gallery.score_all(query)
    if not scores:
        return MatchResult(None, None, None, None, "nobody enrolled")
    top = scores[0]
    second = scores[1].score if len(scores) > 1 else None
    if top.score < threshold:
        return MatchResult(None, None, top.score, second, "too far from everyone")
    if second is not None and top.score - second < margin:
        return MatchResult(None, None, top.score, second, "ambiguous between two people")
    return MatchResult(top.person_id, top.name, top.score, second, "match")


def vote(embedding: np.ndarray, gallery: Gallery, cfg: MatchingConfig) -> MatchResult:
    """Single-frame vote."""
    return _match(embedding, gallery, cfg.t_frame, cfg.margin)


def decide(query: np.ndarray, gallery: Gallery, cfg: MatchingConfig) -> MatchResult:
    """Decision on a track's averaged embedding."""
    return _match(query, gallery, cfg.t_match, cfg.margin)
