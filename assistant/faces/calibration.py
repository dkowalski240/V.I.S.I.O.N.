"""[Vendored from FaceLock, imports adapted.] Threshold calibration math (used by scripts/calibrate.py).

The idea: collect face crops of several people with *your* camera, then
compute two lists of similarity scores:

    genuine  = scores between two crops of the SAME person
    impostor = scores between crops of DIFFERENT people

Plot them and you'll see two humps. Ideally they don't overlap. Thresholds are
then chosen from these real distributions instead of guessed:

* t_match: high enough that (almost) no impostor pair reaches it.
  "FAR" (false accept rate) is the fraction of impostor pairs at or above it.
  For a lock we want FAR close to 0, and accept some "FRR" (false rejects:
  genuine pairs below it), because a false reject just means "look at the
  camera a second longer", while a false accept means a stranger gets in.
* t_new: low enough that a known person's worse frames rarely fall below it
  (otherwise they'd get a duplicate profile).
* t_update: comfortably above t_match, so only clear matches teach the gallery.

Note: these are single-image pairs. The live system averages several frames
per visit, which makes genuine scores higher and steadier than this, so
single-image numbers are a conservative (safe) starting point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def pair_scores(embeddings: np.ndarray, labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """All pairwise cosine similarities, split into (genuine, impostor)."""
    labels = np.asarray(labels)
    sims = embeddings @ embeddings.T
    iu = np.triu_indices(len(labels), k=1)
    same = labels[iu[0]] == labels[iu[1]]
    vals = sims[iu]
    return vals[same], vals[~same]


@dataclass
class Suggestion:
    t_match: float
    t_new: float
    t_update: float
    t_anchor: float
    far_at_match: float
    frr_at_match: float
    impostors_below_new: float
    notes: List[str]

    def as_yaml(self) -> str:
        return (
            "matching:\n"
            f"  t_match: {self.t_match:.2f}\n"
            f"  t_new: {self.t_new:.2f}\n"
            f"  t_update: {self.t_update:.2f}\n"
            f"  t_anchor: {self.t_anchor:.2f}\n"
        )


def _ceil2(x: float) -> float:
    return float(np.ceil(x * 100) / 100)


def _floor2(x: float) -> float:
    return float(np.floor(x * 100) / 100)


# Never suggest a t_match below this. With only a few people in the
# calibration set, the impostor distribution's tail is badly underestimated,
# and a low threshold would let lookalike strangers in.
MIN_T_MATCH = 0.55
MIN_PEOPLE = 5


def suggest_thresholds(
    genuine: np.ndarray,
    impostor: np.ndarray,
    target_far: float = 0.001,
    n_people: int = MIN_PEOPLE,
    t_novel: float = 0.92,
) -> Suggestion:
    if len(genuine) < 10 or len(impostor) < 10:
        raise ValueError("Need more data: at least 2 people with several crops each.")
    notes: List[str] = []

    # --- t_match from the impostor distribution
    if len(impostor) * target_far < 1:
        # Too few impostor pairs to measure such a small FAR. Use the worst
        # impostor seen plus a safety margin instead.
        t_match = float(impostor.max()) + 0.05
        notes.append(
            f"Only {len(impostor)} impostor pairs: can't measure FAR={target_far}. "
            "Used max impostor + 0.05. Add more people for a tighter estimate."
        )
    else:
        t_match = float(np.quantile(impostor, 1 - target_far)) + 0.02
    if t_match < MIN_T_MATCH:
        notes.append(f"Measured t_match {t_match:.2f} raised to the safety floor {MIN_T_MATCH}.")
    t_match = min(max(_ceil2(t_match), MIN_T_MATCH), 0.90)
    if n_people < MIN_PEOPLE:
        notes.append(
            f"Only {n_people} people in the data. Real strangers vary far more than that; "
            f"capture at least {MIN_PEOPLE} people before trusting these numbers."
        )

    # --- t_new from the genuine distribution, kept well below t_match
    t_new = min(float(np.quantile(genuine, 0.01)), t_match - 0.10)
    t_new = max(_floor2(t_new), 0.10)

    # --- t_update: clear matches only. It must stay below t_novel, otherwise
    # no frame could ever be both "confident" and "new" and nothing is learned.
    t_update = max(_ceil2(float(np.quantile(genuine, 0.25))), t_match + 0.10)
    t_update = min(t_update, t_novel - 0.05)
    if t_update < t_match + 0.05:
        notes.append("t_update is close to t_match; consider a lower t_match or better capture quality.")

    far = float((impostor >= t_match).mean())
    frr = float((genuine < t_match).mean())
    below_new = float((impostor < t_new).mean())

    if impostor.max() >= genuine.min():
        notes.append(
            "Genuine and impostor scores overlap (the worst same-person pair scores below the "
            "best different-person pair). Check the crops for mistakes, bad lighting or blur."
        )
    if frr > 0.2:
        notes.append(
            f"{frr:.0%} of single-image genuine pairs fall below t_match. Expect people to need a "
            "moment in front of the camera; improve lighting/camera placement to reduce this."
        )
    if below_new < 0.5:
        notes.append(
            f"Only {below_new:.0%} of impostor pairs fall below t_new, so many strangers will sit "
            "in the gray zone (logged as 'uncertain', no profile created). That is safe but less adaptive."
        )
    return Suggestion(t_match, t_new, t_update, t_update, far, frr, below_new, notes)


def summary(genuine: np.ndarray, impostor: np.ndarray) -> Dict[str, Dict[str, float]]:
    def stats(a):
        return {
            "n": int(len(a)),
            "min": float(a.min()),
            "p01": float(np.quantile(a, 0.01)),
            "median": float(np.median(a)),
            "p99": float(np.quantile(a, 0.99)),
            "max": float(a.max()),
        }

    return {"genuine": stats(genuine), "impostor": stats(impostor)}
