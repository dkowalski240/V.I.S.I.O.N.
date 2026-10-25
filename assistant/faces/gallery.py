"""[Adapted from FaceLock's gallery.py.] In-memory face galleries.

Each person keeps several real embeddings ("exemplars") captured from
different angles during enrollment, and a face is compared to the *closest*
one. That handles glasses on/off or a turned head better than one averaged
vector.

FaceLock also learned new exemplars automatically after confident matches.
That's off here: the assistant only stores what was captured with consent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .embedding import l2_normalize


@dataclass
class GalleryEntry:
    person_id: int
    name: str
    embeddings: np.ndarray  # (k, 512)


@dataclass
class PersonScore:
    person_id: int
    name: str
    score: float  # best cosine similarity over this person's exemplars


class Gallery:
    """Thread-safe cache of everyone's exemplars, rebuilt from storage."""

    def __init__(self):
        self._entries: Dict[int, GalleryEntry] = {}
        self._lock = threading.RLock()

    def load(self, storage) -> None:
        with self._lock:
            self._entries.clear()
            for pid, name, embs, _ids in storage.face_gallery():
                self._entries[pid] = GalleryEntry(pid, name, l2_normalize(embs))

    def set_person(self, person_id: int, name: str, embeddings: np.ndarray) -> None:
        with self._lock:
            self._entries[person_id] = GalleryEntry(person_id, name, l2_normalize(np.asarray(embeddings)))

    def remove(self, person_id: int) -> None:
        with self._lock:
            self._entries.pop(person_id, None)

    def has(self, person_id: int) -> bool:
        with self._lock:
            return person_id in self._entries

    def name(self, person_id: int) -> Optional[str]:
        with self._lock:
            e = self._entries.get(person_id)
            return e.name if e else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def score_all(self, query: np.ndarray) -> List[PersonScore]:
        """Score a unit-vector query against everyone, best first."""
        out = []
        with self._lock:
            for pid, e in self._entries.items():
                out.append(PersonScore(pid, e.name, float(np.max(e.embeddings @ query))))
        out.sort(key=lambda r: r.score, reverse=True)
        return out
