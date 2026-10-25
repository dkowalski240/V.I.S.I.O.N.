"""Enrollment by introduction, with consent, and with other people in view.

Flow (all speech goes to the user's headphones; the user asks the person aloud):

    "Hi, I'm Sarah."                       (ambient speech, parser finds a name)
      -> pick WHICH face said it (FacesModule.pick_speaker): skip people we
         already know, prefer the face whose mouth moved during the speech,
         else the largest, most central face
      -> collect that face's embeddings IN MEMORY (strict quality gate)
      -> headphones: "Heard Sarah. Ask to save?"  or, with several faces in
         view, "Heard Sarah, 2 o'clock. Ask to save?" so the user knows which
         person will be saved (name and direction read back for correction)
    user, aloud: "Okay if I remember you?"  (a question: ignored by the answer parser)
    Sarah: "Sure."                          (ambient) or the user says "yes" with push-to-talk
      -> keep capturing for `capture_s` (more angles/expressions), pick diverse views
      -> save person + embeddings, "Saved Sarah."
    "No" / silence for `consent_timeout_s`  -> everything captured is dropped, "Not saved."
    "It's Sara, actually" (during consent)  -> name changed, asks again
    "No, it's Sara" right after saving       -> renamed (within `correction_window_s`)

Other people in view can't leak into Sarah's profile:
* Only the chosen track's frames are captured.
* Every captured view must resemble Sarah's reference appearance (the frames
  seen while she spoke) by `min_self_similarity`; a frame from someone else
  (e.g. the tracker swapped two faces that crossed) is rejected.
* If her face disappears (someone walks in front), the session waits up to
  `reacquire_s` and picks her up again on whichever new face matches her
  reference by t_match.

Manual fallback: "save this person as Sarah" (push-to-talk) runs the same flow
on the largest, most central unknown face (the user is the one talking, so
mouth motion isn't used), and also asks unless `consent_for_manual` is off.

Nothing reaches the disk until the yes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .. import events as ev
from ..bus import Bus
from ..clock import Clock
from ..config import Config
from ..geometry import clock_phrase
from .embedding import l2_normalize

log = logging.getLogger(__name__)


@dataclass
class _Session:
    name: str
    track_id: int
    started: float
    source: str                     # ambient | ptt | typed
    state: str = "consent"          # consent | capturing
    deadline: float = 0.0
    capture_until: float = 0.0
    samples: List[Tuple[np.ndarray, float, float]] = field(default_factory=list)  # (emb, quality, ts)
    ref: Optional[np.ndarray] = None   # reference appearance of the person being enrolled
    lost_since: Optional[float] = None
    hour: Optional[int] = None         # clock direction when picked
    multi: bool = False                # several faces were in view when picked
    rejected: int = 0                  # views dropped because they looked like someone else
    reacquired: int = 0


class Enrollment:
    def __init__(self, cfg: Config, bus: Bus, storage, faces, clock: Optional[Clock] = None):
        self.cfg = cfg
        self.ecfg = cfg.enrollment
        self.bus = bus
        self.storage = storage
        self.faces = faces
        self.clock = clock or Clock()
        self._lock = threading.RLock()
        self.session: Optional[_Session] = None
        self._last_saved: Optional[Tuple[int, str, float]] = None  # (person_id, name, time)

    # ------------------------------------------------------------------ speech helpers
    def _say(self, text: str, tone: Optional[str] = None, kind: str = ev.KIND_ANSWER) -> None:
        self.bus.publish(ev.SAY, text=text, priority=ev.PRI_ANSWER, pan=0.0, kind=kind, tone=tone)

    def _prompt(self, s: _Session) -> str:
        if s.multi and s.hour is not None and self.ecfg.prompt_multi:
            return self.ecfg.prompt_multi.format(name=s.name, where=clock_phrase(s.hour, self.cfg.direction.style))
        return self.ecfg.prompt.format(name=s.name)

    # ------------------------------------------------------------------ appearance checks
    def _reference(self, s: _Session) -> Optional[np.ndarray]:
        if s.ref is not None:
            return s.ref
        if len(s.samples) >= 2:
            s.ref = l2_normalize(np.mean([e for e, _, _ in s.samples], axis=0))
            return s.ref
        return None

    def _consistent(self, s: _Session, emb: np.ndarray) -> bool:
        ref = self._reference(s)
        return ref is None or float(ref @ emb) >= self.ecfg.min_self_similarity

    # ------------------------------------------------------------------ hooks used by FacesModule
    def wants_track(self, track_id: int) -> bool:
        s = self.session
        return s is not None and s.track_id == track_id

    def target_track(self) -> Optional[int]:
        s = self.session
        return s.track_id if s is not None else None

    def offer(self, track_id: int, embedding: np.ndarray, quality: float, ts: float) -> None:
        with self._lock:
            s = self.session
            if s is None or s.track_id != track_id:
                return
            if not self._consistent(s, embedding):
                s.rejected += 1
                return
            t_novel = self.cfg.faces.matching.t_novel
            for i, (e, q, _) in enumerate(s.samples):
                if float(e @ embedding) >= t_novel:  # near-duplicate: keep the better one
                    if quality > q:
                        s.samples[i] = (embedding, quality, ts)
                    return
            s.samples.append((embedding, quality, ts))
            if len(s.samples) > 3 * self.ecfg.max_exemplars:
                s.samples.sort(key=lambda x: x[1], reverse=True)
                del s.samples[3 * self.ecfg.max_exemplars:]

    def try_reacquire(self, track_id: int, embedding: np.ndarray, now: float) -> bool:
        """The enrolled face vanished; is this new face the same person?"""
        with self._lock:
            s = self.session
            if s is None or track_id == s.track_id:
                return False
            ref = self._reference(s)
            if ref is None or float(ref @ embedding) < self.cfg.faces.matching.t_match:
                return False
            log.info("Enrollment of %s: found them again on track %d (was %d)", s.name, track_id, s.track_id)
            s.track_id = track_id
            s.lost_since = None
            s.reacquired += 1
            return True

    # ------------------------------------------------------------------ inputs
    def busy(self) -> bool:
        return self.session is not None

    def awaiting_answer(self) -> bool:
        s = self.session
        return s is not None and s.state == "consent"

    def start_from_introduction(self, name: str, t0: float, t1: float, source: str = "ambient") -> bool:
        pad = self.ecfg.speaking_pad_s
        speaker = self.faces.pick_speaker(t0 - pad, t1 + pad, name=name, prefer_talking=True)
        return self._start(name, speaker, source)

    def start_manual(self, name: str, now: Optional[float] = None, source: str = "ptt") -> bool:
        now = self.clock.now() if now is None else now
        speaker = self.faces.pick_speaker(now - 1.0, now + 0.1, name=name, prefer_talking=False)
        return self._start(name, speaker, source)

    def _start(self, name: str, speaker, source: str) -> bool:
        now = self.clock.now()
        with self._lock:
            if self.session is not None:
                log.info("Ignoring introduction of %s: already enrolling %s", name, self.session.name)
                return False
            if speaker is None:
                log.info("Introduction of %s but no face to pick", name)
                if source != "ambient":
                    self._say("No face in view.", tone=ev.TONE_ERROR)
                return False
            if speaker.get("person_id") is not None:
                known = speaker.get("name") or "someone I know"
                if speaker.get("reason") == "known_same_name":
                    self._say(f"I already know {known}.")
                else:
                    self._say(f"That looks like {known}. Not saved.")
                return False
            s = _Session(name=name, track_id=speaker["track_id"], started=now, source=source,
                         deadline=now + self.ecfg.consent_timeout_s, hour=speaker.get("hour"),
                         multi=speaker.get("n_faces", 1) > 1)
            seen = self.faces.track_samples(s.track_id)  # frames from while they were speaking
            if seen:
                s.ref = l2_normalize(np.mean([e for e, _, _ in seen], axis=0))
            self.session = s
            log.info("Enrolling %s from track %d (%s; %s, %d faces in view, talk %.3f)", name, s.track_id, source,
                     speaker.get("reason"), speaker.get("n_faces", 1), speaker.get("talk", 0.0))
            if source != "ambient" and not self.ecfg.consent_for_manual:
                self._begin_capture(now)
            else:
                self._say(self._prompt(s), kind=ev.KIND_PROMPT)
            return True

    def on_answer(self, intent, now: Optional[float] = None) -> None:
        """intent.type in yes | no | correction."""
        now = self.clock.now() if now is None else now
        with self._lock:
            s = self.session
            if s is None or s.state != "consent":
                return
            if intent.type == "yes":
                self._begin_capture(now)
            elif intent.type == "no":
                self._discard("Okay, not saved.")
            elif intent.type == "correction":
                s.name = intent.args["name"]
                s.deadline = now + self.ecfg.consent_timeout_s
                self._say(self._prompt(s), kind=ev.KIND_PROMPT)

    def handle_correction(self, new_name: str, now: Optional[float] = None) -> bool:
        """'No, it's Sara' after saving: rename. Returns True if handled."""
        now = self.clock.now() if now is None else now
        with self._lock:
            if self.session is not None and self.session.state == "consent":
                self.session.name = new_name
                self.session.deadline = now + self.ecfg.consent_timeout_s
                self._say(self._prompt(self.session), kind=ev.KIND_PROMPT)
                return True
            if self._last_saved and now - self._last_saved[2] <= self.ecfg.correction_window_s:
                pid, old, _ = self._last_saved
                self.storage.rename_person(pid, new_name)
                self.faces.reload_gallery()
                self._last_saved = (pid, new_name, now)
                self._say(f"Renamed to {new_name}.", tone=ev.TONE_DONE)
                log.info("Renamed %s -> %s", old, new_name)
                return True
        return False

    def cancel(self) -> bool:
        with self._lock:
            if self.session is None:
                return False
            self._discard("Cancelled. Not saved.")
            return True

    # ------------------------------------------------------------------ time-based steps
    def tick(self, now: Optional[float] = None) -> None:
        now = self.clock.now() if now is None else now
        with self._lock:
            s = self.session
            if s is None:
                return
            if self.faces.track_alive(s.track_id):
                s.lost_since = None
            elif s.lost_since is None:
                s.lost_since = now
            lost = s.lost_since is not None and now - s.lost_since >= self.ecfg.reacquire_s
            if s.state == "consent":
                if now > s.deadline:
                    self._discard("No answer. Not saved.")
                elif lost and len(s.samples) < self.ecfg.min_exemplars:
                    self._discard(f"Lost sight of {s.name}. Not saved.")
                return
            # capturing
            if now >= s.capture_until or lost or len(s.samples) >= 3 * self.ecfg.max_exemplars:
                self._finish(now)

    def _begin_capture(self, now: float) -> None:
        s = self.session
        s.state = "capturing"
        s.capture_until = now + self.ecfg.capture_s
        self.bus.publish(ev.SAY, text="", priority=ev.PRI_ANSWER, kind=ev.KIND_ANSWER, tone=ev.TONE_CAPTURE)

    def _select(self, samples) -> List[Tuple[np.ndarray, float, float]]:
        t_novel = self.cfg.faces.matching.t_novel
        chosen: List[Tuple[np.ndarray, float, float]] = []
        for smp in sorted(samples, key=lambda x: x[1], reverse=True):
            if len(chosen) >= self.ecfg.max_exemplars:
                break
            if all(float(c[0] @ smp[0]) < t_novel for c in chosen):
                chosen.append(smp)
        return chosen

    def _finish(self, now: float) -> None:
        s = self.session
        chosen = self._select(s.samples)
        if len(chosen) < self.ecfg.min_exemplars:
            # Too few frames passed the strict gate (dim light, slight turn):
            # top up with this face's recognition-quality frames that still look like them.
            extra = [smp for smp in self.faces.track_samples(s.track_id) if self._consistent(s, smp[0])]
            chosen = self._select(s.samples + extra)
        if len(chosen) < self.ecfg.min_exemplars:
            self._discard(f"Couldn't see {s.name} clearly. Not saved.")
            return
        pid = self.storage.add_person(s.name, [c[0] for c in chosen], [c[1] for c in chosen], now=now)
        self.faces.reload_gallery()
        self.faces.mark_announced(pid, s.track_id, now)
        self._last_saved = (pid, s.name, now)
        self.bus.publish(ev.PERSON_SAVED, person_id=pid, name=s.name, n_exemplars=len(chosen), source=s.source,
                         rejected_views=s.rejected, reacquired=s.reacquired)
        log.info("Saved %s (person %d) with %d views (%d rejected as someone else, re-found %d times)",
                 s.name, pid, len(chosen), s.rejected, s.reacquired)
        name = s.name
        s.samples.clear()
        self.session = None
        self._say(f"Saved {name}.", tone=ev.TONE_DONE)

    def _discard(self, message: str) -> None:
        s = self.session
        if s is not None:
            s.samples.clear()  # the captured embeddings are gone
            s.ref = None
            log.info("Enrollment of %s discarded: %s", s.name, message)
        self.session = None
        self._say(message)

    def status(self) -> Optional[dict]:
        s = self.session
        if s is None:
            return None
        return {"state": s.state, "name": s.name, "track_id": s.track_id, "views": len(s.samples),
                "rejected": s.rejected,
                "remaining": round(max(0.0, (s.deadline if s.state == "consent" else s.capture_until)
                                       - self.clock.now()), 1)}
