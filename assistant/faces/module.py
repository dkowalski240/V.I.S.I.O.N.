"""Faces module: detect -> track -> embed -> match -> announce known people.

Runs on its own thread at up to `faces.max_fps`, always on the newest camera
frame. Per frame:

    detect faces (MTCNN/YuNet)
      -> IoU tracking (one track = one encounter)
      -> record face size/position history (approach speed, speaker picking)
      -> lenient quality gate -> align -> embed (one batched pass)
      -> per-frame vote: which enrolled person is this, if anyone?
      -> identify a track only when `agree_frames` consecutive votes agree AND
         the track's averaged embedding passes t_match + margin
      -> approach estimate from face-size growth
      -> announce once per encounter, with per-person cooldowns,
         publishing a `person_seen` event (the router turns it into speech)

Unknown faces: their embeddings live in the Track object in RAM and vanish
when the track ends ~1 s after the face leaves. Nothing about them is written
anywhere. While an enrollment is running, embeddings of the one face being
enrolled are handed to the enrollment session, which also keeps them in RAM
until the person says yes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .. import events as ev
from ..bus import Bus
from ..clock import Clock
from ..config import Config
from ..geometry import approach_rate, direction_for_x
from ..storage import name_key
from . import talking
from .alignment import align_face, map_points_to_crop
from .gallery import Gallery
from .matcher import decide, vote
from .quality import HINTS, QualityGate, QualityReport
from .tracking import IoUTracker, Sample, Track

log = logging.getLogger(__name__)

NEVER = -1e18


class NullDetector:
    """--stub mode: finds nothing."""

    def detect(self, frame_bgr):
        return []


class NullEmbedder:
    dim = 512

    def embed(self, crops):
        return np.zeros((len(crops), 512), dtype=np.float32)


class FacesModule:
    def __init__(self, cfg: Config, camera, bus: Bus, storage, clock: Optional[Clock] = None,
                 detector=None, embedder=None):
        self.cfg = cfg
        self.fcfg = cfg.faces
        self.camera = camera
        self.bus = bus
        self.storage = storage
        self.clock = clock or Clock()
        self.detector = detector or NullDetector()
        self.embedder = embedder or NullEmbedder()
        self.gate = QualityGate(self.fcfg.quality)
        self.enroll_gate = QualityGate(self.fcfg.enroll_quality)
        self.tracker = IoUTracker(self.fcfg.tracking.iou_threshold, self.fcfg.tracking.timeout_s)
        self.gallery = Gallery()
        self.gallery.load(storage)
        self.enrollment = None  # set by the app: has wants_track(track_id) and offer(...)
        self._lock = threading.RLock()
        self._view_lock = threading.Lock()
        self._view: List[Dict[str, Any]] = []
        self.frame_size = (0, 0)
        self.stats = {"fps": 0.0, "ms": 0.0, "faces": 0}
        self._last_any: Dict[int, float] = {}
        self._last_approach: Dict[int, float] = {}
        # Last time each person was seen on ANY track. A new track of someone
        # seen within announce.encounter_gap_s is the same encounter (they
        # looked down, turned away, or were briefly hidden): not announced again.
        self._person_last_seen: Dict[int, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        log.info("Faces: %d people enrolled", len(self.gallery))

    # ================================================================ thread
    def start(self) -> "FacesModule":
        self._thread = threading.Thread(target=self._run, name="faces", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        period = 1.0 / self.fcfg.max_fps if self.fcfg.max_fps > 0 else 0.0
        last_index, last_t = 0, time.monotonic()
        while not self._stop.is_set():
            frame = self.camera.wait_for(last_index, timeout=0.5)
            if frame is None:
                continue
            last_index = frame.index
            t0 = time.monotonic()
            try:
                self.process_frame(frame.image, frame.timestamp)
            except Exception:
                log.exception("Faces: error while processing a frame")
            dt = time.monotonic() - t0
            self.stats["ms"] = dt * 1000
            now_m = time.monotonic()
            self.stats["fps"] = 0.9 * self.stats["fps"] + 0.1 / max(now_m - last_t, 1e-3)
            last_t = now_m
            if dt < period:
                time.sleep(period - dt)

    # ================================================================ per frame
    def process_frame(self, image: np.ndarray, now: float) -> None:
        h, w = image.shape[:2]
        self.frame_size = (w, h)
        detections = self.detector.detect(image)
        mcfg = self.fcfg.matching
        # Views for an enrollment in progress are handed over AFTER releasing
        # self._lock: the enrollment holds its own lock while calling into this
        # module, so taking both locks in the opposite order here could deadlock.
        offers = []
        with self._lock:
            assignments, _ended = self.tracker.update(detections, now)
            to_embed = []
            enrolling = self.enrollment
            for track, det in assignments:
                x1, y1, x2, y2 = [float(v) for v in det.box]
                track.history.append((now, y2 - y1, ((x1 + x2) / 2) / w, (x2 - x1) * (y2 - y1) / (w * h)))
                reasons, metrics = self.gate.check_geometry(det, image.shape)
                if reasons:
                    track.last_quality = QualityReport(False, 0.0, reasons, metrics)
                    continue
                aligned = align_face(image, det, self.fcfg.alignment)
                talking.update(track, aligned.image, map_points_to_crop(det.landmarks, aligned.dst_to_src), now)
                report = self.gate.assess(det, image.shape, aligned)
                track.last_quality = report
                if not report.ok:
                    continue
                for_enroll = False
                if enrolling is not None and enrolling.wants_track(track.id):
                    for_enroll = self.enroll_gate.assess(det, image.shape, aligned).ok
                to_embed.append((track, aligned, report, for_enroll))

            if to_embed:
                embs = self.embedder.embed([a.image for _, a, _, _ in to_embed])
                for (track, _aligned, report, for_enroll), emb in zip(to_embed, embs):
                    track.add_sample(Sample(emb, report.score, now), self.fcfg.tracking.max_samples)
                    res = vote(emb, self.gallery, mcfg)
                    if res.person_id is not None and res.person_id == track.vote_pid:
                        track.vote_streak += 1
                    else:
                        track.vote_pid = res.person_id
                        track.vote_streak = 1 if res.person_id is not None else 0
                    track.best_score = res.best
                    if for_enroll and enrolling is not None:
                        offers.append((track.id, emb, report.score, now))

            # The face being enrolled vanished (someone walked in front, or the
            # tracker lost it)? Offer every unknown face so the enrollment can
            # find the same person again by appearance.
            reacquire = []
            target = enrolling.target_track() if enrolling is not None else None
            if target is not None and not any(t.id == target for t in self.tracker.tracks):
                reacquire = [(track.id, emb) for (track, _a, _r, _f), emb in zip(to_embed, embs if to_embed else [])
                             if track.person_id is None]

            for track in self.tracker.tracks:
                self._update_identity(track, now)
                self._update_approach(track, now, h)
                self._maybe_announce(track, now)
                if track.person_id is not None and track.last_seen == now:
                    self._person_last_seen[track.person_id] = now
            view = [self._view_of(t, now) for t in self.tracker.tracks]
        if enrolling is not None:
            for offer in offers:
                enrolling.offer(*offer)
            for track_id, emb in reacquire:
                if enrolling.try_reacquire(track_id, emb, now):
                    break
        with self._view_lock:
            self._view = view
            self.stats["faces"] = len(detections)

    def _update_identity(self, track: Track, now: float) -> None:
        mcfg = self.fcfg.matching
        if track.person_id is not None:
            if not self.gallery.has(track.person_id):  # forgotten while in view
                track.reset_identity()
                return
            q = track.query_embedding()
            if q is not None:
                res = decide(q, self.gallery, mcfg)
                if res.person_id == track.person_id:
                    track.confidence = float(res.best)
                elif (track.vote_pid is not None and track.vote_pid != track.person_id
                      and track.vote_streak >= 2 * mcfg.agree_frames):
                    # The tracker probably swapped two people; start over.
                    track.reset_identity()
            return
        if track.vote_pid is None or track.vote_streak < mcfg.agree_frames:
            return
        if len(track.samples) < mcfg.min_samples:
            return
        res = decide(track.query_embedding(), self.gallery, mcfg)
        if res.person_id is not None and res.person_id == track.vote_pid:
            track.person_id = res.person_id
            track.name = res.name
            track.confidence = float(res.best)
            track.identified_at = now
            prev = self._person_last_seen.get(res.person_id)
            if prev is not None and now - prev <= self.fcfg.announce.encounter_gap_s:
                track.announced = True  # same encounter as their previous track: don't re-announce
                track.continuation = True
            log.info("Track %d is %s (%.2f, %d agreeing frames%s)", track.id, res.name, res.best, track.vote_streak,
                     ", same encounter" if track.continuation else "")

    def _update_approach(self, track: Track, now: float, frame_h: int) -> None:
        """Approaching = the face keeps growing. Two guards against false alarms:
        the first `settle_s` of a track are ignored (a face turning back toward the
        camera, or being re-detected, makes its box grow without anyone moving), the
        face must have grown by `min_growth` over the last `lookback_s` (leaning in
        doesn't do that, walking up does), and a face already taller than
        `max_face_frac` of the frame is close by (e.g. the person sitting at the laptop)."""
        a = self.fcfg.approach
        pts = [(t, ht) for t, ht in track.heights() if t >= track.first_seen + a.settle_s]
        rate = approach_rate(pts, now, a.window_s, a.min_points)
        track.approach_rate = rate
        if rate is None:
            track.approaching = track.maybe_approaching = False
            return
        close = bool(track.history) and track.history[-1][1] > a.max_face_frac * max(frame_h, 1)
        # Total growth: walking up makes a face much bigger; leaning in or a jittery box doesn't.
        recent = [ht for t, ht in pts if now - t <= a.lookback_s]
        h_now = float(np.median([ht for _, ht in pts[-3:]]))
        growth = h_now / max(min(recent), 1e-6) if recent else 1.0
        track.maybe_approaching = (not close) and rate >= a.growth_per_s and growth < a.min_growth
        if close:
            track.approaching = False
        elif track.approaching:
            track.approaching = rate >= a.growth_per_s * 0.5  # hysteresis
        else:
            track.approaching = rate >= a.growth_per_s and growth >= a.min_growth

    def _maybe_announce(self, track: Track, now: float) -> None:
        if track.person_id is None or track.last_seen < now:  # only tracks seen this frame
            return
        acfg = self.fcfg.announce
        pid = track.person_id
        approach_ok = now - self._last_approach.get(pid, NEVER) >= acfg.approach_cooldown_s
        if not track.announced:
            if track.approach_rate is None:
                # Give the approach estimate a moment, counted from when one is first possible,
                # so someone walking up gets one "approaching" announcement, not "That's X" first.
                a = self.fcfg.approach
                possible_at = max(track.identified_at or now, track.first_seen + a.settle_s + 0.4 * a.window_s)
                if now < possible_at + acfg.approach_wait_s:
                    return
            elif track.maybe_approaching and now < (track.identified_at or now) + acfg.approach_wait_s + 0.8:
                return  # growing fast but not confirmed yet: wait a moment rather than say "That's X" first
            track.announced = True
            if track.approaching and approach_ok:
                self._emit(track, now, approaching=True)
            elif now - self._last_any.get(pid, NEVER) >= acfg.cooldown_s:
                self._emit(track, now, approaching=False)
        elif (acfg.upgrade_to_approaching and track.approaching and not track.announced_approach
              and approach_ok):
            self._emit(track, now, approaching=True)

    def _emit(self, track: Track, now: float, approaching: bool) -> None:
        pid = track.person_id
        cx = track.history[-1][2] if track.history else 0.5
        d = direction_for_x(cx, self.cfg.direction)
        sure = track.confidence >= self.fcfg.matching.t_sure
        self._last_any[pid] = now
        if approaching:
            self._last_approach[pid] = now
            track.announced_approach = True
        self.bus.publish(
            ev.PERSON_SEEN, ts=now, name=track.name, person_id=pid, track_id=track.id,
            clock_direction=d.hour, angle_deg=round(d.angle_deg, 1), pan=round(d.pan, 3),
            approaching=approaching, confidence=round(track.confidence, 3), sure=sure,
            camera_id=getattr(self.camera, "camera_id", ""),
        )
        try:
            self.storage.touch_person(pid, now)
        except Exception:
            pass

    # ================================================================ views / queries
    def _view_of(self, t: Track, now: float) -> Dict[str, Any]:
        cx = t.history[-1][2] if t.history else 0.5
        d = direction_for_x(cx, self.cfg.direction)
        q = t.last_quality
        if t.person_id is not None:
            state = "known"
        elif t.vote_pid is not None or (t.samples and len(t.samples) < self.fcfg.matching.min_samples):
            state = "checking"
        else:
            state = "unknown"
        return {
            "track_id": t.id, "box": [float(v) for v in t.box], "state": state,
            "name": t.name, "person_id": t.person_id, "confidence": round(t.confidence, 3),
            "best_score": None if t.best_score is None else round(t.best_score, 3),
            "votes": t.vote_streak, "approaching": t.approaching,
            "approach_rate": None if t.approach_rate is None else round(t.approach_rate, 2),
            "hour": d.hour, "pan": d.pan, "last_seen": t.last_seen, "age": round(now - t.first_seen, 1),
            "hint": HINTS.get(q.reasons[0]) if q is not None and q.reasons else None,
            "talk": round(talking.activity(t, now - 0.5, now), 3),
        }

    def view(self) -> List[Dict[str, Any]]:
        with self._view_lock:
            return list(self._view)

    def who_is_here(self, now: float, max_age_s: float = 1.5) -> List[Dict[str, Any]]:
        """Tracks seen in the last `max_age_s`, left to right."""
        v = [t for t in self.view() if now - t["last_seen"] <= max_age_s]
        v.sort(key=lambda t: (t["box"][0] + t["box"][2]) / 2)
        return v

    def pick_speaker(self, t0: float, t1: float, name: Optional[str] = None,
                     prefer_talking: bool = True) -> Optional[Dict[str, Any]]:
        """The face to enroll for speech heard between t0 and t1.

        1. A known face already called `name` -> returned (enrollment says "I already know Sarah").
        2. Known people are skipped: someone introducing a new name isn't someone we know.
        3. With `prefer_talking` (introductions), the face whose mouth moved clearly
           more than everyone else's during the speech wins.
        4. Otherwise the largest, most central face present for most of the window
           (also used for "save this person as X", where the *user* is speaking).

        Returns None if nobody is in view, or if several unknown faces are in view,
        nobody clearly talked, and enrollment.require_talking is on.
        """
        ecfg = self.cfg.enrollment
        with self._lock:
            cands = []
            for t in self.tracker.tracks:
                pts = [p for p in t.history if t0 <= p[0] <= t1]
                if not pts:
                    continue
                area = float(np.mean([p[3] for p in pts]))
                centrality = float(np.mean([1.0 - abs(p[2] - 0.5) for p in pts]))  # 0.5..1
                presence = len(pts) / max(1, len([p for p in t.history if p[0] >= t0]))
                cands.append({"track": t, "size": area * centrality * (0.5 + 0.5 * presence),
                              "talk": talking.activity(t, t0, t1), "cx": pts[-1][2]})
            if not cands:
                return None

            def result(c, reason, n_unknown):
                t = c["track"]
                return {"track_id": t.id, "person_id": t.person_id, "name": t.name, "confidence": t.confidence,
                        "hour": direction_for_x(c["cx"], self.cfg.direction).hour, "n_faces": len(cands),
                        "n_unknown": n_unknown, "reason": reason, "talk": round(c["talk"], 4)}

            if name:
                for c in cands:
                    if c["track"].person_id is not None and name_key(c["track"].name or "") == name_key(name):
                        return result(c, "known_same_name", 0)
            unknown = [c for c in cands if c["track"].person_id is None]
            if not unknown:  # everyone in view is already known
                return result(max(cands, key=lambda c: c["size"]), "all_known", 0)
            if len(unknown) == 1:
                return result(unknown[0], "only_unknown_face", 1)
            if prefer_talking:
                by_talk = sorted(unknown, key=lambda c: c["talk"], reverse=True)
                top, second = by_talk[0]["talk"], by_talk[1]["talk"]
                if top >= ecfg.talk_min and top >= ecfg.talk_ratio * max(second, 1e-6):
                    return result(by_talk[0], "talking", len(unknown))
                if ecfg.require_talking:
                    log.info("Several faces, nobody clearly talking (top %.3f, next %.3f): not guessing", top, second)
                    return None
            return result(max(unknown, key=lambda c: c["size"]), "largest", len(unknown))

    def track_alive(self, track_id: int) -> bool:
        with self._lock:
            return any(t.id == track_id for t in self.tracker.tracks)

    def track_samples(self, track_id: int) -> list:
        """The (lenient-gate) embeddings a live track holds: [(emb, quality, ts)]."""
        with self._lock:
            for t in self.tracker.tracks:
                if t.id == track_id:
                    return [(s.embedding, s.quality, s.timestamp) for s in t.samples]
        return []

    # ================================================================ changes from other modules
    def reload_gallery(self) -> None:
        with self._lock:
            self.gallery.load(self.storage)
            for t in self.tracker.tracks:
                if t.person_id is not None and not self.gallery.has(t.person_id):
                    t.reset_identity()
                elif t.person_id is not None:
                    t.name = self.gallery.name(t.person_id)

    def mark_announced(self, person_id: int, track_id: Optional[int], now: float) -> None:
        """Right after enrollment: don't immediately announce the person we just saved."""
        with self._lock:
            self._last_any[person_id] = now
            self._person_last_seen[person_id] = now
            for t in self.tracker.tracks:
                if t.id == track_id:
                    t.announced = True

    def forget_all_cooldowns(self) -> None:
        with self._lock:
            self._last_any.clear()
            self._last_approach.clear()
            self._person_last_seen.clear()

    def forget_cooldowns(self, person_id: int) -> None:
        with self._lock:
            self._last_any.pop(person_id, None)
            self._last_approach.pop(person_id, None)
            self._person_last_seen.pop(person_id, None)
