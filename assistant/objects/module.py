"""Objects module: remember personal items and where they were last seen.

Enrollment ("remember this, it's my wallet"):
    for `enroll_s` seconds at `enroll_fps`, find the held-up item in each frame
    (detector query = the item's name + generic small-object words; prefer a
    box labelled with the name, near the centre, not on a face; fall back to
    the centre of the frame), embed the crop with open_clip, keep diverse views,
    save them. "Saved your wallet."

Tracking (background, `scan_fps`, 1-2 per second):
    detect candidates (enrolled names + generic object words) and context
    objects (table, laptop, ...) in one OWLv2 pass -> embed up to
    `max_candidates` candidate crops -> match against every enrolled item
    (threshold + margin over the next item) -> an item seen in `agree_scans`
    of the last `agree_window` scans gets a sighting: time, box, snapshot,
    context phrase ("on the table, next to the laptop"), clock direction.
    If it's still in the same place, the latest sighting is refreshed
    instead of adding a new row every second.

Extra context stored with each sighting (see places.py and radio.py):
    place       a room the user named ("this is my bedroom"), recognised from
                the whole frame plus Wi-Fi / Bluetooth
    near        a landmark (bed, couch, door...) from the same frame or, when the
                item's own frame shows none (a wallet on the floor, camera looking
                down), from the scans `context_window_s` before it was first seen
                and after it was last seen. The room is filled in the same way.
                -> "on the floor of your bedroom, near the bed"
    companions  known people in view at that moment ("You were with Sarah.")
    wifi_ssid   the network the laptop was on (tells home from the library)

Place enrollment ("this is my bedroom"): the user turns slowly for
`places.enroll_s` (12 s; "done" ends it early). Whole-frame embeddings spread
over the turn, the landmarks seen (a bed in 2+ frames, ...) and the current
radio snapshot are saved: "Saved your bedroom. I saw a bed, a desk and a door."
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import events as ev
from ..bus import Bus
from ..clock import Clock
from ..config import Config
from ..geometry import box_center_x_norm, direction_for_x, iou
from ..storage import item_key
from .context import describe
from .detector import NullObjectDetector, ObjDet
from .embedder import NullImageEmbedder, crop_with_margin
from .places import PlaceMemory
from ..radio import StaticRadio

log = logging.getLogger(__name__)


class ObjectsModule:
    def __init__(self, cfg: Config, camera, bus: Bus, storage, clock: Optional[Clock] = None,
                 detector=None, embedder=None, faces=None, radio=None):
        self.cfg = cfg
        self.ocfg = cfg.objects
        self.camera = camera
        self.bus = bus
        self.storage = storage
        self.clock = clock or Clock()
        self.detector = detector or NullObjectDetector()
        self.embedder = embedder or NullImageEmbedder()
        self.faces = faces
        self.radio = radio or StaticRadio()
        self.places = PlaceMemory(cfg.places, storage)
        self._lock = threading.RLock()
        self._model_lock = threading.Lock()  # the router thread also embeds ("where am I?")
        self._gallery: Dict[int, Tuple[str, np.ndarray]] = {}
        self._hits: Dict[int, Deque[bool]] = {}
        self._current: Dict[int, Tuple[float, List[float], int, float]] = {}  # id -> (ts, box, hour, score)
        self._last: Dict[int, Tuple[int, float, List[float]]] = {}  # id -> (sighting_id, snapshot_ts, box)
        self._enroll: Optional[dict] = None
        self._view: List[dict] = []
        self._recent: Deque[dict] = deque(maxlen=64)   # per scan: {"ts", "place", "landmarks"}
        self._follow: Dict[int, dict] = {}              # item id -> context still being filled in
        self.room = ""                                  # the saved place the last scan recognised
        self.stats = {"ms": 0.0, "scans": 0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reload_gallery()

    # ================================================================ gallery
    def reload_gallery(self) -> None:
        with self._lock:
            self._gallery = {iid: (name, emb) for iid, name, emb in self.storage.item_gallery()}
            for iid in list(self._current):
                if iid not in self._gallery:
                    self._current.pop(iid, None)
                    self._last.pop(iid, None)
                    self._hits.pop(iid, None)

    def reload_places(self) -> None:
        self.places.reload()

    def _embed(self, crops) -> np.ndarray:
        with self._model_lock:
            return self.embedder.embed(crops)

    def _detect(self, image, queries) -> List[ObjDet]:
        with self._model_lock:
            return self.detector.detect(image, queries)

    def item_names(self) -> List[str]:
        with self._lock:
            return [n for n, _ in self._gallery.values()]

    def queries(self, include_context: bool = True) -> List[str]:
        seen, out = set(), []
        for q in self.item_names() + list(self.ocfg.generic_queries) + (
                list(self.ocfg.context_queries) + list(self.ocfg.landmark_labels) if include_context else []):
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out

    # ================================================================ thread
    def start(self) -> "ObjectsModule":
        self._thread = threading.Thread(target=self._run, name="objects", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        last_index = 0
        while not self._stop.is_set():
            period = 1.0 / (self.ocfg.enroll_fps if self._enroll else max(self.ocfg.scan_fps, 0.1))
            t0 = time.monotonic()
            frame = self.camera.wait_for(last_index, timeout=0.5)
            if frame is not None:
                last_index = frame.index
                try:
                    self.process_frame(frame.image, frame.timestamp)
                except Exception:
                    log.exception("Objects: error while processing a frame")
                self.stats["ms"] = (time.monotonic() - t0) * 1000
            # Sleep in small steps so "remember this" starts promptly.
            deadline = t0 + period
            while not self._stop.is_set() and time.monotonic() < deadline:
                if self._enroll is not None and period > 1.0 / self.ocfg.enroll_fps:
                    break  # enrollment just started: switch to the faster rate now
                time.sleep(0.05)

    def process_frame(self, image: np.ndarray, now: float) -> None:
        e = self._enroll
        if e is not None and e.get("kind") == "place":
            self._place_step(image, now)
        elif e is not None:
            self._enroll_step(image, now)
        else:
            self._scan(image, now)

    # ================================================================ enrollment
    def start_enrollment(self, name: str, now: Optional[float] = None) -> bool:
        now = self.clock.now() if now is None else now
        with self._lock:
            if self._enroll is not None:
                self._say(f"Still learning {self._busy_name()}.")
                return False
            start = now + self.ocfg.enroll_delay_s  # let the prompt finish first
            self._enroll = {"kind": "item", "name": name, "start_at": start, "until": start + self.ocfg.enroll_s,
                            "views": [], "started": now}
        self._say("Hold it up and turn it slowly.", tone=ev.TONE_CAPTURE, kind=ev.KIND_PROMPT)
        log.info("Learning item %r for %.1fs", name, self.ocfg.enroll_s)
        return True

    def cancel_enrollment(self) -> bool:
        with self._lock:
            if self._enroll is None:
                return False
            self._enroll = None
            return True

    def finish_enrollment_early(self, now: Optional[float] = None) -> bool:
        """"Done": end an item or room capture now (the next frame saves what it has)."""
        now = self.clock.now() if now is None else now
        with self._lock:
            if self._enroll is None:
                return False
            self._enroll["until"] = min(self._enroll["until"], max(now, self._enroll["start_at"]))
            return True

    def enrolling(self) -> Optional[str]:
        e = self._enroll
        if not e:
            return None
        return f"place: {e['name']}" if e.get("kind") == "place" else e["name"]

    def _busy_name(self) -> str:
        e = self._enroll
        return f"the {e['name']}" if e.get("kind") == "place" else f"your {e['name']}"

    # ================================================================ places
    def start_place_enrollment(self, name: str, now: Optional[float] = None, mine: Optional[bool] = None) -> bool:
        now = self.clock.now() if now is None else now
        pc = self.cfg.places
        with self._lock:
            if self._enroll is not None:
                self._say(f"Still learning {self._busy_name()}.")
                return False
            start = now + pc.enroll_delay_s
            self._enroll = {"kind": "place", "name": name, "start_at": start, "until": start + pc.enroll_s,
                            "views": [], "started": now, "mine": mine, "marks": {}, "nudged": False}
        self._say("Turn slowly all the way around. Say done to finish.", tone=ev.TONE_CAPTURE, kind=ev.KIND_PROMPT)
        log.info("Learning place %r", name)
        return True

    def _place_step(self, image: np.ndarray, now: float) -> None:
        e = self._enroll
        if e is None or now < e["start_at"]:
            return
        e["views"].append(self._embed([image])[0])
        # Landmarks in view during the turn: frames they appear in.
        for d in self._detect(image, list(self.ocfg.landmark_labels)):
            if d.score >= self.ocfg.landmark_min_score:
                e["marks"].setdefault(d.label.lower(), set()).add(len(e["views"]))
        half = e["start_at"] + (e["until"] - e["start_at"]) / 2
        if not e["nudged"] and now >= half and e["until"] - e["start_at"] >= 6.0:
            e["nudged"] = True
            self._say("Keep turning.", kind=ev.KIND_PROMPT)
        if now >= e["until"]:
            with self._lock:
                e, self._enroll = self._enroll, None
            if e is not None:
                self._finish_place(e, now)

    def _finish_place(self, e: dict, now: float) -> None:
        chosen: List[np.ndarray] = []
        for v in e["views"]:
            if float(np.linalg.norm(v)) > 0.5 and all(float(c @ v) < 0.97 for c in chosen):
                chosen.append(v)
        mx = self.cfg.places.max_views
        if len(chosen) > mx:  # spread over the whole turn, not just its start
            keep = sorted({int(round(i)) for i in np.linspace(0, len(chosen) - 1, mx)})
            chosen = [chosen[i] for i in keep]
        if not chosen:
            self._say(f"Couldn't learn {self.storage.place_label(e['name'])}. Try again.", tone=ev.TONE_ERROR)
            return
        # A landmark must be seen in 2+ frames (one frame can be a false detection).
        marks = {k: len(v) for k, v in e["marks"].items() if len(v) >= 2}
        r = self.radio.latest()
        ble = {a: rssi for a, rssi in r.ble.items()} if self.cfg.places.bluetooth else {}
        self.storage.add_place(e["name"], chosen, wifi_ssid=r.ssid, wifi_bssids=[r.bssid] if r.bssid else [],
                               ble=ble, now=now, mine=e.get("mine"), landmarks=marks)
        self.places.reload()
        text = f"Saved {self.storage.place_label(e['name'])}."
        if marks:
            seen = [f"{'an' if k[0] in 'aeiou' else 'a'} {k}" for k in sorted(marks, key=marks.get, reverse=True)]
            text += " I saw " + (seen[0] if len(seen) == 1 else ", ".join(seen[:-1]) + " and " + seen[-1]) + "."
        self._say(text, tone=ev.TONE_DONE)
        log.info("Saved place %r (%d views from %d frames, landmarks %s, wifi %r, %d Bluetooth devices)",
                 e["name"], len(chosen), len(e["views"]), marks, r.ssid, len(ble))

    def current_place(self) -> Optional[str]:
        """Which saved place does the camera see right now? ("where am I?")"""
        frame = self.camera.latest()
        if frame is None or not self.cfg.places.enabled or len(self.places) == 0:
            return None
        recent = self._recent[-1] if self._recent else None
        marks = [k for k, _ in recent["landmarks"]] if recent and frame.timestamp - recent["ts"] < 2.0 else None
        return self.places.classify(self._embed([frame.image])[0], self.radio.latest(), marks)

    def _place_for(self, image: np.ndarray, landmarks=None) -> str:
        if not self.cfg.places.enabled or len(self.places) == 0:
            return ""
        return self.places.classify(self._embed([image])[0], self.radio.latest(), landmarks) or ""

    def _companions(self, now: float) -> str:
        if self.faces is None or not self.cfg.places.mention_people:
            return ""
        names = []
        for t in self.faces.who_is_here(now):
            if t.get("name") and t["name"] not in names:
                names.append(t["name"])
        return ", ".join(names)

    def _face_boxes(self) -> List[List[float]]:
        if self.faces is None:
            return []
        return [f["box"] for f in self.faces.view()]

    def _pick_held_item(self, dets: List[ObjDet], name: str, w: int, h: int) -> Tuple[List[float], float, str]:
        """-> (box, score, kind) with kind = named | generic | fallback."""
        frame_area = float(w * h)
        faces = self._face_boxes()
        best, best_score = None, 0.0
        for d in dets:
            frac = d.area / frame_area
            if not (0.002 <= frac <= 0.5):
                continue
            if any(iou(d.box, fb) > 0.3 for fb in faces):
                continue
            cx = (d.box[0] + d.box[2]) / 2 / w
            cy = (d.box[1] + d.box[3]) / 2 / h
            centrality = 1.0 - min(1.0, abs(cx - 0.5) + abs(cy - 0.5))
            s = d.score * (1.5 if item_key(d.label) == item_key(name) else 1.0) * (0.5 + centrality)
            if s > best_score:
                best, best_score = d, s
        if best is not None:
            kind = "named" if item_key(best.label) == item_key(name) else "generic"
            return [float(v) for v in best.box], best.score, kind
        # Fallback: the middle of the frame, where a held-up item usually is.
        return [w * 0.3, h * 0.3, w * 0.7, h * 0.7], 0.0, "fallback"

    def _enroll_step(self, image: np.ndarray, now: float) -> None:
        e = self._enroll
        if e is None:
            return
        if now < e["start_at"]:
            return  # still saying "Hold it up..."
        h, w = image.shape[:2]
        queries = [e["name"]] + [q for q in self.ocfg.generic_queries if item_key(q) != item_key(e["name"])]
        dets = self._detect(image, queries)
        box, score, kind = self._pick_held_item(dets, e["name"], w, h)
        emb = self._embed([crop_with_margin(image, box)])[0]
        e["views"].append((emb, score, kind))
        with self._lock:
            self._view = [{"box": box, "item": f"learning {e['name']}", "score": score}]
        if now >= e["until"]:
            self._finish_enrollment(now)

    def _finish_enrollment(self, now: float) -> None:
        with self._lock:
            e, self._enroll = self._enroll, None
        if e is None:
            return
        # Prefer views where the detector really found the named item; use
        # other detections, then centre-of-frame crops, only if needed.
        views = e["views"]
        pool = [v for v in views if v[2] == "named"]
        if len(pool) < 2:
            pool += [v for v in views if v[2] == "generic"]
        if len(pool) < 2:
            pool += [v for v in views if v[2] == "fallback"]
        pool = [v for v in pool if float(np.linalg.norm(v[0])) > 0.5]  # --stub embedder returns zeros
        chosen: List[np.ndarray] = []
        for emb, _, _ in sorted(pool, key=lambda v: v[1], reverse=True):
            if len(chosen) >= self.ocfg.enroll_max_exemplars:
                break
            if all(float(c @ emb) < 0.95 for c in chosen):
                chosen.append(emb)
        if len(chosen) < 2:
            self._say(f"Couldn't see your {e['name']}. Try again.", tone=ev.TONE_ERROR)
            return
        iid = self.storage.add_item(e["name"], chosen, now=now)
        self.reload_gallery()
        self.bus.publish(ev.ITEM_SAVED, item_id=iid, name=e["name"], n_views=len(chosen))
        self._say(f"Saved your {e['name']}.", tone=ev.TONE_DONE)
        kinds = [v[2] for v in views]
        log.info("Saved item %r (%d views kept; of %d frames the detector found it as %r in %d, as another "
                 "object in %d, and %d were centre-of-frame guesses)", e["name"], len(chosen), len(views),
                 e["name"], kinds.count("named"), kinds.count("generic"), kinds.count("fallback"))
        if kinds.count("named") + kinds.count("generic") < 2:
            log.warning("The detector barely saw the %s while learning it: hold it closer, against a plain "
                        "background, in good light, and try again", e["name"])

    # ================================================================ scanning
    def _scan(self, image: np.ndarray, now: float) -> None:
        h, w = image.shape[:2]
        with self._lock:
            gallery = dict(self._gallery)
        context_set = ({q.lower() for q in self.ocfg.context_queries} | {q.lower() for q in self.ocfg.landmark_labels}) \
            - {n.lower() for n, _ in gallery.values()}
        dets = self._detect(image, self.queries())
        ctx = [d for d in dets if d.label.lower() in context_set]
        info = self._scan_info(image, ctx, now)
        frame_area = float(w * h)
        cands = [d for d in dets if d.label.lower() not in context_set and 0.0005 <= d.area / frame_area <= 0.35]
        # One object often gets several boxes ("wallet", "phone", "card" on the same spot);
        # keep the best-scoring one so duplicates don't push a faint wallet out of the top N.
        unique: List[ObjDet] = []
        for d in cands:  # already sorted by score
            if all(iou(d.box, u.box) <= 0.6 for u in unique):
                unique.append(d)
        cands = unique[: self.ocfg.max_candidates]
        view = [{"box": [float(v) for v in d.box], "label": d.label, "item": None, "score": d.score} for d in ctx]
        self.stats["scans"] += 1

        best: Dict[int, Tuple[float, ObjDet]] = {}
        if gallery and cands:
            embs = self._embed([crop_with_margin(image, d.box) for d in cands])
            ids = list(gallery)
            for d, e in zip(cands, embs):
                sims = [float(np.max(gallery[i][1] @ e)) for i in ids]
                order = np.argsort(sims)[::-1]
                top = sims[order[0]]
                second = sims[order[1]] if len(order) > 1 else -1.0
                iid = ids[order[0]]
                if top >= self.ocfg.t_item and top - second >= self.ocfg.item_margin:
                    if iid not in best or top > best[iid][0]:
                        best[iid] = (top, d)
                matched = top >= self.ocfg.t_item
                view.append({"box": [float(v) for v in d.box], "label": d.label,
                             "item": gallery[iid][0] if matched else None, "score": top,
                             # close but under t_item: the debug window shows "wallet? 0.78"
                             "near": gallery[iid][0] if not matched and top >= self.ocfg.t_item - 0.10 else None})

        for iid, (name, _) in gallery.items():
            hits = self._hits.setdefault(iid, deque(maxlen=self.ocfg.agree_window))
            hits.append(iid in best)
            if iid in best and sum(hits) >= self.ocfg.agree_scans:
                sim, d = best[iid]
                self._log_sighting(iid, name, d, sim, image, ctx, now, info)
        self._follow_up(info)
        with self._lock:
            self._view = view

    def _log_sighting(self, iid: int, name: str, det: ObjDet, sim: float, image: np.ndarray,
                      ctx: List[ObjDet], now: float, info: Optional[dict] = None) -> None:
        h, w = image.shape[:2]
        info = info or {"ts": now, "place": "", "landmarks": []}
        box = [float(v) for v in det.box]
        hour = direction_for_x(box_center_x_norm(box, w), self.cfg.direction).hour
        context = describe(box, ctx, self.ocfg.surface_labels, w, h)
        with self._lock:
            self._current[iid] = (now, box, hour, sim)
        companions = self._companions(now)  # cheap: who's in view right now
        last = self._last.get(iid)
        same_place = last is not None and iou(last[2], box) > 0.3
        if same_place and now - last[1] < self.ocfg.sighting_interval_s:
            self.storage.refresh_sighting(last[0], now, box, sim, context or None, companions=companions)
            self._track_context(iid, last[0], False, context, info, now)
            return
        snapshot = self._save_snapshot(image, box, name, now)
        if same_place:
            self.storage.refresh_sighting(last[0], now, box, sim, context or None, snapshot_path=snapshot,
                                          companions=companions)
            sid = last[0]
        else:
            sid = self.storage.add_sighting(iid, now, box, (w, h), sim, snapshot, context,
                                            getattr(self.camera, "camera_id", ""), keep=self.ocfg.keep_sightings,
                                            place=info["place"], companions=companions,
                                            wifi_ssid=self.radio.latest().ssid)
        self._last[iid] = (sid, now, box)
        self._track_context(iid, sid, not same_place, context, info, now)
        f = self._follow.get(iid, {})
        self.bus.publish(ev.OBJECT_SIGHTING, ts=now, item=name, item_id=iid, timestamp=now, bbox=box,
                         snapshot_path=snapshot, context=context, clock_direction=hour, score=round(sim, 3),
                         place=(f.get("place") or ("",))[0], near=(f.get("near") or ("",))[0],
                         companions=companions, camera_id=getattr(self.camera, "camera_id", ""))

    # ------------------------------------------------ room + landmark from nearby frames
    def _scan_info(self, image: np.ndarray, ctx: List[ObjDet], now: float) -> dict:
        """What this scan says about where we are: the room (if any are saved) and
        the landmarks in view. Kept for context_window_s so a sighting can use the
        frames just before it, and offered to recent sightings for the frames after."""
        marks = {m.lower() for m in self.ocfg.landmark_labels}
        best: Dict[str, float] = {}
        for d in ctx:
            lab = d.label.lower()
            if lab in marks and d.score >= self.ocfg.landmark_min_score and d.score > best.get(lab, 0.0):
                best[lab] = d.score
        landmarks = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        info = {"ts": now, "place": self._place_for(image, [k for k, _ in landmarks]), "landmarks": landmarks}
        self._recent.append(info)
        self.room = info["place"]  # shown in the debug window
        while self._recent and self._recent[0]["ts"] < now - 2 * self.ocfg.context_window_s - 1.0:
            self._recent.popleft()
        return info

    def _track_context(self, iid: int, sid: int, new: bool, context: str, info: dict, now: float) -> None:
        f = self._follow.get(iid)
        surface = context[len("on the "):].split(",")[0].strip() if context.startswith("on the ") else ""
        if new or f is None or f["sid"] != sid:
            f = {"sid": sid, "last": now, "surface": surface, "near": None, "place": None}
            self._follow[iid] = f
            for past in list(self._recent):  # the frames just before it was first seen
                if past is not info and now - self.ocfg.context_window_s <= past["ts"] < now:
                    self._consider(f, past, now - past["ts"])
        else:
            f["last"] = now
            f["surface"] = surface or f["surface"]
        self._consider(f, info, 0.0)

    def _follow_up(self, info: dict) -> None:
        """Offer this scan to sightings whose item was seen in the last context_window_s."""
        for iid, f in list(self._follow.items()):
            if f["last"] == info["ts"]:
                continue  # seen in this very scan: _log_sighting already used it
            dt = info["ts"] - f["last"]
            if dt > self.ocfg.context_window_s:
                del self._follow[iid]
                continue
            self._consider(f, info, dt)

    def _consider(self, f: dict, info: dict, dt: float) -> None:
        """Keep the landmark and room seen closest in time to the item (same frame first)."""
        near_changed = place_changed = False
        for label, score in info["landmarks"]:
            if label == f["surface"]:
                continue  # "on the bed, near the bed"
            cur = f["near"]
            if cur is None or (dt, -score) < (cur[1], -cur[2]):
                f["near"], near_changed = (label, dt, score), True
            break  # landmarks are sorted best first
        if info["place"] and (f["place"] is None or dt < f["place"][1]):
            place_changed = f["place"] is None or f["place"][0] != info["place"]
            f["place"] = (info["place"], dt)
        if near_changed or place_changed:
            self.storage.set_sighting_extras(f["sid"], near=f["near"][0] if near_changed else None,
                                             place=f["place"][0] if place_changed else None)
            if dt > 0:
                log.info("Sighting %d: filled in from a scan %.1fs away: room %r, near %r", f["sid"], dt,
                         (f["place"] or ("",))[0], (f["near"] or ("",))[0])

    def _save_snapshot(self, image: np.ndarray, box: List[float], name: str, now: float) -> Optional[str]:
        try:
            h, w = image.shape[:2]
            scale = min(1.0, self.ocfg.snapshot_width / float(w))
            img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            os.makedirs(self.storage.snapshot_dir, exist_ok=True)
            path = os.path.join(str(self.storage.snapshot_dir), f"{item_key(name)}_{int(now * 1000)}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return path
        except Exception:
            log.exception("Could not save snapshot")
            return None

    # ================================================================ queries
    def in_view(self, item_id: int, now: float) -> Optional[int]:
        """Clock hour if the item was seen in the last `in_view_s` seconds."""
        with self._lock:
            cur = self._current.get(item_id)
        if cur is not None and now - cur[0] <= self.ocfg.in_view_s:
            return cur[2]
        return None

    def view(self) -> List[dict]:
        with self._lock:
            return list(self._view)

    def _say(self, text: str, tone: Optional[str] = None, kind: str = ev.KIND_ANSWER) -> None:
        self.bus.publish(ev.SAY, text=text, priority=ev.PRI_ANSWER, kind=kind, tone=tone)
