"""Named places: "this is the kitchen", then "...last seen in the kitchen".

A place is learned from a few seconds of whole-frame CLIP embeddings (the
user turns around slowly) plus the radio context at that moment: the Wi-Fi
access point and, if enabled, the Bluetooth devices nearby.

To classify a frame (when an item is sighted, or "where am I?"):

    score = best visual similarity to the place's views
          + wifi_bonus              if the laptop is on the same access point
          - other_network_penalty   if it's on none of the Wi-Fi networks the place was saved on
          + ble_bonus * share of the place's Bluetooth devices seen now
          + landmark_bonus * share of the landmarks in view now that were seen
                             while the place was learned (a bed -> the bedroom)

The best place wins if score >= t_place and it beats the runner-up by margin.
Saying "this is the kitchen" again adds views and the current network, so a
home with two network names (2.4 / 5 GHz) works after saving it on both.
Vision does most of the work in a home; Wi-Fi tells buildings (home vs. the
library) apart; Bluetooth helps tell similar-looking rooms apart.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np

from ..config import PlacesConfig


class PlaceMemory:
    def __init__(self, cfg: PlacesConfig, storage):
        self.cfg = cfg
        self.storage = storage
        self._lock = threading.Lock()
        self._places: List[Tuple[dict, np.ndarray]] = []
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._places = list(self.storage.place_gallery())

    def __len__(self) -> int:
        with self._lock:
            return len(self._places)

    def scores(self, scene: np.ndarray, radio=None, landmarks=None) -> List[Tuple[str, float]]:
        out = []
        with self._lock:
            places = list(self._places)
        for place, embs in places:
            s = float(np.max(embs @ scene))
            if radio is not None:
                ssids = {x for x in place["wifi_ssid"].split("|") if x}
                if radio.ssid and ssids:
                    if radio.ssid not in ssids:
                        s -= self.cfg.other_network_penalty
                    elif radio.bssid and radio.bssid in place["wifi_bssids"]:
                        s += self.cfg.wifi_bonus
                if radio.ble and place["ble"]:
                    seen = len(set(radio.ble) & set(place["ble"])) / max(1, len(place["ble"]))
                    s += self.cfg.ble_bonus * seen
            seen_marks = set(landmarks or ())
            if seen_marks and place.get("landmarks"):
                s += self.cfg.landmark_bonus * len(seen_marks & set(place["landmarks"])) / len(seen_marks)
            out.append((place["name"], s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def classify(self, scene: np.ndarray, radio=None, landmarks=None) -> Optional[str]:
        sc = self.scores(scene, radio, landmarks)
        if not sc:
            return None
        best, score = sc[0]
        second = sc[1][1] if len(sc) > 1 else -1.0
        if score >= self.cfg.t_place and score - second >= self.cfg.margin:
            return best
        return None
