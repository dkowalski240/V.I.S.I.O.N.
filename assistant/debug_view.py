"""OpenCV debug window: boxes, names, confidence, track ids, item boxes, status.

Colours: green = known (solid when sure), yellow = checking, grey = unknown
(never stored), cyan = enrolled item, magenta = the face being enrolled.
An arrow above a face means it is approaching. Also used for the demo video
(`debug.record_path` writes exactly what this window shows).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, GREY, CYAN, MAGENTA, WHITE, RED = (
    (60, 200, 60), (0, 210, 255), (150, 150, 150), (230, 200, 0), (220, 60, 220), (255, 255, 255), (40, 40, 230)
)


def _label(img, text, x, y, color, scale=0.55):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, color, 1, cv2.LINE_AA)


def draw(frame: np.ndarray, faces: List[Dict], objects: List[Dict], status: List[str],
         enrolling_track: Optional[int] = None, show_unknown: bool = True, t_sure: float = 0.75,
         talk_min: float = 0.015) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f["box"]]
        if f["track_id"] == enrolling_track:
            color, text = MAGENTA, f"#{f['track_id']} enrolling"
        elif f["state"] == "known":
            color = GREEN
            sure = f["confidence"] >= t_sure
            text = f"#{f['track_id']} {f['name']} {f['confidence']:.2f}{'' if sure else ' ?'} {f['hour']}h"
        elif f["state"] == "checking":
            color = YELLOW
            score = f.get("best_score")
            text = f"#{f['track_id']} checking" + (f" {score:.2f}" if score is not None else "") + f" v{f['votes']}"
        else:
            if not show_unknown:
                continue
            color, text = GREY, f"#{f['track_id']} unknown"
        thick = 3 if f["state"] == "known" and f["confidence"] >= t_sure else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        _label(img, text, x1, y1 - 4, color)
        rate = f.get("approach_rate")
        if f.get("hint"):
            _label(img, f["hint"], x1, y2 + 18, GREY, 0.45)
        if rate is not None:
            _label(img, f"{rate:+.2f}/s", x2 - 70, y2 + 18, RED if f["approaching"] else GREY, 0.45)
        if f.get("talk"):
            _label(img, f"talk {f['talk']:.3f}", x1, y2 + 38, YELLOW if f["talk"] >= talk_min else GREY, 0.45)
        if f.get("approaching"):
            cx = (x1 + x2) // 2
            cv2.arrowedLine(img, (cx, max(0, y1 - 40)), (cx, max(0, y1 - 12)), RED, 3, tipLength=0.5)

    for o in objects:
        x1, y1, x2, y2 = [int(v) for v in o["box"]]
        if o.get("item"):
            cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 2)
            _label(img, f"{o['item']} {o['score']:.2f}", x1, y1 - 4, CYAN)
        elif o.get("near"):  # close to a saved item, but under objects.t_item
            cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 1)
            _label(img, f"{o['near']}? {o['score']:.2f}", x1, y1 - 4, YELLOW, 0.45)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
            text = o.get("label", "")
            if o.get("score") is not None and "near" in o:  # an item candidate: show its best similarity
                text += f" {o['score']:.2f}"
            cv2.putText(img, text, (x1 + 2, y1 + 14), FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # clock-direction ruler along the top
    for i in range(1, 12):
        x = int(w * i / 12)
        cv2.line(img, (x, 0), (x, 6), (80, 80, 80), 1)

    # status panel
    if status:
        line_h = 20
        panel_h = line_h * len(status) + 10
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, line in enumerate(status):
            cv2.putText(img, line[:140], (8, h - panel_h + 18 + i * line_h), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return img


def placeholder(text: str, size=(1280, 720)) -> np.ndarray:
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(img, text, (30, size[1] // 2), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
    return img
