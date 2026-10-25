"""Open-vocabulary object detection: find boxes for text queries ("wallet", "table").

OWLv2Detector (default) uses google/owlv2-base-patch16-ensemble through
transformers. Preprocessing is done here with OpenCV, matching
Owlv2ImageProcessor in transformers 5.x: scale to [0, 1], pad bottom/right to
a square with 0, resize to 960x960, normalise with the CLIP mean/std. Boxes
come back as (cx, cy, w, h) relative to the padded square, so they're scaled
by the square's side and clipped to the real frame. Doing this ourselves
skips the library's slow anti-aliasing resize and its API churn.

The detector runs on its own CUDA stream so a 100 ms OWLv2 pass doesn't stall
the face thread's kernels.

YoloWorldDetector is an optional, faster backend (pip install ultralytics).
TODO: untested here (no GPU in the build environment). Note that
ultralytics may download a CLIP text model on first use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


@dataclass
class ObjDet:
    box: np.ndarray   # (4,) float32 x1, y1, x2, y2 in frame pixels
    score: float
    label: str        # the text query that matched

    @property
    def area(self) -> float:
        return float(max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1]))


def owl_preprocess(image_bgr: np.ndarray, size: int = 960) -> Tuple[np.ndarray, int]:
    """-> (1, 3, size, size) float32 array, side of the padded square."""
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[:h, :w] = rgb
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    x = (resized.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return x.transpose(2, 0, 1)[None], side


def boxes_to_frame(pred_boxes: np.ndarray, side: int, w: int, h: int) -> np.ndarray:
    """(N, 4) relative (cx, cy, bw, bh) on the padded square -> frame x1y1x2y2."""
    cx, cy, bw, bh = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x1 = np.clip((cx - bw / 2) * side, 0, w)
    y1 = np.clip((cy - bh / 2) * side, 0, h)
    x2 = np.clip((cx + bw / 2) * side, 0, w)
    y2 = np.clip((cy + bh / 2) * side, 0, h)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.3) -> List[int]:
    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ious = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[ious < iou_thr]
    return keep


class OWLv2Detector:
    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True, threshold: float = 0.12,
                 max_dets: int = 40):
        import torch
        from transformers import AutoTokenizer, Owlv2ForObjectDetection

        self.torch = torch
        self.device = torch.device(device)
        self.half = bool(fp16 and self.device.type == "cuda")
        self.dtype = torch.float16 if self.half else torch.float32
        log.info("Loading OWLv2 %s on %s%s", model_name, self.device, " fp16" if self.half else "")
        model = Owlv2ForObjectDetection.from_pretrained(model_name).eval().to(self.device)
        self.model = model.half() if self.half else model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.size = int(getattr(model.config.vision_config, "image_size", 960))
        self.threshold = threshold
        self.max_dets = max_dets
        self._tok_cache: Optional[Tuple[Tuple[str, ...], object, object]] = None
        self.stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def _tokens(self, queries: Sequence[str]):
        key = tuple(queries)
        if self._tok_cache is None or self._tok_cache[0] != key:
            tok = self.tokenizer(list(key), padding="max_length", max_length=16, truncation=True,
                                 return_tensors="pt")
            self._tok_cache = (key, tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device))
        return self._tok_cache[1], self._tok_cache[2]

    def detect(self, image_bgr: np.ndarray, queries: Sequence[str]) -> List[ObjDet]:
        if not queries:
            return []
        torch = self.torch
        h, w = image_bgr.shape[:2]
        x, side = owl_preprocess(image_bgr, self.size)
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else _nullctx()
        with ctx, torch.inference_mode():
            ids, mask = self._tokens(queries)
            pv = torch.from_numpy(x).to(self.device, dtype=self.dtype)
            out = self.model(input_ids=ids, pixel_values=pv, attention_mask=mask)
            logits = out.logits[0].float()               # (patches, queries)
            pred = out.pred_boxes[0].float()             # (patches, 4)
            probs = torch.sigmoid(logits)
            scores, labels = probs.max(dim=-1)
            keep = scores > self.threshold
            scores, labels, pred = scores[keep].cpu().numpy(), labels[keep].cpu().numpy(), pred[keep].cpu().numpy()
        if len(scores) == 0:
            return []
        boxes = boxes_to_frame(pred, side, w, h)
        valid = (boxes[:, 2] - boxes[:, 0] > 4) & (boxes[:, 3] - boxes[:, 1] > 4)
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        out_dets: List[ObjDet] = []
        for lab in np.unique(labels):  # NMS per query
            idx = np.where(labels == lab)[0]
            for k in nms(boxes[idx], scores[idx], 0.3):
                i = idx[k]
                out_dets.append(ObjDet(boxes[i], float(scores[i]), queries[int(lab)]))
        out_dets.sort(key=lambda d: d.score, reverse=True)
        return out_dets[: self.max_dets]


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class YoloWorldDetector:
    """Optional backend. TODO: untested in this build (no GPU here)."""

    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = True, threshold: float = 0.12):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(model_path)
        self.device = device
        self.half = fp16 and device == "cuda"
        self.threshold = threshold
        self._classes: Tuple[str, ...] = ()

    def detect(self, image_bgr: np.ndarray, queries: Sequence[str]) -> List[ObjDet]:
        if tuple(queries) != self._classes:
            self.model.set_classes(list(queries))
            self._classes = tuple(queries)
        res = self.model.predict(image_bgr, conf=self.threshold, verbose=False, device=self.device, half=self.half)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [ObjDet(b, float(c), self._classes[k]) for b, c, k in zip(boxes, conf, cls)]


class NullObjectDetector:
    def detect(self, image_bgr, queries):
        return []
