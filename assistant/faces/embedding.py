"""[Adapted from FaceLock: adds fp16 on CUDA.] Face embedding: aligned
160x160 crop -> 512-d unit vector (FaceNet / InceptionResnetV1, VGGFace2 weights).

Because every vector has length 1, the dot product of two embeddings is their
cosine similarity (~0.7+ same person, ~0.2 different people; calibrate with
scripts/calibrate_faces.py).

fp16 halves memory and is faster on the RTX GPU. Enroll and recognise with
the same setting; mixing fp16 and fp32 galleries works but costs a little
accuracy.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, Sequence

import cv2
import numpy as np

from ..config import EmbeddingConfig

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


class Embedder(Protocol):
    dim: int

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray: ...


class FaceNetEmbedder:
    dim = EMBEDDING_DIM

    def __init__(self, cfg: EmbeddingConfig):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._torch = torch
        self.device = torch.device(cfg.device)
        self.half = bool(cfg.fp16 and self.device.type == "cuda")
        log.info("Loading FaceNet (%s) on %s%s", cfg.pretrained, self.device, " fp16" if self.half else "")
        model = InceptionResnetV1(pretrained=cfg.pretrained).eval().to(self.device)
        self.model = model.half() if self.half else model

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of crops in one forward pass. Returns (n, 512) float32."""
        if len(crops_bgr) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        batch: List[np.ndarray] = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            # facenet-pytorch "fixed_image_standardization": 0..255 -> about -1..1
            batch.append(((rgb - 127.5) / 128.0).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            emb = self.model(x).float().cpu().numpy()
        return l2_normalize(emb)
