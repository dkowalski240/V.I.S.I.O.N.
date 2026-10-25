"""Image embeddings for item crops (open_clip, default ViT-B-32 laion2b).

Crops are padded to a square (so a long object like a key isn't cut off by
the centre crop) and embedded with the image tower only; the text tower is
dropped to save VRAM. Output: unit vectors, so dot product = cosine.

Same-object-different-view similarities are typically ~0.85-0.95; another
object of the same kind (someone else's black wallet) can also score high,
which is why the threshold, the margin over other items and the
multi-scan agreement exist. (DINOv2 is better at "this exact object" if you
ever need it; it's not in open_clip.)
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


def crop_with_margin(image: np.ndarray, box, margin: float = 0.10) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def pad_square(crop: np.ndarray, value: int = 114) -> np.ndarray:
    h, w = crop.shape[:2]
    side = max(h, w)
    out = np.full((side, side, 3), value, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = crop
    return out


class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                 device: str = "cuda", fp16: bool = True):
        import open_clip
        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.device = torch.device(device)
        half = bool(fp16 and self.device.type == "cuda")
        log.info("Loading open_clip %s (%s) on %s%s", model_name, pretrained, self.device, " fp16" if half else "")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device, precision="fp16" if half else "fp32"
        )
        model.eval()
        for attr in ("transformer", "token_embedding", "ln_final"):  # text tower: not needed
            if hasattr(model, attr):
                try:
                    setattr(model, attr, None)
                except Exception:
                    pass
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model = model
        self.preprocess = preprocess
        self.dtype = torch.float16 if half else torch.float32
        self.dim = int(getattr(model.visual, "output_dim", 512))

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self.torch
        ims = [self.preprocess(self.Image.fromarray(cv2.cvtColor(pad_square(c), cv2.COLOR_BGR2RGB)))
               for c in crops_bgr]
        x = torch.stack(ims).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            f = self.model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return f.cpu().numpy().astype(np.float32)


class NullImageEmbedder:
    dim = 512

    def embed(self, crops):
        return np.zeros((len(crops), 512), dtype=np.float32)
