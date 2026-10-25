"""Optional cloud description of where an item is (OFF by default).

Only runs when `cloud.vlm_enabled: true` AND the user explicitly asks
("where's my wallet in detail?", "describe it"). It sends that one sighting
snapshot (which may show other people) to Claude via the Anthropic API and
speaks the one-sentence answer. Needs `pip install anthropic` and the
ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import cv2

from ..config import CloudConfig

log = logging.getLogger(__name__)

PROMPT = (
    "This photo is from a blind user's wearable camera. Their {item} is inside the green box. "
    "In one short sentence (under 20 words), tell them where it is relative to nearby furniture "
    "and objects, so they can reach for it. No preamble."
)


class CloudVLM:
    def __init__(self, cfg: CloudConfig):
        self.cfg = cfg

    def _image_b64(self, path: str) -> Optional[str]:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = self.cfg.image_max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.standard_b64encode(buf.tobytes()).decode("ascii") if ok else None

    def describe(self, item_name: str, sighting) -> Optional[str]:
        if not self.cfg.vlm_enabled:
            return None
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            return f"Set {self.cfg.api_key_env} to use descriptions."
        if not sighting.snapshot_path or not os.path.exists(sighting.snapshot_path):
            return "No photo of it saved."
        data = self._image_b64(sighting.snapshot_path)
        if data is None:
            return "Couldn't read the photo."
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=key, timeout=self.cfg.timeout_s)
            msg = client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                        {"type": "text", "text": PROMPT.format(item=item_name)},
                    ],
                }],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()
            return text or None
        except Exception as e:
            log.warning("Cloud description failed: %s", e)
            return "Description failed."
