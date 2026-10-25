"""Download every model before the hackathon Wi-Fi gets crowded (~1.9 GB total).

    python scripts/download_models.py                 everything
    python scripts/download_models.py --only piper    one of: facenet owlv2 clip whisper piper yunet

Models land in the usual caches (Hugging Face: %USERPROFILE%\\.cache\\huggingface,
PyTorch hub: %USERPROFILE%\\.cache\\torch) except the Piper voice and YuNet,
which go into models/.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def facenet(cfg):
    from facenet_pytorch import MTCNN, InceptionResnetV1

    InceptionResnetV1(pretrained=cfg.faces.embedding.pretrained)
    MTCNN()  # weights ship inside the package


def owlv2(cfg):
    from transformers import AutoTokenizer, Owlv2ForObjectDetection

    Owlv2ForObjectDetection.from_pretrained(cfg.objects.owlv2_model)
    AutoTokenizer.from_pretrained(cfg.objects.owlv2_model)


def clip(cfg):
    import open_clip

    open_clip.create_model_and_transforms(cfg.objects.embedder_model, pretrained=cfg.objects.embedder_pretrained)


def whisper(cfg):
    from faster_whisper.utils import download_model

    print("  ->", download_model(cfg.stt.model))


def piper(cfg):
    from piper.download_voices import download_voice

    target = Path(cfg.tts.piper_voice)
    target.parent.mkdir(parents=True, exist_ok=True)
    download_voice(target.stem, target.parent)


def yunet(cfg):
    target = Path(cfg.faces.detection.yunet_model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(YUNET_URL, target)


STEPS = {"facenet": facenet, "owlv2": owlv2, "clip": clip, "whisper": whisper, "piper": piper, "yunet": yunet}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(STEPS), action="append")
    args = ap.parse_args()
    from assistant.config import load_config

    cfg = load_config("config.yaml" if os.path.exists("config.yaml") else None)
    failed = []
    for name in args.only or list(STEPS):
        print(f"Downloading {name}...", flush=True)
        try:
            STEPS[name](cfg)
            print(f"  ok: {name}")
        except Exception as e:
            print(f"  FAILED: {name}: {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    print("All models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
