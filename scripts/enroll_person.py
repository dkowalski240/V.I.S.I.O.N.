"""Enroll someone directly from the webcam (no introduction needed).

    python scripts/enroll_person.py --name Sarah [--seconds 5] [--camera 0]

Use it to test recognition (M2) before speech works, or as a demo fallback.
Only run it with the person's agreement. Other people may be in view: the
script locks onto the largest face (green box; others are grey and ignored)
and follows it. Press n to switch to the next face, q to stop early.
Views that don't look like the locked-on person are dropped, so someone
crossing in front can't end up in the profile.
Run it while the assistant is NOT running (or restart the assistant after).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from assistant.config import apply_cpu_fallback, load_config  # noqa: E402
from assistant.faces.alignment import align_face  # noqa: E402
from assistant.faces.embedding import l2_normalize  # noqa: E402
from assistant.faces.follow import FaceFollower  # noqa: E402
from assistant.faces.quality import QualityGate  # noqa: E402
from assistant.storage import Storage  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--camera", default=None, help="webcam index / file (default: camera.source from config)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    if args.cpu:
        apply_cpu_fallback(cfg)
    try:
        import torch

        if not torch.cuda.is_available():
            apply_cpu_fallback(cfg)
    except ImportError:
        pass
    from assistant.faces.detection import create_detector
    from assistant.faces.embedding import FaceNetEmbedder

    detector = create_detector(cfg.faces.detection)
    embedder = FaceNetEmbedder(cfg.faces.embedding)
    gate = QualityGate(cfg.faces.enroll_quality)
    src = args.camera if args.camera is not None else cfg.camera.source
    src = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and sys.platform.startswith("win") \
        else cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    if not cap.isOpened():
        print(f"Could not open camera {src!r}")
        return 1

    follower = FaceFollower()
    samples, rejected = [], 0

    def ref():
        return l2_normalize(np.mean([e for e, _ in samples], axis=0)) if len(samples) >= 2 else None

    def same_person(d) -> bool:  # used to find them again after losing them
        r = ref()
        if r is None:
            return False
        e = embedder.embed([align_face(frame, d, cfg.faces.alignment).image])[0]
        return float(r @ e) >= cfg.faces.matching.t_match

    print(f"Capturing {args.name} for {args.seconds:.0f} s: look at the camera, turn your head slowly...")
    print("Several faces? The green one is being enrolled; press n to switch.")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect(frame)
        target = follower.update(dets, frame.shape[1], same_person)
        msg, color = ("no face" if not dets else "lost them: step back into view"), (0, 0, 255)
        for d in dets:
            if d is not target:
                x1, y1, x2, y2 = d.box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
        if target is not None:
            aligned = align_face(frame, target, cfg.faces.alignment)
            rep = gate.assess(target, frame.shape, aligned)
            color = (0, 165, 255)
            msg = ", ".join(rep.reasons)
            if rep.ok:
                emb = embedder.embed([aligned.image])[0]
                r = ref()
                if r is not None and float(r @ emb) < cfg.enrollment.min_self_similarity:
                    rejected += 1  # a different person on the followed box
                elif all(float(e @ emb) < cfg.faces.matching.t_novel for e, _ in samples):
                    samples.append((emb, rep.score))
                msg, color = f"{len(samples)} views", (0, 200, 0)
            x1, y1, x2, y2 = target.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if len(dets) > 1:
            msg += f"   ({len(dets)} faces: n = next)"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("enroll (n = next face, q = stop)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            follower.next()
            samples.clear()  # start over on the newly chosen face
    cap.release()
    cv2.destroyAllWindows()
    samples.sort(key=lambda s: s[1], reverse=True)
    samples = samples[: cfg.enrollment.max_exemplars]
    if len(samples) < cfg.enrollment.min_exemplars:
        print(f"Only {len(samples)} good views; need {cfg.enrollment.min_exemplars}. Better light, closer, try again.")
        return 1
    store = Storage(cfg.storage.db_path, cfg.storage.snapshot_dir)
    pid = store.add_person(args.name, [s[0] for s in samples], [s[1] for s in samples])
    store.close()
    print(f"Saved {args.name} (person {pid}) with {len(samples)} views"
          f"{f'; ignored {rejected} views of someone else' if rejected else ''}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
