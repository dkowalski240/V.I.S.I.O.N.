"""Measure same-person vs different-person similarity on YOUR webcam and
suggest faces.matching thresholds. (Port of FaceLock's calibrate.py, same math.)

    python scripts/calibrate_faces.py --capture alice      # ~30 s per person, repeat for 3-5+ people
    python scripts/calibrate_faces.py --data data/people --plot calibration.png

--capture saves aligned crops to data/people/<name>/ (only with that person's
agreement; delete the folder afterwards if you like). Other people may be in
view: only the green (followed) face is saved; press n to switch faces. Calibration compares
every pair of crops: same-person pairs ("genuine") vs different-person pairs
("impostor"), and puts t_match safely to the right of the impostor scores,
because a false "That's Sarah" is worse than a miss.
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
from assistant.faces.calibration import pair_scores, suggest_thresholds, summary  # noqa: E402
from assistant.faces.follow import FaceFollower  # noqa: E402
from assistant.faces.quality import QualityGate  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _cfg(args):
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    try:
        import torch

        if not torch.cuda.is_available():
            apply_cpu_fallback(cfg)
    except ImportError:
        apply_cpu_fallback(cfg)
    return cfg


def capture(args) -> int:
    cfg = _cfg(args)
    from assistant.faces.detection import create_detector

    det = create_detector(cfg.faces.detection)
    gate = QualityGate(cfg.faces.quality)
    out = Path(args.data) / args.capture
    out.mkdir(parents=True, exist_ok=True)
    src = cfg.camera.source
    cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW) if str(src).isdigit() and sys.platform.startswith("win") \
        else cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    saved, last, seen = len(list(out.glob("*.png"))), 0.0, time.time()
    follower = FaceFollower()  # other people may be in view: follow one face only
    target = saved + args.count
    print(f"Saving {args.count} crops to {out}. Move around: near/far, left/right, turn your head. q to stop.")
    while saved < target:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        face = follower.update(dets, frame.shape[1])
        if face is None and dets and time.time() - seen > 1.0:
            follower.reset()  # lost for a second: lock onto the largest face again
        for d in dets:
            if d is not face:
                x1, y1, x2, y2 = d.box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
        if face is not None:
            seen = time.time()
            a = align_face(frame, face, cfg.faces.alignment)
            rep = gate.assess(face, frame.shape, a)
            if rep.ok and time.time() - last > 0.4:
                saved += 1
                cv2.imwrite(str(out / f"{saved:04d}.png"), a.image)
                last = time.time()
            x1, y1, x2, y2 = face.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if rep.ok else (0, 165, 255), 2)
        note = f"   ({len(dets)} faces: only the green one is saved, n = next)" if len(dets) > 1 else ""
        cv2.putText(frame, f"{saved}/{target}{note}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("capture (n = next face, q = stop)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            follower.next()
    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} crops in {out}")
    return 0


def calibrate(args) -> int:
    cfg = _cfg(args)
    from assistant.faces.embedding import FaceNetEmbedder

    people = sorted(p for p in Path(args.data).iterdir() if p.is_dir()) if Path(args.data).exists() else []
    if len(people) < 2:
        print(f"Need at least 2 people folders in {args.data} (use --capture NAME)")
        return 1
    emb = FaceNetEmbedder(cfg.faces.embedding)
    size = cfg.faces.alignment.image_size
    E, labels = [], []
    for person in people:
        crops = [cv2.imread(str(p)) for p in sorted(person.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        crops = [c for c in crops if c is not None and c.shape[:2] == (size, size)]
        if len(crops) < 2:
            print(f"  skipping {person.name}: fewer than 2 crops")
            continue
        E.append(emb.embed(crops))
        labels += [person.name] * len(crops)
        print(f"  {person.name}: {len(crops)} crops")
    if len(set(labels)) < 2:
        print("Not enough usable data.")
        return 1
    genuine, impostor = pair_scores(np.vstack(E), labels)
    s = summary(genuine, impostor)
    print(f"\n{'':10}{'pairs':>8}{'min':>8}{'1%':>8}{'median':>8}{'99%':>8}{'max':>8}")
    for k in ("genuine", "impostor"):
        v = s[k]
        print(f"{k:10}{v['n']:>8}{v['min']:>8.3f}{v['p01']:>8.3f}{v['median']:>8.3f}{v['p99']:>8.3f}{v['max']:>8.3f}")
    sug = suggest_thresholds(genuine, impostor, args.target_far, n_people=len(set(labels)),
                             t_novel=cfg.faces.matching.t_novel)
    t_match = sug.t_match
    t_frame = max(0.5, round(t_match - 0.05, 2))
    t_sure = max(round(t_match + 0.05, 2), min(sug.t_update, 0.9))
    for n in sug.notes:
        print(f"  note: {n}")
    print(f"\nFAR at t_match {sug.far_at_match:.4f}, single-image FRR {sug.frr_at_match:.1%}")
    print("\nPaste into config.yaml under faces: -> matching:\n")
    print(f"    t_match: {t_match:.2f}\n    t_frame: {t_frame:.2f}\n    t_sure: {t_sure:.2f}")
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("pip install matplotlib for --plot")
            return 0
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(-0.3, 1.0, 66)
        ax.hist(impostor, bins=bins, alpha=0.6, density=True, label="different people")
        ax.hist(genuine, bins=bins, alpha=0.6, density=True, label="same person")
        for v, name in ((t_frame, "t_frame"), (t_match, "t_match"), (t_sure, "t_sure")):
            ax.axvline(v, color="black", linestyle="--", linewidth=1)
            ax.text(v, ax.get_ylim()[1] * 0.95, f" {name}", rotation=90, va="top", fontsize=9)
        ax.set_xlabel("cosine similarity")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"Saved {args.plot}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", metavar="NAME", help="capture crops of one person from the webcam")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--data", default="data/people")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--target-far", type=float, default=0.001)
    ap.add_argument("--plot")
    args = ap.parse_args()
    return capture(args) if args.capture else calibrate(args)


if __name__ == "__main__":
    sys.exit(main())
