"""Record a session for replay debugging: webcam video + microphone + push-to-talk.

    python scripts/record_session.py --name demo1 [--camera 0] [--seconds 120]

Writes data/sessions/demo1.mp4, demo1.wav and demo1.script.txt (the
push-to-talk windows, so commands replay exactly as spoken). Replay with:

    python -m assistant --replay data/sessions/demo1.mp4 --script data/sessions/demo1.script.txt --db data/replay.db

Press q in the preview window (or Ctrl+C) to stop. Hold the push-to-talk key
(speech_in.ptt_key) while giving commands, as you would live.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from assistant.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config if os.path.exists(args.config) else None)

    import sounddevice as sd
    import soundfile as sf

    out_dir = Path("data/sessions")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.name
    src = args.camera if args.camera is not None else cfg.camera.source
    src = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and sys.platform.startswith("win") \
        else cv2.VideoCapture(src)
    if cfg.camera.mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
    if not cap.isOpened():
        print(f"Could not open camera {src!r}")
        return 1
    ok, frame = cap.read()
    if not ok:
        print("Camera gives no frames")
        return 1
    fps = 15.0  # fixed output rate; frames are duplicated/dropped to keep the timeline exact
    writer = cv2.VideoWriter(str(stem) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))

    # --- push-to-talk windows
    ptt_windows, ptt_state = [], {"down": None}
    t_start = time.time()
    try:
        from pynput import keyboard

        key_name = cfg.speech_in.ptt_key
        target = keyboard.KeyCode.from_char(key_name) if len(key_name) == 1 else getattr(keyboard.Key, key_name)

        def on_press(k):
            if k == target and ptt_state["down"] is None:
                ptt_state["down"] = time.time() - t_start

        def on_release(k):
            if k == target and ptt_state["down"] is not None:
                ptt_windows.append((ptt_state["down"], time.time() - t_start))
                ptt_state["down"] = None

        keyboard.Listener(on_press=on_press, on_release=on_release, daemon=True).start()
        print(f"Push-to-talk key: {key_name}")
    except Exception as e:
        print(f"(push-to-talk not recorded: {e})")

    # --- audio
    wav = sf.SoundFile(str(stem) + ".wav", mode="w", samplerate=16000, channels=1, subtype="PCM_16")
    lock = threading.Lock()

    def audio_cb(indata, frames, time_info, status):
        with lock:
            wav.write(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=512,
                            device=cfg.speech_in.input_device, callback=audio_cb)
    stream.start()
    t_start = time.time()
    print(f"Recording to {stem}.mp4/.wav ... (q to stop)")
    n_written = 0
    try:
        while time.time() - t_start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            due = int((time.time() - t_start) * fps) + 1
            while n_written < due:  # keep video time == wall time
                writer.write(frame)
                n_written += 1
            view = frame.copy()
            rec = f"REC {time.time() - t_start:5.1f}s" + ("  PTT" if ptt_state["down"] is not None else "")
            cv2.putText(view, rec, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("record (q to stop)", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    stream.stop()
    stream.close()
    with lock:
        wav.close()
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    with open(str(stem) + ".script.txt", "w", encoding="utf-8") as fh:
        fh.write(f"# recorded {time.strftime('%Y-%m-%d %H:%M')}; push-to-talk windows (seconds)\n")
        for a, b in ptt_windows:
            fh.write(f"{a:.2f}-{b:.2f}  ptt\n")
        fh.write("# add scripted lines if you like, e.g.:  31.0  type  where's my wallet\n")
    print(f"Saved {stem}.mp4, {stem}.wav, {stem}.script.txt ({len(ptt_windows)} push-to-talk windows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
