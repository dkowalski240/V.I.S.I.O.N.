"""Run the assistant.

    python -m assistant                         live: webcam + mic + push-to-talk
    python -m assistant --stub                  no models: camera, window, bus, speech output (M1 check)
    python -m assistant --replay data/sessions/demo.mp4 [--audio demo.wav] [--script demo.script.txt]
    python -m assistant --source 1              another webcam
    python -m assistant --source DroidCam       a camera picked by name (phone virtual webcam)
    python -m assistant --source http://192.168.1.20:8080/video   phone stream
    python -m assistant --record data/demo.mp4 --record-mic DroidCam   one video: window + phone audio + voice
    python -m assistant --cpu                   every model on the CPU
    python -m assistant --forget-people         start with nobody enrolled (testing introductions)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="assistant", description="Visual memory assistant")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="override camera.source (webcam index, camera name, file or URL)")
    ap.add_argument("--replay", metavar="VIDEO", help="run the full pipeline on a recorded video")
    ap.add_argument("--audio", help="replay audio file (default: VIDEO with .wav extension, if present)")
    ap.add_argument("--script", help="replay script: scripted speech and push-to-talk windows")
    ap.add_argument("--db", help="override storage.db_path (e.g. a scratch DB for replays)")
    ap.add_argument("--stub", action="store_true", help="don't load any model (M1 smoke test)")
    ap.add_argument("--cpu", action="store_true", help="run every model on the CPU")
    ap.add_argument("--no-window", action="store_true", help="no OpenCV window")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--no-speech", action="store_true", help="no microphone / speech-to-text")
    ap.add_argument("--tts", choices=["piper", "pyttsx3", "print"], help="override tts.engine")
    ap.add_argument("--record", help="save one video to this .mp4: debug window + microphone + the assistant's voice")
    ap.add_argument("--record-mic", metavar="MIC",
                    help="microphone for --record: 'assistant' (what it hears), a device name like DroidCam, or 'none'")
    ap.add_argument("--keep-open", action="store_true", help="replay: keep running after the video ends")
    ap.add_argument("--forget-people", action="store_true", help="delete every enrolled person before starting")
    ap.add_argument("--forget-items", action="store_true", help="delete every saved item (and sightings) first")
    ap.add_argument("--forget-places", action="store_true", help="delete every saved place first")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("assistant")

    from .config import apply_cpu_fallback, load_config

    if not os.path.exists(args.config):
        log.warning("Config %s not found; using built-in defaults", args.config)
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    if args.source is not None:
        cfg.camera.source = int(args.source) if args.source.isdigit() else args.source
    if args.record_mic is not None:
        cfg.debug.record_mic = int(args.record_mic) if args.record_mic.isdigit() else args.record_mic
    if args.db:
        cfg.storage.db_path = args.db
    if args.cpu:
        apply_cpu_fallback(cfg)
    if args.no_faces:
        cfg.faces.enabled = False
    if args.no_objects:
        cfg.objects.enabled = False
    if args.no_speech:
        cfg.speech_in.enabled = False
    if args.stub:
        cfg.speech_in.enabled = False

    if args.forget_people or args.forget_items or args.forget_places:
        from .storage import Storage

        st = Storage(cfg.storage.db_path, cfg.storage.snapshot_dir)
        if args.forget_people:
            log.info("Forgot %d people", st.delete_all_people())
        if args.forget_items:
            log.info("Forgot %d items", st.delete_all_items())
        if args.forget_places:
            log.info("Forgot %d places", st.delete_all_places())
        st.close()

    from .app import App, RunOptions

    opts = RunOptions(stub=args.stub, replay_video=args.replay, replay_audio=args.audio, script=args.script,
                      window=not args.no_window, record_path=args.record or "", tts_engine=args.tts,
                      exit_when_done=not args.keep_open)
    try:
        App(cfg, opts).run()
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
