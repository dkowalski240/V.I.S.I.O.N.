"""Check the laptop before the demo: Python, packages, CUDA, every model, camera, audio.

    python scripts/check_setup.py              packages + CUDA
    python scripts/check_setup.py --models     also load every model, time it, measure VRAM
    python scripts/check_setup.py --cameras    list cameras (number, name, resolution, fps) via DirectShow + Media Foundation
    python scripts/check_setup.py --audio      list audio devices, speak a test phrase, beep left then right
    python scripts/check_setup.py --mic        show the microphone level for 5 seconds
    python scripts/check_setup.py --radio      Wi-Fi network and Bluetooth devices (place context)
    python scripts/check_setup.py --all
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OK, BAD, WARN = "  ok  ", "  !!  ", "  ..  "


def check_packages() -> None:
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    for mod, pip_name in [
        ("numpy", "numpy"), ("cv2", "opencv-python"), ("yaml", "PyYAML"), ("torch", "torch"),
        ("torchvision", "torchvision"), ("facenet_pytorch", "facenet-pytorch (--no-deps)"),
        ("transformers", "transformers"), ("open_clip", "open-clip-torch"), ("faster_whisper", "faster-whisper"),
        ("ctranslate2", "ctranslate2"), ("av", "av"), ("onnxruntime", "onnxruntime"), ("piper", "piper-tts"),
        ("pyttsx3", "pyttsx3"), ("sounddevice", "sounddevice"), ("soundfile", "soundfile"), ("pynput", "pynput"),
    ]:
        try:
            m = importlib.import_module(mod)
            print(f"{OK}{mod:16s} {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"{BAD}{mod:16s} missing ({pip_name}): {type(e).__name__}: {e}")
    try:
        import av

        if int(av.__version__.split(".")[0]) >= 19:
            print(f"{WARN}av {av.__version__}: faster-whisper 1.2.1 file decoding needs av<19 (pip install av==18.1.0)")
    except Exception:
        pass
    try:
        import torch

        print(f"     torch build: {torch.__version__}, CUDA runtime {torch.version.cuda}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"{OK}GPU {torch.cuda.get_device_name(0)}: {total / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
        else:
            print(f"{BAD}CUDA not available. You have a CPU-only torch build. Reinstall with:\n"
                  "        pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126")
    except Exception as e:
        print(f"{BAD}torch import failed: {e}")
    for p in ["models/piper/en_US-lessac-medium.onnx", "models/piper/en_US-lessac-medium.onnx.json"]:
        print(f"{OK if os.path.exists(p) else BAD}{p}{'' if os.path.exists(p) else '  (python scripts/download_models.py --only piper)'}")


def check_models() -> None:
    import numpy as np

    from assistant.config import load_config

    cfg = load_config("config.yaml" if os.path.exists("config.yaml") else None)
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        from assistant.config import apply_cpu_fallback

        apply_cpu_fallback(cfg)

    def used():
        if not cuda:
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9

    base = used()
    print(f"\nLoading models (GPU memory in use before: {base:.2f} GB)")
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    def step(name, load, run):
        t0 = time.time()
        try:
            obj = load()
            t_load = time.time() - t0
            run(obj)  # warm-up
            t1 = time.time()
            for _ in range(3):
                run(obj)
            ms = (time.time() - t1) / 3 * 1000
            print(f"{OK}{name:28s} load {t_load:5.1f}s  run {ms:6.1f} ms   GPU in use {used():.2f} GB")
            return obj
        except Exception as e:
            print(f"{BAD}{name:28s} {type(e).__name__}: {e}")
            return None

    from assistant.faces.detection import create_detector
    from assistant.faces.embedding import FaceNetEmbedder

    step("face detector (" + cfg.faces.detection.backend + ")", lambda: create_detector(cfg.faces.detection),
         lambda d: d.detect(frame))
    step("FaceNet", lambda: FaceNetEmbedder(cfg.faces.embedding),
         lambda e: e.embed([np.zeros((160, 160, 3), np.uint8)] * 4))
    from assistant.objects.detector import OWLv2Detector
    from assistant.objects.embedder import ClipEmbedder

    o = cfg.objects
    step("OWLv2", lambda: OWLv2Detector(o.owlv2_model, o.device, o.fp16, o.det_threshold),
         lambda d: d.detect(frame, list(o.generic_queries) + list(o.context_queries)))
    step("open_clip " + o.embedder_model, lambda: ClipEmbedder(o.embedder_model, o.embedder_pretrained, o.device, o.fp16),
         lambda e: e.embed([np.zeros((80, 120, 3), np.uint8)] * 8))
    from assistant.speech.stt import WhisperSTT

    step("faster-whisper " + cfg.stt.model, lambda: WhisperSTT(cfg.stt),
         lambda s: s.transcribe(np.zeros(32000, np.float32)))
    from assistant.speech.vad import SileroVAD

    step("Silero VAD", lambda: SileroVAD(), lambda v: v.prob(np.zeros(512, np.float32)))
    from assistant.speech.tts import make_tts

    step("TTS (" + cfg.tts.engine + ")", lambda: make_tts(cfg.tts), lambda t: t.synth("Sarah, approaching, 10 o'clock"))
    print(f"     total GPU memory used by the assistant: ~{used() - base:.2f} GB (8 GB budget)")


def check_cameras() -> None:
    """Try every camera number through both Windows camera systems (DirectShow and
    Media Foundation): virtual cameras such as DroidCam sometimes appear in only one."""
    import cv2

    try:  # failed opens of empty camera numbers print warnings; we report those ourselves
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    win = sys.platform.startswith("win")
    backends = [("DirectShow", cv2.CAP_DSHOW, "dshow"), ("Media Foundation", cv2.CAP_MSMF, "msmf")] if win \
        else [("default", cv2.CAP_ANY, "any")]
    names = {}
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        for _label, be, _key in backends:
            for c in enumerate_cameras(be):
                names[(be, c.index)] = c.name
    except ImportError:
        print(f"{WARN}pip install cv2-enumerate-cameras==1.4.0 to see camera names")
    except Exception as e:
        print(f"{WARN}couldn't list camera names: {e}")
    from assistant.camera import GREEN_HELP, apply_mode, is_green_blank, open_with_fallbacks, probe
    from assistant.config import load_config

    cfg = load_config("config.yaml" if os.path.exists("config.yaml") else None)

    print("\nCameras")
    found, green = 0, False
    for label, be, _key in backends:
        for i in range(10):
            cap = cv2.VideoCapture(i, be)
            if not cap.isOpened():
                cap.release()
                continue
            found += 1
            name = names.get((be, i), "")
            head = f"camera {i} via {label:16s} {name[:32]:32s}"
            # 1. the camera's own default mode
            t0, n, img = time.time(), 0, None
            while time.time() - t0 < 1.5:
                ok, frame = cap.read()
                if ok and frame is not None:
                    n, img = n + 1, frame
            fps = n / max(time.time() - t0, 1e-3)
            cap.release()
            if img is None:
                print(f"{BAD}{head} opens, but sends no frames")
                continue
            if is_green_blank(img):
                green = True
                print(f"{BAD}{head} {img.shape[1]}x{img.shape[0]}: BLANK GREEN PICTURE in its default mode;"
                      " trying other modes...")
                cap, used = open_with_fallbacks([(i, be, name or f"camera {i}")], cfg.camera, warn=False)
                cap.release()
                print(f"{OK if used else BAD}     {'works as: ' + used if used else 'green in every mode'}")
                continue
            # 2. the mode the assistant asks for (MJPG 1280x720 by default)
            cap = cv2.VideoCapture(i, be)
            apply_mode(cap, cfg.camera.mjpg, (cfg.camera.width, cfg.camera.height), cfg.camera.fps)
            verdict = probe(cap, 1.5)
            cap.release()
            note = "" if verdict == "ok" else (
                f"  (the requested {cfg.camera.width}x{cfg.camera.height} gives "
                f"{'a green picture' if verdict == 'green' else 'no frames'}: the assistant falls back by itself)")
            print(f"{OK}{head} {img.shape[1]}x{img.shape[0]}, ~{fps:.0f} fps{note}")
    if green:
        print(f"\n{WARN}{GREEN_HELP}")
        print("     Check: does the Windows Camera app show the phone's video on DroidCam Video? If it's green\n"
              "     there too, the fix is on the DroidCam side, not in the assistant.")
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
    except Exception:
        pass
    if not found:
        print(f"{BAD}No camera opened.")
    print("\nUse one with:  python -m assistant --source N   (add camera.backend: msmf to config.yaml if it")
    print("only opened via Media Foundation)   or by name:  python -m assistant --source DroidCam")


def check_audio() -> None:
    import numpy as np
    import sounddevice as sd

    from assistant.config import load_config
    from assistant.geometry import pan_gains
    from assistant.speech.tts import make_tts, to_stereo, tone

    print("\nAudio devices")
    print(sd.query_devices())
    cfg = load_config("config.yaml" if os.path.exists("config.yaml") else None)
    tts = make_tts(cfg.tts)
    print(f"Speaking a test phrase with {tts.name}, then a beep LEFT, then RIGHT (check your headphones)")
    audio = tts.synth("Sarah, approaching, 10 o'clock")
    sd.play(to_stereo(audio, pan_gains(-1.0)), tts.sample_rate, device=cfg.tts.output_device)
    sd.wait()
    for pan in (-1.0, 1.0):
        sd.play(to_stereo(tone("person", tts.sample_rate), pan_gains(pan)), tts.sample_rate,
                device=cfg.tts.output_device)
        sd.wait()
        time.sleep(0.3)


def check_radio() -> None:
    from assistant.config import load_config
    from assistant.radio import RadioScanner

    cfg = load_config("config.yaml" if os.path.exists("config.yaml") else None)
    cfg.places.wifi = True
    try:
        import bleak  # noqa: F401

        cfg.places.bluetooth = True
    except ImportError:
        print(f"{WARN}bleak not installed: Bluetooth context off (pip install -r requirements-bluetooth.txt)")
    print("\nWi-Fi / Bluetooth context")
    snap = RadioScanner(cfg.places).scan_once()
    if snap.ssid:
        print(f"{OK}Wi-Fi: {snap.ssid}  access point {snap.bssid}  signal {snap.signal}%")
    else:
        print(f"{BAD}No Wi-Fi name. Windows 11: Settings > Privacy & security > Location: turn on Location and "
              "'Let desktop apps access your location'.")
    if cfg.places.bluetooth:
        print(f"{OK if snap.ble else WARN}Bluetooth: {len(snap.ble)} devices at >= {cfg.places.ble_min_rssi} dBm")


def check_mic() -> None:
    import numpy as np
    import sounddevice as sd

    print("\nMicrophone level for 5 s (speak now)")
    with sd.InputStream(samplerate=16000, channels=1, blocksize=1600) as s:
        t0 = time.time()
        while time.time() - t0 < 5:
            block, _ = s.read(1600)
            lvl = float(np.sqrt(np.mean(block ** 2)))
            print("  " + "#" * min(60, int(lvl * 600)), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--cameras", action="store_true")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--radio", action="store_true", help="Wi-Fi network + Bluetooth devices (place context)")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    check_packages()
    if a.models or a.all:
        check_models()
    if a.cameras or a.all:
        check_cameras()
    if a.audio or a.all:
        check_audio()
    if a.mic or a.all:
        check_mic()
    if a.radio or a.all:
        check_radio()
    return 0


if __name__ == "__main__":
    sys.exit(main())
