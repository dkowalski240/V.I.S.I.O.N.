"""Camera sources: webcam index, video file (replay) or stream URL.

Same idea as FaceLock's camera.py: a background thread always holds the
*newest* frame ("latest frame wins"), so slow models never make the video lag.
Modules call `wait_for(after_index)` to block until a newer frame exists.

    0, "1"                      -> webcam (DirectShow on Windows: opens fast)
    "DroidCam"                  -> the camera whose name contains that text (needs cv2-enumerate-cameras)
    "http://...", "rtsp://..."  -> network stream (phone apps like IP Webcam / DroidCam)
    "path/to/video.mp4"         -> file, paced in real time on the shared clock (replay)

Every frame carries a timestamp from the shared clock. For files that's the
replay clock, so "frame at 12.4 s into the video" and "audio at 12.4 s" line up.

The `camera_id` field exists so a fixed home camera could be added later as a
second CameraSource; nothing downstream assumes there's only one.

TODO: phone streaming works as a URL source but is untuned (latency, reconnect
timing); when you switch to a phone, use direction.mode: angle with its real FOV.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .clock import Clock
from .config import CameraConfig

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray   # BGR
    index: int          # increments for every captured frame
    timestamp: float    # shared clock time when captured (or media time in replay)
    camera_id: str = "wearable"


def classify_source(source) -> str:
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        return "webcam"
    if isinstance(source, str) and "://" in source:
        return "stream"
    if isinstance(source, str) and not os.path.exists(source) and not os.path.splitext(source)[1] \
            and "/" not in source and "\\" not in source:
        return "named"  # e.g. "DroidCam": a camera picked by (part of) its name
    return "file"


def find_cameras_by_name(name: str):
    """[(index, backend, full name)] of every camera whose name contains `name`:
    DirectShow first, then Media Foundation (a virtual camera may be in only one)."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        raise RuntimeError(f"camera.source {name!r} looks like a camera name; picking cameras by name needs "
                           "'pip install cv2-enumerate-cameras==1.4.0' (or use the camera's number)")
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    seen, found = [], []
    for be in backends:
        for c in enumerate_cameras(be):
            seen.append(c.name)
            if name.lower() in c.name.lower():
                found.append((c.index, be, c.name))
    if not found:
        raise RuntimeError(f"No camera named like {name!r}. Cameras found: {sorted(set(seen)) or 'none'}. "
                           "Is the phone app streaming and its Windows client started?")
    return found


def find_camera_by_name(name: str):
    """(index, backend, full name) of the first camera whose name contains `name`."""
    return find_cameras_by_name(name)[0]


def green_fraction(image: np.ndarray) -> float:
    """Share of pixels that are pure dark green (B, R < 30, G > 90): what an empty
    YUV buffer looks like after conversion, i.e. a virtual camera sending nothing."""
    if image is None or image.ndim != 3 or image.size == 0:
        return 0.0
    small = image[::8, ::8]
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    return float(np.mean((g > 90) & (b < 30) & (r < 30)))


def is_green_blank(image: np.ndarray) -> bool:
    return green_fraction(image) > 0.5


GREEN_HELP = ("The camera is sending a blank green picture. For DroidCam that means its webcam output "
              "isn't synced: exit the DroidCam client with File > Exit (not just the window's X), start "
              "it again and reconnect the phone; if it's still green, restart the laptop.")


def probe(cap, timeout_s: float = 2.0) -> str:
    """'ok' (a real picture arrived), 'green' (only blank green frames) or 'none' (no frames)."""
    t0, got = time.monotonic(), False
    while time.monotonic() - t0 < timeout_s:
        ok, image = cap.read()
        if ok and image is not None:
            got = True
            if not is_green_blank(image):
                return "ok"
        else:
            time.sleep(0.02)
    return "green" if got else "none"


def camera_modes(cfg: CameraConfig):
    """(label, ask_for_mjpg, (width, height) or None) to try, in order. Virtual cameras
    (DroidCam, Camo...) can answer a mode they don't like with no frames or a blank
    green picture, and some only work at the size their own app is set to."""
    modes = [(f"{'MJPG ' if cfg.mjpg else ''}{cfg.width}x{cfg.height} (requested)", cfg.mjpg, (cfg.width, cfg.height)),
             ("its default mode", False, None)]
    for size in [(1280, 720), (640, 480), (1920, 1080)]:
        if size != (cfg.width, cfg.height) or cfg.mjpg:
            modes.append((f"{size[0]}x{size[1]}", False, size))
    return modes


def apply_mode(cap, mjpg: bool, size, fps: float) -> None:
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_with_fallbacks(candidates, cfg: CameraConfig, warn: bool = True):
    """Open the first (index, backend, name) candidate and mode that gives a real picture.
    Returns (capture, description of what worked or None). If nothing gives a real
    picture, the first candidate is opened in the requested mode anyway (the phone
    app may reconnect) and a warning says why it looks wrong."""
    tried = []
    modes = camera_modes(cfg)
    if cfg.probe_s <= 0:
        modes = modes[:1]
    for index, backend, name in candidates:
        for label, mjpg, size in modes:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                tried.append(f"{name} via {_backend_name(backend)}: won't open")
                break
            apply_mode(cap, mjpg, size, cfg.fps)
            verdict = probe(cap, cfg.probe_s) if cfg.probe_s > 0 else "ok"
            if verdict == "ok":
                used = f"{name} via {_backend_name(backend)}, {label}"
                if tried:
                    log.info("Camera: using %s (didn't work: %s)", used, "; ".join(tried))
                else:
                    log.info("Camera: %s", used)
                return cap, used
            cap.release()
            tried.append(f"{name} via {_backend_name(backend)}, {label}: "
                         + ("blank green picture" if verdict == "green" else "no frames"))
    index, backend, name = candidates[0]
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        apply_mode(cap, cfg.mjpg, (cfg.width, cfg.height), cfg.fps)
        if not warn:
            pass
        elif any("green" in t for t in tried):
            log.warning("%s Tried: %s", GREEN_HELP, "; ".join(tried))
        else:
            log.warning("%s gives no frames yet. Tried: %s", name, "; ".join(tried))
    return cap, None


_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG,
    "any": cv2.CAP_ANY,
}


def _backend_name(backend: int) -> str:
    return {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation", cv2.CAP_ANY: "default"}.get(
        backend, str(backend))


class CameraSource:
    def __init__(self, cfg: CameraConfig, clock: Optional[Clock] = None):
        self.cfg = cfg
        self.clock = clock or Clock()
        self.kind = classify_source(cfg.source)
        self.camera_id = cfg.camera_id
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.ended = False         # a video file reached its end (and loop_video is off)
        self.connected = False
        self.fps = 0.0             # measured capture rate
        self.problem = ""          # e.g. GREEN_HELP while the camera sends blank green frames
        self._green_since: Optional[float] = None
        self._warned_at = -1e9
        self.size = (0, 0)         # (width, height) of delivered frames

    # ------------------------------------------------------------------ lifecycle
    def _backend(self) -> int:
        name = self.cfg.backend.lower()
        if name != "auto":
            if name not in _BACKENDS:
                raise ValueError(f"camera.backend must be auto or one of {sorted(_BACKENDS)}")
            return _BACKENDS[name]
        if self.kind == "webcam":
            return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        # FFmpeg for streams and files: it decodes iPhone .mov (HEVC/H.264) and applies
        # the video's rotation metadata, so portrait clips come out upright.
        return cv2.CAP_FFMPEG

    def _open(self) -> cv2.VideoCapture:
        src = self.cfg.source
        if self.kind in ("webcam", "named"):
            cap = self._open_device()
        elif self.kind == "stream":
            # Short timeouts so a dropped phone stream reconnects quickly.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(str(src), self._backend())
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if not os.path.exists(str(src)):
                raise FileNotFoundError(f"Video file not found: {src}")
            cap = cv2.VideoCapture(str(src), self._backend())
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source {src!r}. Try another index (scripts/check_setup.py "
                "--cameras lists them) or camera.backend: msmf."
            )
        return cap

    def _open_device(self) -> cv2.VideoCapture:
        """Open a webcam number or a named camera (see open_with_fallbacks)."""
        src = self.cfg.source
        if self.kind == "named":
            candidates = find_cameras_by_name(str(src))
            if self.cfg.backend.lower() != "auto":
                be = self._backend()
                candidates = [c for c in candidates if c[1] == be] or [(candidates[0][0], be, candidates[0][2])]
        else:
            candidates = [(int(src), self._backend(), f"camera {src}")]
        cap, _ = open_with_fallbacks(candidates, self.cfg)
        return cap

    def start(self) -> "CameraSource":
        self._cap = self._open()
        self.connected = True
        target = self._run_file if self.kind == "file" else self._run_live  # named = a live camera
        self._thread = threading.Thread(target=target, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()
        log.info("Camera %s started (%s: %r)", self.camera_id, self.kind, self.cfg.source)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self.connected = False
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ access
    def latest(self) -> Optional[Frame]:
        with self._cond:
            return self._latest

    def wait_for(self, after_index: int, timeout: float = 0.5) -> Optional[Frame]:
        """Block until a frame newer than `after_index` exists (or timeout)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._stop.is_set():
                if self._latest is not None and self._latest.index > after_index:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.ended:
                    return None
                self._cond.wait(remaining)
        return None

    # ------------------------------------------------------------------ internals
    _ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def _deliver(self, image: np.ndarray, index: int, ts: float) -> None:
        if self.cfg.rotate:
            image = cv2.rotate(image, self._ROTATIONS[self.cfg.rotate])
        if self.cfg.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image=image, index=index, timestamp=ts, camera_id=self.camera_id)
        with self._cond:
            self._latest = frame
            self.size = (image.shape[1], image.shape[0])
            self._cond.notify_all()

    def _run_live(self) -> None:
        index, failures = 0, 0
        t_rate, n_rate = time.monotonic(), 0
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                failures += 1
                self.connected = failures < 30
                if failures % 30 == 0:
                    log.warning("Camera read failed %d times in a row", failures)
                    if self.kind == "stream":
                        self._reconnect()
                time.sleep(0.03)
                continue
            failures = 0
            self.connected = True
            if index % 15 == 0:  # about twice a second
                self._watch_picture(image, time.monotonic())
            index += 1
            self._deliver(image, index, self.clock.now())
            n_rate += 1
            if time.monotonic() - t_rate >= 1.0:
                self.fps = n_rate / (time.monotonic() - t_rate)
                t_rate, n_rate = time.monotonic(), 0

    def _watch_picture(self, image: np.ndarray, now: float) -> None:
        """Flag (debug window + log) a camera that keeps sending blank green frames,
        e.g. DroidCam after the phone disconnects or before its output syncs."""
        if is_green_blank(image):
            self._green_since = self._green_since if self._green_since is not None else now
            if now - self._green_since > 3.0:
                self.problem = GREEN_HELP
                if now - self._warned_at > 30.0:
                    log.warning("%s", GREEN_HELP)
                    self._warned_at = now
        else:
            self._green_since, self.problem = None, ""

    def _reconnect(self) -> None:
        log.info("Reconnecting to stream %s", self.cfg.source)
        try:
            self._cap.release()
            time.sleep(1.0)
            self._cap = self._open()
        except RuntimeError as e:
            log.warning("%s", e)

    def _run_file(self) -> None:
        """Pace a video file on the shared clock: frame at media time m is
        released when clock.now() reaches clock.at(m)."""
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        at = getattr(self.clock, "at", None)
        base = self.clock.now() if at is None else None
        offset = 0.0      # media time added each time the file loops
        last_media = 0.0
        index, n_in_pass = 0, 0
        self.fps = fps
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                if self.cfg.loop_video and n_in_pass > 0:
                    offset = last_media + 1.0 / fps
                    n_in_pass = 0
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.ended = True
                log.info("Video file ended after %d frames", index)
                with self._cond:
                    self._cond.notify_all()
                return
            # Frame position from the container; fall back to counting frames.
            pos = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            media_s = offset + (pos if pos > 0 or n_in_pass == 0 else n_in_pass / fps)
            last_media = media_s
            n_in_pass += 1
            ts = at(media_s) if at is not None else base + media_s
            self.clock.sleep_until(ts)
            index += 1
            self._deliver(image, index, ts)


class StaticCamera:
    """A fake camera that serves frames you hand it (tests, scripts)."""

    def __init__(self, camera_id: str = "wearable"):
        self.camera_id = camera_id
        self._latest: Optional[Frame] = None
        self._index = 0
        self.connected = True
        self.ended = False
        self.fps = 0.0
        self.size = (0, 0)
        self.kind = "static"
        self.problem = ""

    def push(self, image: np.ndarray, timestamp: float) -> Frame:
        self._index += 1
        self._latest = Frame(image=image, index=self._index, timestamp=timestamp, camera_id=self.camera_id)
        self.size = (image.shape[1], image.shape[0])
        return self._latest

    def latest(self) -> Optional[Frame]:
        return self._latest

    def wait_for(self, after_index: int, timeout: float = 0.5) -> Optional[Frame]:
        f = self._latest
        return f if f is not None and f.index > after_index else None

    def start(self):
        return self

    def stop(self):
        pass
