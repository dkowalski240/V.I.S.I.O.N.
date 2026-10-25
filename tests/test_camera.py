"""Camera source selection: numbers, names (DroidCam), files and URLs."""

import sys
import types

import cv2
import numpy as np
import pytest

from assistant import camera as cam
from assistant.config import CameraConfig


def test_classify_source(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    assert cam.classify_source(0) == "webcam"
    assert cam.classify_source("2") == "webcam"
    assert cam.classify_source("http://192.168.1.20:4747/video") == "stream"
    assert cam.classify_source("DroidCam") == "named"
    assert cam.classify_source("DroidCam Video") == "named"
    assert cam.classify_source(str(clip)) == "file"
    assert cam.classify_source("data/sessions/missing.mp4") == "file"   # a path, not a name
    assert cam.classify_source(r"C:\clips\demo") == "file"


def _fake_enumerator(monkeypatch, per_backend):
    mod = types.ModuleType("cv2_enumerate_cameras")

    def enumerate_cameras(backend):
        return [types.SimpleNamespace(index=i, name=n, backend=backend)
                for i, n in enumerate(per_backend.get(backend, []))]

    mod.enumerate_cameras = enumerate_cameras
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", mod)
    monkeypatch.setattr(cam.sys, "platform", "win32")


def test_find_camera_by_name_looks_in_media_foundation_too(monkeypatch):
    # The laptop webcam in both systems; the virtual camera only in Media Foundation.
    _fake_enumerator(monkeypatch, {cv2.CAP_DSHOW: ["HD Webcam"],
                                   cv2.CAP_MSMF: ["HD Webcam", "DroidCam Video"]})
    assert cam.find_camera_by_name("droidcam") == (1, cv2.CAP_MSMF, "DroidCam Video")
    assert cam.find_camera_by_name("HD") == (0, cv2.CAP_DSHOW, "HD Webcam")
    with pytest.raises(RuntimeError, match="Cameras found: \\['DroidCam Video', 'HD Webcam'\\]"):
        cam.find_camera_by_name("Camo")


GREEN = np.zeros((480, 640, 3), np.uint8)
GREEN[..., 1] = 135                      # an empty YUV buffer, converted: what DroidCam sends when unsynced
REAL = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)


class FakeCapture:
    """picture(index, backend, mode) -> frame or None; mode = (mjpg, size) last applied."""
    opened = []

    def __init__(self, index, backend, picture):
        self.index, self.backend, self.picture = index, backend, picture
        self.mjpg, self.size = False, None
        self._w = None
        FakeCapture.opened.append((index, backend))

    def isOpened(self):
        return self.picture(self.index, self.backend, None) is not False

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOURCC:
            self.mjpg = True
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._w = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.size = (self._w, int(value))
        return True

    def read(self):
        img = self.picture(self.index, self.backend, (self.mjpg, self.size))
        return (img is not None), img

    def release(self):
        pass


def _install(monkeypatch, picture):
    FakeCapture.opened = []
    monkeypatch.setattr(cam.cv2, "VideoCapture", lambda i, b: FakeCapture(i, b, picture))


def test_green_detection():
    assert cam.is_green_blank(GREEN)
    assert not cam.is_green_blank(REAL)
    assert not cam.is_green_blank(np.zeros((480, 640, 3), np.uint8))       # dark room: a real picture
    plant = REAL.copy()
    plant[:200, :, :] = (40, 160, 50)                                    # lots of (natural) green
    assert not cam.is_green_blank(plant)
    half = REAL.copy()
    half[120:] = GREEN[120:]                                             # mostly green = still broken
    assert cam.is_green_blank(half)


def test_named_source_opens_the_found_camera(monkeypatch):
    _fake_enumerator(monkeypatch, {cv2.CAP_DSHOW: ["HD Webcam", "DroidCam Video"]})
    _install(monkeypatch, lambda i, b, mode: REAL)
    cam.CameraSource(CameraConfig(source="DroidCam", probe_s=0.2))._open()
    cam.CameraSource(CameraConfig(source="DroidCam", backend="msmf", probe_s=0.2))._open()
    assert FakeCapture.opened == [(1, cv2.CAP_DSHOW), (1, cv2.CAP_MSMF)]


def test_green_in_requested_mode_falls_back_to_default_mode(monkeypatch):
    # The virtual camera dislikes MJPG 1280x720 and answers with green; its own mode works.
    _install(monkeypatch, lambda i, b, mode: GREEN if mode and mode[0] else REAL)
    cap, used = cam.open_with_fallbacks([(1, cv2.CAP_DSHOW, "DroidCam Video")], CameraConfig(probe_s=0.2))
    assert used == "DroidCam Video via DirectShow, its default mode"
    assert cap.read()[1] is REAL


def test_wrong_size_falls_back_to_the_size_that_works(monkeypatch):
    # Only the size the phone app is set to (640x480) gives a picture.
    _install(monkeypatch, lambda i, b, mode: REAL if mode and mode[1] == (640, 480) else (GREEN if mode else REAL))
    _, used = cam.open_with_fallbacks([(1, cv2.CAP_DSHOW, "DroidCam Video")], CameraConfig(probe_s=0.2))
    assert used.endswith("640x480")


def test_named_camera_tries_the_other_windows_camera_system(monkeypatch):
    _fake_enumerator(monkeypatch, {cv2.CAP_DSHOW: ["HD Webcam", "DroidCam Video"],
                                   cv2.CAP_MSMF: ["HD Webcam", "DroidCam Video"]})
    _install(monkeypatch, lambda i, b, mode: GREEN if (i, b) == (1, cv2.CAP_DSHOW) else REAL)
    src = cam.CameraSource(CameraConfig(source="DroidCam", probe_s=0.2))
    cap = src._open()
    assert (cap.index, cap.backend) == (1, cv2.CAP_MSMF)


def test_green_everywhere_still_opens_and_explains(monkeypatch, caplog):
    _install(monkeypatch, lambda i, b, mode: GREEN)
    with caplog.at_level("WARNING"):
        cap, used = cam.open_with_fallbacks([(1, cv2.CAP_DSHOW, "DroidCam Video")], CameraConfig(probe_s=0.1))
    assert used is None and cap.isOpened()
    assert "File > Exit" in caplog.text
    # A number that won't open at all is still an error.
    _install(monkeypatch, lambda i, b, mode: False)
    with pytest.raises(RuntimeError, match="Could not open camera source"):
        cam.CameraSource(CameraConfig(source=3, probe_s=0.1))._open()


def test_live_green_picture_is_flagged_and_cleared():
    src = cam.CameraSource(CameraConfig(source=0))
    src._watch_picture(GREEN, 0.0)
    src._watch_picture(GREEN, 2.0)
    assert src.problem == ""             # a moment of green (reconnecting) is fine
    src._watch_picture(GREEN, 3.5)
    assert "File > Exit" in src.problem
    src._watch_picture(REAL, 4.0)
    assert src.problem == ""


def test_named_source_without_the_package_explains(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2_enumerate_cameras", None)   # import fails
    with pytest.raises(RuntimeError, match="cv2-enumerate-cameras"):
        cam.find_camera_by_name("DroidCam")
