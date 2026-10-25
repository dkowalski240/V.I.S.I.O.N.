"""`--record`: one video with the debug window, a microphone and the assistant's voice.

What goes in:

* picture: the annotated debug window, encoded to H.264 by a background thread
  (NVIDIA's hardware encoder when there is one, else x264) at a constant
  `debug.record_fps`. If the window loop runs slower than that, the last frame is
  repeated, so the video always runs in real time and stays in sync with the sound.
* microphone: `debug.record_mic: assistant` records what the assistant itself hears;
  a device name such as `DroidCam` records that microphone instead (the phone's
  audio, from the wearer's point of view); `none` records no microphone.
* voice: every clip the assistant plays, at the moment it starts, cut short if it was
  interrupted, with its left/right pan kept, so a viewer with headphones hears the
  direction as the wearer did.

While running, only the picture goes to disk (NAME.video.mkv, which survives a crash).
The sound is kept in memory. When the assistant stops (q, Esc or Ctrl+C) the sound is
mixed, saved as NAME.audio.wav, and joined with the picture into NAME.mp4 (H.264 +
AAC). The two part files are removed after that succeeds; if it fails they stay and
the log prints the ffmpeg command that joins them.

Every timestamp comes from the shared clock, so picture, microphone and voice line
up the same way the assistant's own modules do.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

OUT_RATE = 48000      # sound track sample rate
AAC_FRAME = 1024


# --------------------------------------------------------------------------- sound helpers
def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampling along axis 0. Plenty for speech, no extra dependency."""
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    t_in = np.arange(len(x), dtype=np.float64)
    if x.ndim == 1:
        return np.interp(t_out, t_in, x).astype(np.float32)
    return np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)


def place(dst: np.ndarray, src: np.ndarray, start: int) -> None:
    """Add `src` into `dst` starting at sample `start` (either may stick out of the other)."""
    if len(src) == 0:
        return
    a, b = max(0, start), min(len(dst), start + len(src))
    if b <= a:
        return
    dst[a:b] += src[a - start:b - start]


class _MicTrack:
    """Microphone blocks laid on one timeline by counting samples (smooth), re-synced
    to the blocks' own timestamps after a gap or stall of more than 0.25 s."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sr: Optional[int] = None
        self.t0 = 0.0
        self.n = 0
        self.chunks: List[Tuple[int, np.ndarray]] = []

    def add(self, block: np.ndarray, ts: float, sr: int) -> None:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        with self.lock:
            if self.sr is None:
                self.sr, self.t0 = int(sr), float(ts)
            elif sr != self.sr:
                block = resample(block, sr, self.sr)
            expected = self.t0 + self.n / self.sr
            if abs(ts - expected) > 0.25:
                self.n = max(0, int(round((ts - self.t0) * self.sr)))
            self.chunks.append((self.n, block))
            self.n += len(block)

    def render(self) -> Optional[Tuple[np.ndarray, float, int]]:
        with self.lock:
            if self.sr is None or not self.chunks:
                return None
            total = max(off + len(b) for off, b in self.chunks)
            out = np.zeros(total, dtype=np.float32)
            for off, b in self.chunks:
                out[off:off + len(b)] = b
            return out, self.t0, self.sr


@dataclass
class _Voice:
    t: float                 # shared-clock time playback started
    audio: np.ndarray        # (n, 2) float32
    sr: int
    end: Optional[float] = None  # set when playback was cut short


# --------------------------------------------------------------------------- the recorder
class Recorder:
    def __init__(self, path: str, cfg, clock):
        self.cfg = cfg
        self.clock = clock
        stem, ext = os.path.splitext(path)
        self.path = path if ext else path + ".mp4"
        self.video_part = stem + ".video.mkv"
        self.audio_part = stem + ".audio.wav"
        self.fps = max(1.0, float(cfg.record_fps))
        self.with_audio = bool(cfg.record_audio)
        self.codec: Optional[str] = None
        self.frames_written = 0
        self.mic = _MicTrack()
        self._voices: List[_Voice] = []
        self._vlock = threading.Lock()
        self._cond = threading.Condition()
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._seq = 0
        self._stopping = False
        self._t0: Optional[float] = None
        self._t_end: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._device_stream = None
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ inputs
    def add_frame(self, img: np.ndarray, t: Optional[float] = None) -> None:
        """Called by the window loop with each drawn frame (no copy: it's a fresh image)."""
        t = self.clock.now() if t is None else t
        with self._cond:
            self._latest = (img, t)
            self._seq += 1
            self._cond.notify()

    def add_mic(self, block: np.ndarray, ts: float, sr: int) -> None:
        """A microphone block whose first sample was heard at shared-clock time `ts`."""
        if self.with_audio:
            self.mic.add(block, ts, sr)

    def add_voice(self, stereo: np.ndarray, sr: int, t: Optional[float] = None) -> _Voice:
        v = _Voice(self.clock.now() if t is None else t, np.asarray(stereo, dtype=np.float32), int(sr))
        if self.with_audio and len(v.audio):
            with self._vlock:
                self._voices.append(v)
        return v

    def cut_voice(self, v: _Voice, t: Optional[float] = None) -> None:
        v.end = self.clock.now() if t is None else t

    def open_mic_device(self, spec) -> None:
        """Record a microphone of our own (e.g. "DroidCam": the phone's audio)."""
        import sounddevice as sd

        from .speech.audio_in import resolve_input_device

        dev = resolve_input_device(spec)
        info = sd.query_devices(dev, "input")
        sr = int(info.get("default_samplerate") or 48000)
        latency = [0.0]

        def callback(indata, frames, time_info, status):
            self.add_mic(indata[:, 0].copy(), self.clock.now() - frames / sr - latency[0], sr)

        stream = sd.InputStream(device=dev, channels=1, samplerate=sr, dtype="float32", callback=callback)
        latency[0] = float(stream.latency or 0.0)
        stream.start()
        self._device_stream = stream
        log.info("Recording microphone: %s (%d Hz)", info.get("name"), sr)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Recorder":
        self._thread = threading.Thread(target=self._encode_loop, name="recorder", daemon=True)
        self._thread.start()
        log.info("Recording to %s (%g fps%s)", self.path, self.fps,
                 ", with sound" if self.with_audio else ", picture only")
        return self

    def finish(self, t_end: Optional[float] = None) -> Optional[str]:
        """Stop, mix the sound and write the final video. Returns its path (None if nothing was saved)."""
        if self._device_stream is not None:
            try:
                self._device_stream.stop()
                self._device_stream.close()
            except Exception:
                pass
            self._device_stream = None
        with self._cond:
            self._t_end = self.clock.now() if t_end is None else t_end
            self._stopping = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=600)
        if self._t0 is None or self.frames_written == 0:
            log.warning("Recording: no frames were captured, nothing saved%s",
                        f" ({self.error})" if self.error else "")
            return None
        duration = self.frames_written / self.fps
        log.info("Saving the recording (%.0f s of video)...", duration)
        audio = self.mix(self._t0, duration) if self.with_audio else None
        if audio is not None:
            try:
                import soundfile as sf

                sf.write(self.audio_part, audio, OUT_RATE)
            except Exception as e:
                log.warning("Couldn't write %s: %s", self.audio_part, e)
        try:
            self._mux(audio)
        except Exception as e:
            log.exception("Recording: joining picture and sound failed (%s)", e)
            log.warning("The parts are kept. Join them with:\n  ffmpeg -i \"%s\" -i \"%s\" -c:v copy -c:a aac "
                        "-shortest \"%s\"", self.video_part, self.audio_part, self.path)
            return None
        for part in (self.video_part, self.audio_part):
            try:
                os.remove(part)
            except OSError:
                pass
        log.info("Saved %s (%.0f s, %s video%s)", self.path, duration, self.codec,
                 " + sound" if audio is not None else "")
        return self.path

    # ------------------------------------------------------------------ picture
    def _codecs(self) -> List[Tuple[str, dict]]:
        # No B-frames ("bf": "0"): every packet then has its decode time = its display time,
        # which keeps the MKV -> MP4 copy at the end simple and exact.
        q = str(int(self.cfg.record_quality))
        return [
            ("h264_nvenc", {"preset": "p4", "rc": "vbr", "cq": q, "bf": "0"}),  # NVIDIA hardware encoder: ~no CPU
            ("libx264", {"preset": "veryfast", "crf": q, "bf": "0"}),
            ("mpeg4", {"qscale": "3", "bf": "0"}),                               # last resort
        ]

    def _open_video(self, w: int, h: int):
        import av

        os.makedirs(os.path.dirname(self.video_part) or ".", exist_ok=True)
        rate = Fraction(self.fps).limit_denominator(1001)
        for name, opts in self._codecs():
            container = None
            try:
                container = av.open(self.video_part, mode="w")
                stream = container.add_stream(name, rate=rate)
                stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
                stream.options = dict(opts)
                if name == "mpeg4":
                    stream.bit_rate = 12_000_000
                stream.codec_context.open()  # fail here, not mid-recording, if this encoder can't run
                self.codec = name
                return container, stream
            except Exception as e:
                log.debug("Encoder %s unavailable: %s", name, e)
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
        raise RuntimeError("no H.264 or MPEG-4 encoder available in PyAV")

    def _encode_loop(self) -> None:
        import av

        container = stream = None
        prev_yuv: Optional[np.ndarray] = None
        last_slot, seen = -1, 0
        w = h = 0

        def encode(yuv: np.ndarray, slot: int) -> None:
            frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
            frame.pts = slot
            frame.time_base = stream.codec_context.time_base
            for pkt in stream.encode(frame):
                container.mux(pkt)
            self.frames_written += 1

        try:
            while True:
                with self._cond:
                    while self._seq == seen and not self._stopping:
                        self._cond.wait(0.5)
                    item, seq, stopping = self._latest, self._seq, self._stopping
                if item is not None and seq != seen:
                    seen = seq
                    img, t = item
                    if self._t0 is None:
                        self._t0 = t
                        h, w = img.shape[0] - img.shape[0] % 2, img.shape[1] - img.shape[1] % 2
                        container, stream = self._open_video(w, h)
                    if img.shape[0] != h or img.shape[1] != w:  # odd size, or the camera changed size
                        import cv2

                        img = cv2.resize(img, (w, h))
                    slot = int((t - self._t0) * self.fps)
                    if slot > last_slot:
                        while prev_yuv is not None and last_slot + 1 < slot:  # loop was slow: repeat
                            last_slot += 1
                            encode(prev_yuv, last_slot)
                        yuv = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:, :, :3]), format="bgr24") \
                            .reformat(format="yuv420p").to_ndarray()
                        last_slot = slot
                        encode(yuv, slot)
                        prev_yuv = yuv
                if stopping:
                    if container is not None and prev_yuv is not None and self._t_end is not None:
                        end_slot = int((self._t_end - self._t0) * self.fps)
                        while last_slot < end_slot:
                            last_slot += 1
                            encode(prev_yuv, last_slot)
                    break
        except Exception as e:
            self.error = str(e)
            log.exception("Recording stopped: %s", e)
        finally:
            if container is not None:
                try:
                    for pkt in stream.encode(None):
                        container.mux(pkt)
                except Exception:
                    log.exception("Recording: flushing the encoder failed")
                container.close()

    # ------------------------------------------------------------------ sound
    def mix(self, t0: float, duration: float) -> np.ndarray:
        """The sound track: microphone in both ears + the assistant's voice with its pan."""
        n = int(round(duration * OUT_RATE))
        out = np.zeros((n, 2), dtype=np.float32)
        mic = self.mic.render()
        if mic is not None:
            audio, mic_t0, sr = mic
            m = resample(audio, sr, OUT_RATE) * float(self.cfg.record_mic_gain)
            start = int(round((mic_t0 + float(self.cfg.record_mic_shift_ms) / 1000.0 - t0) * OUT_RATE))
            place(out, np.repeat(m[:, None], 2, axis=1), start)
        with self._vlock:
            voices = list(self._voices)
        for v in voices:
            a = v.audio if v.audio.ndim == 2 else np.repeat(v.audio[:, None], 2, axis=1)
            if v.end is not None:
                a = a[:max(0, int((v.end - v.t) * v.sr))]
            a = resample(a[:, :2], v.sr, OUT_RATE) * float(self.cfg.record_voice_gain)
            place(out, a, int(round((v.t - t0) * OUT_RATE)))
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def _mux(self, audio: Optional[np.ndarray]) -> None:
        import av

        with av.open(self.video_part) as vin, av.open(self.path, mode="w") as out:
            v_in = vin.streams.video[0]
            v_out = out.add_stream_from_template(v_in)
            a_out = None
            if audio is not None:
                a_out = out.add_stream("aac", rate=OUT_RATE)
                a_out.codec_context.layout = "stereo"
                a_out.bit_rate = 192_000
            tb = Fraction(1, OUT_RATE)
            next_i = 0

            def audio_until(t_sec: float) -> None:
                nonlocal next_i
                while a_out is not None and next_i < len(audio) and next_i / OUT_RATE <= t_sec:
                    chunk = audio[next_i:next_i + AAC_FRAME]
                    if len(chunk) < AAC_FRAME:
                        chunk = np.vstack([chunk, np.zeros((AAC_FRAME - len(chunk), 2), np.float32)])
                    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk.T), format="fltp",
                                                       layout="stereo")
                    frame.sample_rate = OUT_RATE
                    frame.pts = next_i
                    frame.time_base = tb
                    for pkt in a_out.encode(frame):
                        out.mux(pkt)
                    next_i += AAC_FRAME

            for pkt in vin.demux(v_in):
                if pkt.size == 0:  # the demuxer's end-of-stream marker
                    continue
                ts = pkt.pts if pkt.pts is not None else pkt.dts
                if ts is None:
                    continue
                if pkt.dts is None:
                    pkt.dts = pkt.pts  # no B-frames: decode order = display order
                audio_until(float(ts * pkt.time_base))
                pkt.stream = v_out
                out.mux(pkt)
            audio_until(math.inf)
            if a_out is not None:
                for pkt in a_out.encode(None):
                    out.mux(pkt)


class RecordingSink:
    """Wraps the speech output's sink: everything played also goes into the recording."""

    def __init__(self, inner, recorder: Recorder, clock):
        self.inner = inner
        self.recorder = recorder
        self.clock = clock

    def play(self, stereo: np.ndarray, sample_rate: int) -> bool:
        v = self.recorder.add_voice(stereo, sample_rate, self.clock.now())
        finished = self.inner.play(stereo, sample_rate)
        if not finished:
            self.recorder.cut_voice(v, self.clock.now())
        return finished

    def stop(self) -> None:
        self.inner.stop()

    def __getattr__(self, name):
        return getattr(self.inner, name)

