"""Voice activity detection and segmentation.

SileroVAD runs the small Silero v6 model that ships inside faster-whisper
(no download), one 32 ms block at a time on the CPU (~0.1 ms per block). It
carries the model's LSTM state between blocks, so it's a true streaming VAD.

Segmenter turns the block stream into utterances:

* push-to-talk held -> everything from key press (minus a little pre-roll)
  until `ptt_tail_ms` after release is ONE "ptt" segment (a command), whatever
  the VAD says.
* otherwise (if `speech_in.ambient`) -> VAD-based "ambient" segments: start after
  `min_speech_ms` of speech (with `pre_roll_ms` of audio before it), end after
  `end_silence_ms` of silence or at `max_segment_s`.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from ..config import SpeechInConfig
from .audio_in import BLOCK, SAMPLE_RATE

BLOCK_MS = 1000.0 * BLOCK / SAMPLE_RATE  # 32 ms


def silero_model_path() -> str:
    """The VAD model bundled in faster-whisper, found without importing it
    (importing faster_whisper would also load ctranslate2)."""
    import importlib.util

    spec = importlib.util.find_spec("faster_whisper")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("faster-whisper is not installed (it ships the Silero VAD model)")
    assets = os.path.join(list(spec.submodule_search_locations)[0], "assets")
    for name in ("silero_vad_v6.onnx", "silero_vad.onnx"):
        p = os.path.join(assets, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No Silero VAD model found in {assets}")


class SileroVAD:
    CONTEXT = 64

    def __init__(self, path: Optional[str] = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(path or silero_model_path(), providers=["CPUExecutionProvider"],
                                            sess_options=opts)
        names = [i.name for i in self.session.get_inputs()]
        if names[:3] != ["input", "h", "c"]:
            raise RuntimeError(f"Unexpected Silero VAD model inputs {names}")
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def prob(self, block: np.ndarray) -> float:
        x = np.asarray(block, dtype=np.float32).reshape(1, -1)
        inp = np.concatenate([self._ctx, x], axis=1)
        out, self._h, self._c = self.session.run(None, {"input": inp, "h": self._h, "c": self._c})
        self._ctx = x[:, -self.CONTEXT:]
        return float(np.asarray(out).ravel()[0])


class EnergyVAD:
    """Fallback when onnxruntime/faster-whisper aren't installed (and for tests)."""

    def __init__(self, threshold_rms: float = 0.02):
        self.threshold = threshold_rms

    def reset(self) -> None:
        pass

    def prob(self, block: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.asarray(block, dtype=np.float32) ** 2)))
        return 1.0 if rms >= self.threshold else 0.0


@dataclass
class Segment:
    audio: np.ndarray
    t_start: float
    t_end: float
    source: str  # ambient | ptt


class Segmenter:
    def __init__(self, cfg: SpeechInConfig, vad):
        self.cfg = cfg
        self.vad = vad
        self.mode = "idle"  # idle | speech | ptt
        # holds the pre-roll plus the speech run that triggers a segment
        self._pre: Deque = deque(maxlen=max(1, int((cfg.pre_roll_ms + cfg.min_speech_ms) / BLOCK_MS) + 1))
        self._blocks: List[np.ndarray] = []
        self._t_start = 0.0
        self._run = 0          # consecutive speech blocks while idle
        self._silence_ms = 0.0
        self._release_t: Optional[float] = None
        self.last_prob = 0.0

    def _begin(self, ts: float, include_pre: bool = True) -> None:
        pre = list(self._pre) if include_pre else []
        self._blocks = [b for b, _ in pre]
        self._t_start = pre[0][1] if pre else ts

    def _emit(self, t_end: float, source: str) -> Segment:
        audio = np.concatenate(self._blocks) if self._blocks else np.zeros(0, dtype=np.float32)
        seg = Segment(audio, self._t_start, t_end, source)
        self._blocks, self._run, self._silence_ms, self._release_t = [], 0, 0.0, None
        self._pre.clear()
        self.mode = "idle"
        return seg

    def feed(self, block: np.ndarray, ts: float, ptt_down: bool) -> List[Segment]:
        out: List[Segment] = []
        p = self.vad.prob(block)  # always run, so the VAD state stays continuous
        self.last_prob = p
        speech = p >= self.cfg.vad_threshold
        t_end = ts + BLOCK_MS / 1000.0

        # ---- push-to-talk takes over
        if ptt_down and self.mode != "ptt":
            if self.mode == "speech":
                pass  # keep what we have: the command may have started a moment early
            else:
                self._begin(ts)
            self.mode = "ptt"
            self._release_t = None
        if self.mode == "ptt":
            self._blocks.append(block)
            if not ptt_down:
                if self._release_t is None:
                    self._release_t = ts
                if (ts - self._release_t) * 1000 >= self.cfg.ptt_tail_ms:
                    out.append(self._emit(t_end, "ptt"))
            return out

        if not self.cfg.ambient:
            self._pre.append((block, ts))
            return out

        # ---- ambient VAD segmentation
        if self.mode == "idle":
            self._pre.append((block, ts))
            if speech:
                self._run += 1
                if self._run * BLOCK_MS >= self.cfg.min_speech_ms:
                    self._begin(ts)  # pre-roll already includes the speech run
                    self.mode = "speech"
                    self._silence_ms = 0.0
            else:
                self._run = 0
            return out

        # mode == speech
        self._blocks.append(block)
        self._silence_ms = 0.0 if speech else self._silence_ms + BLOCK_MS
        duration = len(self._blocks) * BLOCK / SAMPLE_RATE
        if self._silence_ms >= self.cfg.end_silence_ms or duration >= self.cfg.max_segment_s:
            out.append(self._emit(t_end, "ambient"))
        return out
