"""Wires every module together and runs the debug window on the main thread.

Threads (all daemon threads in one process; the heavy libraries release
Python's GIL while they compute, so a slow model never blocks the camera):

    camera       newest frame wins
    faces        detect/track/embed/match at up to faces.max_fps
    objects      OWLv2 + CLIP scans at objects.scan_fps
    speech-in    mic -> VAD/push-to-talk segments
    stt          Whisper on finished segments only
    router       speech_heard / person_seen -> actions + say
    speech-out   one priority queue, one voice
    main         OpenCV debug window (HighGUI must live on the main thread on Windows)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from . import events as ev
from .bus import Bus
from .camera import CameraSource
from .clock import Clock, ReplayClock
from .config import Config, apply_cpu_fallback
from .debug_view import draw, placeholder
from .faces.enrollment import Enrollment
from .faces.module import FacesModule
from .inputs import ConsoleInput, ScriptPlayer, parse_script
from .intents.router import Router
from .objects.module import ObjectsModule
from .speech.output import SpeechOutput, make_sink
from .speech.tts import PrintTTS, make_tts
from .storage import Storage

log = logging.getLogger("assistant")


@dataclass
class RunOptions:
    stub: bool = False           # no models: camera + window + bus + speech output only (M1 check)
    replay_video: Optional[str] = None
    replay_audio: Optional[str] = None
    script: Optional[str] = None
    window: bool = True
    record_path: str = ""
    tts_engine: Optional[str] = None
    exit_when_done: bool = True  # replay: quit after the video ends and speech finishes


@dataclass
class Models:
    face_detector: object = None
    face_embedder: object = None
    obj_detector: object = None
    obj_embedder: object = None
    stt: object = None
    vad: object = None
    notes: list = field(default_factory=list)


class VRAM:
    """GPU memory as the driver sees it (includes ctranslate2, not only PyTorch)."""

    def __init__(self):
        self.torch = None
        try:
            import torch

            if torch.cuda.is_available():
                self.torch = torch
        except Exception:
            pass

    def used_gb(self) -> Optional[tuple]:
        if self.torch is None:
            return None
        free, total = self.torch.cuda.mem_get_info()
        return (total - free) / 1e9, total / 1e9

    def report(self, label: str) -> str:
        u = self.used_gb()
        msg = f"{label}: GPU memory in use {u[0]:.2f} / {u[1]:.1f} GB" if u else f"{label}: (no CUDA)"
        log.info(msg)
        return msg


def load_models(cfg: Config, stub: bool) -> Models:
    m = Models()
    if stub:
        return m
    uses_cuda = "cuda" in (cfg.faces.detection.device, cfg.faces.embedding.device, cfg.objects.device, cfg.stt.device)
    if uses_cuda:
        try:
            import torch  # first: on Windows this also puts its CUDA DLLs in the process

            ok = torch.cuda.is_available()
        except ImportError:
            ok = False
        if not ok:
            log.warning("CUDA is not available (CPU-only PyTorch?). Running every model on the CPU. "
                        "See README 'Install' for the CUDA build.")
            apply_cpu_fallback(cfg)
    vram = VRAM()
    vram.report("start")
    if cfg.faces.enabled:
        from .faces.detection import create_detector
        from .faces.embedding import FaceNetEmbedder

        m.face_detector = create_detector(cfg.faces.detection)
        m.face_embedder = FaceNetEmbedder(cfg.faces.embedding)
        m.notes.append(vram.report("faces loaded"))
    if cfg.objects.enabled:
        from .objects.detector import OWLv2Detector, YoloWorldDetector
        from .objects.embedder import ClipEmbedder

        o = cfg.objects
        if o.detector == "yolo_world":
            m.obj_detector = YoloWorldDetector(o.yolo_world_model, o.device, o.fp16, o.det_threshold)
        else:
            m.obj_detector = OWLv2Detector(o.owlv2_model, o.device, o.fp16, o.det_threshold)
        m.obj_embedder = ClipEmbedder(o.embedder_model, o.embedder_pretrained, o.device, o.fp16)
        m.notes.append(vram.report("objects loaded"))
    if cfg.speech_in.enabled:
        from .speech.stt import WhisperSTT

        try:
            m.stt = WhisperSTT(cfg.stt)
        except Exception as e:
            if cfg.stt.device != "cuda":
                raise
            log.warning("Whisper on CUDA failed (%s); falling back to CPU int8", e)
            cfg.stt.device, cfg.stt.compute_type = "cpu", "int8"
            m.stt = WhisperSTT(cfg.stt)
        m.notes.append(vram.report("speech-to-text loaded"))
    return m


def make_vad():
    from .speech.vad import EnergyVAD, SileroVAD

    try:
        return SileroVAD()
    except Exception as e:
        log.warning("Silero VAD unavailable (%s); using a simple energy VAD", e)
        return EnergyVAD()


class App:
    def __init__(self, cfg: Config, opts: RunOptions):
        self.cfg = cfg
        self.opts = opts
        replay = opts.replay_video is not None
        self.clock = ReplayClock() if replay else Clock()
        if replay:
            cfg.camera.source = opts.replay_video

        log_path = None
        if cfg.storage.log_events:
            os.makedirs(cfg.storage.log_dir, exist_ok=True)
            log_path = os.path.join(cfg.storage.log_dir, time.strftime("events-%Y%m%d-%H%M%S.jsonl"))
        self.bus = Bus(self.clock, log_path)
        self.storage = Storage(cfg.storage.db_path, cfg.storage.snapshot_dir)
        self.camera = CameraSource(cfg.camera, self.clock)

        self.models = load_models(cfg, opts.stub)

        engine = "print" if opts.stub and opts.tts_engine is None else (opts.tts_engine or cfg.tts.engine)
        tts_cfg = cfg.tts
        if engine != cfg.tts.engine:
            from dataclasses import replace

            tts_cfg = replace(cfg.tts, engine=engine)
        tts = make_tts(tts_cfg) if engine != "print" else PrintTTS(tts_cfg)
        self.output = SpeechOutput(cfg.tts, self.bus, self.clock, tts=tts, sink=make_sink(cfg.tts, tts.name))
        log.info("Speech output: %s", tts.name)

        # --record: debug window + microphone + the assistant's voice in one video (recorder.py)
        self.recorder = None
        self._record_device = None
        record = opts.record_path or cfg.debug.record_path
        if record:
            from .recorder import Recorder, RecordingSink

            self.recorder = Recorder(record, cfg.debug, self.clock)
            self.output.sink = RecordingSink(self.output.sink, self.recorder, self.clock)

        self.faces = None
        self.enrollment = None
        if cfg.faces.enabled:
            self.faces = FacesModule(cfg, self.camera, self.bus, self.storage, self.clock,
                                     self.models.face_detector, self.models.face_embedder)
            self.enrollment = Enrollment(cfg, self.bus, self.storage, self.faces, self.clock)
            self.faces.enrollment = self.enrollment

        self.radio = None
        if cfg.places.enabled and (cfg.places.wifi or cfg.places.bluetooth) and not opts.stub:
            from .radio import RadioScanner

            self.radio = RadioScanner(cfg.places)
        self.objects = None
        if cfg.objects.enabled:
            self.objects = ObjectsModule(cfg, self.camera, self.bus, self.storage, self.clock,
                                         self.models.obj_detector, self.models.obj_embedder, self.faces,
                                         radio=self.radio)

        self.vlm = None
        if cfg.cloud.vlm_enabled:
            from .objects.vlm import CloudVLM

            self.vlm = CloudVLM(cfg.cloud)
            log.warning("Cloud descriptions are ON: asking for detail sends a snapshot off this computer.")

        self.router = Router(cfg, self.bus, self.storage, self.clock, faces=self.faces, objects=self.objects,
                             enrollment=self.enrollment, output=self.output, vlm=self.vlm)

        self.script = None
        if opts.script:
            with open(opts.script, "r", encoding="utf-8") as fh:
                self.script = ScriptPlayer(parse_script(fh.read()), self.bus, self.clock)

        self.speech_in = self._make_speech_in(replay)
        self.console = ConsoleInput(self.bus, self.clock) if cfg.debug.console_input else None
        self._quit_at = None
        if self.recorder is not None and cfg.debug.record_audio:
            self._wire_record_mic(replay)

    def _wire_record_mic(self, replay: bool) -> None:
        mic = self.cfg.debug.record_mic
        name = str(mic).strip().lower() if mic is not None else "none"
        if name in ("none", "off", "", "false"):
            return
        source = getattr(self.speech_in, "source", None)
        if replay or name == "assistant":
            # What the assistant hears: the live microphone, or the replayed recording's sound.
            if source is not None and hasattr(source, "taps"):
                source.taps.append(self.recorder.add_mic)
            else:
                log.info("Recording without a microphone: speech input is off "
                         "(set debug.record_mic to a device name, e.g. DroidCam)")
            return
        self._record_device = mic  # a microphone of its own, opened in start()

    def _make_speech_in(self, replay: bool):
        if not self.cfg.speech_in.enabled or self.models.stt is None:
            return None
        from .speech.audio_in import FileAudioSource, MicSource
        from .speech.input import SpeechInput
        from .speech.ptt import KeyboardPTT, NoPTT, ScriptedPTT
        from .speech.stt import STTWorker

        if replay:
            audio = self.opts.replay_audio
            if audio is None:
                # record_session.py writes NAME.wav next to NAME.mp4; a phone video carries
                # its own audio track, which works just as well.
                stem = os.path.splitext(self.opts.replay_video)[0]
                audio = stem + ".wav" if os.path.exists(stem + ".wav") else self.opts.replay_video
            try:
                source = FileAudioSource(audio, self.clock)
            except Exception as e:
                log.info("Replay without audio (%s): speech input off (scripted speech still works)", e)
                return None
        else:
            source = MicSource(self.cfg.speech_in.input_device, self.clock)

        if self.script is not None:
            ptt = ScriptedPTT(self.script)
        elif replay:
            ptt = NoPTT()
        else:
            try:
                ptt = KeyboardPTT(self.cfg.speech_in.ptt_key, self.bus, self.clock)
            except Exception as e:
                log.warning("Push-to-talk unavailable (%s); type commands in the console instead", e)
                ptt = NoPTT()

        def hotwords() -> str:
            names = [p["name"] for p in self.storage.list_people()] + [i["name"] for i in self.storage.list_items()]
            return " ".join(names[:30])

        worker = STTWorker(self.cfg.stt, self.bus, self.models.stt, hotwords)
        return SpeechInput(self.cfg.speech_in, source, make_vad(), ptt, worker)

    # ------------------------------------------------------------------ run
    def start(self) -> None:
        self.output.start()
        if self.recorder is not None:
            self.recorder.start()
            if self._record_device is not None:
                try:
                    self.recorder.open_mic_device(self._record_device)
                except Exception as e:
                    log.warning("Recording without the %r microphone: %s", self._record_device, e)
        self.camera.start()
        for m in (self.radio, self.faces, self.objects, self.router, self.speech_in, self.script, self.console):
            if m is not None:
                try:
                    m.start()
                except Exception as e:
                    log.error("Could not start %s: %s", type(m).__name__, e)
                    if m is self.speech_in:
                        self.speech_in = None
        self.bus.publish(ev.SAY, text="Ready.", priority=ev.PRI_INFO, kind=ev.KIND_INFO)

    def status_lines(self) -> list:
        lines = []
        f = self.faces.stats if self.faces else {}
        o = self.objects.stats if self.objects else {}
        vram = VRAM().used_gb() if not self.opts.stub else None
        parts = [f"cam {self.camera.fps:.0f}fps"]
        if self.faces:
            parts.append(f"faces {f.get('fps', 0):.0f}fps {f.get('ms', 0):.0f}ms")
        if self.objects:
            parts.append(f"objects {o.get('ms', 0):.0f}ms/scan")
            if getattr(self.objects, "room", ""):
                parts.append(f"room: {self.objects.room}")
        if vram:
            parts.append(f"GPU {vram[0]:.1f}/{vram[1]:.0f}GB")
        if self.output.quiet:
            parts.append("QUIET")
        lines.append(" | ".join(parts))
        if getattr(self.camera, "problem", ""):
            lines.append("CAMERA: blank green picture. DroidCam: File > Exit the client, start it again"
                         " (restart the laptop if still green)")
        if self.speech_in is not None:
            lvl = min(10, int(self.speech_in.level * 100))
            ptt = self.speech_in.ptt.is_down(self.clock.now())
            lines.append(f"mic [{'#' * lvl}{'.' * (10 - lvl)}] vad {self.speech_in.vad_prob:.2f} "
                         f"{self.speech_in.mode}{'  PUSH-TO-TALK' if ptt else ''}"
                         f"{'  transcribing' if self.speech_in.stt_worker.busy else ''}")
        heard = self.router.last_heard
        if heard:
            lines.append(f"heard ({heard.get('source')}): {heard.get('text')}")
        said = self.output.history[-1] if self.output.history else None
        if said and said.text:
            lines.append(f"said: {said.text}")
        if self.enrollment and self.enrollment.status():
            s = self.enrollment.status()
            lines.append(f"enrolling {s['name']}: {s['state']}, {s['views']} views"
                         f"{', ' + str(s['rejected']) + ' rejected' if s['rejected'] else ''}, {s['remaining']}s left")
        if self.objects and self.objects.enrolling():
            lines.append(f"learning item: {self.objects.enrolling()}")
        return lines

    def run(self) -> None:
        self.start()
        window = self.opts.window and self.cfg.debug.window
        record = self.recorder is not None
        ended_at = None
        try:
            while True:
                frame = self.camera.latest()
                if window or record:
                    if frame is None:
                        img = placeholder("Waiting for camera...")
                    else:
                        enr = self.enrollment.session.track_id if self.enrollment and self.enrollment.session else None
                        img = draw(frame.image, self.faces.view() if self.faces else [],
                                   self.objects.view() if self.objects else [], self.status_lines(), enr,
                                   self.cfg.debug.show_unknown, self.cfg.faces.matching.t_sure,
                                   self.cfg.enrollment.talk_min)
                    if record and frame is not None:
                        self.recorder.add_frame(img)  # the recorder keeps its own constant frame rate
                    if window:
                        mh = self.cfg.debug.max_height
                        show = img if not mh or img.shape[0] <= mh else cv2.resize(
                            img, (int(img.shape[1] * mh / img.shape[0]), mh), interpolation=cv2.INTER_AREA)
                        cv2.imshow(self.cfg.debug.window_name, show)
                        key = cv2.waitKey(15) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    else:
                        time.sleep(0.03)
                else:
                    time.sleep(0.05)
                if self.camera.ended and self.opts.exit_when_done:
                    ended_at = ended_at or time.monotonic()
                    if time.monotonic() - ended_at > 2.0 and self.output.idle():
                        log.info("Replay finished")
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self._quit_at = self.clock.now()
            self.shutdown()

    def shutdown(self) -> None:
        log.info("Shutting down...")
        for m in (self.script, self.speech_in, self.router, self.objects, self.faces, self.radio):
            if m is not None:
                try:
                    m.stop()
                except Exception:
                    log.exception("Error stopping %s", type(m).__name__)
        self.camera.stop()
        self.output.stop()
        if self.recorder is not None:
            log.info("Saving the recording; this takes a few seconds, don't close the window...")
            try:
                self.recorder.finish(self._quit_at)
            except Exception:
                log.exception("Saving the recording failed")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.storage.close()
        self.bus.close()
