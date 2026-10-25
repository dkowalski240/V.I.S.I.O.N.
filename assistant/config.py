"""Configuration: typed dataclasses with defaults, loaded from one YAML file.

Every threshold, frame rate, model choice and the cloud toggle lives here.
`config.yaml` mirrors these defaults with comments (tests/test_config.py keeps
the two in sync), and the README explains each value.

Unknown keys in the YAML raise an error, so a typo like `t_mach: 0.7` can't be
silently ignored. (Pattern carried over from FaceLock's config.py.)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml


# --------------------------------------------------------------------------- camera
@dataclass
class CameraConfig:
    # Webcam index (0 = default camera), a camera name ("DroidCam": the first camera
    # whose name contains it), a video file path, or a stream URL (http://..., rtsp://...).
    # Video files are paced in real time (replay mode).
    source: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # auto = DirectShow for webcams on Windows (opens fast), FFmpeg for URLs.
    # Other values: dshow | msmf | ffmpeg | any
    backend: str = "auto"
    mjpg: bool = True          # ask the webcam for MJPG (needed for 720p30 on many webcams)
    # At start, read the camera for up to this long; if it sends no frames or a blank
    # green picture (virtual cameras like DroidCam), retry in the camera's default mode
    # and then via the other Windows camera system. 0 = don't check.
    probe_s: float = 2.0
    mirror: bool = False       # flip left/right if your driver delivers a mirrored image
    rotate: int = 0            # rotate frames clockwise by 0 / 90 / 180 / 270 (phone held sideways etc.)
    loop_video: bool = False   # replay: loop the file instead of stopping at the end
    camera_id: str = "wearable"  # tag on every event; a fixed home camera would use another id


@dataclass
class DirectionConfig:
    # zones: split the frame into equal columns, one clock hour each (left to right).
    #        Use this with a normal webcam (~70 deg), which can't see out to 10 o'clock.
    # angle: true angle from the lens field of view (use with a wide phone lens).
    mode: str = "zones"
    zones: Tuple[int, ...] = (10, 11, 12, 1, 2)
    hfov_deg: float = 70.0      # horizontal field of view, used by angle mode
    pan_strength: float = 1.0   # 0 = no stereo panning, 1 = full left/right at the edges
    # How directions are SAID (the stereo pan works the same either way):
    #   clock: "Sarah, 2 o'clock"     (orientation & mobility convention; precise, must be learned)
    #   words: "Sarah, to your right" (no learning needed; coarser)
    #   both:  "Sarah, 2 o'clock, to your right" (while learning the clock positions)
    # Also switchable by voice: "use clock directions" / "use left and right".
    style: str = "clock"


# --------------------------------------------------------------------------- faces
@dataclass
class FaceDetectionConfig:
    backend: str = "mtcnn"     # mtcnn (bundled with facenet-pytorch) | yunet (needs the .onnx)
    yunet_model_path: str = "models/face_detection_yunet_2023mar.onnx"
    scale: float = 0.75        # detect on a downscaled frame (faster; misses small faces)
    min_confidence: float = 0.85
    min_face_px: int = 40      # smallest face the detector looks for (full-res pixels)
    device: str = "cuda"


@dataclass
class AlignmentConfig:
    image_size: int = 160
    margin: int = 20


@dataclass
class QualityConfig:
    """Same fields as FaceLock's quality gate. Two copies are used: a lenient one
    for recognition (people at a distance, a bit turned) and a strict one for
    enrollment (only clean frames get saved)."""

    min_face_px: int = 48
    min_det_confidence: float = 0.90
    min_sharpness: float = 25.0
    max_roll_deg: float = 35.0
    max_yaw_ratio: float = 0.45
    pitch_ratio_range: Tuple[float, float] = (0.20, 0.85)
    brightness_range: Tuple[float, float] = (35.0, 220.0)
    max_saturated_fraction: float = 0.20
    edge_margin_frac: float = 0.02


def _strict_quality() -> QualityConfig:
    return QualityConfig(
        min_face_px=70,
        min_det_confidence=0.95,
        min_sharpness=35.0,
        max_roll_deg=30.0,
        max_yaw_ratio=0.35,
        pitch_ratio_range=(0.25, 0.80),
        brightness_range=(40.0, 215.0),
        max_saturated_fraction=0.15,
        edge_margin_frac=0.04,
    )


@dataclass
class EmbeddingConfig:
    pretrained: str = "vggface2"   # facenet-pytorch weights: vggface2 | casia-webface
    device: str = "cuda"
    fp16: bool = True              # half precision on CUDA (ignored on CPU)


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.20
    timeout_s: float = 1.0         # an encounter ends after this long without seeing the face
    max_samples: int = 8           # best embeddings kept per track (averaged for the decision)


@dataclass
class MatchingConfig:
    # Cosine similarities (1.0 = identical). Run scripts/calibrate_faces.py on
    # your camera and paste its suggestions. False positives are worse than
    # misses, so these lean conservative.
    t_match: float = 0.65    # averaged track embedding must reach this to be "known"
    t_frame: float = 0.60    # a single frame must reach this to count as a vote
    t_sure: float = 0.75     # at or above this we say "Sarah"; below it "I think that's Sarah"
    margin: float = 0.08     # best person must beat the runner-up by this much
    min_samples: int = 3     # good embeddings needed before deciding
    agree_frames: int = 3    # consecutive frames that must vote for the same person
    t_novel: float = 0.92    # enrollment keeps only frames less similar than this to each other

    def validate(self) -> None:
        if not (0 < self.t_frame <= self.t_match <= self.t_sure < 1):
            raise ValueError(
                f"faces.matching needs 0 < t_frame ({self.t_frame}) <= t_match ({self.t_match}) "
                f"<= t_sure ({self.t_sure}) < 1"
            )
        if not (0 < self.margin < 1):
            raise ValueError("faces.matching.margin must be between 0 and 1")
        if self.agree_frames < 1 or self.min_samples < 1:
            raise ValueError("faces.matching.agree_frames and min_samples must be >= 1")


@dataclass
class AnnounceConfig:
    cooldown_s: float = 60.0          # don't re-announce the same person within this window
    approach_cooldown_s: float = 45.0  # ...but an "approaching" announcement may repeat after this
    # Seen again within this many seconds = the same encounter (they looked down, turned
    # away, were briefly hidden): a new track of them is not announced.
    encounter_gap_s: float = 30.0
    approach_wait_s: float = 0.4      # after recognising, wait this long for an approach estimate
    upgrade_to_approaching: bool = True  # announce again (once) if a passer-by turns toward you
    max_age_s: float = 2.5            # drop a person announcement that couldn't be spoken in time


@dataclass
class ApproachConfig:
    window_s: float = 0.8        # look at this much face-size history
    min_points: int = 4          # need at least this many detections in the window
    growth_per_s: float = 0.30   # d ln(face height)/dt above this = approaching (~35%/s growth)
    settle_s: float = 0.3        # ignore a new track's first moments (box grows as a face turns back in)
    min_growth: float = 1.3      # ...and the face must have grown 1.3x over the last lookback_s
    lookback_s: float = 2.0
    max_face_frac: float = 0.30  # face taller than this share of the frame = already close, not "approaching"


@dataclass
class FacesConfig:
    enabled: bool = True
    max_fps: float = 15.0
    detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    enroll_quality: QualityConfig = field(default_factory=_strict_quality)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    announce: AnnounceConfig = field(default_factory=AnnounceConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)


@dataclass
class EnrollmentConfig:
    # Spoken into the user's headphones after an introduction; the user then
    # asks the person aloud and the person (or the user, with push-to-talk) answers.
    prompt: str = "Heard {name}. Ask to save?"
    consent_timeout_s: float = 12.0   # no yes/no within this = not saved
    capture_s: float = 4.0            # keep collecting views this long after "yes"
    min_exemplars: int = 3            # fewer good views than this = not saved
    max_exemplars: int = 10
    speaking_pad_s: float = 0.6       # widen the speech window when picking the speaker's face
    correction_window_s: float = 15.0  # "no, it's Sara" right after saving renames
    consent_for_manual: bool = True   # "save this person as X" also asks before saving
    # --- several people in view
    prompt_multi: str = "Heard {name}, {where}. Ask to save?"  # says WHICH face will be saved
    talk_min: float = 0.015    # mouth-motion level (debug window "talk") that counts as talking
    talk_ratio: float = 1.5    # the talker must move their mouth this much more than anyone else
    require_talking: bool = False  # several unknown faces, nobody clearly talking -> don't guess
    min_self_similarity: float = 0.5  # every captured view must look like the person being enrolled
    reacquire_s: float = 2.0   # face hidden (someone walked in front)? keep the session this long


# --------------------------------------------------------------------------- objects
@dataclass
class ObjectsConfig:
    enabled: bool = True
    detector: str = "owlv2"     # owlv2 | yolo_world (optional, needs ultralytics)
    owlv2_model: str = "google/owlv2-base-patch16-ensemble"
    yolo_world_model: str = "yolov8s-worldv2.pt"
    device: str = "cuda"
    fp16: bool = True
    scan_fps: float = 1.5       # background scans per second
    det_threshold: float = 0.12  # open-vocabulary detector score threshold
    max_candidates: int = 12    # crops embedded per scan (after merging duplicate boxes)
    embedder_model: str = "ViT-B-32"          # open_clip model name
    embedder_pretrained: str = "laion2b_s34b_b79k"
    t_item: float = 0.82        # crop must be this similar to an enrolled item's views
    item_margin: float = 0.02   # ...and beat the next enrolled item by this much
    agree_scans: int = 2        # seen in this many of the last `agree_window` scans before logging
    agree_window: int = 3
    enroll_delay_s: float = 1.5  # wait for the spoken prompt before capturing
    enroll_s: float = 4.0       # "remember this" captures for this long
    enroll_fps: float = 3.0
    enroll_max_exemplars: int = 8
    keep_sightings: int = 20    # per item
    sighting_interval_s: float = 10.0  # same place: refresh the latest sighting instead of adding rows
    in_view_s: float = 3.0      # seen this recently = "in view right now"
    snapshot_width: int = 640
    generic_queries: Tuple[str, ...] = (
        "keys", "wallet", "earbuds case", "headphones", "phone", "glasses",
        "water bottle", "bag", "remote control", "card",
    )
    context_queries: Tuple[str, ...] = (
        "table", "desk", "counter", "shelf", "bed", "couch", "chair", "floor",
        "laptop", "keyboard", "monitor", "backpack", "sink", "door", "plant", "lamp", "book",
    )
    surface_labels: Tuple[str, ...] = (
        "table", "desk", "counter", "shelf", "bed", "couch", "chair", "floor",
    )
    # Big, fixed things in a room. If the frame where an item was seen shows none (a
    # wallet on the floor, camera pointing down), the scans from context_window_s before
    # and after it are checked for one, and for the room: "on the floor of your bedroom,
    # near the bed". These words are added to the detector's queries automatically.
    landmark_labels: Tuple[str, ...] = (
        "bed", "couch", "desk", "table", "counter", "shelf", "door", "sink",
    )
    landmark_min_score: float = 0.2   # the detector must be at least this sure (it's 0.12 for items)
    context_window_s: float = 3.0     # look this long before the item was first seen and after it was last seen


@dataclass
class PlacesConfig:
    """Where an item was seen, beyond "on the table": named places and who you were with."""
    enabled: bool = True
    enroll_delay_s: float = 1.0   # "this is the kitchen": wait for the prompt, then...
    enroll_s: float = 12.0        # ...capture views for this long while the user turns around ("done" ends it early)
    max_views: int = 20           # views kept, spread over the whole turn
    landmark_bonus: float = 0.04  # scaled by how many of the landmarks in view now were seen when the room was saved
    t_place: float = 0.80         # scene similarity needed to call it a saved place
    margin: float = 0.02          # ...and beat the second-best place by this much
    wifi: bool = True             # read the laptop's Wi-Fi network (Windows 11: Location services must be on)
    wifi_bonus: float = 0.05      # same Wi-Fi access point as when the place was saved
    other_network_penalty: float = 0.20  # on none of the place's Wi-Fi networks: probably elsewhere
    bluetooth: bool = False       # scan nearby Bluetooth LE devices (pip install -r requirements-bluetooth.txt)
    ble_scan_s: float = 4.0
    ble_min_rssi: int = -80       # ignore devices weaker than this (dBm)
    ble_bonus: float = 0.05       # scaled by how many of the place's Bluetooth devices are seen now
    scan_interval_s: float = 20.0  # how often Wi-Fi / Bluetooth are re-read
    mention_people: bool = True   # "...next to the laptop. You were with Sarah."
    mention_network: bool = True  # "That was on the CampusNet network." when it isn't the current one


# --------------------------------------------------------------------------- speech
@dataclass
class SpeechInConfig:
    enabled: bool = True
    input_device: Optional[Union[int, str]] = None  # None = system default microphone
    ambient: bool = True        # always listen for introductions and yes/no answers
    ptt_key: str = "ctrl_r"     # hold to give a command (pynput key name or a single character)
    vad_threshold: float = 0.5
    min_speech_ms: int = 250
    end_silence_ms: int = 600
    pre_roll_ms: int = 300
    max_segment_s: float = 12.0
    ptt_tail_ms: int = 250      # keep recording this long after the key is released


@dataclass
class STTConfig:
    model: str = "small.en"     # faster-whisper size: tiny.en | base.en | small.en | medium.en
    device: str = "cuda"
    compute_type: str = "int8_float16"   # CPU: int8
    beam_size: int = 1
    language: str = "en"
    cpu_threads: int = 4
    no_speech_threshold: float = 0.6
    drop_hallucinations: bool = True     # drop "Thank you." etc. heard in background noise


@dataclass
class TTSConfig:
    engine: str = "piper"       # piper | pyttsx3 | print (no audio; for tests)
    piper_voice: str = "models/piper/en_US-lessac-medium.onnx"
    speed: float = 1.15         # >1 = faster speech
    volume: float = 0.9
    pyttsx3_rate: int = 190
    output_device: Optional[Union[int, str]] = None  # None = system default output
    tones: bool = True
    interrupt_for_approaching: bool = True  # cut off a lower-priority message for an approaching person
    answer_max_age_s: float = 30.0
    info_max_age_s: float = 8.0


# --------------------------------------------------------------------------- storage etc.
@dataclass
class CommandsConfig:
    confirm_forget: bool = True    # "forget Sarah" -> "Forget Sarah? Say yes." first
    confirm_timeout_s: float = 10.0


@dataclass
class StorageConfig:
    db_path: str = "data/assistant.db"
    snapshot_dir: str = "data/snapshots"
    log_dir: str = "data/logs"
    log_events: bool = True     # JSONL log of every bus event (no images, no embeddings)


@dataclass
class CloudConfig:
    # OFF by default: when on, "describe where my keys are" sends that item's
    # snapshot to a vision-language model in the cloud.
    vlm_enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 150
    image_max_side: int = 1024
    timeout_s: float = 20.0


@dataclass
class DebugConfig:
    window: bool = True
    window_name: str = "Visual assistant (q to quit)"
    show_unknown: bool = True   # draw boxes for unknown faces (never stored)
    # --record / record_path (e.g. data/demo.mp4): ONE video with the debug window, a
    # microphone and the assistant's voice (panned), in sync. See assistant/recorder.py.
    record_path: str = ""
    record_fps: float = 30.0
    record_audio: bool = True   # False = picture only
    # "assistant" = what the assistant hears; a device name ("DroidCam") or number =
    # record that microphone instead (the phone's audio); "none" = no microphone.
    record_mic: Union[int, str] = "assistant"
    record_mic_gain: float = 1.0
    record_voice_gain: float = 1.0
    record_mic_shift_ms: float = 0.0  # move the microphone track (negative = earlier) if speech and lips don't line up
    record_quality: int = 20    # H.264 quality: lower = sharper and bigger (18-28 is sensible)
    max_height: int = 720       # shrink the window (not the processing) for tall phone video
    console_input: bool = True  # type commands in the terminal (same as push-to-talk)


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    faces: FacesConfig = field(default_factory=FacesConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    places: PlacesConfig = field(default_factory=PlacesConfig)
    speech_in: SpeechInConfig = field(default_factory=SpeechInConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def validate(self) -> None:
        self.faces.matching.validate()
        if self.direction.mode not in ("zones", "angle"):
            raise ValueError("direction.mode must be 'zones' or 'angle'")
        if self.direction.style not in ("clock", "words", "both"):
            raise ValueError("direction.style must be 'clock', 'words' or 'both'")
        if self.direction.mode == "zones" and not self.direction.zones:
            raise ValueError("direction.zones must list at least one clock hour")
        if not (0 < self.objects.t_item < 1):
            raise ValueError("objects.t_item must be between 0 and 1")
        if self.objects.agree_scans > self.objects.agree_window:
            raise ValueError("objects.agree_scans can't exceed objects.agree_window")
        if self.camera.rotate not in (0, 90, 180, 270):
            raise ValueError("camera.rotate must be 0, 90, 180 or 270")
        if self.tts.engine not in ("piper", "pyttsx3", "print"):
            raise ValueError("tts.engine must be piper, pyttsx3 or print")
        if self.objects.detector not in ("owlv2", "yolo_world"):
            raise ValueError("objects.detector must be owlv2 or yolo_world")


def _merge(dc_instance: Any, overrides: dict, path: str = "") -> Any:
    """Return a copy of a dataclass with values from a (nested) dict applied."""
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected a mapping at '{path or 'root'}', got {type(overrides).__name__}")
    known = {f.name: f for f in dataclasses.fields(dc_instance)}
    updates = {}
    for key, value in overrides.items():
        full = f"{path}.{key}" if path else key
        if key not in known:
            raise ValueError(f"Unknown config key '{full}'. Valid keys here: {sorted(known)}")
        current = getattr(dc_instance, key)
        if dataclasses.is_dataclass(current):
            updates[key] = _merge(current, value or {}, full)
        elif isinstance(current, tuple) and isinstance(value, list):
            updates[key] = tuple(value)
        else:
            updates[key] = value
    return dataclasses.replace(dc_instance, **updates)


def load_config(path: str | Path | None) -> Config:
    """Load a YAML file on top of the defaults. `None` = all defaults."""
    cfg = Config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = _merge(cfg, data)
    cfg.validate()
    return cfg


def apply_cpu_fallback(cfg: Config) -> None:
    """Switch every model to the CPU (used by --cpu and when CUDA is missing)."""
    cfg.faces.detection.device = "cpu"
    cfg.faces.embedding.device = "cpu"
    cfg.objects.device = "cpu"
    cfg.stt.device = "cpu"
    cfg.stt.compute_type = "int8"
