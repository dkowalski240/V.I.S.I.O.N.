# Visual memory assistant

A portable assistant for blind and low-vision users, built on the FaceLock
recognition pipeline. It runs fully offline on a laptop with an RTX GPU and
talks into headphones:

* **Face memory.** Learns people from their spoken introduction ("Hi, I'm Sarah")
  after they agree, then announces them with a clock direction, panned left or
  right: *"Sarah, approaching, 10 o'clock."*
* **Object memory.** "Remember this, it's my wallet." Later: "Where's my wallet?"
  *"Your wallet was last seen 14 minutes ago, on the table, next to the laptop."*

Unknown faces are never stored. Nothing leaves the laptop unless you turn on
the optional cloud description.

---

## Contents

1. [Windows setup](#1-windows-setup)
2. [Run it, milestone by milestone](#2-run-it-milestone-by-milestone)
3. [Live mode and replay mode](#3-live-mode-and-replay-mode) (incl. [testing with an iPhone](#testing-with-an-iphone) and [resetting who it knows](#resetting-who-it-knows))
4. [Running the demo](#4-running-the-demo)
5. [Voice commands](#5-voice-commands)
6. [Configuration reference](#6-configuration-reference)
7. [How it works](#7-how-it-works)
8. [GPU memory and speed](#8-gpu-memory-and-speed)
9. [Privacy](#9-privacy)
10. [Troubleshooting](#10-troubleshooting)
11. [Tests](#11-tests)
12. [Known limits and TODOs](#12-known-limits-and-todos)

---

## 1. Windows setup

Tested target: Windows 11, Python 3.13 (3.11 and 3.12 also work), NVIDIA RTX
4060 Laptop GPU. Open **Anaconda Prompt** (or any terminal where `python` is
3.11+) in this folder.

```bat
:: 1. A fresh virtual environment, separate from FaceLock's
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

:: 2. PyTorch WITH CUDA. The plain "pip install torch" gives a CPU-only build on Windows.
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

:: 3. facenet-pytorch WITHOUT its dependencies (it pins ancient torch/numpy; the code itself is fine)
pip install facenet-pytorch==2.6.0 --no-deps

:: 4. Everything else
pip install -r requirements-dev.txt

:: 5. Download all models now (~1.9 GB) while the Wi-Fi is fast
python scripts\download_models.py

:: 6. Check everything: packages, CUDA, each model's load time / speed / GPU memory, camera, audio
python scripts\check_setup.py --all

:: optional: Bluetooth room fingerprints for place context (then set places.bluetooth: true)
pip install -r requirements-bluetooth.txt
```

Why **cu126** and not the newer CUDA 13 builds: faster-whisper's engine
(ctranslate2 4.8) loads `cublas64_12.dll`, which ships inside the cu126 torch
build. The app imports torch first and adds `torch\lib` to the DLL search path,
so no separate CUDA toolkit install is needed. A CUDA 12.6 build needs NVIDIA
driver 560 or newer; update the driver from MSI Center / GeForce Experience if
`check_setup.py` says CUDA isn't available.

Your FaceLock `.venv` has `torch 2.14.1+cpu`, which is why this uses a new venv.

## 2. Run it, milestone by milestone

Each milestone runs end to end. Run the quick test first; it needs no camera or GPU.

### M1: event bus, module stubs, camera, replay, debug window

```bat
pytest tests\test_geometry_bus.py tests\test_storage_output.py tests\test_config.py
python -m assistant --stub
```

`--stub` loads no models. The window shows the webcam; the terminal prints
`SAY ... Ready.` Type `who's here` and press Enter: you should see
`Nobody in view.` Press **q** in the window to quit.

### M2: face recognition and spoken announcements

```bat
pytest tests\test_faces.py
python scripts\enroll_person.py --name Dan
python -m assistant --no-objects --no-speech
```

`enroll_person.py` captures your face for 5 s (look at the camera, turn your
head a little). Others may be in view: it locks onto the largest face (green
box) and ignores the rest; press **n** to switch faces. Step out of view, come back: you hear *"That's Dan, 12
o'clock"* in the headphones. Walk toward the camera from the left edge:
*"Dan, approaching, 10 o'clock"*, panned left.

### M3: enrollment by introduction, with consent

```bat
pytest tests\test_enrollment.py tests\test_parser.py tests\test_speech_in.py
python -m assistant --no-objects
```

A teammate stands in front of the camera and says **"Hi, I'm Sarah."** The
headphones say *"Heard Sarah. Ask to save?"* The user asks aloud ("Is it okay
if this remembers you?"), Sarah says **"Sure"** (or the user holds right Ctrl
and says "yes"). After ~4 s more of capture: *"Saved Sarah."* If the name was
misheard: "No, it's Sara" (during the question) or, after saving, hold
push-to-talk and say "No, her name is Sara".

No microphone handy? Type in the terminal: `!ambient Hi, I'm Sarah`, then `!yes`.

**Other people in view.** Enrollment picks *which* face said "Hi, I'm Sarah":
people it already knows are skipped, then the face whose mouth moved during the
speech wins (the debug window shows each face's `talk` value), else the
largest, most central face. With more than one face in view the prompt says
where: *"Heard Sarah, 2 o'clock. Ask to save?"*, so the user can check it's the
person in front of them. Only that face is captured, every captured view must
look like her, and if someone walks in front of her the session finds her
again (up to `enrollment.reacquire_s`).

### M4: object enrollment, sighting log, "where is my X?"

```bat
pytest tests\test_objects.py
python -m assistant
```

Hold **right Ctrl**, say **"Remember this, it's my wallet"**, release, then hold
the wallet up to the camera and turn it for 4 s (*"Saved your wallet."*). Put
it on the table in view, wait a few seconds, then hold Ctrl: **"Where's my wallet?"**

**If it learns an item but misses it later**, the debug window tells you which
step fails:

* **No box on the item at all** (not even a thin grey one): the detector isn't
  finding it. Usually a dark item on a dark surface, or too far away. Better
  light, a contrasting surface, closer, or a higher camera resolution helps.
* **A yellow `wallet? 0.78` box**: found and *almost* matched (`objects.t_item`
  is 0.82). The saved views don't look like how it sits now, typically because
  it was learned held in a hand and now lies flat on a table. Say "remember
  this, it's my wallet" again while pointing the camera at it lying where you
  usually leave it: the new views are *added* to the old ones. Lowering
  `t_item` to 0.78 also works, at some risk of naming a look-alike.
* **A grey box with a low number** (0.6x): the saved views are poor. The log line
  after "Saved your wallet" says how many frames the detector really found it
  in; centre-of-frame guesses mean it learned your hand and shirt. Forget it
  and learn it again closer up, against a plain background.

### M5: polish, the full demo

```bat
pytest
python -m assistant --record data\demo.mp4
```

`tests\test_demo_scenario.py` runs all six demo steps from the brief with fake
models. `--record` saves the demo video: see *Recording the demo video* below.

### Recording the demo video

`--record` writes ONE video with three things in sync: the debug window, a
microphone, and the assistant's own voice, panned left/right exactly as the
wearer hears it (headphones in or not: the voice is taken from the assistant,
not from the air).

```bat
:: the phone's audio (DroidCam) as the microphone track
python -m assistant --source DroidCam --record data\demo.mp4 --record-mic DroidCam

:: what the assistant itself hears (its own microphone)
python -m assistant --record data\demo.mp4
```

* Press **q** (or Esc / Ctrl+C) to stop. It then takes a few seconds to save
  `data\demo.mp4` (H.264 + AAC, plays in PowerPoint and every editor). Don't close
  the window while the log says "Saving the recording".
* **Phone audio:** turn DroidCam's audio on in its Windows client; Windows then
  has a microphone with "DroidCam" in its name (`check_setup.py --audio` lists
  them). `--record-mic DroidCam` (or `debug.record_mic: DroidCam`) records it.
  The assistant keeps listening to its own microphone (`speech_in.input_device`).
* **Lip sync:** if voices in the video are early or late against the picture,
  set `debug.record_mic_shift_ms` (e.g. -150 moves the phone audio 150 ms earlier).
* **Levels:** `record_mic_gain` / `record_voice_gain` (1.0 = as recorded).
* While running, only the picture goes to disk (`demo.video.mkv`, survives a
  crash); the sound is joined at the end. If that last step fails, the parts stay
  and the log prints the ffmpeg command that joins them.
* It uses NVIDIA's hardware video encoder when available (almost no CPU), else
  x264. A slow window loop repeats frames rather than speeding the video up, so
  the sound always stays in sync. 30 fps by default (`debug.record_fps`).
* Replays work too: `--replay data\sessions\demo1.mp4 --record data\demo.mp4`
  puts the recording's own sound in the video. A clean way to make the final cut:
  record raw with `record_session.py`, replay until you like the run.

## 3. Live mode and replay mode

**Live** (default): webcam `camera.source`, the default microphone, push-to-talk
on `speech_in.ptt_key`.

```bat
python -m assistant                     :: webcam 0
python -m assistant --source 1          :: another webcam
python -m assistant --source http://192.168.1.20:8080/video   :: an MJPEG / RTSP stream URL
python -m assistant --cpu               :: every model on the CPU (slow, but works anywhere)
```

**Replay** runs the whole pipeline on a recording instead of the camera and
mic, in real time, so you can debug without standing in front of the camera.

```bat
:: record video + mic + your push-to-talk presses
python scripts\record_session.py --name demo1

:: replay it (use a scratch database so replays don't touch your real one)
python -m assistant --replay data\sessions\demo1.mp4 --script data\sessions\demo1.script.txt --db data\replay.db
```

* `--audio FILE` picks the audio (default: the video's name with `.wav`).
* `--script FILE` holds push-to-talk windows and scripted speech:

  ```text
  # seconds      action    text
  12.0           ambient   Hi, I'm Sarah        # as if overheard
  15.5           ambient   Sure
  31.0           type      where's my wallet    # as if said with push-to-talk
  40.2-42.0      ptt                            # hold push-to-talk during this window
  ```
* A replay ends 2 s after the video ends (`--keep-open` to stay).
* Every run writes `data\logs\events-*.jsonl`: every heard sentence, decision and
  spoken message, one JSON object per line (no images, no embeddings).

### Testing with an iPhone

**Live, as a webcam (easiest).** Install **DroidCam** on the iPhone (App Store,
"DroidCam Webcam & OBS Camera") and the **DroidCam client for Windows** from
droidcam.app. Connect them over USB (private: nothing on the network) or Wi-Fi.
The client creates a virtual webcam called **DroidCam Video**, so the assistant
just sees another camera. Pick it by name, so its number doesn't matter:

```bat
pip install cv2-enumerate-cameras==1.4.0     :: already in requirements.txt
python scripts\check_setup.py --cameras      :: lists every camera by name, via both Windows camera systems
python -m assistant --source DroidCam
```

* `--cameras` tries each camera number through DirectShow *and* Media
  Foundation. If DroidCam only opens via Media Foundation, `--source DroidCam`
  handles that for you; with a number instead, also set `camera.backend: msmf`.
* **DroidCam isn't listed at all?** Its virtual webcam isn't installed or isn't
  running yet. Check that the Windows **Camera** app can show "DroidCam Video";
  if it can't, the problem is on the DroidCam side: exit the client (File >
  Exit), start it again, reconnect the phone, and restart the laptop if the
  webcam was just installed (DroidCam's own advice).
* The video is landscape. Set the resolution in the client's Video section;
  DroidCam recommends 1280x720. A wider phone lens is what makes true clock
  directions work: then set `direction.mode: angle` and `hfov_deg` to that
  lens's field of view.
* **Blank green picture** (while the DroidCam client itself shows your video):
  DroidCam's webcam output isn't synced with the client. Exit the client with
  **File > Exit** (closing the window leaves it running), start it again and
  reconnect the phone; if it's still green, restart the laptop. That's
  DroidCam's own fix. The assistant also retries by itself in the camera's
  default mode, at 1280x720, 640x480 and 1920x1080, and (with `--source
  DroidCam`) through the other Windows camera system, and keeps the first that
  gives a real picture. The console says which one. `check_setup.py --cameras`
  does the same and prints the result. If the Windows Camera app is green on
  DroidCam Video too, the fix is on DroidCam's side.
* If frames don't arrive from the virtual camera, set `camera.mjpg: false`,
  `width: 640`, `height: 480`.
* **Hackathon Wi-Fi often blocks phone-to-laptop traffic.** Use USB, or turn on
  the iPhone's Personal Hotspot and connect the laptop to it.
* Camo and iVCam work the same way (a virtual webcam; `--source Camo` / `--source iVCam`).
* Any app that serves an MJPEG or RTSP URL also works with `--source <url>`;
  the URL format is in that app's screen.
* Arrives sideways? `camera.rotate: 90` (or 270).

**Record on the phone, replay on the laptop (best for repeatable tests).** Film
the scenario with the iPhone camera app (Settings > Camera > Formats > *Most
Compatible* gives H.264, and 1080p is plenty), AirDrop or copy the file over,
then:

```bat
python -m assistant --replay IMG_0042.MOV --db data\replay.db --forget-people
```

The video's own sound track is used as the microphone (no separate .wav
needed), so introductions and "yes" work in the replay. Portrait video is
rotated upright automatically and shrunk for the window (`debug.max_height`).
Commands need push-to-talk, which a phone video doesn't have: add them to a
`--script` file (`31.0  type  where's my wallet`), or mark the moments you
spoke a command as `ptt` windows.

### Resetting who it knows

To test introductions over and over:

| how | does |
|---|---|
| say or type **"forget everyone"**, then **"yes"** | removes every enrolled person (also "forget all items", "forget all places", "forget everything") |
| type **`!reset-people`** in the terminal | removes every enrolled person at once, no confirmation |
| **`python -m assistant --forget-people`** | starts with nobody enrolled (`--forget-items`, `--forget-places` too) |
| `python scripts\manage.py list` | shows people, items (with last sighting) and places |
| `python scripts\manage.py rename wallet "black wallet"` | rename a person, item or place (assistant stopped) |
| `python scripts\manage.py forget Sarah` / `forget-people` / `reset` | edits the database while the assistant is stopped |
| `--db data\test.db` | a separate scratch database; delete the file to start over |

After forgetting someone they're treated as a stranger immediately (no
restart), so they can introduce themselves again right away.

## 4. Running the demo

* **Camera direction.** The camera is the user's eyes. Turn the laptop so the
  webcam faces the room, with the user behind or beside the screen. In the raw
  image, left is the camera's left; set `camera.mirror: true` only if your
  driver mirrors it.
* **Clock directions with a webcam.** A webcam sees ~70 degrees, so its edges are
  really only ~11 and 1 o'clock. `direction.mode: zones` (default) spreads the
  frame over 10, 11, 12, 1, 2 so step 3 can say "10 o'clock". With a wide phone
  lens later, switch to `mode: angle` and set `hfov_deg` (e.g. 110-120).
* **Step 1 (introduction).** Sarah should face the camera, about 1-2 m away.
  Other people may be in view; the one whose mouth moves while "Hi, I'm Sarah"
  is said is chosen, and the prompt then names her direction. For the most
  reliable demo, have her a bit closer or more central than bystanders anyway.
  The user asks her aloud after the headphone prompt; she answers "yes" or "sure".
* **Step 3 (approach).** She starts at the left edge, a few metres away, and walks
  toward the camera. Approach = her face growing ~35%/s and at least 1.3x overall,
  while still under 30% of the frame height (someone already close, like the person
  at the laptop, is never "approaching"). The
  announcement comes ~0.3-0.6 s after her face is clear.
* **Step 4 (stranger).** Anyone not enrolled shows as a grey "unknown" box and
  is never announced or stored.
* **Step 6 (forget).** "Forget Sarah" -> *"Forget Sarah? Say yes."* -> "yes".
* Practice with `record_session.py` + replay, and tune thresholds with
  `scripts\calibrate_faces.py` on the venue's lighting.
* Push-to-talk alternative: a Bluetooth selfie button usually sends
  "volume up"; set `speech_in.ptt_key: media_volume_up`.

## 5. Voice commands

Hold push-to-talk (right Ctrl) for commands. Ambient speech is only used for
introductions and yes/no answers, so normal conversation can't trigger a command.
Every command can also be typed in the terminal.

| Say | Does |
|---|---|
| who's here? | names and directions of everyone in view, plus a count of unknown people |
| where's my wallet? / where did I leave my keys? | last sighting: time, surface, room, landmark ("on the floor of your bedroom, near the bed"), or "in view at 12 o'clock" |
| where's my wallet in detail? / describe it | cloud description of the snapshot (only if `cloud.vlm_enabled`) |
| remember this, it's my keys / this is my wallet | learn an item held up to the camera (4 s) |
| save this person as Sarah | manual enrollment of the most central face (still asks first) |
| no, her name is Sara / rename Sierra to Sarah | fix a misheard name |
| rename my wallet to my black wallet / rename the study to my office / rename this room to the guest room | rename an item or a place; views, landmarks and past sightings carry over ("my" = "your office") |
| forget Sarah / forget my keys / forget the kitchen | delete a person, item or place (asks "Say yes." first) |
| forget everyone / forget all items / forget everything | delete them all (asks first) |
| this is my bedroom / this is the guest room / remember this place as my office | name the room you're in: turn slowly, up to 12 s ("my" = later called "your bedroom"); it reads back the landmarks it saw |
| done / that's it | end a room (or item) capture early, e.g. when you've turned all the way round |
| where am I? | "Probably the kitchen." |
| list places | the named places |
| list people / list items | what it knows |
| repeat that | say the last sentence again |
| quiet mode / quiet mode off | only answers are spoken; approaching people become a short panned tone |
| cancel | stop an enrollment or confirmation |
| use left and right / use clock directions / use both | how directions are said, from now on (until restart; set `direction.style` to keep it) |
| help | a short reminder |

Typed only: `!ambient <text>` pretends the text was overheard; `!yes`, `!no` answer a consent question;
`!reset-people` forgets everyone without asking.

Tones: short beep = push-to-talk is listening; low double tone = didn't
understand; rising two-note = saved; long low beep = capturing views.

## 6. Configuration reference

Everything lives in `config.yaml` (it lists every default; delete what you
don't change). Unknown keys are errors, so typos are caught.

### camera
| key | default | meaning |
|---|---|---|
| source | 0 | webcam index, camera name (`DroidCam`: first camera whose name contains it), video file, or stream URL |
| width, height, fps | 1280, 720, 30 | requested webcam mode |
| backend | auto | auto = DirectShow for webcam numbers on Windows, whichever system lists a named camera, FFmpeg for files and URLs; or dshow, msmf, ffmpeg, any |
| mjpg | true | request MJPG from the webcam (needed for 720p at 30 fps on many webcams) |
| probe_s | 2.0 | at start, read the camera this long; no frames or a blank green picture = retry in other modes (default mode, 1280x720, 640x480, 1920x1080) and, for a named camera, the other Windows camera system. 0 = don't check |
| mirror | false | flip left/right |
| rotate | 0 | rotate frames clockwise 0 / 90 / 180 / 270 (a phone stream that arrives sideways) |
| loop_video | false | replay: loop the file |
| camera_id | wearable | tag on every event; a future fixed home camera would get its own id |

### direction
| key | default | meaning |
|---|---|---|
| mode | zones | zones: equal columns, one clock hour each. angle: true angle from `hfov_deg` |
| style | clock | how it's said: clock ("2 o'clock"), words ("to your right"), both. Voice: "use left and right" / "use clock directions" / "use both" |
| zones | [10, 11, 12, 1, 2] | clock hours across the frame, left to right |
| hfov_deg | 70 | horizontal field of view (angle mode) |
| pan_strength | 1.0 | 0 = no stereo panning; 1 = full left/right at the edges |

### faces
| key | default | meaning |
|---|---|---|
| enabled | true | turn face memory off entirely |
| max_fps | 15 | recognition rate cap |
| detection.backend | mtcnn | mtcnn (bundled) or yunet (faster; `download_models.py --only yunet`) |
| detection.scale | 0.75 | detect on a downscaled frame; lower = faster but misses distant faces |
| detection.min_confidence | 0.85 | detector's own cut-off |
| detection.min_face_px | 40 | smallest face looked for (full-resolution pixels) |
| detection.device | cuda | cuda or cpu |
| alignment.image_size, margin | 160, 20 | FaceNet crop size and context margin (leave alone) |
| quality.* | lenient | gate for recognition: min_face_px 48, confidence 0.9, sharpness 25, roll 35 deg, yaw ratio 0.45, pitch ratio 0.2-0.85, brightness 35-220, saturation 0.2, edge margin 0.02 |
| enroll_quality.* | strict | gate for frames saved at enrollment (bigger, sharper, more frontal) |
| embedding.pretrained | vggface2 | FaceNet weights (vggface2 or casia-webface) |
| embedding.device, fp16 | cuda, true | half precision on the GPU |
| tracking.iou_threshold | 0.2 | box overlap needed to call it the same face |
| tracking.timeout_s | 1.0 | an encounter ends after this long unseen (its embeddings are dropped) |
| tracking.max_samples | 8 | best embeddings kept per encounter and averaged |
| matching.t_match | 0.65 | the encounter's averaged embedding must reach this to be "known" |
| matching.t_frame | 0.60 | a single frame must reach this to count as a vote |
| matching.t_sure | 0.75 | at or above: "That's Sarah"; below: "I think that's Sarah" |
| matching.margin | 0.08 | best person must beat the second-best by this much |
| matching.min_samples | 3 | good embeddings needed before deciding |
| matching.agree_frames | 3 | consecutive frames that must vote for the same person |
| matching.t_novel | 0.92 | enrollment keeps only views less similar than this to each other |
| announce.cooldown_s | 60 | don't re-announce the same person within this time |
| announce.encounter_gap_s | 30 | seen again within this time = the same encounter (they looked down or were briefly hidden): not announced again |
| announce.approach_cooldown_s | 45 | an "approaching" announcement may repeat after this |
| announce.approach_wait_s | 0.4 | after recognising someone, wait this long for an approach estimate |
| announce.upgrade_to_approaching | true | announce once more if someone already announced starts approaching |
| announce.max_age_s | 2.5 | drop a person announcement that couldn't be spoken within this time |
| approach.window_s | 0.8 | face-size history used for the approach estimate |
| approach.min_points | 4 | detections needed in that window |
| approach.growth_per_s | 0.30 | growth rate of ln(face height) that counts as approaching (~35%/s) |
| approach.min_growth, lookback_s | 1.3, 2.0 | ...and the face must have grown 1.3x over the last 2 s (leaning in doesn't count) |
| approach.settle_s | 0.3 | ignore a new track's first moments (a face turning back to the camera grows without anyone moving) |
| approach.max_face_frac | 0.30 | a face taller than 30% of the frame is already close: never "approaching" |

### enrollment
| key | default | meaning |
|---|---|---|
| prompt | "Heard {name}. Ask to save?" | headphone prompt after an introduction (name read back) |
| consent_timeout_s | 12 | no answer = nothing saved |
| capture_s | 4 | keep capturing views this long after "yes" |
| min_exemplars, max_exemplars | 3, 10 | views needed / kept |
| speaking_pad_s | 0.6 | widen the speech time window when picking the speaker's face |
| correction_window_s | 15 | "no, it's Sara" right after saving renames |
| consent_for_manual | true | "save this person as X" also asks before saving |
| prompt_multi | "Heard {name}, {where}. Ask to save?" | prompt used when several faces are in view ({where} = clock direction) |
| talk_min | 0.015 | mouth-motion level counted as talking (debug window shows `talk` per face; tune on site) |
| talk_ratio | 1.5 | the talker must move their mouth this many times more than anyone else |
| require_talking | false | several unknown faces and nobody clearly talking: don't guess (stay silent) |
| min_self_similarity | 0.5 | every captured view must be this similar to the person being enrolled |
| reacquire_s | 2.0 | if their face is hidden (someone walks in front), keep looking for them this long |

### objects
| key | default | meaning |
|---|---|---|
| enabled | true | turn object memory off |
| detector | owlv2 | owlv2, or yolo_world (optional, `pip install ultralytics`) |
| owlv2_model | google/owlv2-base-patch16-ensemble | Hugging Face model id |
| yolo_world_model | yolov8s-worldv2.pt | ultralytics weights (yolo_world only) |
| device, fp16 | cuda, true | |
| scan_fps | 1.5 | background scans per second |
| det_threshold | 0.12 | open-vocabulary detector score threshold |
| max_candidates | 12 | candidate crops embedded per scan; boxes overlapping a better one (IoU > 0.6, any label) are merged first |
| embedder_model, embedder_pretrained | ViT-B-32, laion2b_s34b_b79k | open_clip image model |
| t_item | 0.82 | crop must be this similar to an item's saved views |
| item_margin | 0.02 | and beat the next item by this much |
| agree_scans, agree_window | 2, 3 | seen in 2 of the last 3 scans before a sighting is logged |
| enroll_delay_s | 1.5 | wait for the "hold it up" prompt to finish before capturing |
| enroll_s, enroll_fps | 4, 3 | "remember this" capture length and rate |
| enroll_max_exemplars | 8 | views kept per item |
| keep_sightings | 20 | sightings kept per item (older rows and snapshots deleted) |
| sighting_interval_s | 10 | same place: refresh the latest sighting instead of adding rows |
| in_view_s | 3 | seen this recently = "in view right now" |
| snapshot_width | 640 | saved snapshot size |
| generic_queries | keys, wallet, ... | words used to find candidate items |
| context_queries | table, desk, laptop, ... | landmarks used for "on the table, next to the laptop" |
| surface_labels | table, desk, ... | which context words are surfaces ("on the ...") |
| landmark_labels | bed, couch, desk, table, counter, shelf, door, sink | big fixed things for "near the bed"; searched for automatically |
| landmark_min_score | 0.2 | how sure the detector must be about a landmark (items use 0.12) |
| context_window_s | 3.0 | no room / landmark in the item's frame: use the scans this long before it was first seen and after it was last seen |

### speech_in
| key | default | meaning |
|---|---|---|
| enabled | true | microphone + speech-to-text |
| input_device | null | default mic, or an index/name (`check_setup.py --audio` lists them) |
| ambient | true | always listen for introductions and yes/no answers |
| ptt_key | ctrl_r | push-to-talk key: a pynput name (ctrl_r, f8, space, media_volume_up...) or one character |
| vad_threshold | 0.5 | Silero speech probability threshold |
| min_speech_ms | 250 | speech needed to start a segment |
| end_silence_ms | 600 | silence that ends a segment |
| pre_roll_ms | 300 | audio kept from before speech started |
| max_segment_s | 12 | longest segment |
| ptt_tail_ms | 250 | keep recording this long after releasing the key |

### stt
| key | default | meaning |
|---|---|---|
| model | small.en | faster-whisper size; medium.en if names get misheard (+~1 GB GPU memory) |
| device, compute_type | cuda, int8_float16 | on CPU use int8 |
| beam_size | 1 | higher = slightly more accurate, slower |
| language | en | |
| cpu_threads | 4 | CPU mode only |
| no_speech_threshold | 0.6 | Whisper's silence detection |
| drop_hallucinations | true | drop "Thank you." etc. that Whisper invents from background noise (ambient only) |

### tts
| key | default | meaning |
|---|---|---|
| engine | piper | piper (neural, offline), pyttsx3 (Windows voices), print (no audio) |
| piper_voice | models/piper/en_US-lessac-medium.onnx | voice file (`download_models.py --only piper`) |
| speed, volume | 1.15, 0.9 | |
| pyttsx3_rate | 190 | words per minute for pyttsx3 |
| output_device | null | default output (your headphones) |
| tones | true | short tones for listening / saved / error |
| interrupt_for_approaching | true | cut off a lower-priority message for an approaching person (answers resume after) |
| answer_max_age_s, info_max_age_s | 30, 8 | drop answers / info that couldn't be spoken in time |

### places
Extra context for "where's my X?": the named room, who you were with, and the Wi-Fi network.
| key | default | meaning |
|---|---|---|
| enabled | true | |
| enroll_delay_s, enroll_s | 1.0, 12.0 | "this is my bedroom": wait for the prompt, then capture views and landmarks while the user turns (halfway: "Keep turning."; "done" ends it early) |
| max_views | 20 | views kept per place, spread evenly over the turn |
| landmark_bonus | 0.04 | room recognition: + this x the share of landmarks in view now that were seen when the room was learned |
| t_place, margin | 0.80, 0.02 | scene similarity needed to name a place, and its lead over the next place |
| wifi | true | read the laptop's Wi-Fi network (Windows 11 needs Location on, see below) |
| wifi_bonus | 0.05 | same access point as when the place was saved |
| other_network_penalty | 0.20 | on none of the networks the place was saved on: probably a different building |
| bluetooth | false | nearby Bluetooth LE devices as a room fingerprint (`pip install -r requirements-bluetooth.txt`) |
| ble_scan_s, ble_min_rssi, ble_bonus | 4, -80, 0.05 | scan length, weakest device used (dBm), weight |
| scan_interval_s | 20 | how often Wi-Fi / Bluetooth are re-read |
| mention_people | true | "...next to the laptop. You were with Sarah." |
| mention_network | true | "That was on the OSU Library network." (only when it isn't the current network) |

### commands, storage, cloud, debug
| key | default | meaning |
|---|---|---|
| commands.confirm_forget | true | "Forget Sarah? Say yes." first |
| commands.confirm_timeout_s | 10 | no answer = keep |
| storage.db_path | data/assistant.db | SQLite file (people, embeddings, items, sightings, places); older files are upgraded automatically |
| storage.snapshot_dir | data/snapshots | sighting snapshots (JPEG) |
| storage.log_dir, log_events | data/logs, true | JSONL event log per run |
| cloud.vlm_enabled | **false** | allow "in detail" questions to send that item's snapshot to the cloud |
| cloud.provider, model | anthropic, claude-haiku-5-5 | needs `pip install anthropic` and `ANTHROPIC_API_KEY` |
| cloud.max_tokens, image_max_side, timeout_s | 150, 1024, 20 | |
| debug.window | true | OpenCV window |
| debug.show_unknown | true | draw grey boxes for unknown faces (never stored) |
| debug.record_path, record_fps | "", 30 | save one video (debug window + microphone + voice); same as `--record` |
| debug.record_audio | true | false = picture only |
| debug.record_mic | assistant | `assistant` = what the assistant hears; a device name (`DroidCam`) or number; `none`. Also `--record-mic` |
| debug.record_mic_gain, record_voice_gain | 1.0, 1.0 | track levels in the recording |
| debug.record_mic_shift_ms | 0 | move the microphone track (negative = earlier) to fix lip sync |
| debug.record_quality | 20 | H.264 quality: lower = sharper and bigger (18-28) |
| debug.max_height | 720 | shrink the window (not the processing) for tall phone video |
| debug.console_input | true | type commands in the terminal |

## 7. How it works

```
 camera thread ──newest frame──┬──> faces (15 fps): detect → track → quality → align → FaceNet
 (webcam / file / URL)         │      → per-frame vote → 3 agreeing frames + averaged match
                               │      → approach rate (face growth) → person_seen
                               └──> objects (1.5 fps): OWLv2 (items + landmarks) → CLIP crops
                                      → match enrolled items → sighting log + snapshot → object_sighting
 mic → Silero VAD / push-to-talk → segments → Whisper (only when someone spoke) → speech_heard
                                              │
 router: speech_heard (commands, intros, answers), person_seen ──> say ──> speech output
          enrollment state machine (consent)                          priority queue, one voice,
                                                                      stereo pan, tones
```

* **Threads, one process.** PyTorch, ctranslate2 and onnxruntime release
  Python's GIL while computing, so a slow model never blocks the camera, and
  sharing the GPU in one process avoids Windows multiprocessing+CUDA pitfalls.
* **Event bus** (`assistant/bus.py`): small dict messages, as in the brief
  (`person_seen`, `object_sighting`, `speech_heard`, `intent`, `say`), each
  subscriber with its own bounded queue. Frames are not sent on the bus.
* **From FaceLock** (in `assistant/faces/`, imports adapted): detection,
  alignment, quality gate, embedding (+fp16), IoU tracking (+size history,
  votes), exemplar gallery, the open-set matcher and the calibration math. Not
  reused: the door pipeline (it auto-creates profiles for strangers), lock,
  permissions, liveness, web UI.
* **False positives are worse than misses:** a person is only named when 3
  consecutive frames each vote for them (t_frame + margin) AND the averaged
  embedding of the encounter passes t_match + margin.
* **Item context:** each sighting also stores the named place (whole-frame CLIP
  match against rooms the user named, nudged by Wi-Fi access point and
  Bluetooth devices), the known people in view, and the Wi-Fi network:
  *"Your keys were last seen 2 hours ago, on the counter in your kitchen. You
  were with Sarah."* If the network isn't the current one: *"That was on the
  OSU Library network."*
* **Context from the frames around a sighting.** A wallet thrown on the floor is
  often seen with the camera pointing down: just floor, no room, no landmark.
  So every scan also notes the room (if any are saved) and the landmarks in view
  (`objects.landmark_labels`: bed, couch, desk, door...), and keeps them for a
  few seconds. When a sighting's own frame lacks them, the scans up to
  `objects.context_window_s` (3 s) before the item was first seen and after it
  was last seen fill them in, closest in time first: *"on the floor of your
  bedroom, near the bed."* "Near" means *seen within those 3 seconds*, not a
  measured distance: one camera can't measure distance reliably (that would
  need a depth model, and a step-based distance would need calibrating per
  user). A "next to the laptop" from the item's own frame always wins.
* **Renaming** an item or place keeps everything it learned. An item's name is
  also one of the detector's search words, so a name the detector can't picture
  ("dad's gift") relies on the generic words ("wallet") to find it.
* **"Your bedroom" or "the library":** say "this is **my** bedroom" and it's
  "your bedroom" afterwards; "this is the library" stays "the library".
* **Learning a room** takes up to 12 s of turning slowly ("Keep turning." at
  halfway, "done" to stop early). It keeps up to 20 views spread over the whole
  turn and notes the landmarks it saw in at least 2 frames: *"Saved your
  bedroom. I saw a bed, a desk and a door."* Those landmarks then help
  recognise the room (a bed in view points to the bedroom, `landmark_bonus`).
  Didn't make it all the way round? Say "this is my bedroom" again facing the
  part you missed: views and landmarks are added to the room.
* **Which landmark is "near":** each scan offers only its most confident
  landmark (not the surface the item is on). Of those, the one seen closest in
  time to the item wins, measured from when the item was first seen (frames
  before) or last seen (frames after); one in the item's own frame beats both.
* **Directions: clock or words.** `direction.style: clock` says "Sarah, 2
  o'clock"; `words` says "Sarah, to your right"; `both` says both (handy while
  learning). Or say "use left and right" / "use clock directions" at any time.
  Clock positions are a standard orientation-and-mobility convention, and more
  precise, but they are *learned*: someone who never used an analogue clock may
  not know them, and guidance for sighted helpers is to ask first whether the
  person uses them. Hence the choice. Switching to clock directions by voice
  gives a one-line reminder ("12 o'clock is straight ahead, 3 is to your
  right..."). The stereo pan works the same in every style.
* **Future home camera:** `CameraSource` is generic and every event carries
  `camera_id`. A fixed camera would be a second `CameraSource` with its own
  faces/objects module instances sharing the same storage and bus.

Files:

```
assistant/
  __main__.py app.py        command line, wiring, debug window loop
  config.py                 every tunable value (mirrors config.yaml)
  bus.py events.py clock.py event bus, event names, real/replay/fake clocks
  camera.py                 camera sources
  geometry.py               clock directions, pan, approach rate, "14 minutes ago"
  storage.py                SQLite + snapshots
  debug_view.py inputs.py   OpenCV overlay; console + replay script input
  faces/                    FaceLock vision parts + module.py (recognition) + enrollment.py (consent)
                            + talking.py (mouth motion: who is speaking) + follow.py (lock onto one face)
  objects/                  detector.py (OWLv2), embedder.py (CLIP), module.py, context.py, places.py, vlm.py
  radio.py                  Wi-Fi network + Bluetooth devices (place context)
  speech/                   audio_in, vad (Silero), ptt, stt (Whisper), input loop, tts, output
  intents/                  parser.py (rules), router.py, phrases.py (every sentence it says)
scripts/                    check_setup, download_models, enroll_person, record_session, calibrate_faces, manage
tests/                      pytest suite with fake models; rig.py wires a full assistant on a fake clock
```

## 8. GPU memory and speed

Estimates for the RTX 4060 Laptop (8 GB); `scripts\check_setup.py --models`
measures the real numbers on your machine:

| part | model | precision | GPU memory |
|---|---|---|---|
| CUDA context + library workspaces | | | ~0.6 GB |
| face detection + embedding | MTCNN + FaceNet | fp32 / fp16 | ~0.2 GB |
| object detection (1.5 fps) | OWLv2 base-16 ensemble | fp16 | ~1.0-1.5 GB peak |
| item crops | open_clip ViT-B-32 (image tower) | fp16 | ~0.3 GB |
| speech-to-text (only when speaking) | faster-whisper small.en | int8_float16 | ~0.7-1.0 GB |
| VAD, text-to-speech | Silero, Piper | CPU | 0 |
| **total** | | | **~3-4 GB** |

Speed levers: `faces.detection.backend: yunet` (several times faster
detection), `faces.detection.scale: 0.5`, `objects.scan_fps: 1.0`,
`stt.model: base.en`.

## 9. Privacy

* Unknown faces: embeddings exist only in memory while the face is tracked and
  vanish ~1 s after it leaves. No face images are stored for anyone.
* People are saved only after an introduction and a yes. Captured views stay
  in memory until then and are dropped on "no" or no answer.
* "Forget Sarah" deletes her and all her embeddings; "forget my keys" deletes
  the item, its sightings and snapshots.
* Item snapshots (`data\snapshots`) can show other people. They stay local.
  Only with `cloud.vlm_enabled: true` AND an explicit "in detail" request is
  one snapshot sent to the cloud.
* Place context stays local too: the Wi-Fi network name and access-point id,
  and (only if `places.bluetooth` is on) the addresses of nearby Bluetooth
  devices, stored with saved places and sightings.
* `data\` holds biometric data; don't commit it or share the laptop's copy.

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `check_setup.py`: CUDA not available | CPU-only torch: rerun step 2 of setup (cu126 index URL). Update the NVIDIA driver. |
| Whisper error about `cublas64_12.dll` | you installed a cu130 torch: reinstall cu126. Or `pip install nvidia-cublas-cu12`, or set `stt.device: cpu`. |
| `open() got an unexpected keyword argument 'metadata_errors'` | PyAV 19 installed: `pip install av==18.1.0` |
| camera won't open / black frames | `check_setup.py --cameras`, then `--source N`; try `camera.backend: msmf`; close other apps using the webcam |
| low webcam fps | keep `camera.mjpg: true`; good light (webcams drop fps in the dark) |
| no speech output | `check_setup.py --audio`; set `tts.output_device`; `download_models.py --only piper`; or `tts.engine: pyttsx3` |
| push-to-talk does nothing | try `speech_in.ptt_key: f8`; some keyboards' right Ctrl is a Fn combo. Typed commands always work. |
| "I think that's ..." for everyone, or misses | run `scripts\calibrate_faces.py` (3-5+ people) and paste its thresholds |
| a learned item isn't found later | see "If it learns an item but misses it later" under M4: no box = detection, yellow `name? 0.7x` = re-learn it where it usually lies |
| a stranger gets named | raise `faces.matching.t_match` / `t_frame` and `margin`; calibrate |
| names misheard | `stt.model: medium.en`; names you've already enrolled are given to Whisper as hotwords |
| "approaching" for someone who isn't | raise `faces.approach.min_growth` (1.5) or lower `max_face_frac`; the debug window shows each face's growth rate |
| sightings never mention a place | name places first ("this is the kitchen"); `check_setup.py --radio` shows whether Wi-Fi is readable |
| no Wi-Fi name (Windows 11) | Settings > Privacy & security > Location: turn on Location services and "Let desktop apps access your location" |
| DroidCam / phone camera shows no frames | `camera.mjpg: false`, `width: 640`, `height: 480`; on hackathon Wi-Fi use USB or the iPhone's hotspot |
| DroidCam shows a blank green picture (the client shows video) | DroidCam client: File > Exit, start it again, reconnect the phone; still green: restart the laptop. The debug window says CAMERA: blank green picture while it lasts |
| `--cameras` shows only camera 0 while DroidCam streams | `--cameras` now also tries Media Foundation and prints names; run `--source DroidCam`. Not listed anywhere = DroidCam's virtual webcam isn't running: check the Windows Camera app, restart the DroidCam client, then the laptop |

## 11. Tests

```bat
pytest                        :: ~197 tests, a few seconds, no GPU/camera/mic needed
set ASSISTANT_MODEL_TESTS=1 && pytest tests\test_real_models.py   :: real weights (downloads them)
pytest tests\test_demo_scenario.py -v
```

The tests use fake detectors and embedders (FaceLock's trick: a grey square's
grey level encodes who it is), so the real tracking, matching, enrollment,
object logging, routing and speech-queue code all run unchanged.
`tests\rig.py` wires a complete assistant on a fake clock.

## 12. Known limits and TODOs

* Accuracy with the trained weights wasn't measured while building this (the
  build machine had no GPU and couldn't download the weights). The code paths
  were run for real on the CPU: MTCNN (bundled weights) on a sample photo, the
  FaceNet, OWLv2 and open_clip architectures with random weights, and a full
  threaded replay that said "approaching, 10 o'clock" through the real
  detector. On the laptop run `scripts\check_setup.py --models`, the optional
  real-weights tests (`set ASSISTANT_MODEL_TESTS=1` then
  `pytest tests\test_real_models.py`), then calibrate on site.
* TODO: YOLO-World backend is written but untested.
* TODO: wake word (stretch goal); push-to-talk only for now.
* TODO: phone streaming works (virtual webcam or URL) but hasn't been tuned
  (latency, reconnects). Use `direction.mode: angle` with the phone's real FOV.
* "Near the bed" from the frames around a sighting is a time-based guess: if
  you turn round within 3 s and face a bed across the room, it still says "near
  the bed". Lower `objects.context_window_s` (e.g. 1.5) if that happens.
* Place context: room recognition compares whole-frame CLIP embeddings, so two
  rooms that look alike can be confused; Wi-Fi tells buildings apart, Bluetooth
  (if on) helps between rooms. `t_place` is a starting guess: name a room, walk
  around, ask "where am I?" and adjust. Not tested on real hardware yet.
* Bluetooth can't find AirPods or AirTags themselves: they change their address
  every few minutes for privacy, and Find My is Apple-only. It's used only as a
  fingerprint of fixed devices in a room.
* GPS isn't used: the laptop has none and it doesn't work indoors, where items
  get lost. TODO if outdoor places matter: an iPhone Shortcut that sends the
  phone's location to the laptop.
* CLIP can confuse two very similar items (two black wallets). The margin
  between enrolled items helps; DINOv2 embeddings would be stronger.
* Picking the speaker by mouth motion is a heuristic (smiling or chewing also
  moves the mouth), so it's only trusted when one face clearly moves more than
  the others; otherwise the largest, most central unknown face is used and the
  prompt's clock direction lets the user catch a wrong pick. Tune `talk_min` /
  `talk_ratio` with the `talk` values in the debug window. If the user
  introduces themselves, nobody in view is talking, so with
  `require_talking: true` nothing is started.
