"""Visual memory assistant for blind and low-vision users.

Modules (each runs on its own thread and talks over the event bus):

    camera.py          webcam index / video file / stream URL, newest frame wins
    faces/             recognise enrolled people, announce them with clock direction
    faces/enrollment   learn a person from "Hi, I'm Sarah" after they agree
    objects/           remember personal items and where they were last seen
    speech/            mic, VAD, push-to-talk, Whisper, text-to-speech output
    intents/           rule-based parsing and routing of what was heard
    bus.py             dict messages between modules (+ JSONL event log)
    storage.py         SQLite + snapshots, all local
    app.py             wiring + OpenCV debug window
"""

__version__ = "0.1.0"
