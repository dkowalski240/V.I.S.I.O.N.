"""Item context beyond 'on the table': named places, who you were with, Wi-Fi network."""

import sqlite3

import numpy as np
import pytest

from assistant.radio import RadioSnapshot, StaticRadio, parse_netsh, parse_nmcli
from assistant.storage import Storage

from rig import Rig

SARAH = 200
WALLET, TABLE, LAPTOP = (40, 90, 200), (60, 60, 60), (150, 150, 150)
ROOM_OBJECTS = [(TABLE, "table", (200, 450, 1100, 700)), (LAPTOP, "laptop", (700, 380, 950, 520))]
WALLET_ON_TABLE = ROOM_OBJECTS + [(WALLET, "wallet", (560, 470, 660, 520))]
KITCHEN, OFFICE = (30, 140, 70), (150, 60, 40)


def _learn_wallet(rig):
    rig.hear("remember this, it's my wallet", source="ptt")
    rig.run(rig.cfg.objects.enroll_delay_s + rig.cfg.objects.enroll_s + 1.0,
            objects=[(WALLET, "wallet", (560, 250, 720, 360))])
    assert rig.words()[-1] == "Saved your wallet."


def _learn_place(rig, name, colour, command=None):
    rig.background = colour
    rig.hear(command or f"this is the {name}", source="ptt")
    assert rig.words()[-1] == "Turn slowly all the way around. Say done to finish."
    rig.run(rig.cfg.places.enroll_delay_s + rig.cfg.places.enroll_s + 1.0)
    mine = command is not None and " my " in f" {command} "
    assert rig.words()[-1] == f"Saved {'your' if mine else 'the'} {name}."


def test_place_and_companion_in_the_answer(storage):
    rig = Rig(storage)
    rig.enroll_directly("Sarah", SARAH)
    _learn_place(rig, "kitchen", KITCHEN)
    _learn_wallet(rig)
    rig.background = KITCHEN
    rig.run(3.0, [(SARAH, 950, 100, 150)], WALLET_ON_TABLE)   # Sarah is there when it's put down
    rig.background = OFFICE
    rig.run(2.0)                                             # walk to another room
    rig.clock.advance(14 * 60)
    rig.hear("where's my wallet?", source="ptt")
    assert rig.words()[-1] == ("Your wallet was last seen 14 minutes ago, on the table in the kitchen, "
                               "next to the laptop. You were with Sarah.")


def test_where_am_i(storage):
    rig = Rig(storage)
    rig.hear("where am I?", source="ptt")
    assert rig.words()[-1] == "No places saved yet. Say: this is the kitchen."
    _learn_place(rig, "kitchen", KITCHEN)
    _learn_place(rig, "office", OFFICE, command="remember this place as my office")
    rig.background = KITCHEN
    rig.run(0.5)
    rig.hear("where am I?", source="ptt")
    assert rig.words()[-1] == "Probably the kitchen."
    rig.background = (10, 10, 200)  # somewhere new
    rig.run(0.5)
    rig.hear("what room is this", source="ptt")
    assert rig.words()[-1] == "I don't recognise this place."
    rig.hear("list places", source="ptt")
    assert rig.words()[-1] == "Saved places: kitchen and office."


def test_left_on_another_wifi_network(storage):
    radio = StaticRadio(RadioSnapshot(ssid="OSU Library", bssid="aa:bb:cc:dd:ee:01"))
    rig = Rig(storage, radio=radio)
    _learn_wallet(rig)
    rig.run(3.0, objects=WALLET_ON_TABLE)
    rig.run(2.0)
    radio.snap = RadioSnapshot(ssid="HomeNet", bssid="11:22:33:44:55:66")  # now at home
    rig.clock.advance(3 * 3600)
    rig.hear("where's my wallet", source="ptt")
    assert rig.words()[-1] == ("Your wallet was last seen 3 hours ago, on the table, next to the laptop. "
                               "That was on the OSU Library network.")


def test_wifi_tells_lookalike_rooms_in_different_buildings_apart(storage):
    radio = StaticRadio(RadioSnapshot(ssid="HomeNet", bssid="11:22:33:44:55:66"))
    rig = Rig(storage, radio=radio)
    _learn_place(rig, "kitchen", KITCHEN)
    radio.snap = RadioSnapshot(ssid="OSU Library", bssid="aa:bb:cc:dd:ee:01")
    rig.background = KITCHEN  # looks the same, but it's a different building
    rig.run(0.5)
    rig.hear("where am I", source="ptt")
    assert rig.words()[-1] == "I don't recognise this place."


def test_forget_everyone_and_reset_shortcut(storage):
    rig = Rig(storage)
    rig.enroll_directly("Sarah", SARAH)
    rig.enroll_directly("Mike", 120)
    rig.hear("forget everyone", source="ptt")
    assert rig.words()[-1] == "Forget all 2 people? Say yes."
    rig.hear("no", source="ptt")
    assert rig.words()[-1] == "Kept all 2 people." and len(storage.list_people()) == 2
    rig.hear("forget everyone", source="ptt")
    rig.hear("yes", source="ptt")
    assert rig.words()[-1] == "Forgot all 2 people." and storage.list_people() == []
    rig.enroll_directly("Sarah", SARAH)
    rig.hear("!reset-people", source="typed")
    assert rig.words()[-1] == "Forgot 1 person." and storage.list_people() == []
    rig.run(2.0, [(SARAH, 560, 200, 160)])
    assert rig.faces.view()[0]["state"] != "known"


def test_forget_a_place_and_everything(storage):
    rig = Rig(storage)
    _learn_place(rig, "kitchen", KITCHEN)
    rig.hear("forget the kitchen", source="ptt")
    assert rig.words()[-1] == "Forget the kitchen? Say yes."
    rig.hear("yes", source="ptt")
    assert storage.list_places() == []
    rig.enroll_directly("Sarah", SARAH)
    _learn_wallet(rig)
    rig.hear("forget everything", source="ptt")
    rig.hear("yes", source="ptt")
    assert storage.list_people() == [] and storage.list_items() == []


def test_netsh_and_nmcli_parsing():
    netsh = """
There is 1 interface on the system:

    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6E AX211 160MHz
    State                  : connected
    SSID                   : OSU Library
    AP BSSID               : AA:BB:CC:DD:EE:01
    Band                   : 5 GHz
    Signal                 : 86%
"""
    assert parse_netsh(netsh) == {"ssid": "OSU Library", "bssid": "aa:bb:cc:dd:ee:01", "signal": "86"}
    nm = "no:Other:11\\:22\\:33\\:44\\:55\\:66:40\nyes:HomeNet:AA\\:BB\\:CC\\:DD\\:EE\\:FF:70\n"
    assert parse_nmcli(nm) == {"ssid": "HomeNet", "bssid": "aa:bb:cc:dd:ee:ff", "signal": "70"}


def test_old_database_is_upgraded(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
                            created_at REAL NOT NULL);
        CREATE TABLE sightings (id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL, ts REAL NOT NULL,
            first_ts REAL NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, frame_w INTEGER, frame_h INTEGER, score REAL,
            snapshot_path TEXT, context TEXT NOT NULL DEFAULT '', camera_id TEXT NOT NULL DEFAULT '');
        INSERT INTO items (name, key, created_at) VALUES ('keys', 'keys', 1.0);
        INSERT INTO sightings (item_id, ts, first_ts, context) VALUES (1, 5.0, 5.0, 'on the desk');
    """)
    con.commit()
    con.close()
    st = Storage(str(db), str(tmp_path / "snaps"))
    s = st.last_sighting(1)
    assert s.context == "on the desk" and s.place == "" and s.companions == ""
    st.close()


def test_camera_rotate_for_sideways_phone_video(tmp_path):
    import cv2

    from assistant.camera import CameraSource
    from assistant.clock import Clock
    from assistant.config import CameraConfig

    path = str(tmp_path / "side.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(5):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    cam = CameraSource(CameraConfig(source=path, rotate=90), Clock()).start()
    f = cam.wait_for(0, timeout=3)
    cam.stop()
    assert f is not None and f.image.shape[:2] == (320, 240)


def test_phone_video_audio_track_is_used_for_replay(tmp_path):
    av = pytest.importorskip("av")
    from assistant.speech.audio_in import load_audio

    path = str(tmp_path / "clip.mp4")
    c = av.open(path, "w")
    a = c.add_stream("aac", rate=44100)
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.float32)
    for k in range(0, len(tone) - 1024, 1024):
        fr = av.AudioFrame.from_ndarray(tone[None, k:k + 1024], format="fltp", layout="mono")
        fr.sample_rate = 44100
        for pkt in a.encode(fr):
            c.mux(pkt)
    for pkt in a.encode():
        c.mux(pkt)
    c.close()
    audio = load_audio(path)
    assert 0.8 < len(audio) / 16000 < 1.2 and np.abs(audio).max() > 0.1


def test_a_place_remembers_every_network_it_was_saved_on(storage):
    radio = StaticRadio(RadioSnapshot(ssid="HomeNet", bssid="11:22:33:44:55:66"))
    rig = Rig(storage, radio=radio)
    _learn_place(rig, "kitchen", KITCHEN)
    radio.snap = RadioSnapshot(ssid="HomeNet-5G", bssid="11:22:33:44:55:67")
    _learn_place(rig, "kitchen", KITCHEN)   # saved again on the other band
    assert storage.find_place("kitchen")["wifi_ssid"] == "HomeNet|HomeNet-5G"
    rig.run(0.5)
    rig.hear("where am I", source="ptt")
    assert rig.words()[-1] == "Probably the kitchen."
