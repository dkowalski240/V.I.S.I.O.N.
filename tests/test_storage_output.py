import os

import numpy as np

from assistant import events as ev
from assistant.bus import Bus
from assistant.clock import FakeClock
from assistant.config import TTSConfig
from assistant.speech.output import NullSink, SpeechOutput
from assistant.speech.tts import PrintTTS
from assistant.storage import item_key

from conftest import random_identity


def test_item_keys_are_forgiving():
    assert item_key("my keys") == item_key("Keys") == item_key("the key") == "keys"
    assert item_key("AirPods") == item_key("air pods") == item_key("earbuds") == "airpods"
    assert item_key("wallets") == item_key("wallet")
    assert item_key("glasses") == "glasses"


def test_people_and_items_roundtrip(storage, rng):
    pid = storage.add_person("Sarah", [random_identity(rng) for _ in range(4)], [0.9] * 4, now=1.0)
    assert storage.find_person("sarah")["id"] == pid
    gal = list(storage.face_gallery())
    assert gal[0][0] == pid and gal[0][2].shape == (4, 512)
    storage.rename_person(pid, "Sara")
    assert storage.find_person("Sara") is not None
    storage.delete_person(pid)
    assert list(storage.face_gallery()) == []

    iid = storage.add_item("my wallet", [random_identity(rng)], now=1.0)
    assert storage.find_item("wallet")["name"] == "wallet"
    assert storage.add_item("Wallet", [random_identity(rng)]) == iid  # same item, more views
    assert list(storage.item_gallery())[0][2].shape == (2, 512)


def test_sightings_keep_last_n_and_delete_snapshots(storage, rng, tmp_path):
    iid = storage.add_item("keys", [random_identity(rng)])
    paths = []
    for i in range(5):
        p = tmp_path / f"s{i}.jpg"
        p.write_bytes(b"x")
        paths.append(str(p))
        storage.add_sighting(iid, 100.0 + i, (0, 0, 10, 10), (640, 480), 0.9, str(p), "on the table", "cam", keep=3)
    assert [s.ts for s in storage.sightings(iid)] == [104.0, 103.0, 102.0]
    assert not os.path.exists(paths[0]) and os.path.exists(paths[4])
    last = storage.last_sighting(iid)
    storage.refresh_sighting(last.id, 200.0, (1, 1, 11, 11), 0.95)
    assert storage.last_sighting(iid).ts == 200.0
    storage.delete_item(iid)
    assert not any(os.path.exists(p) for p in paths)


def _out(**kw):
    clock = FakeClock()
    bus = Bus(clock)
    out = SpeechOutput(TTSConfig(engine="print", **kw), bus, clock, tts=PrintTTS(), sink=NullSink())
    return out, clock


def _play_all(out):
    while out.process_one():
        pass
    return [s.text for s in out.history]


def test_priority_order_and_stale_messages_dropped():
    out, clock = _out()
    out.say("info", ev.PRI_INFO)
    out.say("Sarah, 2 o'clock", ev.PRI_PERSON, max_age_s=2.5)
    out.say("answer", ev.PRI_ANSWER)
    out.say("Mike, approaching, 10 o'clock", ev.PRI_APPROACH, kind=ev.KIND_APPROACH)
    assert _play_all(out) == ["Mike, approaching, 10 o'clock", "answer", "Sarah, 2 o'clock", "info"]

    out.say("old news", ev.PRI_PERSON, max_age_s=2.5)
    clock.advance(5)
    assert _play_all(out)[-1] == "info"  # "old news" was dropped


def test_quiet_mode_keeps_answers_and_turns_approach_into_a_tone():
    out, _ = _out()
    out.set_quiet(True)
    out.say("Sarah, 2 o'clock", ev.PRI_PERSON)
    out.say("Mike, approaching, 10 o'clock", ev.PRI_APPROACH, pan=-1.0, kind=ev.KIND_APPROACH)
    out.say("Your keys were last seen just now.", ev.PRI_ANSWER)
    while out.process_one():
        pass
    spoken = list(out.history)
    assert [s.text for s in spoken] == ["", "Your keys were last seen just now."]
    assert spoken[0].tone == ev.TONE_PERSON and spoken[0].pan == -1.0


def test_pan_puts_left_people_in_the_left_ear():
    out, _ = _out()
    out.say("x", ev.PRI_PERSON, pan=-1.0, tone=ev.TONE_PERSON)
    out.process_one()
    _, right_minus_left = out.sink.played[-1]
    assert right_minus_left < 0


def test_repeat_and_dedupe():
    out, _ = _out()
    out.say("Saved Sarah.", ev.PRI_ANSWER)
    out.say("Saved Sarah.", ev.PRI_ANSWER)  # duplicate while queued
    _play_all(out)
    assert len(out.history) == 1
    out.repeat()
    assert _play_all(out)[-1] == "Saved Sarah."


def test_approaching_person_interrupts_an_answer_which_is_resumed():
    import threading
    import time

    clock = FakeClock()
    out = SpeechOutput(TTSConfig(engine="print"), None, clock, tts=PrintTTS(seconds_per_word=0.2),
                       sink=NullSink(realtime=True))
    out.say("a long answer with many many words in it", ev.PRI_ANSWER)
    t = threading.Thread(target=out.process_one)
    t.start()
    time.sleep(0.2)
    out.say("Sarah, approaching, 10 o'clock", ev.PRI_APPROACH, kind=ev.KIND_APPROACH)
    t.join()
    assert out.history[-1].interrupted
    while out.process_one():
        pass
    texts = [s.text for s in out.history]
    assert texts[1:] == ["Sarah, approaching, 10 o'clock", "a long answer with many many words in it"]
