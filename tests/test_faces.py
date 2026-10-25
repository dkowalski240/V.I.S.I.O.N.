"""Recognition + announcement stories."""

from rig import Rig

SARAH, STRANGER = 200, 90
CENTER = (560, 200, 160)  # x, y, size: middle of a 1280x720 frame


def face(grey, x, y, size):
    return [(grey, x, y, size)]


def test_stranger_is_never_announced_or_stored(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    for i in range(45):  # walks past left to right for 3 s
        rig.step(face(STRANGER, 100 + i * 20, 200, 150))
    rig.run(2.0)  # leaves
    assert rig.person_events() == []
    assert rig.words() == []
    assert [p["name"] for p in storage.list_people()] == ["Sarah"]
    assert rig.faces.tracker.tracks == []  # its embeddings are gone with the track


def test_known_person_announced_once_with_cooldown(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    rig.run(4.0, face(SARAH, *CENTER))
    assert rig.words() == ["That's Sarah, 12 o'clock"]
    rig.run(2.0)                          # leaves: encounter ends
    rig.run(2.0, face(SARAH, *CENTER))    # back within the cooldown: silent
    assert rig.words() == ["That's Sarah, 12 o'clock"]
    rig.run(61.0)
    rig.run(1.5, face(SARAH, *CENTER))    # after the cooldown: announced again
    assert rig.words().count("That's Sarah, 12 o'clock") == 2


def test_needs_several_agreeing_frames(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    agree = rig.cfg.faces.matching.agree_frames
    for _ in range(agree - 1):
        rig.step(face(SARAH, *CENTER))
    assert rig.faces.view()[0]["state"] != "known"
    rig.run(1.0, face(SARAH, *CENTER))
    assert rig.faces.view()[0]["state"] == "known"


def test_approaching_from_the_left_is_announced_fast_and_panned_left(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    t0 = rig.clock.now()
    size = 80.0
    for _ in range(30):
        rig.step(face(SARAH, 40, 200, size))
        size *= 1.04  # ~80%/s growth: walking toward the camera
        if rig.words():
            break
    assert rig.words() == ["Sarah, approaching, 10 o'clock"]
    assert rig.clock.now() - t0 <= 1.0
    ev = [m for m in rig.said_events if m.get("text") == "Sarah, approaching, 10 o'clock"][0]
    assert ev["pan"] < -0.5 and ev["priority"] == 0


def test_low_confidence_wording(storage):
    rig = Rig(storage, objects=False)
    rig.cfg.faces.matching.t_sure = 0.99  # nothing is "sure" now
    rig.enroll_directly("Sarah", SARAH)
    rig.run(3.0, face(SARAH, *CENTER))
    assert rig.words() == ["I think that's Sarah, 12 o'clock"]


def test_forget_with_confirmation_stops_announcements(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    rig.hear("forget Sarah", source="ptt")
    assert rig.words()[-1] == "Forget Sarah? Say yes."
    rig.hear("yes", source="ptt")
    assert rig.words()[-1] == "Forgot Sarah."
    assert storage.list_people() == []
    rig.run(3.0, face(SARAH, *CENTER))
    assert rig.words()[-1] == "Forgot Sarah."
    assert rig.person_events() == []


def test_who_is_here(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    rig.run(2.0, [(SARAH, 40, 200, 150), (STRANGER, 1000, 200, 150)])
    rig.hear("who's here?", source="ptt")
    assert rig.words()[-1] == "Sarah at 10 o'clock. One person I don't know."


def _seated(rig, seconds, size, rng, x=500, look_down_every=40.0, look_down_for=1.5):
    """Someone sitting in front of the camera: small jitter, posture drift, leaning in now and
    then, and looking down at the keyboard (face lost) every `look_down_every` seconds."""
    import math

    for k in range(int(seconds * 15)):
        sec = k / 15
        phase = sec % look_down_every
        if 20 <= phase < 20 + look_down_for:
            rig.step([])
            continue
        s = size * (1 + 0.03 * math.sin(sec * 0.7)) * (1 + rng.normal(0, 0.015))
        if 20 + look_down_for <= phase < 20 + look_down_for + 0.5:  # head coming back up: box grows
            s *= 0.75 + 0.5 * (phase - 20 - look_down_for)
        if 30 <= phase < 30.6:  # leans toward the screen
            s *= 1 + 0.25 * (phase - 30) / 0.6
        rig.step([(SARAH, x + rng.normal(0, 3), 150, s)])


def test_sitting_still_at_the_laptop_is_announced_once(storage):
    import numpy as np

    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    _seated(rig, 150, 280, np.random.default_rng(1))   # face ~40% of the frame height: close
    assert rig.words() == ["That's Sarah, 12 o'clock"]


def test_sitting_farther_away_and_looking_down_is_announced_once(storage):
    import numpy as np

    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    _seated(rig, 150, 120, np.random.default_rng(2))   # ~1.5 m away
    assert rig.words() == ["That's Sarah, 12 o'clock"]


def test_coming_back_after_a_long_absence_is_announced_again(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    rig.run(3.0, face(SARAH, *CENTER))
    rig.run(70.0)                         # gone for longer than the cooldown
    rig.run(3.0, face(SARAH, *CENTER))
    assert rig.words() == ["That's Sarah, 12 o'clock", "That's Sarah, 12 o'clock"]


def test_brief_absence_then_a_real_approach_is_announced(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    rig.run(3.0, face(SARAH, 900, 250, 90))   # standing across the room
    rig.run(5.0)                              # steps out of view briefly (same encounter)
    size = 80.0
    for _ in range(30):                       # walks up from the left
        rig.step(face(SARAH, 40, 220, size))
        size *= 1.04
    assert rig.words() == ["That's Sarah, 1 o'clock", "Sarah, approaching, 10 o'clock"]
