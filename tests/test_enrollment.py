"""Enrollment by introduction, with consent."""

from rig import Rig

SARAH, OTHER = 200, 120
CENTER = [(SARAH, 560, 200, 180)]


def _intro(rig, text="Hi, I'm Sarah."):
    rig.run(1.0, CENTER)
    rig.hear(text, source="ambient", spoken_for=1.0)


def test_introduction_consent_yes_saves_several_views(storage):
    rig = Rig(storage, objects=False)
    _intro(rig)
    assert rig.words()[-1] == "Heard Sarah. Ask to save?"
    rig.run(1.0, CENTER)
    rig.hear("Is it okay if my assistant remembers you?")  # the user asking aloud: not an answer
    assert rig.enrollment.awaiting_answer()
    rig.hear("Sure.")
    rig.run(5.0, CENTER)
    assert rig.words()[-1] == "Saved Sarah."
    people = storage.list_people()
    assert [p["name"] for p in people] == ["Sarah"]
    assert people[0]["n_exemplars"] >= rig.cfg.enrollment.min_exemplars
    # just saved: not immediately announced, but recognised
    assert "That's Sarah, 12 o'clock" not in rig.words()
    assert rig.faces.view()[0]["name"] == "Sarah"


def test_no_means_nothing_is_stored(storage):
    rig = Rig(storage, objects=False)
    _intro(rig)
    rig.run(1.0, CENTER)
    rig.hear("No thanks.")
    assert rig.words()[-1] == "Okay, not saved."
    rig.run(5.0, CENTER)
    assert storage.list_people() == []
    assert rig.enrollment.session is None


def test_silence_times_out(storage):
    rig = Rig(storage, objects=False)
    _intro(rig)
    rig.run(rig.cfg.enrollment.consent_timeout_s + 1, CENTER)
    assert rig.words()[-1] == "No answer. Not saved."
    assert storage.list_people() == []


def test_name_correction_during_consent_and_after_saving(storage):
    rig = Rig(storage, objects=False)
    _intro(rig, "Hi, I'm Sierra.")
    rig.hear("No, it's Sara.")
    assert rig.words()[-1] == "Heard Sara. Ask to save?"
    rig.hear("Yeah.", source="ptt")  # the user can answer with push-to-talk too
    rig.run(5.0, CENTER)
    assert rig.words()[-1] == "Saved Sara."
    rig.hear("No, her name is Sarah", source="ptt")
    assert rig.words()[-1] == "Renamed to Sarah."
    assert [p["name"] for p in storage.list_people()] == ["Sarah"]


def test_ordinary_chatter_does_not_start_enrollment(storage):
    rig = Rig(storage, objects=False)
    rig.run(1.0, CENTER)
    for text in ["I'm fine, thanks.", "I'm going to grab lunch.", "This is great!", "where's my wallet"]:
        rig.hear(text)
    assert rig.enrollment.session is None
    assert rig.words() == []


def test_introduction_with_no_face_is_ignored(storage):
    rig = Rig(storage, objects=False)
    rig.run(1.0)
    rig.hear("Hi, I'm Sarah.")
    assert rig.enrollment.session is None and rig.words() == []


def test_known_face_is_not_saved_again(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Mike", SARAH)
    rig.run(2.0, CENTER)
    rig.hear("Hi, I'm Sarah.")
    assert rig.words()[-1] == "That looks like Mike. Not saved."
    assert [p["name"] for p in storage.list_people()] == ["Mike"]


def test_manual_save_command_also_asks(storage):
    rig = Rig(storage, objects=False)
    rig.run(1.0, CENTER)
    rig.hear("save this person as Sarah", source="ptt")
    assert rig.words()[-1] == "Heard Sarah. Ask to save?"
    rig.hear("Yes.")
    rig.run(5.0, CENTER)
    assert rig.words()[-1] == "Saved Sarah."


def test_picks_the_speaker_not_the_bystander(storage):
    rig = Rig(storage, objects=False)
    scene = [(SARAH, 520, 180, 220), (OTHER, 60, 300, 90)]  # big+central vs small+edge
    rig.run(1.0, scene)
    rig.hear("Hi, I'm Sarah.")
    rig.hear("Sure.")
    rig.run(5.0, scene)
    assert rig.words()[-1] == "Saved Sarah."
    rig.run(2.0)
    rig.faces.forget_cooldowns(storage.find_person("Sarah")["id"])
    rig.person_events()  # drain
    rig.run(2.0, [(OTHER, 560, 200, 180)])  # the bystander alone: not recognised as Sarah
    assert rig.person_events() == []
    assert rig.faces.view()[0]["state"] != "known"


def test_strict_gate_too_strict_falls_back_to_recognition_frames(storage):
    rig = Rig(storage, objects=False)
    rig.cfg.faces.enroll_quality.min_face_px = 10_000  # nothing passes the strict gate
    rig.faces.enroll_gate.cfg = rig.cfg.faces.enroll_quality
    _intro(rig)
    rig.hear("Sure.")
    rig.run(5.0, CENTER)
    assert rig.words()[-1] == "Saved Sarah."
    assert storage.list_people()[0]["n_exemplars"] >= rig.cfg.enrollment.min_exemplars
