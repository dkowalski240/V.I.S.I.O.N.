"""Enrollment with other people in view."""

import numpy as np

from assistant.faces import talking
from assistant.faces.tracking import Track

from rig import Rig

SARAH, OTHER, MIKE = 200, 120, 60


def _saved_views(storage, name):
    for _pid, n, embs, _ids in storage.face_gallery():
        if n == name:
            return embs
    return None


def _all_like(embs, identity, other):
    return all(float(e @ identity) > 0.7 for e in embs) and all(float(e @ other) < 0.3 for e in embs)


def test_the_talker_wins_over_a_bigger_silent_bystander(storage):
    rig = Rig(storage, objects=False)
    scene = [(OTHER, 480, 150, 240, False), (SARAH, 1000, 250, 140, True)]
    rig.run(1.0, scene)
    rig.hear("Hi, I'm Sarah.")
    assert rig.words()[-1] == "Heard Sarah, 2 o'clock. Ask to save?"   # says WHICH face
    rig.hear("Sure.")
    rig.run(5.0, scene)
    assert rig.words()[-1] == "Saved Sarah."
    assert _all_like(_saved_views(storage, "Sarah"), rig.scene.identity(SARAH), rig.scene.identity(OTHER))
    rig.run(2.0)
    rig.run(2.0, [(OTHER, 560, 200, 180)])   # the bystander alone is not "Sarah"
    assert rig.faces.view()[0]["state"] != "known"


def test_a_known_person_in_the_middle_does_not_block_enrollment(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Mike", MIKE)
    scene = [(MIKE, 520, 180, 220), (SARAH, 60, 250, 140)]   # nobody's mouth visibly moving
    rig.run(1.0, scene)
    rig.hear("Hi, I'm Sarah.")
    assert rig.words()[-1] == "Heard Sarah, 10 o'clock. Ask to save?"
    rig.hear("Yes.")
    rig.run(5.0, scene)
    assert rig.words()[-1] == "Saved Sarah."
    assert sorted(p["name"] for p in storage.list_people()) == ["Mike", "Sarah"]
    assert _all_like(_saved_views(storage, "Sarah"), rig.scene.identity(SARAH), rig.scene.identity(MIKE))


def test_manual_save_skips_known_faces(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Mike", MIKE)
    scene = [(MIKE, 520, 180, 220), (SARAH, 1000, 250, 140)]
    rig.run(1.0, scene)
    rig.hear("save this person as Sarah", source="ptt")
    assert rig.words()[-1] == "Heard Sarah, 2 o'clock. Ask to save?"


def test_finds_them_again_after_someone_walks_in_front(storage):
    rig = Rig(storage, objects=False)
    rig.run(1.0, [(SARAH, 560, 200, 160)])
    rig.hear("Hi, I'm Sarah.")
    # Someone steps in front: her face vanishes and reappears elsewhere as a new track.
    rig.run(2.5, [(SARAH, 120, 260, 150), (OTHER, 900, 220, 150)])
    assert rig.enrollment.session is not None and rig.enrollment.session.reacquired >= 1
    rig.hear("Sure.")
    rig.run(5.0, [(SARAH, 120, 260, 150), (OTHER, 900, 220, 150)])
    assert rig.words()[-1] == "Saved Sarah."
    assert _all_like(_saved_views(storage, "Sarah"), rig.scene.identity(SARAH), rig.scene.identity(OTHER))


def test_views_of_someone_else_on_the_same_track_are_rejected(storage):
    rig = Rig(storage, objects=False)
    box = (560, 200, 160)
    rig.run(1.0, [(SARAH, *box)])
    rig.hear("Hi, I'm Sarah.")
    rig.hear("Sure.")
    rig.run(1.0, [(SARAH, *box)])
    rig.run(3.5, [(OTHER, *box)])  # the tracker keeps the same track, but it's a different face now
    assert rig.words()[-1] == "Saved Sarah."
    saved = [m for m in rig.events.drain() if m["type"] == "person_saved"][-1]
    assert saved["rejected_views"] > 0
    assert _all_like(_saved_views(storage, "Sarah"), rig.scene.identity(SARAH), rig.scene.identity(OTHER))


def test_require_talking_does_not_guess_between_silent_strangers(storage):
    rig = Rig(storage, objects=False)
    rig.cfg.enrollment.require_talking = True
    scene = [(OTHER, 200, 200, 160), (SARAH, 800, 200, 160)]
    rig.run(1.0, scene)
    rig.hear("Hi, I'm Sarah.")
    assert rig.enrollment.session is None and rig.words() == []
    rig.run(1.0, [(OTHER, 200, 200, 160), (SARAH, 800, 200, 160, True)])  # now she's visibly talking
    rig.hear("Hi, I'm Sarah.")
    assert rig.words()[-1] == "Heard Sarah, 1 o'clock. Ask to save?"


def _drawn_face(mouth_open: bool) -> np.ndarray:
    """A smooth, face-like crop: skin, eyes, nose, closed or open mouth."""
    import cv2

    f = np.full((180, 180, 3), 170, np.uint8)
    cv2.ellipse(f, (80, 90), (60, 78), 0, 0, 360, (150, 160, 190), -1)
    for x in (48, 112):
        cv2.circle(f, (x, 64), 8, (40, 40, 40), -1)
    cv2.line(f, (80, 70), (80, 98), (120, 130, 160), 3)
    if mouth_open:
        cv2.ellipse(f, (80, 128), (24, 10), 0, 0, 360, (30, 20, 60), -1)
    else:
        cv2.line(f, (56, 128), (104, 128), (60, 50, 90), 3)
    return cv2.GaussianBlur(f, (0, 0), 1.2)


def test_mouth_activity_ignores_head_motion():
    pts = np.array([[48, 64], [112, 64], [80, 96], [56, 128], [104, 128]], dtype=np.float64)
    closed, opened = _drawn_face(False), _drawn_face(True)

    def run(frames):
        t = Track(id=1, box=np.zeros(4), first_seen=0, last_seen=0)
        for i, f in enumerate(frames):
            talking.update(t, f, pts, i / 15)
        return talking.activity(t, 0, 10)

    still = [closed[:160, :160]] * 6
    speaking = [(opened if i % 2 else closed)[:160, :160] for i in range(6)]
    nodding = [closed[(i % 2) * 3:(i % 2) * 3 + 160, :160] for i in range(6)]  # whole face moves 3 px
    assert run(still) == 0.0
    assert run(speaking) > 0.015 * 3          # well above the default talk_min
    assert run(nodding) < run(speaking) / 5   # head motion mostly cancelled by the eye reference
