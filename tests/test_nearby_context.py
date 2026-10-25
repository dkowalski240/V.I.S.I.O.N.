"""'Where's my wallet?' with the room and a landmark from the frames around the sighting,
and plain-word directions."""

import numpy as np
import pytest

from assistant.geometry import at_phrase, clock_phrase, direction_words
from assistant.intents.phrases import location_phrase

from rig import Rig

WALLET, FLOOR, BED, LAPTOP = (40, 90, 200), (90, 70, 50), (200, 180, 230), (150, 150, 150)
BEDROOM, HALL = (30, 140, 70), (120, 120, 20)
SARAH = 200
# Camera pointing down: only the floor and the wallet (in the middle, so the room can't be told).
LOOKING_DOWN = [(FLOOR, "floor", (0, 200, 1280, 720)), (WALLET, "wallet", (560, 330, 720, 410))]
LOOKING_UP = [(BED, "bed", (60, 250, 420, 700))]          # the room, with the bed at the side


def _setup(storage, room_command="this is my bedroom"):
    rig = Rig(storage)
    rig.background = BEDROOM
    rig.hear(room_command, source="ptt")
    rig.run(rig.cfg.places.enroll_delay_s + rig.cfg.places.enroll_s + 1.0)
    rig.hear("remember this, it's my wallet", source="ptt")
    rig.run(rig.cfg.objects.enroll_delay_s + rig.cfg.objects.enroll_s + 1.0,
            objects=[(WALLET, "wallet", (560, 250, 720, 360))])
    assert rig.words()[-1] == "Saved your wallet."
    return rig


def _ask(rig, minutes=5):
    rig.clock.advance(minutes * 60)
    rig.hear("where's my wallet?", source="ptt")
    return rig.words()[-1]


def test_room_and_landmark_from_the_frames_after(storage):
    rig = _setup(storage)
    rig.run(2.0, objects=LOOKING_DOWN)       # threw it on the floor
    rig.run(2.0, objects=LOOKING_UP)         # looked up: the bedroom, the bed
    assert _ask(rig) == "Your wallet was last seen 5 minutes ago, on the floor of your bedroom, near the bed."


def test_room_and_landmark_from_the_frames_before(storage):
    rig = _setup(storage)
    rig.run(2.0, objects=LOOKING_UP)         # walked in, saw the bed...
    rig.run(2.0, objects=LOOKING_DOWN)       # ...then dropped the wallet and looked down at it
    rig.background = HALL
    rig.run(1.0)
    assert _ask(rig) == "Your wallet was last seen 5 minutes ago, on the floor of your bedroom, near the bed."


def test_frames_long_after_are_not_used(storage):
    rig = _setup(storage)
    rig.background = HALL                    # somewhere it doesn't know, nothing landmark-like
    rig.run(rig.cfg.objects.context_window_s + 1.0)
    rig.run(2.0, objects=LOOKING_DOWN)
    rig.run(rig.cfg.objects.context_window_s + 1.0)   # walked on...
    rig.background = BEDROOM                 # ...and much later sees a bed
    rig.run(2.0, objects=LOOKING_UP)
    assert _ask(rig) == "Your wallet was last seen 5 minutes ago, on the floor."


def test_landmark_in_the_same_frame(storage):
    rig = _setup(storage, room_command="this is the study")
    rig.background = BEDROOM
    rig.run(2.0, objects=LOOKING_UP + [(FLOOR, "floor", (420, 200, 1280, 720)),
                                       (WALLET, "wallet", (900, 330, 1060, 410))])
    rig.run(1.0)
    answer = _ask(rig)
    assert answer.startswith("Your wallet was last seen 5 minutes ago, on the floor")
    assert answer.endswith("near the bed.")


def test_next_to_in_the_same_frame_wins():
    assert location_phrase("on the table, next to the laptop", "the kitchen", "couch") == \
        "on the table in the kitchen, next to the laptop"


@pytest.mark.parametrize("context,place,near,expected", [
    ("on the floor", "your bedroom", "bed", "on the floor of your bedroom, near the bed"),
    ("", "your bedroom", "bed", "in your bedroom, near the bed"),
    ("on the bed", "your bedroom", "bed", "on the bed in your bedroom"),
    ("on the floor", "", "bed", "on the floor, near the bed"),
    ("next to the laptop", "", "door", "next to the laptop"),
    ("on the desk", "the library", "", "on the desk in the library"),
    ("", "", "", ""),
])
def test_location_phrase(context, place, near, expected):
    assert location_phrase(context, place, near) == expected


def test_direction_words():
    assert direction_words(12) == "straight ahead"
    assert direction_words(10) == "to your left"
    assert direction_words(1) == "slightly to your right"
    assert clock_phrase(2, "words") == "to your right"
    assert clock_phrase(2, "both") == "2 o'clock, to your right"
    assert at_phrase(2) == "at 2 o'clock"
    assert at_phrase(11, "words") == "slightly to your left"


def test_switching_direction_style_by_voice(storage):
    rig = Rig(storage, objects=False)
    rig.enroll_directly("Sarah", SARAH)
    rig.run(2.0, [(SARAH, 40, 200, 150)])
    rig.hear("who's here?", source="ptt")
    assert rig.words()[-1] == "Sarah at 10 o'clock."
    rig.hear("use left and right", source="ptt")
    assert rig.words()[-1] == "Okay, I'll say left and right."
    rig.hear("who's here?", source="ptt")
    assert rig.words()[-1] == "Sarah to your left."
    rig.hear("use clock directions", source="ptt")
    assert rig.words()[-1].startswith("Clock directions. 12 o'clock is straight ahead")
    rig.hear("where is Sarah", source="ptt")
    assert rig.words()[-1] == "Sarah at 10 o'clock."


def test_announcements_follow_the_style(storage):
    rig = Rig(storage, objects=False)
    rig.cfg.direction.style = "words"
    rig.enroll_directly("Sarah", SARAH)
    rig.run(3.0, [(SARAH, 1000, 200, 150)])
    assert "That's Sarah, to your right" in rig.words()


# ------------------------------------------------------------------ learning a room
DESK = (20, 60, 160)


def test_learning_a_room_records_its_landmarks(storage):
    rig = Rig(storage)
    rig.background = BEDROOM
    rig.hear("this is my bedroom", source="ptt")
    assert rig.words()[-1] == "Turn slowly all the way around. Say done to finish."
    rig.run(rig.cfg.places.enroll_delay_s + 1.0)
    rig.run(5.0, objects=LOOKING_UP)                              # the bed for a while...
    rig.run(3.0, objects=[(DESK, "desk", (800, 300, 1200, 600))])  # ...then the desk
    assert "Keep turning." in rig.words()                         # halfway
    rig.run(rig.cfg.places.enroll_s)                              # rest of the turn
    assert rig.words()[-1] == "Saved your bedroom. I saw a bed and a desk."
    assert set(storage.find_place("bedroom")["landmarks"]) == {"bed", "desk"}


def test_done_ends_room_learning_early(storage):
    rig = Rig(storage)
    rig.background = BEDROOM
    rig.hear("this is the study", source="ptt")
    rig.run(rig.cfg.places.enroll_delay_s + 3.0)
    rig.hear("done", source="ptt")
    rig.run(0.5)
    assert rig.words()[-1] == "Saved the study."
    assert rig.objects.enrolling() is None


def test_room_views_are_spread_over_the_whole_turn(storage):
    rig = Rig(storage)
    rig.cfg.places.max_views = 4
    colours = [(30, 140, 70), (150, 60, 40), (40, 40, 200), (200, 200, 40), (100, 30, 160), (20, 200, 200)]
    rig.background = colours[0]
    rig.hear("this is my bedroom", source="ptt")
    rig.run(rig.cfg.places.enroll_delay_s)
    per = rig.cfg.places.enroll_s / len(colours)
    for c in colours:            # a different part of the room every couple of seconds
        rig.background = c
        rig.run(per)
    rig.run(1.0)
    [(place, views)] = list(storage.place_gallery())
    assert len(views) == 4
    last = rig.scene.color_vec(np.array(colours[-1], float))
    first = rig.scene.color_vec(np.array(colours[0], float))
    assert max(views @ last) > 0.8 and max(views @ first) > 0.8   # both ends of the turn kept


def test_room_landmarks_help_tell_rooms_apart(storage):
    from assistant.config import PlacesConfig
    from assistant.objects.places import PlaceMemory

    view = np.ones(512, np.float32) / np.sqrt(512)                  # two rooms that look the same
    storage.add_place("bedroom", [view], landmarks={"bed": 6}, mine=True)
    storage.add_place("kitchen", [view], landmarks={"sink": 4, "counter": 5})
    pm = PlaceMemory(PlacesConfig(), storage)
    assert pm.classify(view) is None                                 # can't tell from the picture alone
    assert pm.classify(view, None, ["bed"]) == "bedroom"
    assert pm.classify(view, None, ["sink"]) == "kitchen"
