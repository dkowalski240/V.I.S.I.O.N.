"""Object enrollment, sighting log and 'where is my X?'."""

from assistant.objects.context import describe
from assistant.objects.detector import ObjDet, boxes_to_frame, nms, owl_preprocess

import numpy as np

from rig import Rig

WALLET = (40, 90, 200)
OTHER_WALLET = (200, 200, 40)
TABLE = (60, 60, 60)
LAPTOP = (150, 150, 150)

HELD = [(WALLET, "wallet", (560, 300, 720, 400))]
ON_TABLE = [
    (TABLE, "table", (200, 450, 1100, 700)),
    (LAPTOP, "laptop", (700, 380, 950, 520)),
    (WALLET, "wallet", (560, 470, 660, 520)),
]


def _enroll_wallet(rig):
    rig.hear("remember this, it's my wallet", source="ptt")
    assert rig.words()[-1] == "Hold it up and turn it slowly."
    rig.run(rig.cfg.objects.enroll_delay_s + rig.cfg.objects.enroll_s + 1.0, objects=HELD)
    assert rig.words()[-1] == "Saved your wallet."


def test_enroll_then_sighting_with_context_and_where_query(storage):
    rig = Rig(storage)
    _enroll_wallet(rig)
    rig.run(3.0, objects=ON_TABLE)
    item = storage.find_item("wallet")
    s = storage.last_sighting(item["id"])
    assert s is not None and s.context == "on the table, next to the laptop"
    assert s.snapshot_path and s.snapshot_path.endswith(".jpg")

    rig.hear("where's my wallet?", source="ptt")
    assert rig.words()[-1] == "Your wallet is in view at 12 o'clock, on the table, next to the laptop."

    rig.run(2.0)                 # look away
    rig.clock.advance(14 * 60)   # 14 minutes later
    rig.hear("Where's my wallet?", source="ptt")
    assert rig.words()[-1] == "Your wallet was last seen 14 minutes ago, on the table, next to the laptop."


def test_same_place_refreshes_instead_of_adding_rows(storage):
    rig = Rig(storage)
    _enroll_wallet(rig)
    rig.run(8.0, objects=ON_TABLE)
    item = storage.find_item("wallet")
    assert len(storage.sightings(item["id"])) == 1
    moved = [ON_TABLE[0], ON_TABLE[1], (WALLET, "wallet", (250, 480, 350, 530))]
    rig.run(3.0, objects=moved)
    assert len(storage.sightings(item["id"])) == 2


def test_someone_elses_wallet_is_not_mine(storage):
    rig = Rig(storage)
    _enroll_wallet(rig)
    rig.run(4.0, objects=[ON_TABLE[0], (OTHER_WALLET, "wallet", (560, 470, 660, 520))])
    assert storage.last_sighting(storage.find_item("wallet")["id"]) is None


def test_unknown_item_and_never_seen(storage):
    rig = Rig(storage)
    rig.hear("where are my keys", source="ptt")
    assert rig.words()[-1] == "I don't know your keys yet."
    storage.add_item("keys", [np.ones(512, dtype=np.float32) / np.sqrt(512)])
    rig.objects.reload_gallery()
    rig.hear("where are my keys", source="ptt")
    assert rig.words()[-1] == "I haven't seen your keys yet."


def test_forget_item(storage):
    rig = Rig(storage)
    _enroll_wallet(rig)
    rig.run(3.0, objects=ON_TABLE)
    rig.hear("forget my wallet", source="ptt")
    assert rig.words()[-1] == "Forget your wallet? Say yes."
    rig.hear("yes", source="ptt")
    assert rig.words()[-1] == "Forgot your wallet."
    assert storage.find_item("wallet") is None


def test_context_phrase_rules():
    table = ObjDet(np.array([0, 300, 1000, 700], np.float32), 0.5, "table")
    desk = ObjDet(np.array([100, 350, 600, 700], np.float32), 0.5, "desk")
    lamp = ObjDet(np.array([900, 100, 1000, 400], np.float32), 0.5, "lamp")
    item = (200, 380, 260, 420)
    assert describe(item, [table, desk, lamp], ["table", "desk"], 1280, 720) == "on the desk"
    assert describe((850, 330, 890, 380), [table, lamp], ["table"], 1280, 720) == "on the table, next to the lamp"
    assert describe(item, [], ["table"], 1280, 720) == ""


def test_owl_preprocessing_and_box_mapping():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    x, side = owl_preprocess(img, 960)
    assert x.shape == (1, 3, 960, 960) and side == 1280
    # a box centred in the top-left quarter of the padded square
    b = boxes_to_frame(np.array([[0.25, 0.25, 0.1, 0.1]]), side, 1280, 720)[0]
    assert np.allclose(b, [256, 256, 384, 384])  # 0.25*1280 -/+ 0.05*1280
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]))) == [0, 2]


def test_duplicate_boxes_dont_crowd_out_the_item(storage):
    # The detector often puts several labels on one object; with only 3 candidate
    # slots, four boxes on the laptop would leave no room for the wallet.
    rig = Rig(storage)
    _enroll_wallet(rig)
    rig.cfg.objects.max_candidates = 3
    crowd = [(LAPTOP, lab, (700 + k, 380, 950 + k, 520)) for k, lab in
             enumerate(["phone", "card", "bag", "remote control"])]
    rig.run(3.0, objects=[ON_TABLE[0]] + crowd + [(WALLET, "wallet", (560, 470, 660, 520))])
    assert storage.last_sighting(storage.find_item("wallet")["id"]) is not None


def test_near_miss_is_shown_in_the_debug_view(storage):
    rig = Rig(storage)
    _enroll_wallet(rig)
    rig.run(1.0, objects=ON_TABLE)
    seen = [o for o in rig.objects.view() if o.get("item") == "wallet"]
    assert seen and seen[0]["near"] is None
    rig.cfg.objects.t_item = seen[0]["score"] + 0.04   # just out of reach
    rig.run(1.0, objects=ON_TABLE)
    near = [o for o in rig.objects.view() if o.get("near") == "wallet"]
    assert near and near[0]["item"] is None
