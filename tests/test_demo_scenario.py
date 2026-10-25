"""The acceptance test from the brief, end to end with fake models.

1. A teammate walks up and says "Hi, I'm Sarah." The assistant asks; she says yes.
2. "remember this, it's my wallet", held up, then put on a table.
3. Later Sarah approaches from the left -> "Sarah, approaching, 10 o'clock",
   panned left, within ~1 s of her face being clearly visible.
4. A stranger walks past -> nothing announced.
5. "where's my wallet?" -> correct location and time.
6. "forget Sarah" -> she is no longer announced.
"""

from rig import Rig

SARAH, STRANGER = 200, 90
WALLET, TABLE, LAPTOP = (40, 90, 200), (60, 60, 60), (150, 150, 150)
ROOM = [(TABLE, "table", (200, 450, 1100, 700)), (LAPTOP, "laptop", (700, 380, 950, 520))]
WALLET_ON_TABLE = ROOM + [(WALLET, "wallet", (560, 470, 660, 520))]


def test_demo_scenario(storage):
    rig = Rig(storage)

    # 1. Introduction with consent (the user asks her aloud; she answers).
    sarah_close = [(SARAH, 540, 150, 200)]
    rig.run(1.5, sarah_close, ROOM)
    rig.hear("Hi, I'm Sarah.")
    assert rig.words()[-1] == "Heard Sarah. Ask to save?"
    rig.run(1.0, sarah_close, ROOM)
    rig.hear("Is it okay if it remembers you?")
    rig.hear("Yes, sure.")
    rig.run(5.0, sarah_close, ROOM)
    assert rig.words()[-1] == "Saved Sarah."
    rig.run(2.0, objects=ROOM)  # she walks off

    # 2. Remember the wallet, then put it on the table.
    rig.hear("Remember this, it's my wallet.", source="ptt")
    rig.run(rig.cfg.objects.enroll_delay_s + rig.cfg.objects.enroll_s + 1.0, objects=ROOM + [(WALLET, "wallet", (560, 250, 720, 360))])
    assert rig.words()[-1] == "Saved your wallet."
    rig.run(3.0, objects=WALLET_ON_TABLE)
    put_down_at = rig.clock.now()
    rig.run(2.0, objects=ROOM)  # camera turns away from the table

    # 3. Later, Sarah approaches from the left.
    rig.run(30.0)
    n_before = len(rig.words())
    t_visible = rig.clock.now()
    size = 90.0
    announced_at = None
    for _ in range(45):
        rig.step([(SARAH, 30, 220, size)])
        size *= 1.035
        if len(rig.words()) > n_before and announced_at is None:
            announced_at = rig.clock.now()
    new = rig.words()[n_before:]
    assert new == ["Sarah, approaching, 10 o'clock"], new
    assert announced_at - t_visible <= 1.0
    ev = [m for m in rig.said_events if m.get("text") == "Sarah, approaching, 10 o'clock"][-1]
    assert ev["pan"] < -0.5
    rig.run(2.0)

    # 4. A stranger walks past: nothing.
    n_before = len(rig.words())
    for i in range(40):
        rig.step([(STRANGER, 100 + i * 25, 220, 140)])
    rig.run(2.0)
    assert rig.words()[n_before:] == []

    # 5. Where's my wallet?
    rig.clock.advance(14 * 60)
    rig.hear("Where's my wallet?", source="ptt")
    minutes = round((rig.clock.now() - put_down_at) / 60)
    assert rig.words()[-1] == f"Your wallet was last seen {minutes} minutes ago, on the table, next to the laptop."

    # 6. Forget Sarah: no more announcements.
    rig.hear("Forget Sarah.", source="ptt")
    rig.hear("Yes.", source="ptt")
    assert rig.words()[-1] == "Forgot Sarah."
    n_before = len(rig.words())
    size = 90.0
    for _ in range(45):
        rig.step([(SARAH, 30, 220, size)])
        size *= 1.035
    assert rig.words()[n_before:] == []
