import json
import math

from assistant.bus import Bus
from assistant.config import DirectionConfig
from assistant.geometry import approach_rate, direction_for_x, pan_gains, time_ago_phrase


def test_zones_map_frame_to_10_through_2():
    cfg = DirectionConfig(mode="zones")
    hours = [direction_for_x(x, cfg).hour for x in (0.02, 0.3, 0.5, 0.7, 0.98)]
    assert hours == [10, 11, 12, 1, 2]
    assert direction_for_x(0.05, cfg).pan < -0.9
    assert direction_for_x(0.95, cfg).pan > 0.9
    assert abs(direction_for_x(0.5, cfg).pan) < 1e-6


def test_angle_mode_is_honest_about_a_narrow_lens():
    narrow = DirectionConfig(mode="angle", hfov_deg=70)
    assert direction_for_x(0.0, narrow).hour == 11      # -35 deg rounds to 11 o'clock
    assert direction_for_x(0.5, narrow).hour == 12
    wide = DirectionConfig(mode="angle", hfov_deg=120)
    assert direction_for_x(0.0, wide).hour == 10        # -60 deg
    assert direction_for_x(1.0, wide).hour == 2


def test_constant_power_pan():
    l, r = pan_gains(-1)
    assert l > 0.99 and r < 0.01
    l, r = pan_gains(0)
    assert math.isclose(l * l + r * r, 1.0, rel_tol=1e-6) and math.isclose(l, r)


def test_approach_rate_growing_vs_constant_face():
    growing = [(t / 15, 80 * math.exp(0.6 * t / 15)) for t in range(15)]
    assert approach_rate(growing, now=14 / 15, window_s=0.8) > 0.5
    passing = [(t / 15, 80.0) for t in range(15)]
    assert abs(approach_rate(passing, now=14 / 15, window_s=0.8)) < 1e-6
    assert approach_rate(growing[:2], now=1 / 15, window_s=0.8) is None


def test_time_phrases():
    assert time_ago_phrase(10) == "just now"
    assert time_ago_phrase(14 * 60) == "14 minutes ago"
    assert time_ago_phrase(65) == "1 minute ago"
    assert time_ago_phrase(3 * 3600) == "3 hours ago"


def test_bus_fanout_filtering_and_drop_oldest(tmp_path):
    log_path = tmp_path / "events.jsonl"
    bus = Bus(log_path=str(log_path))
    a = bus.subscribe("say")
    b = bus.subscribe()  # everything
    small = bus.subscribe("person_seen", maxsize=2)
    bus.publish("say", text="hi")
    for i in range(3):
        bus.publish("person_seen", name=f"p{i}")
    assert [m["text"] for m in a.drain()] == ["hi"]
    assert len(b.drain()) == 4
    assert [m["name"] for m in small.drain()] == ["p1", "p2"]  # oldest dropped
    bus.close()
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert lines[0]["type"] == "say" and "ts" in lines[0]
