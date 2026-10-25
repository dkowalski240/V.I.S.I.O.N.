import numpy as np

from assistant.debug_view import draw, placeholder


def test_draw_overlay_keeps_frame_size():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    faces = [
        {"track_id": 1, "box": [100, 100, 250, 250], "state": "known", "name": "Sarah", "confidence": 0.81,
         "hour": 10, "votes": 5, "approaching": True, "approach_rate": 0.6, "hint": None, "best_score": 0.8},
        {"track_id": 2, "box": [700, 100, 820, 220], "state": "unknown", "name": None, "confidence": 0.0,
         "hour": 1, "votes": 0, "approaching": False, "approach_rate": None, "hint": "Step closer",
         "best_score": None},
        {"track_id": 3, "box": [900, 100, 990, 190], "state": "checking", "name": None, "confidence": 0.0,
         "hour": 2, "votes": 1, "approaching": False, "approach_rate": 0.0, "hint": None, "best_score": 0.55},
    ]
    objects = [{"box": [400, 500, 500, 560], "item": "wallet", "score": 0.9, "label": "wallet"},
               {"box": [200, 450, 1100, 700], "item": None, "score": 0.3, "label": "table"},
               {"box": [600, 500, 680, 560], "item": None, "near": "keys", "score": 0.77, "label": "keys"},
               {"box": [700, 500, 760, 560], "item": None, "near": None, "score": 0.41, "label": "phone"}]
    out = draw(frame, faces, objects, ["faces 15fps", "heard: hi"], enrolling_track=3)
    assert out.shape == frame.shape and out.any()
    assert placeholder("x").shape == (720, 1280, 3)
