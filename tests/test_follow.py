from assistant.faces.follow import FaceFollower

from conftest import face_detection


def test_follower_locks_on_largest_and_follows_it():
    f = FaceFollower()
    big, small = face_detection(500, 200, 200), face_detection(50, 200, 100)
    assert f.update([small, big], 1280) is big
    moved_big, moved_small = face_detection(520, 205, 200), face_detection(60, 200, 100)
    assert f.update([moved_small, moved_big], 1280) is moved_big   # follows it, ignores the other
    closer_small = face_detection(40, 150, 260)                     # the other one gets bigger: still follow ours
    assert f.update([closer_small, face_detection(530, 205, 200)], 1280).box[0] == 530
    f.next()
    assert f.update([face_detection(40, 150, 260), face_detection(530, 205, 200)], 1280).box[0] == 40


def test_follower_finds_them_again_by_appearance():
    f = FaceFollower()
    f.update([face_detection(500, 200, 200)], 1280)
    elsewhere = face_detection(100, 300, 150)
    assert f.update([elsewhere], 1280) is None
    assert f.update([elsewhere], 1280, same_person=lambda d: True) is elsewhere
