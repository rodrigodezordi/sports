import cv2
import numpy as np

from sports.common.camera import CameraTracker
from sports.common.team_color import ColorTeamClassifier


def _scene(seed=0, size=(480, 640)):
    rng = np.random.default_rng(seed)
    img = (rng.random((size[0] // 8, size[1] // 8, 3)) * 255).astype(np.uint8)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_tracker_follows_pan_and_zoom():
    ref = _scene()
    to_pitch = np.diag([10.0, 10.0, 1.0])  # 1 px of the reference = 10 cm
    tracker = CameraTracker(ref, to_pitch)
    # the camera pans right and zooms in a little: frame = M @ reference
    m = np.array([[1.1, 0.0, -40.0], [0.0, 1.1, -10.0], [0.0, 0.0, 1.0]])
    frame = cv2.warpPerspective(ref, m, (640, 480))
    vt = tracker.update(frame)
    assert vt is not None
    p_frame = np.array([[320.0, 240.0]], np.float32)
    expected = (np.linalg.inv(m) @ [320, 240, 1])[:2] * 10
    assert np.allclose(vt.transform_points(p_frame)[0], expected, atol=15)  # within 1.5 px


def test_tracker_reuses_last_good_then_gives_up():
    ref = _scene()
    tracker = CameraTracker(ref, np.eye(3), max_age=2)
    assert tracker.update(ref) is not None
    blank = np.zeros_like(ref)
    assert tracker.update(blank) is not None
    assert tracker.update(blank) is not None
    assert tracker.update(blank) is None


def _player(shirt_bgr, grass=(60, 160, 70), rng=None):
    crop = np.full((40, 16, 3), grass, np.uint8)
    crop[6:20, 3:13] = shirt_bgr
    crop[20:34, 4:12] = (30, 30, 30)
    if rng is not None:
        crop = np.clip(crop + rng.normal(0, 6, crop.shape), 0, 255).astype(np.uint8)
    return crop


def test_color_teams_split_and_referee_is_far():
    rng = np.random.default_rng(1)
    red = [_player((40, 40, 190), rng=rng) for _ in range(20)]
    green = [_player((80, 110, 20), rng=rng) for _ in range(20)]
    clf = ColorTeamClassifier().fit(red + green)
    pr, pg = clf.predict(red), clf.predict(green)
    assert len(set(pr)) == 1 and len(set(pg)) == 1 and pr[0] != pg[0]
    referee = [_player((200, 120, 40))]  # blue shirt
    assert clf.distance(referee)[0] > clf.threshold
    assert (clf.distance(red) < clf.threshold).all()
