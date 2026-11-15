import numpy as np

from sports.common.shots import detect_shots

FPS = 25
PLAYERS = {7: (0, (95, 35)), 9: (1, (60, 20)), 3: (0, (80, 50)), 4: (1, (100, 60))}


def _segment(log, f0, n, start, vel, rng):
    x = np.array(start, float)
    v = np.array(vel, float) / FPS
    for f in range(f0, f0 + n):
        bx, by = x + rng.normal(0, 0.05, 2)
        dists = {tid: np.hypot(bx - p[0], by - p[1]) for tid, (_, p) in PLAYERS.items()}
        owner = min(dists, key=dists.get)
        log.append(dict(frame=f, t=f / FPS, x_m=bx, y_m=by, owner_id=owner,
                        owner_team=PLAYERS[owner][0], owner_dist_m=dists[owner]))
        x = x + v


def _match():
    rng = np.random.default_rng(1)
    log = []
    _segment(log, 0, 25, (80.5, 50), (0, 0), rng)           # player 3 holds the ball
    _segment(log, 25, 50, (80.5, 50), (4.5, -4.65), rng)    # slow pass to player 7
    _segment(log, 75, 25, (95.5, 35), (0, 0), rng)          # player 7 holds
    _segment(log, 100, 20, (95.5, 35), (25, 0), rng)        # shot on target, right goal
    _segment(log, 160, 25, (60.5, 20), (0, 0), rng)         # player 9 holds
    _segment(log, 185, 25, (60.5, 20), (0, 20), rng)        # fast lateral pass
    _segment(log, 250, 25, (100.5, 60), (0, 0), rng)        # player 4 holds
    _segment(log, 275, 15, (100.5, 60), (22, -5), rng)      # fast, misses goal by far
    _segment(log, 300, 25, (100.5, 60), (0, 0), rng)
    _segment(log, 325, 15, (100.5, 60), (20, -12), rng)     # fast, misses goal by far
    _segment(log, 350, 25, (100.5, 60), (0, 0), rng)
    _segment(log, 375, 15, (100.5, 60), (20, -19), rng)     # shot just wide
    return log


def test_detects_shots_and_ignores_passes():
    shots = detect_shots(_match(), FPS, pitch_length_m=120, pitch_width_m=70)
    assert len(shots) == 2
    first, second = shots.iloc[0], shots.iloc[1]
    assert (first.player_id, first.team, first.goal, first.outcome) == (7, 0, "right", "on_target")
    assert (second.player_id, second.team, second.outcome) == (4, 1, "off_target")
    assert abs(first.ball_speed_kmh - 90) < 10
    assert abs(first.x - 95.5) < 1 and abs(first.distance_m - 24.5) < 1


def test_team_override_and_empty_log():
    shots = detect_shots(_match(), FPS, 120, 70, team_of={7: 1, 4: 0})
    assert shots.team.tolist() == [1, 0]
    assert detect_shots([], FPS, 120, 70).empty
