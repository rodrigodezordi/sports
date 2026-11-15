"""
Shot-on-goal detection from the ball trajectory in pitch coordinates.

A shot is an event defined by geometric rules on the ball path, the same idea
used to call basketball shots by following the ball through the rim:

1. possession: the ball was within `possession_radius_m` of a player during the
   last `possession_memory_s` seconds (that player is the shooter);
2. speed: the ball leaves at `min_speed_ms` or more;
3. zone: it starts within `max_dist_m` of the centre of the goal;
4. direction: extending the path, it crosses the goal line within the goal
   mouth plus `mouth_margin_m` on each side (strong passes and switches of play
   are rejected);
5. one shot per play: events closer than `min_gap_s` count once.

The outcome is "on_target" when the extended path passes between the posts,
"off_target" otherwise. From a single broadcast-style camera this is an
estimate: an airborne ball is projected incorrectly by a ground homography, so
each shot should be checked against the video.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

GOAL_WIDTH_M = 7.32


@dataclass
class ShotDetectorConfig:
    possession_radius_m: float = 2.0
    possession_memory_s: float = 1.0
    min_speed_ms: float = 12.0
    max_speed_ms: float = 45.0
    max_dist_m: float = 35.0
    mouth_margin_m: float = 4.0
    min_gap_s: float = 2.0
    speed_window_s: float = 0.2
    max_gap_fill_s: float = 0.4
    goal_width_m: float = GOAL_WIDTH_M


def ball_track(ball_log: Iterable[Dict], fps: float,
               config: Optional[ShotDetectorConfig] = None) -> pd.DataFrame:
    """
    Ball path per frame with short gaps interpolated, light smoothing and
    velocity.

    Args:
        ball_log: One dict per frame where the ball was seen, with keys
            `frame`, `t` (seconds), `x_m`, `y_m` (pitch metres).
        fps: Frame rate of the frames in `ball_log`.

    Returns:
        DataFrame indexed by frame with `t, x_m, y_m, vx, vy, speed`.
    """
    config = config or ShotDetectorConfig()
    raw = pd.DataFrame(list(ball_log))
    if raw.empty:
        return pd.DataFrame(columns=["t", "x_m", "y_m", "vx", "vy", "speed"])
    raw = raw.groupby("frame")[["t", "x_m", "y_m"]].first().sort_index()
    step = int(np.median(np.diff(raw.index))) if len(raw) > 1 else 1
    b = raw.reindex(range(raw.index.min(), raw.index.max() + 1, max(step, 1)))
    fps_eff = fps / max(step, 1)
    b = b.interpolate(limit=max(1, int(config.max_gap_fill_s * fps_eff)),
                      limit_area="inside")
    b[["x_m", "y_m"]] = b[["x_m", "y_m"]].rolling(
        3, center=True, min_periods=1).median()
    k = max(1, int(config.speed_window_s * fps_eff))
    dt = b.t.diff(k)
    b["vx"], b["vy"] = b.x_m.diff(k) / dt, b.y_m.diff(k) / dt
    b["speed"] = np.hypot(b.vx, b.vy)
    b.attrs["k"] = k
    return b.dropna(subset=["x_m"])


def detect_shots(
    ball_log: Iterable[Dict],
    fps: float,
    pitch_length_m: float,
    pitch_width_m: float,
    team_of: Optional[Dict[int, int]] = None,
    config: Optional[ShotDetectorConfig] = None
) -> pd.DataFrame:
    """
    Detect shots on goal.

    Args:
        ball_log: One dict per frame where the ball was seen, with keys
            `frame`, `t`, `x_m`, `y_m` and the nearest player in that frame:
            `owner_id`, `owner_team`, `owner_dist_m` (None when no player).
        fps: Frame rate of the frames in `ball_log`.
        pitch_length_m, pitch_width_m: Pitch size in metres (goals are at
            x = 0 and x = pitch_length_m).
        team_of: Optional {tracker_id: team} to override per-frame teams
            (e.g. a majority vote over the whole video).

    Returns:
        One row per shot: frame, t, player_id, team, goal ("left"/"right"),
        x, y (where the shot started), distance_m, angle_deg (angle of view of
        the goal), ball_speed_kmh and outcome ("on_target"/"off_target").
    """
    config = config or ShotDetectorConfig()
    ball_log = list(ball_log)
    columns = ["frame", "t", "player_id", "team", "goal", "x", "y",
               "distance_m", "angle_deg", "ball_speed_kmh", "outcome"]
    ball = ball_track(ball_log, fps, config)
    if ball.empty:
        return pd.DataFrame(columns=columns)
    k = ball.attrs.get("k", 1)

    raw = pd.DataFrame(ball_log).groupby("frame").first()
    owners = raw[raw.owner_dist_m.astype(float) <= config.possession_radius_m]
    owners = owners.dropna(subset=["owner_id"])

    half_mouth = config.goal_width_m / 2
    goals = {"left": (0.0, pitch_width_m / 2),
             "right": (pitch_length_m, pitch_width_m / 2)}
    shots: List[Dict] = []
    last_t = -np.inf
    for frame, r in ball.iterrows():
        if not (config.min_speed_ms <= r.speed <= config.max_speed_ms):
            continue
        if r.t - last_t < config.min_gap_s:
            continue
        side, (gx, gy) = min(
            goals.items(), key=lambda kv: np.hypot(r.x_m - kv[1][0], r.y_m - kv[1][1]))
        if np.hypot(r.x_m - gx, r.y_m - gy) > config.max_dist_m:
            continue
        if r.vx == 0 or np.sign(gx - r.x_m) != np.sign(r.vx):
            continue
        y_cross = r.y_m + r.vy * (gx - r.x_m) / r.vx
        if abs(y_cross - gy) > half_mouth + config.mouth_margin_m:
            continue
        prev = owners[(owners.index < frame) &
                      (owners.index >= frame - config.possession_memory_s * fps)]
        if prev.empty:
            continue
        who = prev.iloc[-1]
        player_id = int(who.owner_id)
        team = (team_of or {}).get(player_id, who.owner_team)

        upto = ball.loc[:frame]
        start = upto.iloc[-(k + 1)] if len(upto) > k else upto.iloc[0]
        x0, y0 = float(start.x_m), float(start.y_m)
        dx, dy = x0 - gx, y0 - gy
        angle = np.arctan2(config.goal_width_m * abs(dx),
                           dx ** 2 + dy ** 2 - half_mouth ** 2) % np.pi
        peak = ball.loc[frame:frame + 0.5 * fps, "speed"].clip(
            upper=config.max_speed_ms).max()
        shots.append(dict(
            frame=int(frame), t=round(float(r.t), 2), player_id=player_id,
            team=None if team is None else int(team), goal=side,
            x=round(x0, 1), y=round(y0, 1),
            distance_m=round(float(np.hypot(dx, dy)), 1),
            angle_deg=round(float(np.degrees(angle)), 1),
            ball_speed_kmh=round(float(peak) * 3.6, 1),
            outcome="on_target" if abs(y_cross - gy) <= half_mouth else "off_target"))
        last_t = r.t
    return pd.DataFrame(shots, columns=columns)
