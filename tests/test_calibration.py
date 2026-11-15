import numpy as np

from sports.common.calibration import points_well_spread
from sports.configs.soccer import SoccerPitchConfiguration

V = np.array(SoccerPitchConfiguration().vertices, dtype=float)


def test_rejects_points_on_one_line():
    # only the halfway line (vertices 14-17) is visible
    assert not points_well_spread(V[[13, 14, 15, 16]])


def test_accepts_halfway_line_plus_centre_circle():
    assert points_well_spread(V[[13, 14, 15, 16, 30, 31]])


def test_needs_four_points():
    assert not points_well_spread(V[[0, 5, 24]])
